//! GPU presentation equivalence for shared/full-to-regional transitions (#151).
//!
//! The renderer may borrow a complete engine height texture for a full present,
//! then switch to its local alternating slots for warm regional edits. Every leg
//! below compares that regional path with a full shared presentation of the same
//! complete source texture. The exact raster-frame comparison observes both height
//! displacement and lighting normals.

use terra_core::tiling::SampleRect;
use terra_core::{quality::PreviewQuality, terrain_plan::FieldSlot};
use terra_gpu::output_identity::*;
use terra_render::{GpuContext, HeightPresentGeom, TerrainRenderer, ViewportRendererMode};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const VIEW_W: u32 = 128;
const VIEW_H: u32 = 128;
const WORLD_SIZE: (f32, f32) = (1024.0, 1024.0);

struct HeightSource {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl HeightSource {
    fn new(gpu: &terra_test_gpu::TestGpu, width: u32, height: u32) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gpu-region-height-source"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let source = Self {
            texture,
            view,
            width,
            height,
        };
        let baseline: Vec<f32> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let u = x as f32 / width.saturating_sub(1).max(1) as f32;
                    let v = y as f32 / height.saturating_sub(1).max(1) as f32;
                    40.0 + u * 115.0
                        + v * 35.0
                        + (u * std::f32::consts::TAU * 3.0).sin() * 18.0
                        + (v * std::f32::consts::TAU * 2.0).cos() * 12.0
                })
            })
            .collect();
        source.write_rect(
            gpu,
            SampleRect {
                x: 0,
                y: 0,
                w: width,
                h: height,
            },
            &baseline,
        );
        source
    }

    fn geom(&self) -> HeightPresentGeom {
        HeightPresentGeom {
            width: self.width,
            height: self.height,
            world_size: WORLD_SIZE,
            // Fixed across the test so presentation routing cannot move the camera.
            height_range: (0.0, 320.0),
            dx: WORLD_SIZE.0 / self.width.max(1) as f32,
            dz: WORLD_SIZE.1 / self.height.max(1) as f32,
        }
    }

    fn write_patch(&self, gpu: &terra_test_gpu::TestGpu, rect: SampleRect, lift: f32) {
        let values: Vec<f32> = (0..rect.h)
            .flat_map(|y| {
                (0..rect.w).map(move |x| {
                    let u = x as f32 / rect.w.saturating_sub(1).max(1) as f32;
                    let v = y as f32 / rect.h.saturating_sub(1).max(1) as f32;
                    let ridge = (u * std::f32::consts::PI).sin() * (v * std::f32::consts::PI).sin();
                    65.0 + lift * ridge
                })
            })
            .collect();
        self.write_rect(gpu, rect, &values);
    }

    fn write_rect(&self, gpu: &terra_test_gpu::TestGpu, rect: SampleRect, values: &[f32]) {
        assert_eq!(values.len(), (rect.w * rect.h) as usize);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect.x,
                    y: rect.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(values),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(rect.w * 4),
                rows_per_image: Some(rect.h),
            },
            wgpu::Extent3d {
                width: rect.w,
                height: rect.h,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(std::iter::empty());
    }
}

fn renderer(ctx: &GpuContext) -> TerrainRenderer {
    let mut renderer = TerrainRenderer::new_headless(ctx, VIEW_W, VIEW_H);
    renderer.set_renderer_mode(ViewportRendererMode::Raster);
    renderer
}

fn output_identity(id: u64, coverage: GpuOutputCoverage) -> GpuTerrainOutputIdentity {
    GpuTerrainOutputIdentity {
        output: GpuOutputId(id),
        frame_id: id,
        generation: 7,
        evaluation_id: id,
        plan_revision: 11,
        requested_quality: PreviewQuality::Full,
        actual_quality: PreviewQuality::Full,
        intent: GpuEvaluationIntent::InteractiveLocal,
        selected_field: GpuSelectedFieldIdentity {
            selected: FieldSlot::from_index(1),
            expected_final: FieldSlot::from_index(1),
            resource_incarnation: GpuResourceIncarnation(20),
            physical_allocation: 0,
        },
        output_resource: GpuOutputResourceIdentity {
            device_generation: 3,
            incarnation: GpuResourceIncarnation(30),
            slot: GpuOutputSlot::Ping,
        },
        extent: (96, 96),
        coverage,
        completeness: GpuOutputCompleteness::Complete,
        invalidation: GpuInvalidationKind::Regional,
        last_write: GpuLastWriteIdentity {
            serial: GpuSubmissionSerial(id),
            completion: GpuSubmissionCompletion::Submitted,
        },
    }
}

fn expectations() -> terra_render::TerrainPresentationExpectations {
    terra_render::TerrainPresentationExpectations {
        plan_revision: 11,
        generation: 7,
        extent: (96, 96),
    }
}

fn drain_probes(
    gpu: &terra_test_gpu::TestGpu,
    renderer: &mut TerrainRenderer,
) -> Vec<terra_render::TerrainIntegrityProbeResult> {
    let mut completed = renderer.poll_integrity_probes();
    for _ in 0..8 {
        if !completed.is_empty() {
            break;
        }
        let _ = gpu.device.poll(wgpu::Maintain::Wait);
        completed.extend(renderer.poll_integrity_probes());
    }
    completed
}

fn assert_frames_equal(
    gpu: &terra_test_gpu::TestGpu,
    actual: &mut TerrainRenderer,
    oracle: &mut TerrainRenderer,
    label: &str,
) {
    // Presentation should frame both renderers identically; copy explicitly so
    // this test isolates height/normal state from any future camera policy change.
    oracle.camera = actual.camera.clone();
    let actual_target = gpu.target(VIEW_W, VIEW_H, FORMAT);
    let oracle_target = gpu.target(VIEW_W, VIEW_H, FORMAT);
    actual.render_to_view(&actual_target.view, VIEW_W, VIEW_H);
    oracle.render_to_view(&oracle_target.view, VIEW_W, VIEW_H);
    let actual_pixels = gpu.read_rgba8(&actual_target);
    let oracle_pixels = gpu.read_rgba8(&oracle_target);

    let mut count = 0u32;
    let mut first = None;
    for y in 0..VIEW_H {
        for x in 0..VIEW_W {
            let a = actual_pixels.get(x, y);
            let b = oracle_pixels.get(x, y);
            if a != b {
                count += 1;
                if first.is_none() {
                    first = Some((x, y, a, b));
                }
            }
        }
    }
    assert_eq!(
        count, 0,
        "{label}: regional presentation differs from full oracle in {count} pixel(s); \
         first mismatch {first:?}"
    );
}

#[test]
fn gpu_regional_present_matches_full_across_baseline_transitions() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let ctx = GpuContext::new(gpu.device.clone(), gpu.queue.clone(), FORMAT);

    let source = HeightSource::new(gpu, 96, 96);
    let mut actual = renderer(&ctx);
    let mut oracle = renderer(&ctx);
    actual.present_gpu_height_shared(&source.texture, &source.view, source.geom(), None);
    oracle.present_gpu_height_shared(&source.texture, &source.view, source.geom(), None);

    // Shared full -> first partial: the local baseline must be rebuilt from the
    // complete source rather than seeded from zero/stale internal height slots.
    let west = SampleRect {
        x: 12,
        y: 18,
        w: 19,
        h: 17,
    };
    source.write_patch(gpu, west, 125.0);
    actual.present_gpu_height_region(&source.texture, source.geom(), Some(west));
    oracle.present_gpu_height_shared(&source.texture, &source.view, source.geom(), None);
    assert_frames_equal(gpu, &mut actual, &mut oracle, "shared -> regional");

    // A disjoint second edit alternates the renderer slot. The first edit's height
    // and normals must remain current outside this new padded dirty rectangle.
    let east = SampleRect {
        x: 65,
        y: 58,
        w: 18,
        h: 21,
    };
    source.write_patch(gpu, east, 155.0);
    actual.present_gpu_height_region(&source.texture, source.geom(), Some(east));
    oracle.present_gpu_height_shared(&source.texture, &source.view, source.geom(), None);
    assert_frames_equal(gpu, &mut actual, &mut oracle, "second disjoint regional");

    // A regional call is also legal as the very first presentation. It must not
    // expose the renderer's zero-initialized slots outside the requested rect.
    let mut first_actual = renderer(&ctx);
    let mut first_oracle = renderer(&ctx);
    first_actual.present_gpu_height_region(&source.texture, source.geom(), Some(west));
    first_oracle.present_gpu_height_shared(&source.texture, &source.view, source.geom(), None);
    assert_frames_equal(
        gpu,
        &mut first_actual,
        &mut first_oracle,
        "first regional present",
    );

    // Resize invalidates both old slots. A partial request at the new size must
    // promote once and reconstruct the complete larger field.
    let resized = HeightSource::new(gpu, 144, 128);
    let resized_dirty = SampleRect {
        x: 91,
        y: 27,
        w: 23,
        h: 19,
    };
    resized.write_patch(gpu, resized_dirty, 145.0);
    actual.present_gpu_height_region(&resized.texture, resized.geom(), Some(resized_dirty));
    oracle.present_gpu_height_shared(&resized.texture, &resized.view, resized.geom(), None);
    assert_frames_equal(gpu, &mut actual, &mut oracle, "resize -> regional");
}

#[test]
fn traced_regional_transition_detects_outside_region_corruption_asynchronously() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let ctx = GpuContext::new(gpu.device.clone(), gpu.queue.clone(), FORMAT);
    let source = HeightSource::new(gpu, 96, 96);
    let mut renderer = renderer(&ctx);

    let full = output_identity(1, GpuOutputCoverage::WholeField);
    let record = renderer.present_gpu_height_shared_traced(
        &source.texture,
        &source.view,
        source.geom(),
        None,
        full,
        expectations(),
    );
    assert_eq!(record.shadow_diagnostic, None);
    assert!(drain_probes(gpu, &mut renderer)
        .into_iter()
        .all(|result| result.passed));

    let west = SampleRect {
        x: 12,
        y: 18,
        w: 19,
        h: 17,
    };
    source.write_patch(gpu, west, 125.0);
    let first_patch = output_identity(
        2,
        GpuOutputCoverage::Patch {
            rect: west,
            expected_base: Some(full.output),
        },
    );
    let record = renderer.present_gpu_height_region_traced(
        &source.texture,
        source.geom(),
        Some(west),
        first_patch,
        expectations(),
    );
    assert_eq!(
        record.actual_mode,
        terra_render::TerrainPresentationMode::FullCopy
    );
    assert_eq!(record.shadow_diagnostic, None);
    assert!(drain_probes(gpu, &mut renderer)
        .into_iter()
        .all(|result| result.passed));

    let east = SampleRect {
        x: 65,
        y: 58,
        w: 18,
        h: 21,
    };
    source.write_patch(gpu, east, 155.0);
    // Probe index zero deterministically maps to (31, 61) for a 96x96 field,
    // outside `east`; corrupt that point after the bounded edit.
    source.write_rect(
        gpu,
        SampleRect {
            x: 31,
            y: 61,
            w: 1,
            h: 1,
        },
        &[10_000.0],
    );
    let second_patch = output_identity(
        3,
        GpuOutputCoverage::Patch {
            rect: east,
            expected_base: Some(first_patch.output),
        },
    );
    let record = renderer.present_gpu_height_region_traced(
        &source.texture,
        source.geom(),
        Some(east),
        second_patch,
        expectations(),
    );
    assert_eq!(
        record.actual_mode,
        terra_render::TerrainPresentationMode::RegionalCopy
    );
    assert_eq!(record.shadow_diagnostic, None);
    let results = drain_probes(gpu, &mut renderer);
    let failure = results
        .into_iter()
        .find(|result| !result.passed)
        .expect("outside-region corruption must be detected");
    assert_eq!(failure.first_failing_probe, Some(0));
    assert!(failure.max_delta > 1_000.0);
}

#[test]
fn traced_regional_transition_recovers_after_an_unpresented_output() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let ctx = GpuContext::new(gpu.device.clone(), gpu.queue.clone(), FORMAT);
    let source = HeightSource::new(gpu, 96, 96);
    let mut actual = renderer(&ctx);
    let mut oracle = renderer(&ctx);

    // Establish a coherent renderer-local output 37. Output 38 is then written
    // into the complete engine source but intentionally never presented, matching
    // an evaluation superseded while the camera/input generation advances.
    let output_37 = output_identity(37, GpuOutputCoverage::WholeField);
    let initial = actual.present_gpu_height_region_traced(
        &source.texture,
        source.geom(),
        None,
        output_37,
        expectations(),
    );
    assert_eq!(
        initial.actual_mode,
        terra_render::TerrainPresentationMode::FullCopy
    );

    let skipped_rect = SampleRect {
        x: 12,
        y: 18,
        w: 19,
        h: 17,
    };
    source.write_patch(gpu, skipped_rect, 125.0);
    let _unpresented_output_38 = output_identity(
        38,
        GpuOutputCoverage::Patch {
            rect: skipped_rect,
            expected_base: Some(output_37.output),
        },
    );

    let current_rect = SampleRect {
        x: 65,
        y: 58,
        w: 18,
        h: 21,
    };
    source.write_patch(gpu, current_rect, 155.0);
    let output_39 = output_identity(
        39,
        GpuOutputCoverage::Patch {
            rect: current_rect,
            expected_base: Some(GpuOutputId(38)),
        },
    );
    let recovered = actual.present_gpu_height_region_traced(
        &source.texture,
        source.geom(),
        Some(current_rect),
        output_39,
        expectations(),
    );

    assert_eq!(
        recovered.requested_mode,
        terra_render::TerrainPresentationMode::RegionalCopy
    );
    assert_eq!(
        recovered.actual_mode,
        terra_render::TerrainPresentationMode::FullCopy
    );
    assert_eq!(recovered.shadow_diagnostic, None);
    assert_eq!(recovered.baseline_after.identity.output, output_39.output);

    oracle.present_gpu_height_shared(&source.texture, &source.view, source.geom(), None);
    assert_frames_equal(gpu, &mut actual, &mut oracle, "unpresented output recovery");

    // Recovery is one-shot: the next compatible patch returns to the bounded path.
    let next_rect = SampleRect {
        x: 42,
        y: 23,
        w: 11,
        h: 13,
    };
    source.write_patch(gpu, next_rect, 95.0);
    let output_40 = output_identity(
        40,
        GpuOutputCoverage::Patch {
            rect: next_rect,
            expected_base: Some(output_39.output),
        },
    );
    let resumed = actual.present_gpu_height_region_traced(
        &source.texture,
        source.geom(),
        Some(next_rect),
        output_40,
        expectations(),
    );
    assert_eq!(
        resumed.actual_mode,
        terra_render::TerrainPresentationMode::RegionalCopy
    );
    assert_eq!(resumed.shadow_diagnostic, None);
}
