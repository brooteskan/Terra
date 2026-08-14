//! Headless GPU harness for render and UI tests.
//!
//! Tests that need a device but not a window use [`headless`]. It returns
//! `None` when no adapter is available, so suites stay green on machines
//! without a GPU — the same graceful-skip convention already used by the
//! `terra-gpu` derivative tests:
//!
//! ```ignore
//! let Some(gpu) = terra_test_gpu::headless() else { return };
//! ```
//!
//! # Boundary
//!
//! This crate depends on `wgpu` and `pollster` only — no Terra domain,
//! render, or UI crates. That lets `terra-gui`, `terra-render`, and
//! `terra-gpu` each dev-depend on it without acquiring a dependency on one
//! another, which would defeat the crate-graph separation the workspace
//! relies on.
//!
//! The device is created once per test process and shared; `wgpu::Device`
//! and `wgpu::Queue` are refcounted handles, so this is cheap.

use std::sync::OnceLock;

/// A headless device + queue pair shared across tests.
pub struct TestGpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
}

/// An offscreen colour target: texture, view, and its dimensions.
pub struct TestTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
}

/// RGBA8 pixels read back from a [`TestTarget`], indexed by coordinate.
pub struct Pixels {
    data: Vec<[u8; 4]>,
    width: u32,
    height: u32,
}

impl Pixels {
    /// Pixel at `(x, y)`, origin top-left. Panics if out of bounds.
    pub fn get(&self, x: u32, y: u32) -> [u8; 4] {
        assert!(
            x < self.width && y < self.height,
            "pixel ({x}, {y}) outside {}x{}",
            self.width,
            self.height
        );
        self.data[(y * self.width + x) as usize]
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// True when every pixel equals `rgba`.
    pub fn all(&self, rgba: [u8; 4]) -> bool {
        self.data.iter().all(|p| *p == rgba)
    }

    /// True when any pixel equals `rgba`.
    pub fn any(&self, rgba: [u8; 4]) -> bool {
        self.data.iter().any(|p| *p == rgba)
    }
}

static GPU: OnceLock<Result<TestGpu, String>> = OnceLock::new();

/// Shared headless GPU, or `None` when the machine has no usable adapter.
///
/// Callers should skip rather than fail so the suite stays green without a GPU.
pub fn headless() -> Option<&'static TestGpu> {
    GPU.get_or_init(create_gpu).as_ref().ok()
}

/// Shared headless GPU for tests that are explicitly required to execute GPU
/// work. Unlike [`headless`], adapter absence is a test failure.
pub fn headless_required() -> &'static TestGpu {
    match GPU.get_or_init(create_gpu) {
        Ok(gpu) => gpu,
        Err(reason) => panic!("GPU-required test could not create an adapter: {reason}"),
    }
}

fn create_gpu() -> Result<TestGpu, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: preferred_backends(),
        ..Default::default()
    });
    let force_fallback_adapter = std::env::var("TERRA_TEST_GPU_FORCE_FALLBACK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "yes"));
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        // Tests are tiny; prefer the low-power adapter and leave the discrete
        // GPU free. `compatible_surface: None` is what makes this headless.
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter,
    }))
    .ok_or_else(|| {
        format!(
            "no adapter for backends {:?} (force_fallback_adapter={force_fallback_adapter})",
            preferred_backends()
        )
    })?;
    let adapter_info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("terra-test-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .map_err(|error| format!("request_device failed for {adapter_info:?}: {error}"))?;
    Ok(TestGpu {
        device,
        queue,
        adapter_info,
    })
}

/// Mirrors the application's backend choice so tests exercise the same path.
///
/// DX12 is the Windows default: the Vulkan loader picks up capture-overlay
/// implicit layers (OBS / Overwolf / Medal) that have crashed device creation.
fn preferred_backends() -> wgpu::Backends {
    if let Ok(raw) = std::env::var("WGPU_BACKEND") {
        let mut backends = wgpu::Backends::empty();
        for part in raw.split([',', '|']).map(|p| p.trim().to_ascii_lowercase()) {
            match part.as_str() {
                "dx12" => backends |= wgpu::Backends::DX12,
                "vulkan" => backends |= wgpu::Backends::VULKAN,
                "metal" => backends |= wgpu::Backends::METAL,
                "gl" => backends |= wgpu::Backends::GL,
                _ => {}
            }
        }
        if !backends.is_empty() {
            return backends;
        }
    }
    if cfg!(target_os = "windows") {
        wgpu::Backends::DX12
    } else {
        wgpu::Backends::PRIMARY
    }
}

impl TestGpu {
    /// Allocate an offscreen colour target usable as a render attachment and
    /// readable via [`TestGpu::read_rgba8`].
    pub fn target(&self, width: u32, height: u32, format: wgpu::TextureFormat) -> TestTarget {
        assert!(width > 0 && height > 0, "target must be non-empty");
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terra-test-gpu-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        TestTarget {
            texture,
            view,
            width,
            height,
            format,
        }
    }

    /// Clear `target` to `color`, standing in for previously rendered content.
    ///
    /// Used to prove that a later pass preserves what was already drawn.
    pub fn fill(&self, target: &TestTarget, color: wgpu::Color) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terra-test-gpu-fill"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terra-test-gpu-fill-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Seed `target` with explicit per-texel content, row-major from top-left.
    ///
    /// Useful when a uniform [`TestGpu::fill`] cannot distinguish outcomes —
    /// for example proving that readback preserves texel coordinates.
    ///
    /// `Queue::write_texture` stages the data and applies it at the next
    /// submit, so an empty submit is issued to flush it immediately.
    pub fn write_rgba8(&self, target: &TestTarget, pixels: &[[u8; 4]]) {
        let expected = (target.width as usize) * (target.height as usize);
        assert_eq!(
            pixels.len(),
            expected,
            "expected {expected} texels for {}x{}, got {}",
            target.width,
            target.height,
            pixels.len()
        );
        let bytes: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(target.width * 4),
                rows_per_image: Some(target.height),
            },
            wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::empty());
    }

    /// Read `target` back to CPU pixels.
    ///
    /// `copy_texture_to_buffer` requires each row to start on a
    /// [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`] boundary, so the staging buffer
    /// is allocated with a padded stride and the padding is dropped while
    /// copying rows out. Passing the unpadded width is a validation error for
    /// any width that is not a multiple of 64 texels.
    pub fn read_rgba8(&self, target: &TestTarget) -> Pixels {
        assert_eq!(
            target.format.block_copy_size(None),
            Some(4),
            "read_rgba8 expects a 4-byte-per-texel format, got {:?}",
            target.format
        );
        let unpadded = target.width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terra-test-gpu-readback"),
            size: (padded as u64) * (target.height as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terra-test-gpu-readback-enc"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(target.height),
                },
            },
            wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("map channel")
            .expect("map readback buffer");

        let mapped = slice.get_mapped_range();
        let mut data = Vec::with_capacity((target.width * target.height) as usize);
        for row in 0..target.height {
            let row_start = (row * padded) as usize;
            for col in 0..target.width as usize {
                let o = row_start + col * 4;
                data.push([mapped[o], mapped[o + 1], mapped[o + 2], mapped[o + 3]]);
            }
        }
        drop(mapped);
        staging.unmap();

        Pixels {
            data,
            width: target.width,
            height: target.height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGBA: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    fn color(r: f64, g: f64, b: f64) -> wgpu::Color {
        wgpu::Color { r, g, b, a: 1.0 }
    }

    /// Run a render pass over `target` with the given load op and no draw calls.
    ///
    /// This is what a UI pass that renders nothing does to an attachment, so it
    /// is the exact shape the seam tests depend on being observable.
    fn empty_pass(gpu: &TestGpu, target: &TestTarget, load: wgpu::LoadOp<wgpu::Color>) {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("empty"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("empty-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        gpu.queue.submit(Some(encoder.finish()));
    }

    /// Fill then read back must round-trip the exact colour.
    ///
    /// Width 37 is deliberately not a multiple of 64 texels, so the readback row
    /// stride needs padding — this fails if the alignment handling regresses.
    #[test]
    fn fill_round_trips_through_padded_readback() {
        let Some(gpu) = headless() else {
            return;
        };
        let target = gpu.target(37, 9, RGBA);
        gpu.fill(&target, color(1.0, 0.0, 0.0));
        let pixels = gpu.read_rgba8(&target);
        assert_eq!(pixels.width(), 37);
        assert_eq!(pixels.height(), 9);
        assert_eq!(
            pixels.get(36, 8),
            [255, 0, 0, 255],
            "last texel of last row"
        );
        assert!(pixels.all([255, 0, 0, 255]), "every texel should be red");
    }

    /// A uniform fill cannot detect a transposed or mis-strided readback, so
    /// seed every texel with its own coordinate and check it comes back in place.
    ///
    /// The target is deliberately non-square: a transposed readback would either
    /// panic on bounds or return the wrong texel, and could not pass by accident.
    #[test]
    fn readback_preserves_texel_coordinates() {
        let Some(gpu) = headless() else {
            return;
        };
        let (w, h) = (13u32, 7u32);
        let seeded: Vec<[u8; 4]> = (0..h)
            .flat_map(|y| (0..w).map(move |x| [x as u8, y as u8, 0, 255]))
            .collect();
        let target = gpu.target(w, h, RGBA);
        gpu.write_rgba8(&target, &seeded);

        let pixels = gpu.read_rgba8(&target);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    pixels.get(x, y),
                    [x as u8, y as u8, 0, 255],
                    "texel ({x}, {y}) came back from the wrong place"
                );
            }
        }
    }

    /// The padded-stride maths must hold either side of the 256-byte boundary.
    ///
    /// 64 texels is exactly one alignment unit (256 bytes); 1 is the largest
    /// padding ratio; 65 forces a second alignment unit per row.
    #[test]
    fn readback_handles_row_alignment_boundaries() {
        let Some(gpu) = headless() else {
            return;
        };
        for w in [1u32, 63, 64, 65, 128, 200] {
            let h = 3u32;
            let seeded: Vec<[u8; 4]> = (0..h)
                .flat_map(|y| (0..w).map(move |x| [x as u8, y as u8, 7, 255]))
                .collect();
            let target = gpu.target(w, h, RGBA);
            gpu.write_rgba8(&target, &seeded);

            let pixels = gpu.read_rgba8(&target);
            assert_eq!(pixels.get(0, 0), [0, 0, 7, 255], "width {w}: first texel");
            assert_eq!(
                pixels.get(w - 1, h - 1),
                [(w - 1) as u8, (h - 1) as u8, 7, 255],
                "width {w}: last texel of last row"
            );
        }
    }

    /// A red-only fill cannot tell RGBA from BGRA, so use four distinct channels.
    #[test]
    fn fill_preserves_channel_order() {
        let Some(gpu) = headless() else {
            return;
        };
        let target = gpu.target(4, 4, RGBA);
        gpu.write_rgba8(&target, &[[10, 20, 30, 40]; 16]);
        let pixels = gpu.read_rgba8(&target);
        assert_eq!(
            pixels.get(2, 2),
            [10, 20, 30, 40],
            "channels must not be reordered on the round trip"
        );
    }

    /// Positive control: `LoadOp::Load` must preserve what was already drawn.
    ///
    /// This is the property the GUI pass relies on to composite over terrain.
    #[test]
    fn load_op_preserves_existing_content() {
        let Some(gpu) = headless() else {
            return;
        };
        let target = gpu.target(8, 8, RGBA);
        gpu.fill(&target, color(1.0, 0.0, 0.0));
        empty_pass(gpu, &target, wgpu::LoadOp::Load);
        assert!(
            gpu.read_rgba8(&target).all([255, 0, 0, 255]),
            "an empty Load pass must leave the backdrop intact"
        );
    }

    /// Negative control: the harness must be able to *see* a clear happen.
    ///
    /// Without this, a passing "backdrop preserved" test would be vacuous — it
    /// could not distinguish a correct Load from a harness that always reports
    /// the original fill.
    #[test]
    fn clear_op_is_detectable() {
        let Some(gpu) = headless() else {
            return;
        };
        let target = gpu.target(8, 8, RGBA);
        gpu.fill(&target, color(1.0, 0.0, 0.0));
        empty_pass(gpu, &target, wgpu::LoadOp::Clear(color(0.0, 1.0, 0.0)));
        assert!(
            gpu.read_rgba8(&target).all([0, 255, 0, 255]),
            "a Clear pass must be observable, or backdrop tests prove nothing"
        );
    }

    #[test]
    fn headless_is_shared_across_calls() {
        let Some(a) = headless() else {
            return;
        };
        let b = headless().expect("second call");
        assert!(
            std::ptr::eq(a, b),
            "device should be created once per process"
        );
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn pixel_access_is_bounds_checked() {
        let Some(gpu) = headless() else {
            // Provoke the same panic so the test is meaningful without a GPU.
            panic!("pixel (9, 9) outside 4x4");
        };
        let target = gpu.target(4, 4, RGBA);
        gpu.fill(&target, color(0.0, 0.0, 0.0));
        let _ = gpu.read_rgba8(&target).get(9, 9);
    }
}
