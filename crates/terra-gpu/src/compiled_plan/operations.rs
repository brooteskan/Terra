use bytemuck::{Pod, Zeroable};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use terra_core::ids::OutputId;
use terra_core::layer::BlendMode;
use terra_core::mask::{Distribution, MaskAsset, MaskCombine, MaskOp, MaskSource};
use terra_core::terrain_plan::{FieldSlot, GroupCompositeMode, SeedSource};
use wgpu::util::DeviceExt;

use super::{GpuPlanResourceError, GpuPlanResources};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FillUniform {
    width: u32,
    height: u32,
    value: f32,
    _pad: f32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CopyUniform {
    width: u32,
    height: u32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GroupUniform {
    width: u32,
    height: u32,
    opacity: f32,
    blend_mode: u32,
    composite_mode: u32,
    _pad: [u32; 3],
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AuxUniform {
    width: u32,
    height: u32,
    opacity: f32,
    has_parent: u32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaskBakeUniform {
    width: u32,
    height: u32,
    mode: u32,
    dz: f32,
    dx: f32,
    value: f32,
    range_min: f32,
    range_max: f32,
    invert: f32,
    strength: f32,
    frequency: f32,
    seed: f32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaskProgramUniform {
    width: u32,
    height: u32,
    mode: u32,
    radius: u32,
    a: f32,
    b: f32,
    c: f32,
    _pad: f32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

struct Pipe {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// Authored parameters required by a `CompositeGroup` operation. The plan owns
/// field wiring and ordering; the document continues to own these mutable values.
#[derive(Debug, Clone, Copy)]
pub struct GpuGroupCompositeParams {
    pub blend: BlendMode,
    pub opacity: f32,
    pub mode: GroupCompositeMode,
}

/// Field-addressable plan primitives. The existing flat executor can retain its
/// `TexSlot` wrappers until #144 adopts this backend.
pub struct GpuPlanOperations {
    fill: Pipe,
    copy: Pipe,
    group: Pipe,
    aux: Pipe,
    mask_bake: Pipe,
    mask_program: Pipe,
    mask_scratch: RefCell<Option<MaskScratchSet>>,
    mask_scratch_allocations: Cell<u32>,
    mask_scratch_reuses: Cell<u32>,
}

impl GpuPlanOperations {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            fill: make_pipe(
                device,
                "compiled-plan-fill",
                include_str!("../shaders/compiled_plan_fill.wgsl"),
                &[uniform_entry(0), storage_write_entry(1)],
            ),
            copy: make_pipe(
                device,
                "compiled-plan-copy",
                include_str!("../shaders/compiled_plan_copy.wgsl"),
                &[
                    uniform_entry(0),
                    texture_read_entry(1),
                    storage_write_entry(2),
                ],
            ),
            group: make_pipe(
                device,
                "compiled-plan-group-composite",
                include_str!("../shaders/compiled_plan_group_composite.wgsl"),
                &[
                    uniform_entry(0),
                    texture_read_entry(1),
                    texture_read_entry(2),
                    texture_read_entry(3),
                    texture_read_entry(4),
                    storage_write_entry(5),
                ],
            ),
            aux: make_pipe(
                device,
                "compiled-plan-aux-composite",
                include_str!("../shaders/compiled_plan_aux_composite.wgsl"),
                &[
                    uniform_entry(0),
                    texture_read_entry(1),
                    texture_read_entry(2),
                    texture_read_entry(3),
                    storage_write_entry(4),
                ],
            ),
            mask_bake: make_pipe(
                device,
                "compiled-plan-mask-bake",
                include_str!("../shaders/mask_bake.wgsl"),
                &[
                    uniform_entry(0),
                    texture_read_entry(1),
                    storage_write_entry(2),
                ],
            ),
            mask_program: make_pipe(
                device,
                "compiled-plan-mask-program",
                include_str!("../shaders/mask_program.wgsl"),
                &[
                    uniform_entry(0),
                    texture_read_entry(1),
                    texture_read_entry(2),
                    storage_write_entry(3),
                ],
            ),
            mask_scratch: RefCell::new(None),
            mask_scratch_allocations: Cell::new(0),
            mask_scratch_reuses: Cell::new(0),
        }
    }

    /// Reset transient operation statistics before an evaluator records a plan.
    pub fn begin_evaluation(&self) {
        self.mask_scratch_allocations.set(0);
        self.mask_scratch_reuses.set(0);
    }

    /// Return mask scratch allocation and reuse counts for the active evaluation.
    pub fn mask_scratch_stats(&self) -> (u32, u32) {
        (
            self.mask_scratch_allocations.get(),
            self.mask_scratch_reuses.get(),
        )
    }

    pub fn seed_field(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        source: SeedSource,
        output: FieldSlot,
    ) -> Result<(), GpuPlanOperationError> {
        self.seed_field_region(device, encoder, resources, source, output, None)
    }

    pub fn seed_field_region(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        source: SeedSource,
        output: FieldSlot,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<(), GpuPlanOperationError> {
        match source {
            SeedSource::Zero => {
                self.fill_field_region(device, encoder, resources, output, 0.0, region)
            }
            SeedSource::Copy(source) | SeedSource::Selected(source) => {
                self.copy_field_region(device, encoder, resources, source, output, region)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_distribution(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        input_height: FieldSlot,
        output_mask: FieldSlot,
        distribution: &Distribution,
        mask_assets: &[MaskAsset],
        dx: f32,
        dz: f32,
    ) -> Result<(), GpuPlanOperationError> {
        self.evaluate_distribution_region(
            device,
            encoder,
            resources,
            input_height,
            output_mask,
            distribution,
            mask_assets,
            dx,
            dz,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_distribution_region(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        input_height: FieldSlot,
        output_mask: FieldSlot,
        distribution: &Distribution,
        mask_assets: &[MaskAsset],
        dx: f32,
        dz: f32,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<(), GpuPlanOperationError> {
        self.evaluate_distribution_resolved_region(
            device,
            encoder,
            resources,
            input_height,
            output_mask,
            distribution,
            mask_assets,
            &HashMap::new(),
            dx,
            dz,
            region,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_distribution_resolved_region(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        input_height: FieldSlot,
        output_mask: FieldSlot,
        distribution: &Distribution,
        mask_assets: &[MaskAsset],
        published_outputs: &HashMap<OutputId, FieldSlot>,
        dx: f32,
        dz: f32,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<(), GpuPlanOperationError> {
        if !distribution.nodes.is_empty() {
            return Err(GpuPlanOperationError::UnsupportedMaskNodes);
        }
        if !dx.is_finite() || !dz.is_finite() || dx <= 0.0 || dz <= 0.0 {
            return Err(GpuPlanOperationError::InvalidCellSize);
        }
        ensure_distinct(resources, &[input_height], &[output_mask])?;
        if distribution.entries.is_empty() {
            return self.fill_field_region(device, encoder, resources, output_mask, 1.0, region);
        }

        let key = resources.key();
        let region = normalized_region(key.width, key.height, region);
        {
            let mut scratch = self.mask_scratch.borrow_mut();
            if scratch
                .as_ref()
                .is_none_or(|scratch| scratch.width != key.width || scratch.height != key.height)
            {
                *scratch = Some(MaskScratchSet::new(device, key.width, key.height));
                self.mask_scratch_allocations.set(4);
            } else {
                self.mask_scratch_reuses
                    .set(self.mask_scratch_reuses.get().saturating_add(1));
            }
        }
        let scratch = self.mask_scratch.borrow();
        let scratch = scratch.as_ref().expect("mask scratch was realized");
        let accum_a = &scratch.accum_a;
        let accum_b = &scratch.accum_b;
        let work_a = &scratch.work_a;
        let work_b = &scratch.work_b;
        record_fill_view_region(
            device,
            encoder,
            &self.fill,
            &accum_a.view,
            key.width,
            key.height,
            1.0,
            region,
        );
        let mut accumulator_is_a = true;

        for entry in &distribution.entries {
            let asset = mask_assets
                .iter()
                .find(|asset| asset.id == entry.mask.id)
                .ok_or(GpuPlanOperationError::MissingMaskAsset(entry.mask.id))?;
            let (source_field, mode, value, range_min, range_max) = match asset.source {
                MaskSource::Constant(value) => (input_height, 0, value, 0.0, 1.0),
                MaskSource::Height { min, max } => (input_height, 1, 0.0, min, max),
                MaskSource::Slope { min_deg, max_deg } => (input_height, 2, 0.0, min_deg, max_deg),
                MaskSource::LayerOutput { output_id } => {
                    let Some(source) = published_outputs.get(&output_id).copied() else {
                        return Err(GpuPlanOperationError::UnsupportedMaskSource(format!(
                            "unresolved LayerOutput({output_id:?})"
                        )));
                    };
                    (source, 3, 0.0, 0.0, 1.0)
                }
                ref source => {
                    return Err(GpuPlanOperationError::UnsupportedMaskSource(format!(
                        "{source:?}"
                    )));
                }
            };
            self.record_mask_bake(
                device,
                encoder,
                resources.view(source_field)?,
                &work_a.view,
                MaskBakeUniform {
                    width: key.width,
                    height: key.height,
                    mode,
                    dz,
                    dx,
                    value,
                    range_min,
                    range_max,
                    invert: if entry.mask.invert { 1.0 } else { 0.0 },
                    strength: entry.mask.strength,
                    frequency: 0.0,
                    seed: 0.0,
                    region_x: region.0,
                    region_y: region.1,
                    region_w: region.2,
                    region_h: region.3,
                },
            );
            let mut entry_is_a = true;
            for operation in &asset.ops {
                let (source, destination) = if entry_is_a {
                    (&work_a.view, &work_b.view)
                } else {
                    (&work_b.view, &work_a.view)
                };
                self.record_mask_program(
                    device,
                    encoder,
                    source,
                    source,
                    destination,
                    mask_operation_uniform(*operation, key.width, key.height, region)?,
                );
                entry_is_a = !entry_is_a;
            }
            let entry_view = if entry_is_a {
                &work_a.view
            } else {
                &work_b.view
            };
            let (accumulator, destination) = if accumulator_is_a {
                (&accum_a.view, &accum_b.view)
            } else {
                (&accum_b.view, &accum_a.view)
            };
            self.record_mask_program(
                device,
                encoder,
                accumulator,
                entry_view,
                destination,
                mask_combine_uniform(entry.combine, key.width, key.height, region),
            );
            accumulator_is_a = !accumulator_is_a;
        }

        let accumulator = if accumulator_is_a {
            &accum_a.view
        } else {
            &accum_b.view
        };
        record_copy_views_region(
            device,
            encoder,
            &self.copy,
            accumulator,
            resources.view(output_mask)?,
            key.width,
            key.height,
            region,
        );
        Ok(())
    }

    pub fn fill_field(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        output: FieldSlot,
        value: f32,
    ) -> Result<(), GpuPlanOperationError> {
        self.fill_field_region(device, encoder, resources, output, value, None)
    }

    pub fn fill_field_region(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        output: FieldSlot,
        value: f32,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<(), GpuPlanOperationError> {
        if !value.is_finite() {
            return Err(GpuPlanOperationError::NonFiniteValue);
        }
        let key = resources.key();
        let region = normalized_region(key.width, key.height, region);
        let uniform = uniform_buffer(
            device,
            "compiled-plan-fill-uniform",
            &FillUniform {
                width: key.width,
                height: key.height,
                value,
                _pad: 0.0,
                region_x: region.0,
                region_y: region.1,
                region_w: region.2,
                region_h: region.3,
            },
        );
        let output = resources.view(output)?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compiled-plan-fill-bind-group"),
            layout: &self.fill.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output),
                },
            ],
        });
        dispatch(encoder, &self.fill, &bind_group, region.2, region.3);
        Ok(())
    }

    pub fn copy_field(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        source: FieldSlot,
        output: FieldSlot,
    ) -> Result<(), GpuPlanOperationError> {
        self.copy_field_region(device, encoder, resources, source, output, None)
    }

    pub fn copy_field_region(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        source: FieldSlot,
        output: FieldSlot,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<(), GpuPlanOperationError> {
        ensure_distinct(resources, &[source], &[output])?;
        let key = resources.key();
        let region = normalized_region(key.width, key.height, region);
        let uniform = uniform_buffer(
            device,
            "compiled-plan-copy-uniform",
            &CopyUniform {
                width: key.width,
                height: key.height,
                region_x: region.0,
                region_y: region.1,
                region_w: region.2,
                region_h: region.3,
                _pad: [0; 2],
            },
        );
        let source = resources.view(source)?;
        let output = resources.view(output)?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compiled-plan-copy-bind-group"),
            layout: &self.copy.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(output),
                },
            ],
        });
        dispatch(encoder, &self.copy, &bind_group, region.2, region.3);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn composite_group(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        parent: FieldSlot,
        private_seed: FieldSlot,
        child_output: FieldSlot,
        mask: FieldSlot,
        output: FieldSlot,
        params: GpuGroupCompositeParams,
    ) -> Result<(), GpuPlanOperationError> {
        self.composite_group_region(
            device,
            encoder,
            resources,
            parent,
            private_seed,
            child_output,
            mask,
            output,
            params,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn composite_group_region(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        parent: FieldSlot,
        private_seed: FieldSlot,
        child_output: FieldSlot,
        mask: FieldSlot,
        output: FieldSlot,
        params: GpuGroupCompositeParams,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<(), GpuPlanOperationError> {
        if !params.opacity.is_finite() {
            return Err(GpuPlanOperationError::NonFiniteOpacity);
        }
        ensure_distinct(
            resources,
            &[parent, private_seed, child_output, mask],
            &[output],
        )?;
        let blend_mode = crate::graph::gpu_blend_mode(params.blend)
            .ok_or(GpuPlanOperationError::UnsupportedBlend(params.blend))?;
        let key = resources.key();
        let region = normalized_region(key.width, key.height, region);
        let uniform = uniform_buffer(
            device,
            "compiled-plan-group-uniform",
            &GroupUniform {
                width: key.width,
                height: key.height,
                opacity: params.opacity,
                blend_mode,
                composite_mode: match params.mode {
                    GroupCompositeMode::Standard => 0,
                    GroupCompositeMode::BiomeHeightDelta => 1,
                },
                _pad: [0; 3],
                region_x: region.0,
                region_y: region.1,
                region_w: region.2,
                region_h: region.3,
            },
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compiled-plan-group-bind-group"),
            layout: &self.group.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                texture_entry(1, resources.view(parent)?),
                texture_entry(2, resources.view(private_seed)?),
                texture_entry(3, resources.view(child_output)?),
                texture_entry(4, resources.view(mask)?),
                texture_entry(5, resources.view(output)?),
            ],
        });
        dispatch(encoder, &self.group, &bind_group, region.2, region.3);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn composite_aux(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        parent: Option<FieldSlot>,
        child: FieldSlot,
        mask: FieldSlot,
        output: FieldSlot,
        opacity: f32,
    ) -> Result<(), GpuPlanOperationError> {
        self.composite_aux_region(
            device, encoder, resources, parent, child, mask, output, opacity, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn composite_aux_region(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        parent: Option<FieldSlot>,
        child: FieldSlot,
        mask: FieldSlot,
        output: FieldSlot,
        opacity: f32,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<(), GpuPlanOperationError> {
        if !opacity.is_finite() {
            return Err(GpuPlanOperationError::NonFiniteOpacity);
        }
        let parent_binding = parent.unwrap_or(child);
        let mut inputs = vec![child, mask];
        if let Some(parent) = parent {
            inputs.push(parent);
        }
        ensure_distinct(resources, &inputs, &[output])?;
        let key = resources.key();
        let region = normalized_region(key.width, key.height, region);
        let uniform = uniform_buffer(
            device,
            "compiled-plan-aux-uniform",
            &AuxUniform {
                width: key.width,
                height: key.height,
                opacity,
                has_parent: u32::from(parent.is_some()),
                region_x: region.0,
                region_y: region.1,
                region_w: region.2,
                region_h: region.3,
            },
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compiled-plan-aux-bind-group"),
            layout: &self.aux.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                texture_entry(1, resources.view(parent_binding)?),
                texture_entry(2, resources.view(child)?),
                texture_entry(3, resources.view(mask)?),
                texture_entry(4, resources.view(output)?),
            ],
        });
        dispatch(encoder, &self.aux, &bind_group, region.2, region.3);
        Ok(())
    }

    fn record_mask_bake(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input_height: &wgpu::TextureView,
        output: &wgpu::TextureView,
        value: MaskBakeUniform,
    ) {
        let width = value.region_w;
        let height = value.region_h;
        let uniform = uniform_buffer(device, "compiled-plan-mask-bake-uniform", &value);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compiled-plan-mask-bake-bind-group"),
            layout: &self.mask_bake.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                texture_entry(1, input_height),
                texture_entry(2, output),
            ],
        });
        dispatch(encoder, &self.mask_bake, &bind_group, width, height);
    }

    #[allow(clippy::too_many_arguments)]
    fn record_mask_program(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source_a: &wgpu::TextureView,
        source_b: &wgpu::TextureView,
        output: &wgpu::TextureView,
        value: MaskProgramUniform,
    ) {
        let width = value.region_w;
        let height = value.region_h;
        let uniform = uniform_buffer(device, "compiled-plan-mask-program-uniform", &value);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compiled-plan-mask-program-bind-group"),
            layout: &self.mask_program.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                texture_entry(1, source_a),
                texture_entry(2, source_b),
                texture_entry(3, output),
            ],
        });
        dispatch(encoder, &self.mask_program, &bind_group, width, height);
    }
}

struct ScratchTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct MaskScratchSet {
    width: u32,
    height: u32,
    accum_a: ScratchTexture,
    accum_b: ScratchTexture,
    work_a: ScratchTexture,
    work_b: ScratchTexture,
}

impl MaskScratchSet {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            accum_a: scalar_scratch(device, "compiled-plan-mask-accum-a", width, height),
            accum_b: scalar_scratch(device, "compiled-plan-mask-accum-b", width, height),
            work_a: scalar_scratch(device, "compiled-plan-mask-work-a", width, height),
            work_b: scalar_scratch(device, "compiled-plan-mask-work-b", width, height),
        }
    }
}

fn scalar_scratch(device: &wgpu::Device, label: &str, width: u32, height: u32) -> ScratchTexture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    ScratchTexture {
        _texture: texture,
        view,
    }
}

#[allow(clippy::too_many_arguments)]
fn record_fill_view_region(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipe: &Pipe,
    output: &wgpu::TextureView,
    width: u32,
    height: u32,
    value: f32,
    region: (u32, u32, u32, u32),
) {
    let uniform = uniform_buffer(
        device,
        "compiled-plan-fill-view-uniform",
        &FillUniform {
            width,
            height,
            value,
            _pad: 0.0,
            region_x: region.0,
            region_y: region.1,
            region_w: region.2,
            region_h: region.3,
        },
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compiled-plan-fill-view-bind-group"),
        layout: &pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            texture_entry(1, output),
        ],
    });
    dispatch(encoder, pipe, &bind_group, region.2, region.3);
}

#[allow(clippy::too_many_arguments)]
fn record_copy_views_region(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipe: &Pipe,
    source: &wgpu::TextureView,
    output: &wgpu::TextureView,
    width: u32,
    height: u32,
    region: (u32, u32, u32, u32),
) {
    let uniform = uniform_buffer(
        device,
        "compiled-plan-copy-view-uniform",
        &CopyUniform {
            width,
            height,
            region_x: region.0,
            region_y: region.1,
            region_w: region.2,
            region_h: region.3,
            _pad: [0; 2],
        },
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compiled-plan-copy-view-bind-group"),
        layout: &pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            texture_entry(1, source),
            texture_entry(2, output),
        ],
    });
    dispatch(encoder, pipe, &bind_group, region.2, region.3);
}

fn mask_operation_uniform(
    operation: MaskOp,
    width: u32,
    height: u32,
    region: (u32, u32, u32, u32),
) -> Result<MaskProgramUniform, GpuPlanOperationError> {
    let (mode, radius, a, b, c) = match operation {
        MaskOp::Add { amount } => (0, 0, amount, 0.0, 0.0),
        MaskOp::Subtract { amount } => (1, 0, amount, 0.0, 0.0),
        MaskOp::Multiply { amount } => (2, 0, amount, 0.0, 0.0),
        MaskOp::Min { value } => (3, 0, value, 0.0, 0.0),
        MaskOp::Max { value } => (4, 0, value, 0.0, 0.0),
        MaskOp::Invert => (5, 0, 0.0, 0.0, 0.0),
        MaskOp::Clamp { min, max } => (6, 0, min, max, 0.0),
        MaskOp::Levels {
            in_black,
            in_white,
            gamma,
        } => (7, 0, in_black, in_white, gamma),
        MaskOp::Smoothstep { edge0, edge1 } => (8, 0, edge0, edge1, 0.0),
        MaskOp::Blur { radius } if radius <= 16 => (9, radius, 0.0, 0.0, 0.0),
        MaskOp::Blur { radius } => return Err(GpuPlanOperationError::MaskBlurRadius(radius)),
        MaskOp::Remap { out_min, out_max } => (10, 0, out_min, out_max, 0.0),
    };
    Ok(MaskProgramUniform {
        width,
        height,
        mode,
        radius,
        a,
        b,
        c,
        _pad: 0.0,
        region_x: region.0,
        region_y: region.1,
        region_w: region.2,
        region_h: region.3,
    })
}

fn mask_combine_uniform(
    mode: MaskCombine,
    width: u32,
    height: u32,
    region: (u32, u32, u32, u32),
) -> MaskProgramUniform {
    let mode = match mode {
        MaskCombine::Multiply => 20,
        MaskCombine::Add => 21,
        MaskCombine::Subtract => 22,
        MaskCombine::Min => 23,
        MaskCombine::Max => 24,
        MaskCombine::Replace => 25,
        MaskCombine::Invert => 26,
        MaskCombine::PaintOverride => 27,
    };
    MaskProgramUniform {
        width,
        height,
        mode,
        radius: 0,
        a: 0.0,
        b: 0.0,
        c: 0.0,
        _pad: 0.0,
        region_x: region.0,
        region_y: region.1,
        region_w: region.2,
        region_h: region.3,
    }
}

fn normalized_region(
    width: u32,
    height: u32,
    region: Option<(u32, u32, u32, u32)>,
) -> (u32, u32, u32, u32) {
    debug_assert!(width > 0 && height > 0);
    let Some((x, y, region_width, region_height)) = region else {
        return (0, 0, width, height);
    };
    let x = x.min(width.saturating_sub(1));
    let y = y.min(height.saturating_sub(1));
    let region_width = region_width.max(1).min(width - x);
    let region_height = region_height.max(1).min(height - y);
    (x, y, region_width, region_height)
}

fn ensure_distinct(
    resources: &GpuPlanResources,
    inputs: &[FieldSlot],
    outputs: &[FieldSlot],
) -> Result<(), GpuPlanOperationError> {
    for input in inputs {
        let input_binding = resources
            .layout()
            .binding(*input)
            .ok_or(GpuPlanResourceError::MissingBinding(*input))?;
        for output in outputs {
            let output_binding = resources
                .layout()
                .binding(*output)
                .ok_or(GpuPlanResourceError::MissingBinding(*output))?;
            if input_binding.physical == output_binding.physical {
                return Err(GpuPlanOperationError::AliasedReadWrite {
                    input: *input,
                    output: *output,
                });
            }
        }
    }
    Ok(())
}

fn make_pipe(
    device: &wgpu::Device,
    label: &str,
    shader: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Pipe {
    terra_core::shader_progress::record_shader_compiled();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Pipe { pipeline, layout }
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipe: &Pipe,
    bind_group: &wgpu::BindGroup,
    width: u32,
    height: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("compiled-plan-operation"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipe.pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
}

fn uniform_buffer<T: Pod>(device: &wgpu::Device, label: &str, value: &T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_read_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_write_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::R32Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn texture_entry<'a>(binding: u32, view: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuPlanOperationError {
    #[error(transparent)]
    Resource(#[from] GpuPlanResourceError),
    #[error("compiled plan operation received a non-finite value")]
    NonFiniteValue,
    #[error("compiled plan group opacity is not finite")]
    NonFiniteOpacity,
    #[error("compiled plan group blend {0:?} is not supported by the GPU")]
    UnsupportedBlend(BlendMode),
    #[error("compiled plan operation aliases input {input:?} with output {output:?}")]
    AliasedReadWrite { input: FieldSlot, output: FieldSlot },
    #[error("compiled plan distribution nodes are not GPU-resident")]
    UnsupportedMaskNodes,
    #[error("compiled plan mask asset {0:?} is missing")]
    MissingMaskAsset(terra_core::mask::MaskId),
    #[error("compiled plan mask source {0} is not GPU-resident")]
    UnsupportedMaskSource(String),
    #[error("compiled plan mask blur radius {0} exceeds the GPU limit of 16")]
    MaskBlurRadius(u32),
    #[error("compiled plan mask evaluation requires finite positive cell sizes")]
    InvalidCellSize,
}
