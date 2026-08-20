use std::collections::HashSet;

use terra_core::terrain_plan::{
    CompiledTerrainPlan, FieldSlot, PlanOpId, PlanStructureSignature, TerrainOpKind,
};

/// One physical scalar texture allocated for one or more logical fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuPhysicalFieldId(usize);

impl GpuPhysicalFieldId {
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Whether content must survive between plan executions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFieldResidency {
    Persistent,
    Transient,
}

/// Binding from one plan-local field to physical storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuFieldBinding {
    pub physical: GpuPhysicalFieldId,
    pub residency: GpuFieldResidency,
}

/// Allocation metadata. Transient allocations can own several fields whose
/// inclusive operation lifetimes do not overlap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPlanAllocation {
    pub id: GpuPhysicalFieldId,
    pub residency: GpuFieldResidency,
    pub fields: Vec<FieldSlot>,
}

/// Deterministic logical-to-physical layout for one compiled plan structure.
#[derive(Debug, Clone)]
pub struct GpuPlanResourceLayout {
    structure_signature: PlanStructureSignature,
    bindings: Vec<Option<GpuFieldBinding>>,
    allocations: Vec<GpuPlanAllocation>,
    persistent_fields: usize,
    transient_fields: usize,
}

impl GpuPlanResourceLayout {
    pub fn build(plan: &CompiledTerrainPlan) -> Result<Self, GpuPlanResourceError> {
        let field_count = plan.fields().len();
        let mut persistent = HashSet::new();
        let mut published = HashSet::new();
        let mut private = HashSet::new();

        persistent.insert(plan.final_height());
        for (operation_index, operation) in plan.operations().iter().enumerate() {
            match &operation.kind {
                TerrainOpKind::Seed { output, .. }
                    if operation.origin == terra_core::terrain_plan::PlanOrigin::Root =>
                {
                    persistent.insert(*output);
                }
                TerrainOpKind::RunLayerKernel {
                    output_candidate,
                    output_fields,
                    ..
                } => {
                    persistent.insert(*output_candidate);
                    persistent.extend(output_fields.iter().copied());
                }
                TerrainOpKind::CompositeLayer { output, .. } => {
                    persistent.insert(*output);
                }
                TerrainOpKind::CompositeGroup {
                    private_seed,
                    child_output,
                    output,
                    ..
                } => {
                    persistent.insert(*output);

                    let seed_operation = plan
                        .provenance()
                        .producer_of(*private_seed)
                        .ok_or(GpuPlanResourceError::MissingProducer(*private_seed))?
                        .index();
                    for field in plan.fields() {
                        let Some(lifetime) = plan.analysis().lifetime(field.slot) else {
                            return Err(GpuPlanResourceError::MissingLifetime(field.slot));
                        };
                        if lifetime.first_operation.index() >= seed_operation
                            && lifetime.last_operation.index() <= operation_index
                            && field.slot != *output
                        {
                            private.insert(field.slot);
                        }
                    }
                    // A group mask is evaluated against the parent before the seed;
                    // it is scratch, but not part of the isolated private slice.
                    private.insert(*private_seed);
                    private.insert(*child_output);
                }
                TerrainOpKind::CompositeAuxField { composite, .. } => {
                    persistent.insert(composite.output);
                }
                TerrainOpKind::PublishOutput { source, .. } => {
                    published.insert(*source);
                }
                _ => {}
            }
        }

        // Private working fields are reconstructable scratch. Final and named
        // outputs override that rule because they are observable after execution.
        for field in &private {
            persistent.remove(field);
        }
        persistent.extend(published);
        persistent.insert(plan.final_height());

        let mut bindings = vec![None; field_count];
        let mut allocations = Vec::new();
        let mut persistent_fields = 0;
        let mut transient_fields = 0;

        for field in plan.fields() {
            if !plan.analysis().field_is_live(field.slot) || !persistent.contains(&field.slot) {
                continue;
            }
            persistent_fields += 1;
            let id = GpuPhysicalFieldId(allocations.len());
            allocations.push(GpuPlanAllocation {
                id,
                residency: GpuFieldResidency::Persistent,
                fields: vec![field.slot],
            });
            bindings[field.slot.index()] = Some(GpuFieldBinding {
                physical: id,
                residency: GpuFieldResidency::Persistent,
            });
        }

        let mut transient_lifetimes = Vec::new();
        for field in plan.fields() {
            if !plan.analysis().field_is_live(field.slot) || persistent.contains(&field.slot) {
                continue;
            }
            let lifetime = plan
                .analysis()
                .lifetime(field.slot)
                .ok_or(GpuPlanResourceError::MissingLifetime(field.slot))?;
            if lifetime.first_operation.index() > lifetime.last_operation.index() {
                return Err(GpuPlanResourceError::InvalidLifetime(field.slot));
            }
            transient_lifetimes.push((
                field.slot,
                lifetime.first_operation.index(),
                lifetime.last_operation.index(),
            ));
        }
        transient_lifetimes.sort_by_key(|(slot, first, _)| (*first, slot.index()));

        // (physical id, inclusive last use). Strictly-less-than is essential:
        // an input dying in operation N cannot alias an output born in N.
        let mut transient_allocations: Vec<(GpuPhysicalFieldId, usize)> = Vec::new();
        for (field, first, last) in transient_lifetimes {
            transient_fields += 1;
            let reusable = transient_allocations
                .iter_mut()
                .filter(|(_, previous_last)| *previous_last < first)
                .min_by_key(|(id, _)| id.index());
            let id = if let Some((id, previous_last)) = reusable {
                *previous_last = last;
                *id
            } else {
                let id = GpuPhysicalFieldId(allocations.len());
                allocations.push(GpuPlanAllocation {
                    id,
                    residency: GpuFieldResidency::Transient,
                    fields: Vec::new(),
                });
                transient_allocations.push((id, last));
                id
            };
            allocations[id.index()].fields.push(field);
            bindings[field.index()] = Some(GpuFieldBinding {
                physical: id,
                residency: GpuFieldResidency::Transient,
            });
        }

        let layout = Self {
            structure_signature: plan.structure_signature(),
            bindings,
            allocations,
            persistent_fields,
            transient_fields,
        };
        layout.validate_aliases(plan)?;
        Ok(layout)
    }

    pub const fn structure_signature(&self) -> PlanStructureSignature {
        self.structure_signature
    }

    pub fn binding(&self, field: FieldSlot) -> Option<GpuFieldBinding> {
        self.bindings.get(field.index()).copied().flatten()
    }

    pub fn allocations(&self) -> &[GpuPlanAllocation] {
        &self.allocations
    }

    pub const fn persistent_field_count(&self) -> usize {
        self.persistent_fields
    }

    pub const fn transient_field_count(&self) -> usize {
        self.transient_fields
    }

    pub fn transient_allocation_count(&self) -> usize {
        self.allocations
            .iter()
            .filter(|allocation| allocation.residency == GpuFieldResidency::Transient)
            .count()
    }

    /// Expand a semantic dirty-operation selection with producers required to
    /// reconstruct transient inputs. Persistent checkpoints stop the walk.
    /// This is backend materialization, not semantic dirty propagation.
    pub fn materialization_operations(
        &self,
        plan: &CompiledTerrainPlan,
        requested: &[PlanOpId],
    ) -> Vec<PlanOpId> {
        let mut selected: HashSet<PlanOpId> = requested.iter().copied().collect();
        let mut pending: Vec<PlanOpId> = requested.to_vec();
        while let Some(operation) = pending.pop() {
            for field in plan.analysis().inputs(operation) {
                let Some(binding) = self.binding(*field) else {
                    continue;
                };
                if binding.residency == GpuFieldResidency::Persistent {
                    continue;
                }
                let Some(producer) = plan.provenance().producer_of(*field) else {
                    continue;
                };
                if selected.insert(producer) {
                    pending.push(producer);
                }
            }
        }
        let mut selected: Vec<_> = selected.into_iter().collect();
        selected.sort_by_key(|operation| operation.index());
        selected
    }

    fn validate_aliases(&self, plan: &CompiledTerrainPlan) -> Result<(), GpuPlanResourceError> {
        for (operation_index, _) in plan.operations().iter().enumerate() {
            let operation = terra_core::terrain_plan::PlanOpId::from_index(operation_index);
            if !plan.analysis().operation_is_live(operation) {
                continue;
            }
            let inputs = plan
                .analysis()
                .inputs(operation)
                .iter()
                .copied()
                .filter(|field| plan.analysis().field_is_live(*field))
                .map(|field| {
                    self.binding(field)
                        .map(|binding| (field, binding.physical))
                        .ok_or(GpuPlanResourceError::MissingBinding(field))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = plan
                .analysis()
                .outputs(operation)
                .iter()
                .copied()
                .filter(|field| plan.analysis().field_is_live(*field))
                .map(|field| {
                    self.binding(field)
                        .map(|binding| (field, binding.physical))
                        .ok_or(GpuPlanResourceError::MissingBinding(field))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let input_physical: Vec<_> = inputs.iter().map(|(_, physical)| *physical).collect();
            let output_physical: Vec<_> = outputs.iter().map(|(_, physical)| *physical).collect();
            if let Some(hazard) = operation_alias_hazard(&input_physical, &output_physical) {
                let field = match hazard {
                    OperationAliasHazard::ReadWrite { input } => inputs[input].0,
                    OperationAliasHazard::WriteWrite { output } => outputs[output].0,
                };
                return Err(GpuPlanResourceError::AliasingHazard {
                    operation: operation_index,
                    field,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationAliasHazard {
    ReadWrite { input: usize },
    WriteWrite { output: usize },
}

fn operation_alias_hazard(
    inputs: &[GpuPhysicalFieldId],
    outputs: &[GpuPhysicalFieldId],
) -> Option<OperationAliasHazard> {
    let mut written = HashSet::new();
    for (output, physical) in outputs.iter().copied().enumerate() {
        if !written.insert(physical) {
            return Some(OperationAliasHazard::WriteWrite { output });
        }
    }
    inputs
        .iter()
        .position(|physical| written.contains(physical))
        .map(|input| OperationAliasHazard::ReadWrite { input })
}

/// Resource compatibility key. A new engine owns a new device generation, so
/// device replacement never accidentally reuses textures from the old device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuPlanResourceKey {
    pub width: u32,
    pub height: u32,
    pub device_generation: u64,
    pub format: wgpu::TextureFormat,
}

impl GpuPlanResourceKey {
    pub const fn new(width: u32, height: u32, device_generation: u64) -> Self {
        Self {
            width,
            height,
            device_generation,
            format: wgpu::TextureFormat::R32Float,
        }
    }

    pub const fn with_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.format = format;
        self
    }
}

struct ScalarTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// Concrete R32Float textures for a compiled resource layout.
pub struct GpuPlanResources {
    key: GpuPlanResourceKey,
    layout: GpuPlanResourceLayout,
    textures: Vec<ScalarTexture>,
}

impl GpuPlanResources {
    fn realize(
        device: &wgpu::Device,
        plan: &CompiledTerrainPlan,
        key: GpuPlanResourceKey,
    ) -> Result<Self, GpuPlanResourceError> {
        if key.width == 0 || key.height == 0 {
            return Err(GpuPlanResourceError::EmptyExtent);
        }
        if key.format != wgpu::TextureFormat::R32Float {
            return Err(GpuPlanResourceError::UnsupportedFormat(key.format));
        }
        let max = device.limits().max_texture_dimension_2d;
        if key.width > max || key.height > max {
            return Err(GpuPlanResourceError::ExtentExceedsDevice {
                width: key.width,
                height: key.height,
                limit: max,
            });
        }
        let _bytes = u64::from(key.width)
            .checked_mul(u64::from(key.height))
            .and_then(|texels| texels.checked_mul(4))
            .ok_or(GpuPlanResourceError::SizeOverflow)?;
        let layout = GpuPlanResourceLayout::build(plan)?;
        let textures = layout
            .allocations()
            .iter()
            .map(|allocation| {
                let label = format!("compiled-plan-field-{}", allocation.id.index());
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&label),
                    size: wgpu::Extent3d {
                        width: key.width,
                        height: key.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: key.format,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                ScalarTexture { texture, view }
            })
            .collect();
        Ok(Self {
            key,
            layout,
            textures,
        })
    }

    pub const fn key(&self) -> GpuPlanResourceKey {
        self.key
    }

    pub const fn layout(&self) -> &GpuPlanResourceLayout {
        &self.layout
    }

    pub fn view(&self, field: FieldSlot) -> Result<&wgpu::TextureView, GpuPlanResourceError> {
        let binding = self
            .layout
            .binding(field)
            .ok_or(GpuPlanResourceError::MissingBinding(field))?;
        Ok(&self.textures[binding.physical.index()].view)
    }

    pub fn texture(&self, field: FieldSlot) -> Result<&wgpu::Texture, GpuPlanResourceError> {
        let binding = self
            .layout
            .binding(field)
            .ok_or(GpuPlanResourceError::MissingBinding(field))?;
        Ok(&self.textures[binding.physical.index()].texture)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuPlanResourceCacheStats {
    pub realizations: u64,
    pub warm_hits: u64,
    pub failed_realizations: u64,
    pub staged_candidates: u64,
    pub committed_candidates: u64,
}

/// Last-good resource owner. Replacement is transactional: a candidate is
/// fully validated and allocated before `current` is changed.
#[derive(Default)]
pub struct GpuPlanResourceCache {
    current: Option<GpuPlanResources>,
    stats: GpuPlanResourceCacheStats,
}

impl GpuPlanResourceCache {
    pub fn realize(
        &mut self,
        device: &wgpu::Device,
        plan: &CompiledTerrainPlan,
        key: GpuPlanResourceKey,
    ) -> Result<&GpuPlanResources, GpuPlanResourceError> {
        if self.current.as_ref().is_some_and(|current| {
            current.key == key && current.layout.structure_signature() == plan.structure_signature()
        }) {
            self.stats.warm_hits += 1;
            return Ok(self.current.as_ref().expect("checked above"));
        }
        let candidate = match GpuPlanResources::realize(device, plan, key) {
            Ok(candidate) => candidate,
            Err(error) => {
                self.stats.failed_realizations += 1;
                return Err(error);
            }
        };
        self.current = Some(candidate);
        self.stats.realizations += 1;
        Ok(self.current.as_ref().expect("candidate installed"))
    }

    pub const fn current(&self) -> Option<&GpuPlanResources> {
        self.current.as_ref()
    }

    /// Temporarily transfer ownership of the active realization to an executor.
    /// Recording failures can restore it without publishing a replacement, while
    /// successful executions return it through [`Self::commit_candidate`].
    pub fn take_current(&mut self) -> Option<GpuPlanResources> {
        self.current.take()
    }

    /// Restore an execution candidate without counting it as a publication.
    pub fn restore_current(&mut self, resources: GpuPlanResources) {
        debug_assert!(self.current.is_none());
        self.current = Some(resources);
    }

    /// Build an isolated execution candidate. Recording into this resource set
    /// cannot mutate `current`; callers commit only after successful validation
    /// and submission, or simply drop the candidate on failure.
    pub fn stage_candidate(
        &mut self,
        device: &wgpu::Device,
        plan: &CompiledTerrainPlan,
        key: GpuPlanResourceKey,
    ) -> Result<GpuPlanResources, GpuPlanResourceError> {
        match GpuPlanResources::realize(device, plan, key) {
            Ok(candidate) => {
                self.stats.staged_candidates += 1;
                Ok(candidate)
            }
            Err(error) => {
                self.stats.failed_realizations += 1;
                Err(error)
            }
        }
    }

    pub fn commit_candidate(&mut self, candidate: GpuPlanResources) -> &GpuPlanResources {
        self.current = Some(candidate);
        self.stats.committed_candidates += 1;
        self.current.as_ref().expect("candidate installed")
    }

    pub const fn stats(&self) -> GpuPlanResourceCacheStats {
        self.stats
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GpuPlanResourceError {
    #[error("compiled plan field {0:?} has no producer")]
    MissingProducer(FieldSlot),
    #[error("compiled plan field {0:?} has no lifetime")]
    MissingLifetime(FieldSlot),
    #[error("compiled plan field {0:?} has an invalid lifetime")]
    InvalidLifetime(FieldSlot),
    #[error("compiled plan field {0:?} has no physical binding")]
    MissingBinding(FieldSlot),
    #[error("operation {operation} aliases a simultaneously live field at {field:?}")]
    AliasingHazard { operation: usize, field: FieldSlot },
    #[error("compiled plan resources require a non-empty extent")]
    EmptyExtent,
    #[error("compiled plan extent {width}x{height} exceeds device limit {limit}")]
    ExtentExceedsDevice { width: u32, height: u32, limit: u32 },
    #[error("compiled plan resource byte size overflow")]
    SizeOverflow,
    #[error("compiled plan scalar field format {0:?} is unsupported")]
    UnsupportedFormat(wgpu::TextureFormat),
}

#[cfg(test)]
mod tests {
    use super::{operation_alias_hazard, GpuPhysicalFieldId, OperationAliasHazard};

    #[test]
    fn duplicate_read_only_allocations_are_valid() {
        assert_eq!(
            operation_alias_hazard(
                &[GpuPhysicalFieldId(3), GpuPhysicalFieldId(3)],
                &[GpuPhysicalFieldId(4)],
            ),
            None
        );
    }

    #[test]
    fn read_write_allocation_alias_is_rejected() {
        assert_eq!(
            operation_alias_hazard(
                &[GpuPhysicalFieldId(3), GpuPhysicalFieldId(4)],
                &[GpuPhysicalFieldId(4)],
            ),
            Some(OperationAliasHazard::ReadWrite { input: 1 })
        );
    }

    #[test]
    fn write_write_allocation_alias_is_rejected() {
        assert_eq!(
            operation_alias_hazard(
                &[GpuPhysicalFieldId(2)],
                &[GpuPhysicalFieldId(3), GpuPhysicalFieldId(3)],
            ),
            Some(OperationAliasHazard::WriteWrite { output: 1 })
        );
    }
}
