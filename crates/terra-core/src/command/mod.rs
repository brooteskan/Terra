//! Command-based undo/redo, including bounded snapshots for explicitly resized owned rasters.

use crate::authoring::SculptStroke;
use crate::layer::{
    BlendMode, GridDimensions, Layer, LayerGroup, LayerId, LayerKind, LayerStack, SculptParams,
    StackNode,
};
use crate::mask::{MaskAsset, MaskId, PaintBuffer};
use crate::raster::{RasterResizeError, RasterResizeLimits};
use serde::{Deserialize, Serialize};
use std::mem;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OwnedRasterTarget {
    SculptBase(LayerId),
    PaintedMask(MaskId),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StoredRaster {
    Sculpt(SculptParams),
    Mask(PaintBuffer),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandImpact {
    None,
    Layer(LayerId),
    Masks,
}

#[derive(Debug, thiserror::Error)]
pub enum ResizeRasterSourceError {
    #[error("the selected source no longer exists")]
    MissingTarget,
    #[error("the selected source is not an editable stored raster")]
    NotEditable,
    #[error("source already has the requested dimensions")]
    Unchanged,
    #[error(transparent)]
    Resize(#[from] RasterResizeError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EditorCommand {
    AddLayer {
        layer: Layer,
        index: usize,
    },
    RemoveLayer {
        id: LayerId,
        node: StackNode,
        /// Sibling index within `parent` (root when `parent` is `None`).
        index: usize,
        /// Parent group when the node lived nested in the WC tree.
        #[serde(default)]
        parent: Option<LayerId>,
    },
    Reorder {
        from: usize,
        to: usize,
    },
    SetEnabled {
        id: LayerId,
        enabled: bool,
        previous: bool,
    },
    SetOpacity {
        id: LayerId,
        opacity: f32,
        previous: f32,
    },
    SetBlend {
        id: LayerId,
        blend: BlendMode,
        previous: BlendMode,
    },
    SetKind {
        id: LayerId,
        kind: LayerKind,
        previous: LayerKind,
    },
    Rename {
        id: LayerId,
        name: String,
        previous: String,
    },
    Duplicate {
        source: LayerId,
        new_id: LayerId,
    },
    SetLocked {
        id: LayerId,
        locked: bool,
        previous: bool,
    },
    SetSolo {
        id: LayerId,
        solo: bool,
        previous: bool,
    },
    SetColorTag {
        id: LayerId,
        tag: u8,
        previous: u8,
    },
    SetCached {
        id: LayerId,
        cached: bool,
        previous: bool,
    },
    AddGroup {
        name: String,
        id: LayerId,
        index: usize,
    },
    /// Records an artist action whose data cannot yet be restored by undo.
    Annotate {
        label: String,
    },
    /// Develop Apply Where / local placement (undo restores prior placement + masks).
    SetOperationPlacement {
        id: LayerId,
        placement: crate::operation_placement::OperationPlacement,
        previous: crate::operation_placement::OperationPlacement,
        previous_masks: crate::mask::Distribution,
    },
    /// Swap-based snapshot for an explicitly resized owned raster source.
    ResizeRasterSource {
        target: OwnedRasterTarget,
        stored: StoredRaster,
    },
    SetStrokeEnabled {
        id: LayerId,
        index: usize,
        enabled: bool,
        previous: bool,
    },
    RemoveStroke {
        id: LayerId,
        index: usize,
        stroke: SculptStroke,
    },
}

pub struct CommandHistory {
    undo_stack: Vec<EditorCommand>,
    redo_stack: Vec<EditorCommand>,
    snapshots: Vec<(String, usize)>,
    last_coalesce_key: Option<(u64, &'static str)>,
    pub limit: usize,
    /// Maximum retained raster snapshot payload across the undo stack.
    pub raster_byte_limit: usize,
}

impl Default for CommandHistory {
    fn default() -> Self {
        Self::new(128)
    }
}

impl CommandHistory {
    pub fn new(limit: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            snapshots: Vec::new(),
            last_coalesce_key: None,
            limit,
            raster_byte_limit: 512 * crate::raster::MIB,
        }
    }

    pub fn push_executed(&mut self, cmd: EditorCommand) {
        self.last_coalesce_key = None;
        self.undo_stack.push(cmd);
        self.trim_undo();
        self.redo_stack.clear();
    }

    /// Push a command that was already applied, replacing the preceding command
    /// when it belongs to the same continuous control interaction.
    pub fn push_coalesced(
        &mut self,
        cmd: EditorCommand,
        coalesce_key: Option<(u64, &'static str)>,
    ) {
        if coalesce_key.is_some() && coalesce_key == self.last_coalesce_key {
            if let Some(last) = self.undo_stack.last_mut() {
                // Preserve the first command's `previous` value so one Undo
                // restores the value from before the entire drag began.
                match (last, cmd) {
                    (
                        EditorCommand::SetOpacity { opacity: prior, .. },
                        EditorCommand::SetOpacity { opacity, .. },
                    ) => *prior = opacity,
                    (
                        EditorCommand::SetKind { kind: prior, .. },
                        EditorCommand::SetKind { kind, .. },
                    ) => *prior = kind,
                    (prior, replacement) => *prior = replacement,
                }
                self.redo_stack.clear();
                return;
            }
        }
        self.undo_stack.push(cmd);
        self.trim_undo();
        self.redo_stack.clear();
        self.last_coalesce_key = coalesce_key;
    }

    pub fn mark_snapshot(&mut self, name: impl Into<String>) {
        self.snapshots.push((name.into(), self.undo_stack.len()));
    }

    pub fn snapshots(&self) -> &[(String, usize)] {
        &self.snapshots
    }

    /// Undo labels ordered from oldest to newest.
    pub fn undo_descriptions(&self) -> Vec<String> {
        self.undo_stack
            .iter()
            .map(EditorCommand::describe)
            .collect()
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Cheap change fingerprint for UI cache invalidation.
    pub fn ui_fingerprint(&self) -> (usize, usize) {
        (self.undo_stack.len(), self.redo_stack.len())
    }

    pub fn undo(&mut self, stack: &mut LayerStack) -> Option<LayerId> {
        self.last_coalesce_key = None;
        let cmd = self.undo_stack.pop()?;
        let dirty = invert(&cmd, stack);
        self.redo_stack.push(cmd);
        dirty
    }

    pub fn redo(&mut self, stack: &mut LayerStack) -> Option<LayerId> {
        self.last_coalesce_key = None;
        let cmd = self.redo_stack.pop()?;
        let dirty = apply(&cmd, stack);
        self.undo_stack.push(cmd);
        self.trim_undo();
        dirty
    }

    pub fn undo_document(
        &mut self,
        stack: &mut LayerStack,
        masks: &mut [MaskAsset],
    ) -> Option<CommandImpact> {
        self.last_coalesce_key = None;
        let mut cmd = self.undo_stack.pop()?;
        let impact = match &mut cmd {
            EditorCommand::ResizeRasterSource { target, stored } => {
                swap_raster_source(stack, masks, *target, stored)
            }
            _ => invert(&cmd, stack)
                .map(CommandImpact::Layer)
                .unwrap_or(CommandImpact::None),
        };
        self.redo_stack.push(cmd);
        Some(impact)
    }

    pub fn redo_document(
        &mut self,
        stack: &mut LayerStack,
        masks: &mut [MaskAsset],
    ) -> Option<CommandImpact> {
        self.last_coalesce_key = None;
        let mut cmd = self.redo_stack.pop()?;
        let impact = match &mut cmd {
            EditorCommand::ResizeRasterSource { target, stored } => {
                swap_raster_source(stack, masks, *target, stored)
            }
            _ => apply(&cmd, stack)
                .map(CommandImpact::Layer)
                .unwrap_or(CommandImpact::None),
        };
        self.undo_stack.push(cmd);
        self.trim_undo();
        Some(impact)
    }

    fn trim_undo(&mut self) {
        while self.undo_stack.len() > self.limit
            || (self.undo_stack.len() > 1
                && self
                    .undo_stack
                    .iter()
                    .map(EditorCommand::raster_payload_bytes)
                    .sum::<usize>()
                    > self.raster_byte_limit)
        {
            if self.undo_stack.is_empty() {
                break;
            }
            self.undo_stack.remove(0);
        }
    }
}

pub fn resize_raster_source(
    stack: &mut LayerStack,
    masks: &mut [MaskAsset],
    target: OwnedRasterTarget,
    dimensions: GridDimensions,
    limits: RasterResizeLimits,
) -> Result<EditorCommand, ResizeRasterSourceError> {
    let stored = match target {
        OwnedRasterTarget::SculptBase(id) => {
            let layer = stack
                .find_mut(id)
                .ok_or(ResizeRasterSourceError::MissingTarget)?;
            let LayerKind::SculptBase(params) = &mut layer.kind else {
                return Err(ResizeRasterSourceError::NotEditable);
            };
            if params.dimensions() == dimensions {
                return Err(ResizeRasterSourceError::Unchanged);
            }
            let replacement = params.resized(dimensions, limits)?;
            StoredRaster::Sculpt(mem::replace(params, replacement))
        }
        OwnedRasterTarget::PaintedMask(id) => {
            let asset = masks
                .iter_mut()
                .find(|asset| asset.id == id)
                .ok_or(ResizeRasterSourceError::MissingTarget)?;
            if !asset.is_painted() {
                return Err(ResizeRasterSourceError::NotEditable);
            }
            let paint = asset
                .paint
                .as_mut()
                .ok_or(ResizeRasterSourceError::NotEditable)?;
            if paint.dimensions() == dimensions {
                return Err(ResizeRasterSourceError::Unchanged);
            }
            let replacement = paint.resized(dimensions, limits)?;
            StoredRaster::Mask(mem::replace(paint, replacement))
        }
    };
    Ok(EditorCommand::ResizeRasterSource { target, stored })
}

fn swap_raster_source(
    stack: &mut LayerStack,
    masks: &mut [MaskAsset],
    target: OwnedRasterTarget,
    stored: &mut StoredRaster,
) -> CommandImpact {
    match (target, stored) {
        (OwnedRasterTarget::SculptBase(id), StoredRaster::Sculpt(snapshot)) => {
            let Some(layer) = stack.find_mut(id) else {
                return CommandImpact::None;
            };
            let LayerKind::SculptBase(current) = &mut layer.kind else {
                return CommandImpact::None;
            };
            mem::swap(current, snapshot);
            CommandImpact::Layer(id)
        }
        (OwnedRasterTarget::PaintedMask(id), StoredRaster::Mask(snapshot)) => {
            let Some(asset) = masks.iter_mut().find(|asset| asset.id == id) else {
                return CommandImpact::None;
            };
            let Some(current) = asset.paint.as_mut() else {
                return CommandImpact::None;
            };
            mem::swap(current, snapshot);
            CommandImpact::Masks
        }
        _ => CommandImpact::None,
    }
}

impl EditorCommand {
    /// Classify this undoable edit at the authored-tree/compiled-plan boundary.
    /// Numeric changes inside the same layer operation shape remain patchable;
    /// topology, dependency, and operation-I/O changes advance plan structure.
    pub fn terrain_edit_class(&self, stack: &LayerStack) -> crate::terrain_plan::TerrainEditClass {
        use crate::deps::NodeRef;
        use crate::field_data::FieldId;
        use crate::terrain_plan::{PlanDirtyScope, TerrainEditClass};

        match self {
            Self::AddLayer { .. }
            | Self::RemoveLayer { .. }
            | Self::Reorder { .. }
            | Self::Duplicate { .. }
            | Self::AddGroup { .. }
            | Self::SetEnabled { .. }
            | Self::SetSolo { .. }
            | Self::SetOperationPlacement { .. } => TerrainEditClass::Structure,
            Self::SetKind { id, kind, previous } => {
                if layer_kind_plan_shape(kind) == layer_kind_plan_shape(previous) {
                    TerrainEditClass::Parameters {
                        owner: NodeRef::Layer(*id),
                    }
                } else {
                    TerrainEditClass::Structure
                }
            }
            Self::SetOpacity { id, .. } | Self::SetBlend { id, .. } => {
                let owner = if stack.find_group(*id).is_some() {
                    NodeRef::Group(*id)
                } else {
                    NodeRef::Layer(*id)
                };
                TerrainEditClass::Parameters { owner }
            }
            Self::ResizeRasterSource { target, .. } => {
                let (owner, fields) = match target {
                    OwnedRasterTarget::SculptBase(id) => {
                        (NodeRef::Layer(*id), vec![FieldId::Height])
                    }
                    OwnedRasterTarget::PaintedMask(id) => (NodeRef::Mask(*id), Vec::new()),
                };
                TerrainEditClass::Content {
                    owner,
                    fields,
                    scope: PlanDirtyScope::FullField,
                }
            }
            Self::SetStrokeEnabled { id, .. } | Self::RemoveStroke { id, .. } => {
                TerrainEditClass::Content {
                    owner: NodeRef::Layer(*id),
                    fields: vec![FieldId::Height],
                    scope: PlanDirtyScope::FullField,
                }
            }
            Self::SetCached { .. } => TerrainEditClass::Resources,
            Self::Rename { .. }
            | Self::SetLocked { .. }
            | Self::SetColorTag { .. }
            | Self::Annotate { .. } => TerrainEditClass::ViewOnly,
        }
    }

    fn raster_payload_bytes(&self) -> usize {
        match self {
            Self::ResizeRasterSource {
                stored: StoredRaster::Sculpt(params),
                ..
            } => params
                .samples
                .len()
                .saturating_mul(std::mem::size_of::<f32>()),
            Self::ResizeRasterSource {
                stored: StoredRaster::Mask(paint),
                ..
            } => paint
                .samples
                .len()
                .saturating_mul(std::mem::size_of::<f32>()),
            _ => 0,
        }
    }

    /// Concise, artist-facing description suitable for the History panel.
    pub fn describe(&self) -> String {
        match self {
            Self::AddLayer { layer, .. } => format!("Added {}", layer.common.name),
            Self::RemoveLayer { .. } => "Removed Layer".into(),
            Self::Reorder { .. } => "Reordered Layers".into(),
            Self::SetEnabled { enabled, .. } => if *enabled {
                "Enabled Layer"
            } else {
                "Disabled Layer"
            }
            .into(),
            Self::SetOpacity { .. } => "Changed Opacity".into(),
            Self::SetBlend { .. } => "Changed Blend Mode".into(),
            Self::SetKind { .. } => "Changed Layer Type".into(),
            Self::Rename { name, .. } => format!("Renamed to {}", name),
            Self::Duplicate { .. } => "Duplicated Layer".into(),
            Self::SetLocked { locked, .. } => if *locked {
                "Locked Layer"
            } else {
                "Unlocked Layer"
            }
            .into(),
            Self::SetSolo { solo, .. } => if *solo {
                "Soloed Layer"
            } else {
                "Unsoloed Layer"
            }
            .into(),
            Self::SetColorTag { .. } => "Changed Color Tag".into(),
            Self::SetCached { cached, .. } => if *cached {
                "Cached Layer"
            } else {
                "Uncached Layer"
            }
            .into(),
            Self::AddGroup { name, .. } => format!("Added Group {}", name),
            Self::Annotate { label } => label.clone(),
            Self::SetOperationPlacement { .. } => "Changed Apply Where".into(),
            Self::ResizeRasterSource { .. } => "Resized Source Resolution".into(),
            Self::SetStrokeEnabled { enabled, .. } => if *enabled {
                "Enabled Stroke"
            } else {
                "Disabled Stroke"
            }
            .into(),
            Self::RemoveStroke { stroke, .. } => {
                format!("Deleted {} Stroke", stroke.kind.label())
            }
        }
    }
}

fn layer_kind_plan_shape(
    kind: &LayerKind,
) -> (
    String,
    Vec<crate::field_data::FieldId>,
    Vec<crate::field_data::FieldId>,
    Vec<crate::field_data::FieldId>,
) {
    (
        kind.type_id().into(),
        kind.required_fields(),
        kind.optional_fields(),
        kind.produced_fields(),
    )
}

pub fn apply(cmd: &EditorCommand, stack: &mut LayerStack) -> Option<LayerId> {
    match cmd {
        EditorCommand::AddLayer { layer, index } => {
            let id = layer.id();
            let idx = (*index).min(stack.nodes.len());
            stack.nodes.insert(idx, StackNode::Layer(layer.clone()));
            Some(id)
        }
        EditorCommand::RemoveLayer { id, .. } => {
            stack.remove(*id);
            stack.layer_ids().first().copied()
        }
        EditorCommand::Reorder { from, to } => {
            stack.reorder(*from, *to);
            None
        }
        EditorCommand::SetEnabled { id, enabled, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.enabled = *enabled;
            } else if let Some(g) = stack.find_group_mut(*id) {
                g.enabled = *enabled;
            }
            Some(*id)
        }
        EditorCommand::SetOpacity { id, opacity, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.opacity = *opacity;
            } else if let Some(g) = stack.find_group_mut(*id) {
                g.opacity = *opacity;
            }
            Some(*id)
        }
        EditorCommand::SetBlend { id, blend, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.blend = *blend;
            } else if let Some(g) = stack.find_group_mut(*id) {
                g.blend = *blend;
            }
            Some(*id)
        }
        EditorCommand::SetKind { id, kind, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.kind = kind.clone();
            }
            Some(*id)
        }
        EditorCommand::Rename { id, name, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.name = name.clone();
            } else if let Some(g) = stack.find_group_mut(*id) {
                // Structural folders/sections keep fixed labels.
                if !matches!(
                    g.group_kind,
                    crate::layer::GroupKind::BiomeSection(_)
                        | crate::layer::GroupKind::CategoryFolder
                ) {
                    g.name = name.clone();
                }
            }
            Some(*id)
        }
        EditorCommand::Duplicate { source, .. } => stack.duplicate(*source),
        EditorCommand::SetLocked { id, locked, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.locked = *locked;
            }
            Some(*id)
        }
        EditorCommand::SetSolo { id, solo, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.solo = *solo;
            }
            Some(*id)
        }
        EditorCommand::SetColorTag { id, tag, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.color_tag = *tag;
            }
            Some(*id)
        }
        EditorCommand::SetCached { id, cached, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.cached = *cached;
            }
            Some(*id)
        }
        EditorCommand::AddGroup { name, id, index } => {
            let mut g = LayerGroup::new(name.clone());
            g.id = *id;
            let idx = (*index).min(stack.nodes.len());
            stack.nodes.insert(idx, StackNode::Group(g));
            Some(*id)
        }
        EditorCommand::Annotate { .. } => None,
        EditorCommand::SetOperationPlacement { id, placement, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.operation_placement = placement.clone();
                l.sync_operation_placement_masks();
            }
            Some(*id)
        }
        EditorCommand::ResizeRasterSource { .. } => None,
        EditorCommand::SetStrokeEnabled {
            id, index, enabled, ..
        } => {
            if let Some(l) = stack.find_mut(*id) {
                if let LayerKind::SculptStrokes(p) = &mut l.kind {
                    if let Some(s) = p.strokes.get_mut(*index) {
                        s.enabled = *enabled;
                    }
                }
            }
            Some(*id)
        }
        EditorCommand::RemoveStroke { id, index, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                if let LayerKind::SculptStrokes(p) = &mut l.kind {
                    let i = (*index).min(p.strokes.len().saturating_sub(1));
                    if i < p.strokes.len() {
                        p.strokes.remove(i);
                    }
                }
            }
            Some(*id)
        }
    }
}

fn invert(cmd: &EditorCommand, stack: &mut LayerStack) -> Option<LayerId> {
    match cmd {
        EditorCommand::AddLayer { layer, .. } => {
            stack.remove(layer.id());
            None
        }
        EditorCommand::RemoveLayer {
            node,
            index,
            parent,
            ..
        } => {
            let idx = *index;
            if let Some(pid) = parent {
                if let Some(group) = stack.find_group_mut(*pid) {
                    let idx = idx.min(group.children.len());
                    group.children.insert(idx, node.clone());
                } else {
                    // Parent gone — restore at root.
                    let idx = idx.min(stack.nodes.len());
                    stack.nodes.insert(idx, node.clone());
                }
            } else {
                let idx = idx.min(stack.nodes.len());
                stack.nodes.insert(idx, node.clone());
            }
            match node {
                StackNode::Layer(l) => Some(l.id()),
                StackNode::Group(g) => Some(g.id),
            }
        }
        EditorCommand::Reorder { from, to } => {
            stack.reorder(*to, *from);
            None
        }
        EditorCommand::SetEnabled { id, previous, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.enabled = *previous;
            } else if let Some(g) = stack.find_group_mut(*id) {
                g.enabled = *previous;
            }
            Some(*id)
        }
        EditorCommand::SetOpacity { id, previous, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.opacity = *previous;
            } else if let Some(g) = stack.find_group_mut(*id) {
                g.opacity = *previous;
            }
            Some(*id)
        }
        EditorCommand::SetBlend { id, previous, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.blend = *previous;
            } else if let Some(g) = stack.find_group_mut(*id) {
                g.blend = *previous;
            }
            Some(*id)
        }
        EditorCommand::SetKind { id, previous, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.kind = previous.clone();
            }
            Some(*id)
        }
        EditorCommand::Rename { id, previous, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.name = previous.clone();
            } else if let Some(g) = stack.find_group_mut(*id) {
                if !matches!(
                    g.group_kind,
                    crate::layer::GroupKind::BiomeSection(_)
                        | crate::layer::GroupKind::CategoryFolder
                ) {
                    g.name = previous.clone();
                }
            }
            Some(*id)
        }
        EditorCommand::Duplicate { new_id, .. } => {
            stack.remove(*new_id);
            None
        }
        EditorCommand::SetLocked { id, previous, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.locked = *previous;
            }
            Some(*id)
        }
        EditorCommand::SetSolo { id, previous, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.solo = *previous;
            }
            Some(*id)
        }
        EditorCommand::SetColorTag { id, previous, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.color_tag = *previous;
            }
            Some(*id)
        }
        EditorCommand::SetCached { id, previous, .. } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.cached = *previous;
            }
            Some(*id)
        }
        EditorCommand::AddGroup { id, .. } => {
            stack.remove(*id);
            None
        }
        EditorCommand::Annotate { .. } => None,
        EditorCommand::SetOperationPlacement {
            id,
            previous,
            previous_masks,
            ..
        } => {
            if let Some(l) = stack.find_mut(*id) {
                l.common.operation_placement = previous.clone();
                l.common.masks = previous_masks.clone();
            }
            Some(*id)
        }
        EditorCommand::ResizeRasterSource { .. } => None,
        EditorCommand::SetStrokeEnabled {
            id,
            index,
            previous,
            ..
        } => {
            if let Some(l) = stack.find_mut(*id) {
                if let LayerKind::SculptStrokes(p) = &mut l.kind {
                    if let Some(s) = p.strokes.get_mut(*index) {
                        s.enabled = *previous;
                    }
                }
            }
            Some(*id)
        }
        EditorCommand::RemoveStroke {
            id, index, stroke, ..
        } => {
            if let Some(l) = stack.find_mut(*id) {
                if let LayerKind::SculptStrokes(p) = &mut l.kind {
                    let i = (*index).min(p.strokes.len());
                    p.strokes.insert(i, stroke.clone());
                }
            }
            Some(*id)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::TerrainDocument;
    use crate::layer::FlatParams;
    use crate::mask::MaskAsset;

    #[test]
    fn undo_redo_opacity() {
        let mut stack = LayerStack::new();
        let layer = Layer::new("A", LayerKind::Flat(FlatParams { height: 1.0 }));
        let id = layer.id();
        stack.push(layer);
        let mut hist = CommandHistory::new(32);
        let cmd = EditorCommand::SetOpacity {
            id,
            opacity: 0.5,
            previous: 1.0,
        };
        apply(&cmd, &mut stack);
        hist.push_executed(cmd);
        assert_eq!(stack.find(id).unwrap().common.opacity, 0.5);
        hist.undo(&mut stack);
        assert_eq!(stack.find(id).unwrap().common.opacity, 1.0);
        hist.redo(&mut stack);
        assert_eq!(stack.find(id).unwrap().common.opacity, 0.5);
    }

    #[test]
    fn coalesced_opacity_undo_restores_drag_start() {
        let mut stack = LayerStack::new();
        let layer = Layer::new("A", LayerKind::Flat(FlatParams { height: 1.0 }));
        let id = layer.id();
        stack.push(layer);
        let mut hist = CommandHistory::new(32);

        let first = EditorCommand::SetOpacity {
            id,
            opacity: 0.8,
            previous: 1.0,
        };
        apply(&first, &mut stack);
        hist.push_coalesced(first, Some((1, "opacity")));
        let second = EditorCommand::SetOpacity {
            id,
            opacity: 0.4,
            previous: 0.8,
        };
        apply(&second, &mut stack);
        hist.push_coalesced(second, Some((1, "opacity")));

        hist.undo(&mut stack);
        assert_eq!(stack.find(id).unwrap().common.opacity, 1.0);
    }

    #[test]
    fn sculpt_resize_undo_redo_swaps_complete_rectangular_buffers() {
        let mut document = TerrainDocument::default();
        let id = document
            .stack
            .flatten_layers()
            .into_iter()
            .find(|layer| layer.kind.is_sculpt_base())
            .unwrap()
            .id();
        let mut history = CommandHistory::new(8);
        let command = resize_raster_source(
            &mut document.stack,
            &mut document.masks,
            OwnedRasterTarget::SculptBase(id),
            GridDimensions::new(256, 128),
            RasterResizeLimits::default(),
        )
        .unwrap();
        history.push_executed(command);
        let dimensions = |doc: &TerrainDocument| match &doc.stack.find(id).unwrap().kind {
            LayerKind::SculptBase(params) => params.dimensions(),
            _ => unreachable!(),
        };
        assert_eq!(dimensions(&document), GridDimensions::new(256, 128));
        assert_eq!(
            history.undo_document(&mut document.stack, &mut document.masks),
            Some(CommandImpact::Layer(id))
        );
        assert_eq!(dimensions(&document), GridDimensions::square(512));
        assert_eq!(
            history.redo_document(&mut document.stack, &mut document.masks),
            Some(CommandImpact::Layer(id))
        );
        assert_eq!(dimensions(&document), GridDimensions::new(256, 128));
    }

    #[test]
    fn painted_mask_resize_is_undoable_and_reports_mask_impact() {
        let mut document = TerrainDocument::default();
        let asset = MaskAsset::new_painted(MaskId::new(), "Paint", 256);
        let id = asset.id;
        document.masks.push(asset);
        let command = resize_raster_source(
            &mut document.stack,
            &mut document.masks,
            OwnedRasterTarget::PaintedMask(id),
            GridDimensions::new(128, 512),
            RasterResizeLimits::default(),
        )
        .unwrap();
        let mut history = CommandHistory::new(8);
        history.push_executed(command);
        assert_eq!(
            document.masks[0].paint.as_ref().unwrap().dimensions(),
            GridDimensions::new(128, 512)
        );
        assert_eq!(
            history.undo_document(&mut document.stack, &mut document.masks),
            Some(CommandImpact::Masks)
        );
        assert_eq!(
            document.masks[0].paint.as_ref().unwrap().dimensions(),
            GridDimensions::square(256)
        );
    }
}
