//! Editor session: document + undo stacks (non-region).

use crate::command::CommandHistory;
use crate::layer::LayerId;
use crate::mask::MaskId;
use crate::rebuild_state::RebuildFeedbackState;
use crate::simulation_scenario::SimulationScenarioCommand;
use crate::world_rules::WorldRuleCommand;
use serde::{Deserialize, Serialize};

use super::TerrainDocument;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskPaintStrokeUndo {
    pub label: String,
    pub mask_id: MaskId,
    pub before_samples: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct PaintStrokeUndo {
    pub label: String,
    pub placement_id: crate::biome_paint::BiomeLayerId,
    pub biome_id: LayerId,
    pub before_channels: Vec<crate::biome_paint::BiomeWeightChannel>,
    pub before_pages: Vec<crate::sparse_paint::PaintPage>,
    pub used_sparse: bool,
}

pub struct EditorSession {
    pub document: TerrainDocument,
    pub history: CommandHistory,
    pub outdated_sim_layers: Vec<LayerId>,
    pub rebuild_feedback: RebuildFeedbackState,
    /// UI hint: next frame should schedule a rebuild.
    pub dirty_eval: bool,
    world_rule_undo: Vec<WorldRuleCommand>,
    world_rule_redo: Vec<WorldRuleCommand>,
    scenario_undo: Vec<SimulationScenarioCommand>,
    scenario_redo: Vec<SimulationScenarioCommand>,
    mask_paint_undo: Vec<MaskPaintStrokeUndo>,
    paint_undo: Vec<PaintStrokeUndo>,
}

impl Default for EditorSession {
    fn default() -> Self {
        Self::new()
    }
}

impl EditorSession {
    pub fn new() -> Self {
        Self::with_document(TerrainDocument::new_default())
    }

    pub fn with_document(document: TerrainDocument) -> Self {
        Self {
            document,
            history: CommandHistory::new(256),
            outdated_sim_layers: Vec::new(),
            rebuild_feedback: RebuildFeedbackState::default(),
            dirty_eval: false,
            world_rule_undo: Vec::new(),
            world_rule_redo: Vec::new(),
            scenario_undo: Vec::new(),
            scenario_redo: Vec::new(),
            mask_paint_undo: Vec::new(),
            paint_undo: Vec::new(),
        }
    }

    pub fn push_world_rule_command(&mut self, cmd: WorldRuleCommand) {
        cmd.apply(&mut self.document.world_rules);
        self.world_rule_undo.push(cmd);
        self.world_rule_redo.clear();
    }

    pub fn undo_world_rule(&mut self) -> bool {
        let Some(cmd) = self.world_rule_undo.pop() else {
            return false;
        };
        cmd.invert(&mut self.document.world_rules);
        self.world_rule_redo.push(cmd);
        true
    }

    pub fn redo_world_rule(&mut self) -> bool {
        let Some(cmd) = self.world_rule_redo.pop() else {
            return false;
        };
        cmd.apply(&mut self.document.world_rules);
        self.world_rule_undo.push(cmd);
        true
    }

    pub fn push_scenario_command(&mut self, cmd: SimulationScenarioCommand) {
        cmd.apply(&mut self.document.simulation_scenarios);
        self.scenario_undo.push(cmd);
        self.scenario_redo.clear();
    }

    pub fn undo_scenario(&mut self) -> bool {
        let Some(cmd) = self.scenario_undo.pop() else {
            return false;
        };
        cmd.invert(&mut self.document.simulation_scenarios);
        self.scenario_redo.push(cmd);
        true
    }

    pub fn redo_scenario(&mut self) -> bool {
        let Some(cmd) = self.scenario_redo.pop() else {
            return false;
        };
        cmd.apply(&mut self.document.simulation_scenarios);
        self.scenario_undo.push(cmd);
        true
    }

    pub fn push_mask_paint_undo(&mut self, stroke: MaskPaintStrokeUndo) {
        self.mask_paint_undo.push(stroke);
    }

    pub fn undo_mask_paint(&mut self) -> Option<MaskPaintStrokeUndo> {
        let stroke = self.mask_paint_undo.pop()?;
        if let Some(asset) = self
            .document
            .masks
            .iter_mut()
            .find(|m| m.id == stroke.mask_id)
        {
            if let Some(paint) = asset.paint.as_mut() {
                if paint.width == stroke.width && paint.height == stroke.height {
                    paint.samples = stroke.before_samples.clone();
                }
            }
        }
        Some(stroke)
    }

    pub fn push_paint_undo(&mut self, stroke: PaintStrokeUndo) {
        self.paint_undo.push(stroke);
    }

    pub fn undo_paint_stroke(&mut self) -> bool {
        let Some(stroke) = self.paint_undo.pop() else {
            return false;
        };
        if let Some(layer) = self
            .document
            .biome_layers
            .iter_mut()
            .find(|l| l.id == stroke.placement_id)
        {
            layer.channels = stroke.before_channels;
        }
        if stroke.used_sparse {
            let key = crate::sparse_paint::SparsePaintChannelKey {
                placement_id: stroke.placement_id,
                biome_id: stroke.biome_id,
            };
            self.document
                .sparse_paint
                .restore_pages(key, &stroke.before_pages);
        }
        true
    }
}
