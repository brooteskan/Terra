//! Deterministic cross-crate regression documents.

use crate::document::TerrainDocument;
use crate::heightfield::HeightfieldMetrics;
use crate::layer::{
    BiomeSection, GroupInputMode, Layer, LayerGroup, LayerId, LayerKind, LayerStack, SculptParams,
    SculptStrokeParams, StackCategory, StackNode, VolcanoParams,
};
use crate::mask::{DistNode, Distribution};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Untitled6Variant {
    SupportedTree,
    EmptyIsolatedBiome,
    MaskedComposite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Untitled6Ids {
    pub base: LayerId,
    pub semantic_sculpt: LayerId,
    pub sculpt_strokes: LayerId,
    pub biome: LayerId,
    pub volcano: Option<LayerId>,
}

/// Build the document shape that originally exposed #148:
/// Base; Terrain/Semantic Sculpt/SculptStrokes; and an isolated Default biome
/// whose Filters section contains Volcano. IDs are returned instead of being
/// hard-coded so each test may safely own and mutate its document.
pub fn untitled6_document(
    resolution: u32,
    variant: Untitled6Variant,
) -> (TerrainDocument, Untitled6Ids) {
    let mut stack = LayerStack::new();

    let base = Layer::new(
        "Base",
        LayerKind::SculptBase(SculptParams::filled(resolution, 20.0)),
    );
    let base_id = base.id();
    stack.push(base);

    let mut terrain = LayerGroup::category_folder(StackCategory::Shape);
    let mut semantic_sculpt = LayerGroup::new("Semantic Sculpt");
    let semantic_sculpt_id = semantic_sculpt.id;
    let strokes = Layer::new(
        "SculptStrokes",
        LayerKind::SculptStrokes(SculptStrokeParams::default()),
    );
    let sculpt_strokes_id = strokes.id();
    semantic_sculpt.children.push(StackNode::Layer(strokes));
    terrain.children.push(StackNode::Group(semantic_sculpt));
    stack.push_group(terrain);

    let mut biomes = LayerGroup::category_folder(StackCategory::Surface);
    let mut biome = LayerGroup::biome("Default biome");
    biome.input_mode = GroupInputMode::CopyInput;
    let biome_id = biome.id;
    let volcano_id = if variant == Untitled6Variant::EmptyIsolatedBiome {
        None
    } else {
        let volcano = Layer::new("Volcano", LayerKind::Volcano(VolcanoParams::default()));
        let id = volcano.id();
        biome
            .find_section_mut(BiomeSection::Filters)
            .expect("biome constructor creates Filters")
            .children
            .push(StackNode::Layer(volcano));
        Some(id)
    };
    if variant == Untitled6Variant::MaskedComposite {
        biome.opacity = 0.65;
        biome.masks = Distribution::from_nodes(vec![DistNode::slope(8.0, 42.0)]);
    }
    biomes.children.push(StackNode::Group(biome));
    stack.push_group(biomes);

    let metrics = HeightfieldMetrics::new(resolution, resolution, 4096.0, 4096.0);
    let mut document = TerrainDocument::new_default();
    document.name = "Untitled6 regression".into();
    document.metrics = metrics;
    document.preview_resolution = resolution;
    document.stack = stack;
    document.masks.clear();
    document.selected = Some(base_id);
    document.active_biome = Some(biome_id);

    (
        document,
        Untitled6Ids {
            base: base_id,
            semantic_sculpt: semantic_sculpt_id,
            sculpt_strokes: sculpt_strokes_id,
            biome: biome_id,
            volcano: volcano_id,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain_plan::{TerrainEditClass, TerrainPlanCache};

    #[test]
    fn variants_compile_as_one_stable_authored_tree() {
        for variant in [
            Untitled6Variant::SupportedTree,
            Untitled6Variant::EmptyIsolatedBiome,
            Untitled6Variant::MaskedComposite,
        ] {
            let (document, _) = untitled6_document(32, variant);
            let mut cache = TerrainPlanCache::new();
            cache
                .update(
                    &document.stack,
                    &document.masks,
                    &[TerrainEditClass::Structure],
                )
                .expect("fixture must compile");
            let first = cache.stats().snapshot();
            cache
                .acquire(&document.stack, &document.masks)
                .expect("warm fixture plan");
            let warm = cache.stats().snapshot();
            assert_eq!(first.plan_compiles, 1);
            assert_eq!(first.authored_tree_walks, 1);
            assert_eq!(first.dependency_builds, 1);
            assert_eq!(warm.plan_compiles, first.plan_compiles);
            assert_eq!(warm.authored_tree_walks, first.authored_tree_walks);
            assert_eq!(warm.dependency_builds, first.dependency_builds);
            assert_eq!(warm.plan_cache_hits, first.plan_cache_hits + 1);
        }
    }
}
