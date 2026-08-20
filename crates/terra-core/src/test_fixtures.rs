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
    ProductionTopology,
    EmptyIsolatedBiome,
    MaskedComposite,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Untitled6Ids {
    pub base: LayerId,
    pub semantic_sculpt: LayerId,
    pub default_biome: LayerId,
    pub empty_biomes: Vec<LayerId>,
    pub volcano: Option<LayerId>,
}

/// Build the document shape that exposed #148 and the production alias failure
/// in #150. The production variant contains a 512x512 embedded Base raster,
/// Semantic Sculpt as a direct SculptStrokes layer, Default/Volcano, then the
/// empty isolated Water, Beach, Grassland, and Rock biome siblings.
pub fn untitled6_document(
    resolution: u32,
    variant: Untitled6Variant,
) -> (TerrainDocument, Untitled6Ids) {
    let mut stack = LayerStack::new();

    let base = Layer::new(
        "Base",
        LayerKind::SculptBase(SculptParams::filled(512, 20.0)),
    );
    let base_id = base.id();
    stack.push(base);

    let mut terrain = LayerGroup::category_folder(StackCategory::Shape);
    let semantic_sculpt = Layer::new(
        "Semantic Sculpt",
        LayerKind::SculptStrokes(SculptStrokeParams::default()),
    );
    let semantic_sculpt_id = semantic_sculpt.id();
    terrain.children.push(StackNode::Layer(semantic_sculpt));
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
    let mut empty_biomes = Vec::new();
    if variant == Untitled6Variant::ProductionTopology {
        for name in ["Water", "Beach", "Grassland", "Rock"] {
            let mut biome = LayerGroup::biome(name);
            biome.input_mode = GroupInputMode::CopyInput;
            empty_biomes.push(biome.id);
            biomes.children.push(StackNode::Group(biome));
        }
    }
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
            default_biome: biome_id,
            empty_biomes,
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
            Untitled6Variant::ProductionTopology,
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

    #[test]
    fn production_variant_matches_the_saved_document_topology() {
        let (document, ids) = untitled6_document(4096, Untitled6Variant::ProductionTopology);
        let base = document.stack.find(ids.base).expect("Base");
        let LayerKind::SculptBase(params) = &base.kind else {
            panic!("Base must remain a SculptBase layer");
        };
        assert_eq!((params.width, params.height), (512, 512));

        let semantic = document
            .stack
            .find(ids.semantic_sculpt)
            .expect("Semantic Sculpt");
        assert_eq!(semantic.common.name, "Semantic Sculpt");
        assert!(matches!(semantic.kind, LayerKind::SculptStrokes(_)));

        let surface = document
            .stack
            .find_category(StackCategory::Surface)
            .expect("Biomes category");
        let biome_names: Vec<_> = surface
            .children
            .iter()
            .filter_map(|node| match node {
                StackNode::Group(group) if group.is_biome() => Some(group.name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            biome_names,
            ["Default biome", "Water", "Beach", "Grassland", "Rock"]
        );
        assert_eq!(ids.empty_biomes.len(), 4);
        assert!(ids.volcano.is_some());
    }
}
