//! Terrain Recipe view — stack-only execution order.

use terra_core::layer::{FlatParams, Layer, LayerKind, LayerStack, StackNode};
use terra_core::terrain_recipe::{
    build_terrain_recipe_from_stack, recipe_matches_stack, RecipeItemKind,
};

fn flat(name: &str, h: f32) -> Layer {
    Layer::new(name, LayerKind::Flat(FlatParams { height: h }))
}

#[test]
fn recipe_mirrors_terrain_stack() {
    let mut stack = LayerStack::new();
    stack.nodes.push(StackNode::Layer(flat("Foundation", 0.0)));
    stack.nodes.push(StackNode::Layer(flat("Climate", 1.0)));
    stack.nodes.push(StackNode::Layer(flat("Sea", 2.0)));
    stack.nodes.push(StackNode::Layer(flat("Hydrology", 3.0)));

    let recipe = build_terrain_recipe_from_stack(&stack);
    assert!(recipe_matches_stack(&recipe, &stack));

    let names: Vec<&str> = recipe.iter().map(|i| i.name.as_str()).collect();
    assert_eq!(names[0], "TERRAIN");
    assert!(names.windows(2).any(|w| w == ["Foundation", "Climate"]));
    assert!(!names.contains(&"REGIONS"));
    assert!(recipe.iter().all(|i| matches!(
        i.kind,
        RecipeItemKind::GlobalSection
            | RecipeItemKind::Layer
            | RecipeItemKind::Group
            | RecipeItemKind::FlowArrow
    )));
}

#[test]
fn recipe_layer_selection_targets_exist() {
    let mut stack = LayerStack::new();
    let layer = flat("Hydrology", 1.0);
    let id = layer.id();
    stack.nodes.push(StackNode::Layer(layer));
    let recipe = build_terrain_recipe_from_stack(&stack);
    let item = recipe
        .iter()
        .find(|i| i.layer_id == Some(id))
        .expect("layer row");
    assert_eq!(item.kind, RecipeItemKind::Layer);
}
