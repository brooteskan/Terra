use super::*;

#[test]
fn gpu_required_authored_stack_matches_cpu_with_named_tolerance() {
    let metrics = HeightfieldMetrics::new(32, 32, 320.0, 80.0);
    let mask = MaskAsset::new(MaskId::new(), "constant", MaskSource::Constant(0.6));
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "base",
        LayerKind::Flat(FlatParams { height: 10.0 }),
    ));

    let mut ramp = Layer::new(
        "ramp",
        LayerKind::Ramp(RampParams {
            height_min: 0.0,
            height_max: 12.0,
            direction: 0.0,
        }),
    );
    ramp.common.blend = BlendMode::Add;
    ramp.common.opacity = 0.4;
    stack.push(ramp);

    let mut masked = Layer::new("masked add", LayerKind::Flat(FlatParams { height: 5.0 }));
    masked.common.blend = BlendMode::Add;
    let mut mask_ref = MaskRef::new(mask.id);
    mask_ref.strength = 0.75;
    mask_ref.invert = true;
    masked.common.masks.push(mask_ref);
    stack.push(masked);

    let cpu = cpu_oracle(&stack, std::slice::from_ref(&mask), metrics);
    let gpu = gpu_eval(&stack, std::slice::from_ref(&mask), metrics);
    assert_field_parity("stack.flat-ramp-constant-mask", &gpu, &cpu, EXACT_HEIGHT);
}

#[test]
fn gpu_required_solo_tree_matrix_matches_cpu_oracle() {
    let metrics = HeightfieldMetrics::new(24, 24, 240.0, 240.0);
    let additive = |name: &str, height: f32, solo: bool| {
        let mut layer = Layer::new(name, LayerKind::Flat(FlatParams { height }));
        layer.common.blend = BlendMode::Add;
        layer.common.solo = solo;
        layer
    };
    let mut fixtures = Vec::new();

    let mut root = LayerStack::new();
    root.push(Layer::new(
        "excluded base",
        LayerKind::Flat(FlatParams { height: 100.0 }),
    ));
    root.push(additive("root solo", 20.0, true));
    root.push(additive("excluded sibling", 50.0, false));
    fixtures.push(("solo.root", root));

    let mut folder = LayerGroup::new("Folder");
    folder
        .children
        .push(StackNode::Layer(additive("excluded child", 11.0, false)));
    folder
        .children
        .push(StackNode::Layer(additive("folder solo", 7.0, true)));
    let mut pass_through = LayerStack::new();
    pass_through.push(additive("excluded root", 90.0, false));
    pass_through.push_group(folder);
    fixtures.push(("solo.pass-through", pass_through));

    let mut isolated = LayerGroup::isolated("Isolated");
    isolated.opacity = 0.5;
    isolated
        .children
        .push(StackNode::Layer(additive("excluded private", 70.0, false)));
    isolated
        .children
        .push(StackNode::Layer(additive("isolated solo", 30.0, true)));
    let mut isolated_stack = LayerStack::new();
    isolated_stack.push(additive("excluded parent", 10.0, false));
    isolated_stack.push_group(isolated);
    fixtures.push(("solo.isolated", isolated_stack));

    let mut first = LayerGroup::new("First");
    first
        .children
        .push(StackNode::Layer(additive("first solo", 2.0, true)));
    first
        .children
        .push(StackNode::Layer(additive("first excluded", 3.0, false)));
    let mut second = LayerGroup::new("Second");
    second
        .children
        .push(StackNode::Layer(additive("second excluded", 6.0, false)));
    second
        .children
        .push(StackNode::Layer(additive("second solo", 5.0, true)));
    let mut multiple = LayerStack::new();
    multiple.push_group(first);
    multiple.push(additive("root excluded", 100.0, false));
    multiple.push_group(second);
    fixtures.push(("solo.multiple", multiple));

    let mut inner = LayerGroup::new("Inner");
    inner
        .children
        .push(StackNode::Layer(additive("inner excluded", 8.0, false)));
    inner
        .children
        .push(StackNode::Layer(additive("inner solo", 4.0, true)));
    let mut outer = LayerGroup::new("Outer");
    outer
        .children
        .push(StackNode::Layer(additive("outer solo", 3.0, true)));
    outer.children.push(StackNode::Group(inner));
    let mut nested = LayerStack::new();
    nested.push_group(outer);
    fixtures.push(("solo.nested", nested));

    let mut disabled = additive("disabled solo", 25.0, true);
    disabled.common.enabled = false;
    let mut disabled_stack = LayerStack::new();
    disabled_stack.push(additive("excluded by disabled solo", 100.0, false));
    disabled_stack.push(disabled);
    fixtures.push(("solo.disabled", disabled_stack));

    for (name, stack) in fixtures {
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        assert_field_parity(name, &gpu, &cpu, EXACT_HEIGHT);
    }
}

#[test]
fn gpu_required_import_heightmap_matches_cpu_oracle() {
    let source = TempHeightmap::new();
    let metrics = HeightfieldMetrics::new(31, 19, 310.0, 95.0);
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "import",
        LayerKind::ImportHeightmap(ImportHeightmapParams {
            path: source.path(),
            height_scale: 173.0,
            height_offset: -21.5,
        }),
    ));
    let cpu = cpu_oracle(&stack, &[], metrics);
    let gpu = gpu_eval(&stack, &[], metrics);
    assert_field_parity(
        "asset.heightmap-sample.import",
        &gpu,
        &cpu,
        HEIGHTMAP_SAMPLE_PREVIEW,
    );
}

#[test]
fn gpu_required_transformed_stamp2d_matches_cpu_oracle() {
    let source = TempHeightmap::new();
    let metrics = HeightfieldMetrics::new(37, 23, 370.0, 138.0);
    for blend in [
        BlendMode::Normal,
        BlendMode::Replace,
        BlendMode::Interpolate,
        BlendMode::Add,
        BlendMode::Subtract,
        BlendMode::Multiply,
        BlendMode::Min,
        BlendMode::Max,
        BlendMode::Overlay,
        BlendMode::HeightBlend,
        BlendMode::SmoothMaximum,
        BlendMode::SmoothMinimum,
        BlendMode::SmoothUnion,
        BlendMode::SmoothSubtraction,
    ] {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "base",
            LayerKind::Flat(FlatParams { height: 12.0 }),
        ));
        let mut stamp = Layer::new(
            "stamp",
            LayerKind::Stamp2d(Stamp2dParams {
                heightmap: ImportHeightmapParams {
                    path: source.path(),
                    height_scale: 91.0,
                    height_offset: 4.0,
                },
            }),
        );
        stamp.common.shape_transform = Some(ShapeTransform {
            offset_x: 27.0,
            offset_z: -9.0,
            scale: 0.63,
            rotation_deg: 31.0,
            blend_size: 0.28,
            blend_roundness: 0.42,
        });
        stamp.common.opacity = 0.73;
        stamp.common.blend = blend;
        stack.push(stamp);

        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        for (x, y) in [(0, 0), (36, 0), (0, 22), (36, 22)] {
            assert_eq!(cpu.get(x, y), 12.0, "CPU {blend:?} outside footprint");
            assert!(
                (gpu.get(x, y) - 12.0).abs() < 0.001,
                "GPU {blend:?} changed ({x},{y}) outside footprint"
            );
        }
        assert_field_parity(
            &format!("asset.heightmap-sample.stamp2d.{blend:?}"),
            &gpu,
            &cpu,
            HEIGHTMAP_SAMPLE_PREVIEW,
        );
    }
}
