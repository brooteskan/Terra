//! Ratchet every production wgpu pipeline-creation site into compilation telemetry.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Counts {
    render: usize,
    compute: usize,
}

#[test]
fn every_production_pipeline_creation_is_labeled_and_tracked() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("terra-app lives under workspace/crates")
        .to_path_buf();
    let roots = ["terra-gpu", "terra-gpu-eval", "terra-render", "terra-gui"];
    let mut observed = BTreeMap::<String, Counts>::new();

    for root in roots {
        collect_rs(
            &workspace.join("crates").join(root).join("src"),
            &mut |path| {
                let source = fs::read_to_string(path).expect("production Rust source is readable");
                let relative = path
                    .strip_prefix(&workspace)
                    .expect("source belongs to workspace")
                    .to_string_lossy()
                    .replace('\\', "/");
                let counts = Counts {
                    render: inspect_calls(&source, "create_render_pipeline(", path),
                    compute: inspect_calls(&source, "create_compute_pipeline(", path),
                };
                if counts != Counts::default() {
                    observed.insert(relative, counts);
                }
            },
        );
    }

    let expected = BTreeMap::from([
        (
            "crates/terra-gpu-eval/src/engine/pipelines/mod.rs".into(),
            Counts {
                render: 0,
                compute: 1,
            },
        ),
        (
            "crates/terra-gpu/src/compiled_plan/operations.rs".into(),
            Counts {
                render: 0,
                compute: 1,
            },
        ),
        (
            "crates/terra-gpu/src/derivatives.rs".into(),
            Counts {
                render: 0,
                compute: 1,
            },
        ),
        (
            "crates/terra-gpu/src/pyramid.rs".into(),
            Counts {
                render: 0,
                compute: 1,
            },
        ),
        (
            "crates/terra-gpu/src/tile_cache.rs".into(),
            Counts {
                render: 0,
                compute: 1,
            },
        ),
        (
            "crates/terra-gui/src/renderer.rs".into(),
            Counts {
                render: 1,
                compute: 0,
            },
        ),
        (
            "crates/terra-render/src/brush.rs".into(),
            Counts {
                render: 1,
                compute: 1,
            },
        ),
        (
            "crates/terra-render/src/guides.rs".into(),
            Counts {
                render: 1,
                compute: 0,
            },
        ),
        (
            "crates/terra-render/src/height_gpu.rs".into(),
            Counts {
                render: 0,
                compute: 1,
            },
        ),
        (
            "crates/terra-render/src/integrity_probe.rs".into(),
            Counts {
                render: 0,
                compute: 1,
            },
        ),
        (
            "crates/terra-render/src/lib.rs".into(),
            Counts {
                render: 1,
                compute: 0,
            },
        ),
        (
            "crates/terra-render/src/overhang.rs".into(),
            Counts {
                render: 1,
                compute: 0,
            },
        ),
        (
            "crates/terra-render/src/path_tracer.rs".into(),
            Counts {
                render: 0,
                compute: 1,
            },
        ),
        (
            "crates/terra-render/src/progressive.rs".into(),
            Counts {
                render: 3,
                compute: 0,
            },
        ),
        (
            "crates/terra-render/src/shadows.rs".into(),
            Counts {
                render: 1,
                compute: 0,
            },
        ),
        (
            "crates/terra-render/src/terrain_pipeline.rs".into(),
            Counts {
                render: 3,
                compute: 0,
            },
        ),
        (
            "crates/terra-render/src/vegetation.rs".into(),
            Counts {
                render: 1,
                compute: 0,
            },
        ),
    ]);

    assert_eq!(
        observed, expected,
        "pipeline creation inventory changed; instrument every new site and update this explicit ratchet"
    );
}

fn inspect_calls(source: &str, needle: &str, path: &Path) -> usize {
    let mut count = 0;
    for (offset, _) in source.match_indices(needle) {
        count += 1;
        let mut context_start = offset.saturating_sub(1_200);
        while !source.is_char_boundary(context_start) {
            context_start += 1;
        }
        let context = &source[context_start..offset];
        assert!(
            context.contains("terra_telemetry::measure(")
                || context.contains(".render_pipeline(")
                || context.contains(".compute_pipeline("),
            "{} has an untracked {needle} call near byte {offset}",
            path.display()
        );
        let tail = &source[offset..source.len().min(offset + 350)];
        assert!(
            tail.contains("label: Some("),
            "{} has an unlabeled {needle} descriptor near byte {offset}",
            path.display()
        );
    }
    count
}

fn collect_rs(directory: &Path, visit: &mut impl FnMut(&Path)) {
    for entry in fs::read_dir(directory).expect("source directory is readable") {
        let path = entry.expect("directory entry is readable").path();
        if path.is_dir() {
            collect_rs(&path, visit);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            visit(&path);
        }
    }
}
