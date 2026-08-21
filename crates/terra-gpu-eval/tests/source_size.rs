use std::path::{Path, PathBuf};

const SPLIT_CANDIDATE_LINE_LIMIT: usize = 1_500;

#[test]
fn rust_sources_stay_below_the_split_candidate_limit() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut rust_sources = Vec::new();
    collect_rust_sources(&crate_root.join("src"), &mut rust_sources);
    collect_rust_sources(&crate_root.join("tests"), &mut rust_sources);

    let mut oversized = rust_sources
        .into_iter()
        .filter_map(|path| {
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
            let line_count = source.lines().count();
            (line_count > SPLIT_CANDIDATE_LINE_LIMIT).then_some((path, line_count))
        })
        .collect::<Vec<_>>();
    oversized.sort_by(|left, right| left.0.cmp(&right.0));

    assert!(
        oversized.is_empty(),
        "terra-gpu-eval Rust sources exceeded the {SPLIT_CANDIDATE_LINE_LIMIT}-line split-candidate limit:\n{}",
        oversized
            .iter()
            .map(|(path, lines)| format!("  {}: {lines} lines", path.display()))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn collect_rust_sources(directory: &Path, sources: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", directory.display()));
    for entry in entries {
        let path = entry.expect("source directory entry").path();
        if path.is_dir() {
            collect_rust_sources(&path, sources);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            sources.push(path);
        }
    }
}
