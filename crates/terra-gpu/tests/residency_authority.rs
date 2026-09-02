//! GPU terrain residency authority contract (#171).
//!
//! The shader-visible atlas page table is residency truth. `TileResidencyCache`
//! is its one mutable CPU policy mirror, owned by `GpuTileAtlas`. Planning names
//! and algorithms are deliberately unrestricted; this test discovers persistent
//! tile-keyed stores by source shape instead of preserving a historical symbol
//! blacklist.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const APPROVED_TILE_STORES: &[(&str, &str)] = &[(
    "TileResidencyCache",
    "crates/terra-core/src/terrain/cache.rs",
)];
const APPROVED_CACHE_OWNERS: &[(&str, &str)] =
    &[("GpuTileAtlas", "crates/terra-gpu/src/tile_cache.rs")];

struct SourceEvidence {
    path: &'static str,
    needle: &'static str,
}

struct TestEvidence {
    path: &'static str,
    test_name: &'static str,
    planner_needle: &'static str,
    result_needle: &'static str,
}

struct ApprovedDemandPlanner {
    name: &'static str,
    definition: &'static str,
    justification: &'static str,
    production_consumer: SourceEvidence,
    behavior_test: TestEvidence,
    bounded_work_test: TestEvidence,
}

/// A planner is added here only when its output is live in production and its
/// behavior tests prove downstream effect and demand-bounded work. Empty today:
/// issue #171 removes the prohibition without restoring the former planner.
const APPROVED_DEMAND_PLANNERS: &[ApprovedDemandPlanner] = &[];

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Candidate {
    name: String,
    path: String,
    line: usize,
}

#[derive(Clone, Debug)]
struct StructDef {
    name: String,
    line: usize,
    offset: usize,
}

#[derive(Clone, Debug)]
struct SourceFile {
    path: String,
    production: String,
}

#[test]
fn gpu_atlas_has_one_cpu_residency_policy_store() {
    let files = workspace_source_files();
    let stores = discover_tile_stores(&files);
    let owners = discover_cache_owners(&files);
    let actual_stores: BTreeSet<_> = stores
        .iter()
        .map(|candidate| (candidate.name.as_str(), candidate.path.as_str()))
        .collect();
    let actual_owners: BTreeSet<_> = owners
        .iter()
        .map(|candidate| (candidate.name.as_str(), candidate.path.as_str()))
        .collect();
    let expected_stores: BTreeSet<_> = APPROVED_TILE_STORES.iter().copied().collect();
    let expected_owners: BTreeSet<_> = APPROVED_CACHE_OWNERS.iter().copied().collect();

    assert_eq!(
        actual_stores, expected_stores,
        "persistent mutable TerrainTileKey collections changed: {stores:#?}. A transient demand \
         index may be inventoried only with a production consumer and a bounded-work behavior \
         test; a second residency database is forbidden"
    );
    assert_eq!(
        actual_owners, expected_owners,
        "TileResidencyCache ownership changed: {owners:#?}; GpuTileAtlas must remain its sole owner"
    );

    let construction_sites: Vec<_> = files
        .iter()
        .flat_map(|file| {
            file.production
                .match_indices("TileResidencyCache::new(")
                .map(move |(offset, _)| {
                    let line = file.production[..offset]
                        .bytes()
                        .filter(|byte| *byte == b'\n')
                        .count()
                        + 1;
                    format!("{}:{line}", file.path)
                })
        })
        .collect();
    assert_eq!(
        construction_sites.len(),
        1,
        "TileResidencyCache must have one production construction site in GpuTileAtlas; found \
         {construction_sites:?}"
    );
    assert!(construction_sites[0].starts_with("crates/terra-gpu/src/tile_cache.rs:"));

    let atlas = files
        .iter()
        .find(|file| file.path == "crates/terra-gpu/src/tile_cache.rs")
        .expect("GpuTileAtlas source is scanned");
    for evidence in [
        "residency",
        "insert",
        "write_page_entry",
        "page_table",
        "virtual_page_table",
        "configure_hierarchy",
        "invalidate_key",
    ] {
        assert!(
            contains_ident(&atlas.production, evidence),
            "GpuTileAtlas must couple CPU policy and page-table updates; missing `{evidence}`"
        );
    }
}

#[test]
fn store_detector_is_shape_based_not_name_based() {
    let fixture = production_source(
        r#"
        struct EntirelyNewName {
            pages: std::collections::HashMap<TerrainTileKey, PageState>,
        }

        struct ViewportTilePlan {
            ordered: Vec<TerrainTileKey>,
        }

        fn best_resident_ancestor() {}
        fn projected_error_px() -> f32 { 0.0 }
        "#,
    );
    let files = vec![SourceFile {
        path: "fixture.rs".into(),
        production: fixture,
    }];
    let stores = discover_tile_stores(&files);
    assert_eq!(stores.len(), 1);
    assert_eq!(stores[0].name, "EntirelyNewName");
}

#[test]
fn planner_detector_requires_tile_output_and_view_input_not_historical_names() {
    let fixture = production_source(
        r#"
        fn frobnicate(camera: &OrbitCamera) -> Vec<TerrainTileKey> {
            let _ = camera;
            Vec::new()
        }

        fn plan_resident_tiles() {}
        fn update_visible_tile_plan() {}
        "#,
    );
    let planners = discover_demand_planners("fixture.rs", &fixture);
    assert_eq!(planners.len(), 1);
    assert_eq!(planners[0].name, "frobnicate");
}

#[test]
fn demand_planner_inventory_requires_live_bounded_consumers() {
    let planners: Vec<_> = workspace_source_files()
        .iter()
        .flat_map(|file| discover_demand_planners(&file.path, &file.production))
        .collect();
    let actual: BTreeSet<_> = planners
        .iter()
        .map(|planner| (planner.name.as_str(), planner.path.as_str()))
        .collect();
    let approved: BTreeSet<_> = APPROVED_DEMAND_PLANNERS
        .iter()
        .map(|planner| (planner.name, planner.definition))
        .collect();
    assert_eq!(
        actual, approved,
        "terrain demand planner inventory drifted: {planners:#?}. Add an evidence-bearing entry \
         showing selected keys affect requested, uploaded, or rendered pages, plus a large-world \
         bounded-work behavior test"
    );

    for planner in APPROVED_DEMAND_PLANNERS {
        assert!(
            !planner.justification.trim().is_empty(),
            "planner `{}` needs an architectural justification",
            planner.name
        );
        assert_ne!(
            planner.definition, planner.production_consumer.path,
            "planner `{}` consumer must be outside its defining file",
            planner.name
        );
        let caller = production_source(&read_workspace(planner.production_consumer.path));
        assert!(
            caller.contains(planner.production_consumer.needle),
            "planner `{}` has no live production consumer `{}` in {}",
            planner.name,
            planner.production_consumer.needle,
            planner.production_consumer.path
        );
        validate_test_evidence(planner, &planner.behavior_test, "downstream behavior");
        validate_test_evidence(planner, &planner.bounded_work_test, "bounded work");
    }
}

#[test]
fn terrain_work_scheduler_is_bounded_demand_not_residency_or_fictional_execution() {
    let scheduler = read_workspace("crates/terra-core/src/terrain/work.rs");
    for forbidden in [
        "TerrainWorkKind",
        "TerrainWorkExecutor",
        "TilePageHandle",
        "TileResidencyCache",
        "GpuTileAtlas",
    ] {
        assert!(
            !scheduler.contains(forbidden),
            "terrain work policy must not own residency or fictional executors: {forbidden}"
        );
    }
    for evidence in [
        "capacity",
        "dequeue_budgeted",
        "duplicate_demand_is_one_work_item",
        "aging_live_request_cannot_starve_under_repeated_new_demand",
    ] {
        assert!(
            scheduler.contains(evidence),
            "missing scheduler evidence: {evidence}"
        );
    }
    let consumer = read_workspace("crates/terra-app/src/app/eval.rs");
    for evidence in [
        "terrain_tile_scheduler.reconcile",
        "terrain_tile_scheduler.dequeue_budgeted",
        "publish_pyramid_tile_current",
        "upload_height_tile_current",
        "production_scheduler_choice_controls_first_uploaded_page",
    ] {
        assert!(
            consumer.contains(evidence),
            "missing production evidence: {evidence}"
        );
    }
}

fn validate_test_evidence(planner: &ApprovedDemandPlanner, evidence: &TestEvidence, purpose: &str) {
    let source = sanitize_rust(&read_workspace(evidence.path));
    let (_, body) = find_function_item(&source, evidence.test_name).unwrap_or_else(|| {
        panic!(
            "planner `{}` {purpose} test `{}` is missing from {}",
            planner.name, evidence.test_name, evidence.path
        )
    });
    assert!(
        body.contains(evidence.planner_needle),
        "planner `{}` {purpose} test must invoke `{}`",
        planner.name,
        evidence.planner_needle
    );
    assert!(
        body.contains(evidence.result_needle) && contains_assertion(&body),
        "planner `{}` {purpose} test must assert `{}`",
        planner.name,
        evidence.result_needle
    );
}

fn discover_tile_stores(files: &[SourceFile]) -> Vec<Candidate> {
    discover_struct_candidates(files, |item| {
        contains_ident(item, "TerrainTileKey")
            && ["HashMap", "BTreeMap", "HashSet", "BTreeSet"]
                .iter()
                .any(|collection| contains_ident(item, collection))
    })
}

fn discover_cache_owners(files: &[SourceFile]) -> Vec<Candidate> {
    discover_struct_candidates(files, |item| {
        contains_ident(item, "TileResidencyCache")
            && !item
                .split_whitespace()
                .collect::<Vec<_>>()
                .windows(2)
                .any(|pair| pair == ["struct", "TileResidencyCache"])
    })
}

fn discover_struct_candidates(
    files: &[SourceFile],
    predicate: impl Fn(&str) -> bool,
) -> Vec<Candidate> {
    let mut found = Vec::new();
    for file in files {
        for definition in struct_defs(&file.production) {
            let item = extract_item(&file.production, definition.offset).unwrap_or_default();
            if predicate(&item) {
                found.push(Candidate {
                    name: definition.name,
                    path: file.path.clone(),
                    line: definition.line,
                });
            }
        }
    }
    found.sort();
    found
}

fn discover_demand_planners(path: &str, source: &str) -> Vec<Candidate> {
    let mut found = Vec::new();
    for (name, line, offset) in function_defs(source) {
        let item = extract_item(source, offset).unwrap_or_default();
        let has_tile_output = item.contains("->")
            && contains_ident(&item, "TerrainTileKey")
            && ["Vec", "HashSet", "BTreeSet"]
                .iter()
                .any(|collection| contains_ident(&item, collection));
        let has_view_input = ["OrbitCamera", "Camera", "Viewport"]
            .iter()
            .any(|input| contains_ident(&item, input));
        if has_tile_output && has_view_input {
            found.push(Candidate {
                name,
                path: path.into(),
                line,
            });
        }
    }
    found
}

fn struct_defs(source: &str) -> Vec<StructDef> {
    let mut definitions = Vec::new();
    let mut offset = 0;
    for (line_index, line) in source.lines().enumerate() {
        let tokens = identifiers(line);
        if let Some(index) = tokens.iter().position(|token| *token == "struct") {
            if let Some(name) = tokens.get(index + 1) {
                definitions.push(StructDef {
                    name: (*name).into(),
                    line: line_index + 1,
                    offset,
                });
            }
        }
        offset += line.len() + 1;
    }
    definitions
}

fn function_defs(source: &str) -> Vec<(String, usize, usize)> {
    let mut definitions = Vec::new();
    let mut offset = 0;
    for (line_index, line) in source.lines().enumerate() {
        let tokens = identifiers(line);
        if let Some(index) = tokens.iter().position(|token| *token == "fn") {
            if let Some(name) = tokens.get(index + 1) {
                definitions.push(((*name).into(), line_index + 1, offset));
            }
        }
        offset += line.len() + 1;
    }
    definitions
}

fn find_function_item(source: &str, name: &str) -> Option<(usize, String)> {
    function_defs(source)
        .into_iter()
        .find(|(candidate, _, _)| candidate == name)
        .and_then(|(_, _, offset)| extract_item(source, offset).map(|item| (offset, item)))
}

fn extract_item(source: &str, start: usize) -> Option<String> {
    let tail = source.get(start..)?;
    let open = tail.find('{');
    let semicolon = tail.find(';');
    if semicolon.is_some_and(|semicolon| open.is_none_or(|open| semicolon < open)) {
        return Some(tail[..=semicolon?].into());
    }
    let open = open?;
    let mut depth = 0_i32;
    for (relative, ch) in tail[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(tail[..open + relative + 1].into());
                }
            }
            _ => {}
        }
    }
    None
}

fn contains_ident(source: &str, ident: &str) -> bool {
    identifiers(source).contains(&ident)
}

fn contains_assertion(source: &str) -> bool {
    identifiers(source)
        .iter()
        .any(|token| token.starts_with("assert"))
}

fn identifiers(source: &str) -> Vec<&str> {
    source
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|token| !token.is_empty())
        .collect()
}

fn workspace_source_files() -> Vec<SourceFile> {
    let workspace = workspace_root();
    let mut paths = Vec::new();
    collect_rs(&workspace.join("crates"), &mut paths);
    paths.sort();
    paths
        .into_iter()
        .filter(|path| {
            path.components()
                .any(|component| component.as_os_str() == "src")
        })
        .map(|path| SourceFile {
            path: path
                .strip_prefix(&workspace)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/"),
            production: production_source(
                &fs::read_to_string(&path)
                    .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display())),
            ),
        })
        .collect()
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn read_workspace(relative: &str) -> String {
    let path = workspace_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn collect_rs(directory: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", directory.display()))
        .flatten()
    {
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn production_source(source: &str) -> String {
    let sanitized = sanitize_rust(source);
    let mut out = String::with_capacity(sanitized.len());
    let mut depth = 0_i32;
    let mut pending_test = false;
    let mut skip_base = None;
    for line in sanitized.split_inclusive('\n') {
        let delta = brace_delta(line);
        let trimmed = line.trim();
        if let Some(base) = skip_base {
            push_blank_line(&mut out, line);
            depth += delta;
            if depth <= base {
                skip_base = None;
            }
            continue;
        }
        if trimmed.contains("#[cfg(test)]") || trimmed.contains("#[test]") {
            push_blank_line(&mut out, line);
            let base = depth;
            depth += delta;
            if depth > base {
                skip_base = Some(base);
            } else {
                pending_test = true;
            }
            continue;
        }
        if pending_test {
            push_blank_line(&mut out, line);
            if trimmed.is_empty() || trimmed.starts_with("#[") {
                depth += delta;
                continue;
            }
            pending_test = false;
            let base = depth;
            depth += delta;
            if depth > base {
                skip_base = Some(base);
            }
            continue;
        }
        out.push_str(line);
        depth += delta;
    }
    out
}

fn sanitize_rust(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = bytes.to_vec();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index..].starts_with(b"//") {
            let end = bytes[index..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |offset| index + offset);
            blank(&mut out, index, end);
            index = end;
        } else if bytes[index..].starts_with(b"/*") {
            let start = index;
            index += 2;
            let mut depth = 1;
            while index < bytes.len() && depth > 0 {
                if bytes[index..].starts_with(b"/*") {
                    depth += 1;
                    index += 2;
                } else if bytes[index..].starts_with(b"*/") {
                    depth -= 1;
                    index += 2;
                } else {
                    index += 1;
                }
            }
            blank(&mut out, start, index);
        } else if bytes[index] == b'"' {
            let start = index;
            index += 1;
            while index < bytes.len() {
                if bytes[index] == b'\\' {
                    index = (index + 2).min(bytes.len());
                } else if bytes[index] == b'"' {
                    index += 1;
                    break;
                } else {
                    index += 1;
                }
            }
            blank(&mut out, start, index);
        } else {
            index += 1;
        }
    }
    String::from_utf8(out).expect("sanitized Rust remains UTF-8")
}

fn blank(out: &mut [u8], start: usize, end: usize) {
    for byte in &mut out[start..end] {
        if *byte != b'\n' && *byte != b'\r' {
            *byte = b' ';
        }
    }
}

fn push_blank_line(out: &mut String, line: &str) {
    for byte in line.bytes() {
        out.push(if byte == b'\n' { '\n' } else { ' ' });
    }
}

fn brace_delta(line: &str) -> i32 {
    line.bytes().filter(|byte| *byte == b'{').count() as i32
        - line.bytes().filter(|byte| *byte == b'}').count() as i32
}
