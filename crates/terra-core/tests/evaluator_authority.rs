//! B1 evaluator-authority ratchet (audit B1-G1; protects B1-D1 through B1-D4).
//!
//! `StackEvaluator` is terra-core's sole CPU layer-stack execution authority.
//! `EvalScheduler` and `EvalWorker` are approved orchestration wrappers: they
//! may retain state, schedule work, and route results, but they must delegate
//! terrain production to `StackEvaluator` rather than own a second dispatcher.
//!
//! This source-level guard deliberately uses a small lexical scanner rather
//! than a Rust parser. It enforces four review tripwires:
//!
//! - the discovered authority-sensitive surface exactly matches
//!   `APPROVED_EXECUTION_SEAMS`;
//! - every approved seam has a justification, a production caller outside its
//!   defining module, and a result-level test;
//! - a new evaluator/executor, graph compiler, operator executor, tile
//!   executor, or owner of `ProcessorRegistry` cannot land silently when it
//!   accepts or owns `LayerStack` / `ProcessorRegistry`;
//! - the evaluator generations retired by #53-#56 stay absent.
//!
//! An intentional new orchestration seam must be added to the inventory with
//! honest evidence. A replacement CPU authority additionally requires changing
//! the explicit single-authority assertion below, making that architectural
//! decision visible in review.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SeamRole {
    Authority,
    Orchestrator,
}

struct SourceEvidence {
    path: &'static str,
    needle: &'static str,
}

struct ResultTestEvidence {
    path: &'static str,
    test_name: &'static str,
    seam_needle: &'static str,
    result_needle: &'static str,
}

struct ApprovedSeam {
    name: &'static str,
    definition: &'static str,
    role: SeamRole,
    justification: &'static str,
    production_caller: SourceEvidence,
    result_test: ResultTestEvidence,
}

/// The complete approved terra-core layer-stack execution surface.
///
/// Entries are deliberately evidence-bearing. `authority_inventory_is_honest`
/// rejects duplicates, blank reasons, missing definitions, stale callers, stale
/// tests, and any discovered source candidate not represented here.
const APPROVED_EXECUTION_SEAMS: &[ApprovedSeam] = &[
    ApprovedSeam {
        name: "StackEvaluator",
        definition: "crates/terra-core/src/eval/mod.rs",
        role: SeamRole::Authority,
        justification: "sole CPU authority; owns ProcessorRegistry and performs the authored LayerStack tree walk",
        production_caller: SourceEvidence {
            path: "crates/terra-core/src/document/mod.rs",
            needle: "StackEvaluator::new",
        },
        result_test: ResultTestEvidence {
            path: "crates/terra-core/tests/tropical_island_workflow.rs",
            test_name: "tropical_island_evaluates_with_biome_content",
            seam_needle: "StackEvaluator::new",
            result_needle: "height.min_max",
        },
    },
    ApprovedSeam {
        name: "EvalScheduler",
        definition: "crates/terra-core/src/eval/scheduler.rs",
        role: SeamRole::Orchestrator,
        justification: "quality, cancellation, last-good, and cache orchestration around its owned StackEvaluator",
        production_caller: SourceEvidence {
            path: "crates/terra-app/src/app/eval.rs",
            needle: "self.scheduler.evaluator.evaluate_suffix",
        },
        result_test: ResultTestEvidence {
            path: "crates/terra-core/tests/evaluator_authority.rs",
            test_name: "eval_scheduler_routes_a_returned_terrain_result",
            seam_needle: "EvalScheduler::new",
            result_needle: "result.get(4, 4)",
        },
    },
    ApprovedSeam {
        name: "EvalWorker",
        definition: "crates/terra-core/src/eval/worker.rs",
        role: SeamRole::Orchestrator,
        justification: "background job transport whose worker thread owns and invokes StackEvaluator",
        production_caller: SourceEvidence {
            path: "crates/terra-app/src/app/eval.rs",
            needle: "self.eval_worker.submit",
        },
        result_test: ResultTestEvidence {
            path: "crates/terra-core/src/eval/worker.rs",
            test_name: "worker_height_mask_uses_layer_input_not_previous_frame",
            seam_needle: "EvalWorker::spawn",
            result_needle: "r.height.get",
        },
    },
];

const RETIRED_PATHS: &[&str] = &[
    "crates/terra-core/src/terrain_eval",
    "crates/terra-core/src/fields/context.rs",
    "crates/terra-core/src/domain/pipeline.rs",
    "crates/terra-core/src/terrain/executor.rs",
    "crates/terra-core/src/terrain/work.rs",
];

const RETIRED_SYMBOLS: &[&str] = &[
    "EvalGraph",
    "TerrainContext",
    "TerrainPipelineExecutor",
    "TerrainPipelineStage",
    "RebuildReason",
    "TerrainWorkScheduler",
    "TerrainWorkItem",
    "execute_vector_height_tile",
    "publish_fallback_result",
    "terrain_eval",
];

/// These names are specifically retired from the CPU evaluator. The GPU engine
/// legitimately has its own `last_graph`, so a workspace-wide ban would conflate
/// the live GPU authority with the removed terra-core graph generation.
const RETIRED_EVAL_SYMBOLS: &[&str] = &["last_graph", "compile_graph", "compile_eval_graph"];

#[test]
fn authority_inventory_is_honest() {
    let scan = Scan::core();
    let candidates = scan.execution_candidates();
    let actual: BTreeSet<(String, String)> = candidates
        .iter()
        .map(|candidate| (candidate.name.clone(), candidate.path.clone()))
        .collect();
    let expected: BTreeSet<(String, String)> = APPROVED_EXECUTION_SEAMS
        .iter()
        .map(|seam| (seam.name.to_string(), seam.definition.to_string()))
        .collect();

    let mut violations = Vec::new();
    for (name, path) in actual.difference(&expected) {
        let details = candidates
            .iter()
            .find(|candidate| candidate.name == *name && candidate.path == *path)
            .map(Candidate::location)
            .unwrap_or_else(|| format!("{path} ({name})"));
        violations.push(format!(
            "unapproved execution candidate {details}; delete it, route through StackEvaluator, \
             or add a justified inventory entry with production and result-test evidence"
        ));
    }
    for (name, path) in expected.difference(&actual) {
        violations.push(format!(
            "APPROVED_EXECUTION_SEAMS entry `{name}` at {path} is stale or no longer matches \
             an authority-sensitive source shape; update or remove it"
        ));
    }

    let mut seen_names = BTreeSet::new();
    let mut seen_definitions = BTreeSet::new();
    for seam in APPROVED_EXECUTION_SEAMS {
        if !seen_names.insert(seam.name) {
            violations.push(format!(
                "APPROVED_EXECUTION_SEAMS lists `{}` more than once",
                seam.name
            ));
        }
        if !seen_definitions.insert(seam.definition) {
            violations.push(format!(
                "APPROVED_EXECUTION_SEAMS lists definition {} more than once",
                seam.definition
            ));
        }
        if seam.justification.trim().is_empty() {
            violations.push(format!(
                "APPROVED_EXECUTION_SEAMS entry `{}` has an empty justification",
                seam.name
            ));
        }

        validate_seam_shape(&scan, seam, &mut violations);
        validate_production_evidence(seam, &mut violations);
        validate_result_test_evidence(seam, &mut violations);
    }

    let authorities: Vec<_> = APPROVED_EXECUTION_SEAMS
        .iter()
        .filter(|seam| seam.role == SeamRole::Authority)
        .map(|seam| seam.name)
        .collect();
    if authorities != ["StackEvaluator"] {
        violations.push(format!(
            "terra-core must have exactly one CPU authority named StackEvaluator; inventory has {authorities:?}"
        ));
    }

    assert!(
        violations.is_empty(),
        "evaluator-authority inventory drifted:\n  {}",
        violations.join("\n  ")
    );
}

#[test]
fn retired_evaluator_generations_stay_absent() {
    let root = workspace_root();
    let mut violations = Vec::new();

    for relative in RETIRED_PATHS {
        let path = root.join(relative);
        if path.exists() {
            violations.push(format!(
                "{} reintroduced retired B1 evaluator scaffolding",
                path.display()
            ));
        }
    }

    for file in workspace_source_files() {
        for (line_index, line) in file.production.lines().enumerate() {
            let tokens = identifiers(line);
            for symbol in RETIRED_SYMBOLS {
                if tokens.contains(symbol) {
                    violations.push(format!(
                        "{}:{} references retired symbol `{symbol}`",
                        file.path,
                        line_index + 1
                    ));
                }
            }
        }
    }

    let eval_path = root.join("crates/terra-core/src/eval/mod.rs");
    let eval_source = production_source(&read(&eval_path));
    for symbol in RETIRED_EVAL_SYMBOLS.iter().copied().chain(["terrain_eval"]) {
        if contains_ident(&eval_source, symbol) {
            violations.push(format!(
                "{} references retired `{symbol}`; StackEvaluator must execute its tree walk directly",
                eval_path.display()
            ));
        }
    }
    if !contains_fn_named(&eval_source, &["evaluate_nodes"]) {
        violations.push(format!(
            "{} no longer contains StackEvaluator::evaluate_nodes",
            eval_path.display()
        ));
    }

    let pyramid_path = root.join("crates/terra-core/src/terrain/pyramid.rs");
    let pyramid = production_source(&read(&pyramid_path));
    if !pyramid.contains("pub fn publish_resident") || !pyramid.contains("handle: TilePageHandle") {
        violations.push(format!(
            "{} must require a TilePageHandle when publishing resident terrain",
            pyramid_path.display()
        ));
    }

    assert!(
        violations.is_empty(),
        "retired evaluator generation returned:\n  {}",
        violations.join("\n  ")
    );
}

/// Result-level proof that the approved scheduler wrapper returns terrain made
/// by its `StackEvaluator`, rather than isolated scheduling metadata.
#[test]
fn eval_scheduler_routes_a_returned_terrain_result() {
    use terra_core::analyze::LevelStepSettings;
    use terra_core::eval::EvalScheduler;
    use terra_core::heightfield::HeightfieldMetrics;
    use terra_core::layer::{FlatParams, Layer, LayerKind, LayerStack};

    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Flat authority fixture",
        LayerKind::Flat(FlatParams { height: 17.0 }),
    ));

    let mut scheduler = EvalScheduler::new();
    let token = scheduler.request_rebuild();
    let result = scheduler
        .run_step(
            &stack,
            HeightfieldMetrics::new(8, 8, 80.0, 80.0),
            8,
            8,
            token,
            &LevelStepSettings::default(),
            &[],
            &HashMap::new(),
        )
        .expect("scheduled evaluation")
        .expect("scheduled terrain result");

    assert_eq!(result.get(4, 4), 17.0);
}

#[test]
fn scanner_recognizes_authority_sensitive_shapes() {
    let source = r#"
        pub struct ShadowEvaluator { registry: ProcessorRegistry }

        pub struct ShadowExecutor;
        impl ShadowExecutor {
            pub fn run(&mut self, stack: &LayerStack) {}
        }

        fn compile_execution_graph(
            stack: &LayerStack,
        ) -> Graph { todo!() }

        pub struct LayerOperator;
        impl LayerOperator {
            pub fn execute(&self, stack: &LayerStack) {}
        }

        pub fn evaluate_height_tile(stack: &LayerStack) {}
    "#;
    let names: BTreeSet<_> = discover_candidates("fixture.rs", &production_source(source))
        .into_iter()
        .map(|candidate| candidate.name)
        .collect();

    assert_eq!(
        names,
        BTreeSet::from([
            "LayerOperator".to_string(),
            "ShadowEvaluator".to_string(),
            "ShadowExecutor".to_string(),
            "compile_execution_graph".to_string(),
            "evaluate_height_tile".to_string(),
        ])
    );
}

#[test]
fn scanner_ignores_non_production_decoys() {
    let source = r#"
        // pub struct CommentEvaluator { stack: LayerStack }
        const EXAMPLE: &str = "pub struct StringExecutor { registry: ProcessorRegistry }";

        #[cfg(test)]
        mod tests {
            pub struct TestEvaluator { stack: LayerStack }
            fn compile_test_graph(stack: &LayerStack) {}
        }

        pub struct DependencyGraph;
        impl DependencyGraph {
            pub fn build_from_stack(stack: &LayerStack) -> Self { Self }
        }

        pub struct LandscapeEvolutionOperator;
        impl LandscapeEvolutionOperator {
            pub fn evaluate(&self, input: LandscapeEvolutionInput) {}
        }
    "#;

    assert!(
        discover_candidates("fixture.rs", &production_source(source)).is_empty(),
        "comments, strings, cfg(test), dependency graphs, and non-stack operators are not CPU authorities"
    );
}

fn validate_seam_shape(scan: &Scan, seam: &ApprovedSeam, violations: &mut Vec<String>) {
    let Some(file) = scan.files.iter().find(|file| file.path == seam.definition) else {
        violations.push(format!(
            "approved seam `{}` definition {} does not exist",
            seam.name, seam.definition
        ));
        return;
    };
    let definitions: Vec<_> = public_structs(&file.production)
        .into_iter()
        .filter(|definition| definition.name == seam.name)
        .collect();
    if definitions.len() != 1 {
        violations.push(format!(
            "approved seam `{}` must have exactly one public struct definition in {}; found {}",
            seam.name,
            seam.definition,
            definitions.len()
        ));
        return;
    }

    let context = associated_type_source(&file.production, &definitions[0]);
    match seam.role {
        SeamRole::Authority => {
            if !contains_ident(&context, "ProcessorRegistry")
                || !contains_fn_named(&context, &["evaluate_nodes"])
            {
                violations.push(format!(
                    "authority `{}` must own ProcessorRegistry and expose evaluate_nodes",
                    seam.name
                ));
            }
        }
        SeamRole::Orchestrator => {
            if !contains_ident(&context, "StackEvaluator") {
                violations.push(format!(
                    "orchestrator `{}` no longer routes through StackEvaluator",
                    seam.name
                ));
            }
            if contains_ident(&context, "ProcessorRegistry") {
                violations.push(format!(
                    "orchestrator `{}` owns or invokes ProcessorRegistry directly; only StackEvaluator may own layer dispatch",
                    seam.name
                ));
            }
        }
    }
}

fn validate_production_evidence(seam: &ApprovedSeam, violations: &mut Vec<String>) {
    let evidence = &seam.production_caller;
    if !evidence.path.starts_with("crates/")
        || !evidence.path.contains("/src/")
        || evidence.path.contains("/tests/")
    {
        violations.push(format!(
            "`{}` production evidence must point below crates/*/src: {}",
            seam.name, evidence.path
        ));
        return;
    }
    if evidence.path == seam.definition {
        violations.push(format!(
            "`{}` production caller must be outside its defining module",
            seam.name
        ));
    }
    if evidence.needle.trim().is_empty() {
        violations.push(format!(
            "`{}` production caller evidence has an empty needle",
            seam.name
        ));
        return;
    }

    let path = workspace_root().join(evidence.path);
    if !path.is_file() {
        violations.push(format!(
            "`{}` production caller {} does not exist",
            seam.name,
            path.display()
        ));
        return;
    }
    let source = production_source(&read(&path));
    if !source.contains(evidence.needle) {
        violations.push(format!(
            "`{}` production caller evidence is stale: {} no longer contains `{}` outside tests/comments/strings",
            seam.name, evidence.path, evidence.needle
        ));
    }
}

fn validate_result_test_evidence(seam: &ApprovedSeam, violations: &mut Vec<String>) {
    let evidence = &seam.result_test;
    for (label, value) in [
        ("test name", evidence.test_name),
        ("seam needle", evidence.seam_needle),
        ("result needle", evidence.result_needle),
    ] {
        if value.trim().is_empty() {
            violations.push(format!(
                "`{}` result-test evidence has an empty {label}",
                seam.name
            ));
        }
    }

    let path = workspace_root().join(evidence.path);
    if !path.is_file() {
        violations.push(format!(
            "`{}` result-test file {} does not exist",
            seam.name,
            path.display()
        ));
        return;
    }
    let source = stripped_source(&read(&path));
    let Some((offset, test_body)) = find_function_item(&source, evidence.test_name) else {
        violations.push(format!(
            "`{}` result test `{}` is missing from {}",
            seam.name, evidence.test_name, evidence.path
        ));
        return;
    };
    if !has_test_attribute(&source[..offset]) {
        violations.push(format!(
            "`{}` evidence function `{}` is not marked #[test]",
            seam.name, evidence.test_name
        ));
    }
    if !test_body.contains(evidence.seam_needle) {
        violations.push(format!(
            "`{}` result test `{}` no longer routes through `{}`",
            seam.name, evidence.test_name, evidence.seam_needle
        ));
    }
    if !test_body.contains(evidence.result_needle) || !contains_assertion(&test_body) {
        violations.push(format!(
            "`{}` result test `{}` must assert returned terrain using `{}`",
            seam.name, evidence.test_name, evidence.result_needle
        ));
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Candidate {
    name: String,
    path: String,
    line: usize,
    kind: &'static str,
}

impl Candidate {
    fn location(&self) -> String {
        format!(
            "`{}` ({}) at {}:{}",
            self.name, self.kind, self.path, self.line
        )
    }
}

struct StructDef {
    name: String,
    line: usize,
    offset: usize,
}

struct SourceFile {
    path: String,
    production: String,
}

struct Scan {
    files: Vec<SourceFile>,
}

impl Scan {
    fn core() -> Self {
        let root = workspace_root();
        let source_root = root.join("crates/terra-core/src");
        Self {
            files: source_files_under(&source_root, &root),
        }
    }

    fn execution_candidates(&self) -> Vec<Candidate> {
        let mut candidates = Vec::new();
        for file in &self.files {
            candidates.extend(discover_candidates(&file.path, &file.production));
        }
        candidates.sort_by(|a, b| (&a.path, a.line, &a.name).cmp(&(&b.path, b.line, &b.name)));
        candidates
    }
}

fn discover_candidates(path: &str, source: &str) -> Vec<Candidate> {
    let mut found: BTreeMap<(String, String), Candidate> = BTreeMap::new();
    let approved_names: BTreeSet<_> = APPROVED_EXECUTION_SEAMS
        .iter()
        .map(|seam| seam.name)
        .collect();

    for definition in public_structs(source) {
        let context = associated_type_source(source, &definition);
        let definition_item = extract_item(source, definition.offset).unwrap_or_default();
        let sensitive = contains_sensitive_type(&context);
        let approved = approved_names.contains(definition.name.as_str());
        let evaluator_or_executor =
            definition.name.ends_with("Evaluator") || definition.name.ends_with("Executor");
        let owns_registry = definition.name != "ProcessorRegistry"
            && contains_ident(&definition_item, "ProcessorRegistry");
        let operator_executor = definition.name.ends_with("Operator")
            && contains_fn_named(&context, &["evaluate", "execute"]);

        let kind = if evaluator_or_executor && sensitive {
            Some("public evaluator/executor")
        } else if owns_registry {
            Some("ProcessorRegistry owner")
        } else if operator_executor && sensitive {
            Some("operator executor")
        } else if approved {
            Some("approved orchestration wrapper")
        } else {
            None
        };

        if let Some(kind) = kind {
            insert_candidate(
                &mut found,
                Candidate {
                    name: definition.name,
                    path: path.to_string(),
                    line: definition.line,
                    kind,
                },
            );
        }
    }

    for (name, line, offset) in function_defs(source) {
        let lower = name.to_ascii_lowercase();
        let item = extract_item(source, offset).unwrap_or_default();
        if !contains_sensitive_type(&item) {
            continue;
        }
        let graph_compiler = lower.contains("compile") && lower.contains("graph");
        let tile_executor =
            lower.contains("tile") && (lower.contains("execute") || lower.contains("evaluate"));
        let operator_executor =
            lower.contains("operator") && (lower.contains("execute") || lower.contains("evaluate"));
        let kind = if graph_compiler {
            Some("execution graph compiler")
        } else if tile_executor {
            Some("tile-layer executor")
        } else if operator_executor {
            Some("operator executor")
        } else {
            None
        };
        if let Some(kind) = kind {
            insert_candidate(
                &mut found,
                Candidate {
                    name,
                    path: path.to_string(),
                    line,
                    kind,
                },
            );
        }
    }

    found.into_values().collect()
}

fn insert_candidate(found: &mut BTreeMap<(String, String), Candidate>, candidate: Candidate) {
    found
        .entry((candidate.name.clone(), candidate.path.clone()))
        .or_insert(candidate);
}

fn public_structs(source: &str) -> Vec<StructDef> {
    let mut definitions = Vec::new();
    let mut offset = 0;
    for (line_index, line) in source.lines().enumerate() {
        let tokens = identifiers(line);
        if let Some(index) = tokens.iter().position(|token| *token == "struct") {
            let is_public = index > 0 && tokens[index - 1] == "pub";
            if is_public {
                if let Some(name) = tokens.get(index + 1) {
                    definitions.push(StructDef {
                        name: (*name).to_string(),
                        line: line_index + 1,
                        offset,
                    });
                }
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
                definitions.push(((*name).to_string(), line_index + 1, offset));
            }
        }
        offset += line.len() + 1;
    }
    definitions
}

fn associated_type_source(source: &str, definition: &StructDef) -> String {
    let mut associated = extract_item(source, definition.offset).unwrap_or_default();
    let mut offset = 0;
    for line in source.lines() {
        let tokens = identifiers(line);
        if let Some(index) = tokens.iter().position(|token| *token == "impl") {
            if tokens[index + 1..].contains(&definition.name.as_str()) {
                if let Some(item) = extract_item(source, offset) {
                    associated.push('\n');
                    associated.push_str(&item);
                }
            }
        }
        offset += line.len() + 1;
    }
    associated
}

fn extract_item(source: &str, start: usize) -> Option<String> {
    let tail = source.get(start..)?;
    let open = tail.find('{');
    let semicolon = tail.find(';');
    if semicolon.is_some_and(|semicolon| open.is_none_or(|open| semicolon < open)) {
        let end = semicolon? + 1;
        return Some(tail[..end].to_string());
    }

    let open = open?;
    let mut depth = 0_i32;
    for (relative, ch) in tail[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let end = open + relative + ch.len_utf8();
                    return Some(tail[..end].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn find_function_item(source: &str, name: &str) -> Option<(usize, String)> {
    for (candidate, _, offset) in function_defs(source) {
        if candidate == name {
            return extract_item(source, offset).map(|item| (offset, item));
        }
    }
    None
}

fn has_test_attribute(prefix: &str) -> bool {
    prefix
        .lines()
        .rev()
        .take(4)
        .any(|line| line.trim() == "#[test]")
}

fn contains_sensitive_type(source: &str) -> bool {
    contains_ident(source, "LayerStack") || contains_ident(source, "ProcessorRegistry")
}

fn contains_fn_named(source: &str, names: &[&str]) -> bool {
    function_defs(source)
        .iter()
        .any(|(name, _, _)| names.contains(&name.as_str()))
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

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn workspace_root() -> PathBuf {
    manifest_dir().join("..").join("..")
}

fn workspace_source_files() -> Vec<SourceFile> {
    let root = workspace_root();
    source_files_under(&root.join("crates"), &root)
}

fn source_files_under(directory: &Path, workspace: &Path) -> Vec<SourceFile> {
    let mut paths = Vec::new();
    collect_rs(directory, &mut paths);
    paths.sort();
    paths
        .into_iter()
        .filter(|path| {
            path.components()
                .any(|component| component.as_os_str() == "src")
        })
        .map(|path| SourceFile {
            path: relative_path(workspace, &path),
            production: production_source(&read(&path)),
        })
        .collect()
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn read(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Strip comments/string contents and omit `#[cfg(test)]` items. This is a
/// deliberately small source tripwire, not a Rust parser.
fn production_source(source: &str) -> String {
    let stripped = stripped_source(source);
    let mut output = String::new();
    let mut pending_test_item = false;
    let mut skipped_depth: Option<i32> = None;

    for line in stripped.lines() {
        let delta = brace_delta(line);

        if let Some(depth) = skipped_depth.as_mut() {
            *depth += delta;
            if *depth <= 0 {
                skipped_depth = None;
            }
            output.push('\n');
            continue;
        }

        let trimmed = line.trim_start();
        if trimmed.starts_with("#[cfg(test)]") {
            pending_test_item = true;
            output.push('\n');
            continue;
        }

        if pending_test_item {
            if trimmed.is_empty() || trimmed.starts_with("#[") {
                output.push('\n');
                continue;
            }
            if delta > 0 {
                skipped_depth = Some(delta);
            }
            pending_test_item = false;
            output.push('\n');
            continue;
        }

        output.push_str(line);
        output.push('\n');
    }

    output
}

fn stripped_source(source: &str) -> String {
    let mut output = String::new();
    for line in source.lines() {
        output.push_str(&strip_comments_and_strings(line));
        output.push('\n');
    }
    output
}

fn brace_delta(line: &str) -> i32 {
    line.bytes().fold(0, |depth, byte| match byte {
        b'{' => depth + 1,
        b'}' => depth - 1,
        _ => depth,
    })
}

fn strip_comments_and_strings(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;

    while let Some(ch) = chars.next() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            out.push(' ');
            continue;
        }

        if ch == '"' {
            in_string = true;
            out.push(' ');
        } else if ch == '/' && chars.peek() == Some(&'/') {
            break;
        } else {
            out.push(ch);
        }
    }

    out
}
