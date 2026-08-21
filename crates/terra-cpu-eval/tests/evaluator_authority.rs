//! B1 evaluator-authority ratchet (audit B1-G1, extended by B1-G2; protects
//! B1-D1 through B1-D6).
//!
//! `StackEvaluator` is the workspace's sole CPU layer-stack execution authority.
//! `EvalWorker` is the approved orchestration wrapper: it may retain state,
//! schedule work, and route results, but it must delegate terrain production to
//! `StackEvaluator` rather than own a second dispatcher. (`EvalScheduler` is
//! interactive eval-session state — token mint, quality ladder, last-good — not
//! an execution seam; B1-D5 removed its only orchestration method, so it no
//! longer appears on this surface.)
//!
//! This source-level guard deliberately uses a small lexical scanner rather
//! than a Rust parser. It scans every crate's `src/` (B1-G2) and enforces:
//!
//! - the discovered authority-sensitive surface across the whole workspace
//!   exactly matches `APPROVED_EXECUTION_SEAMS` plus `DEFERRED_CANDIDATES`;
//! - every approved seam names each of its production entry methods with a live
//!   external call site (outside its defining file and outside `cfg(test)`),
//!   accounts for every `pub fn` it exposes (as an entry point or a justified
//!   internal method), and has a result-level test;
//! - a new evaluator/executor, graph compiler, operator executor, tile
//!   executor, or owner of `ProcessorRegistry` cannot land silently in *any*
//!   crate when it accepts or owns `LayerStack` / `ProcessorRegistry`;
//! - each `DEFERRED_CANDIDATES` entry stays honest (really discovered, reasoned,
//!   and never also approved), so a follow-up cannot quietly forget to claim it;
//! - the evaluator generations retired by #53-#56, and the eval entry points
//!   retired by B1-D5 (#89), stay absent from production.
//!
//! An intentional new orchestration seam must be added to the inventory with
//! honest evidence. A replacement CPU authority additionally requires changing
//! the explicit single-authority assertion below, making that architectural
//! decision visible in review.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SeamRole {
    Authority,
    Orchestrator,
    /// A free-function compiler that lowers a `LayerStack` into an execution plan
    /// another executor consumes (e.g. the GPU preview's `compile_gpu_graph`). It
    /// owns no `ProcessorRegistry` and performs no tree walk; its honesty is a live
    /// call site plus a result test, and it is validated as a `pub fn`, not a type.
    Planner,
}

struct SourceEvidence {
    path: &'static str,
    needle: &'static str,
}

/// A production entry method on a seam paired with a live call site: `caller`'s
/// needle must appear in `caller.path`'s production source (comments, strings,
/// and `cfg(test)` stripped), outside the seam's own defining file. This is the
/// B1-G2 upgrade over the old single `production_caller` — evidence now proves a
/// live call *chain* per entry, not one needle that a dead method could satisfy.
struct EntryPoint {
    method: &'static str,
    caller: SourceEvidence,
}

/// A `pub fn` on a seam that is deliberately not a production entry point — an
/// internal primitive, or a method reached only through `Drop`/another method.
/// It carries no external-caller requirement, only a justification: the same
/// visible-in-review exception the dead-seam guard's `ALLOWED_INERT` uses.
struct InternalMethod {
    method: &'static str,
    justification: &'static str,
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
    /// Every production entry method, each with a live external call site.
    /// Together with `internal_methods` this must account for *every* `pub fn`
    /// the seam exposes — a new public method cannot land without a declaration.
    entry_points: &'static [EntryPoint],
    /// Public methods that are intentionally not entry points.
    internal_methods: &'static [InternalMethod],
    result_test: ResultTestEvidence,
}

/// A workspace-discovered authority-shaped candidate whose full inventory entry
/// is intentionally deferred to a named follow-up. It counts as represented (the
/// honesty scan does not fail on it), but `deferred_candidates_are_honest` keeps
/// it from rotting: a stale entry (candidate vanished/renamed), a blank reason,
/// or one that also became an approved seam all fail — forcing the list to
/// shrink as the follow-ups land.
struct DeferredCandidate {
    name: &'static str,
    path: &'static str,
    reason: &'static str,
}

/// The complete approved workspace layer-stack execution surface.
///
/// Entries are deliberately evidence-bearing. `authority_inventory_is_honest`
/// rejects duplicates, blank reasons, missing definitions, stale entry callers,
/// undeclared `pub fn`s, stale result tests, and any discovered source candidate
/// (in any crate) not represented here or in `DEFERRED_CANDIDATES`.
const APPROVED_EXECUTION_SEAMS: &[ApprovedSeam] = &[
    ApprovedSeam {
        name: "StackEvaluator",
        definition: "crates/terra-cpu-eval/src/lib.rs",
        role: SeamRole::Authority,
        justification: "sole CPU authority; owns ProcessorRegistry and performs the authored LayerStack tree walk",
        entry_points: &[
            EntryPoint {
                method: "new",
                caller: SourceEvidence {
                    path: "crates/terra-io/src/lib.rs",
                    needle: "StackEvaluator::new",
                },
            },
            EntryPoint {
                method: "rebuild_all",
                caller: SourceEvidence {
                    path: "crates/terra-io/src/lib.rs",
                    needle: "evaluator.rebuild_all(",
                },
            },
            EntryPoint {
                method: "rebuild_incremental",
                caller: SourceEvidence {
                    path: "crates/terra-cpu-eval/src/worker.rs",
                    needle: "evaluator.rebuild_incremental(",
                },
            },
            EntryPoint {
                method: "mark_dirty_from",
                caller: SourceEvidence {
                    path: "crates/terra-app/src/app/eval.rs",
                    needle: "evaluator.mark_dirty_from(",
                },
            },
            EntryPoint {
                method: "mark_dirty_from_stage",
                caller: SourceEvidence {
                    path: "crates/terra-app/src/app/eval.rs",
                    needle: "evaluator.mark_dirty_from_stage(",
                },
            },
            EntryPoint {
                method: "mark_dirty_from_eval_stage",
                caller: SourceEvidence {
                    path: "crates/terra-app/src/app/actions/scenarios.rs",
                    needle: "mark_dirty_from_eval_stage(",
                },
            },
            EntryPoint {
                method: "mark_all_dirty",
                caller: SourceEvidence {
                    path: "crates/terra-app/src/app/eval.rs",
                    needle: "evaluator.mark_all_dirty(",
                },
            },
            EntryPoint {
                method: "clear_project_caches",
                caller: SourceEvidence {
                    path: "crates/terra-app/src/app/project.rs",
                    needle: "evaluator.clear_project_caches(",
                },
            },
            EntryPoint {
                method: "mark_dirty_from_region",
                caller: SourceEvidence {
                    path: "crates/terra-cpu-eval/src/worker.rs",
                    needle: "evaluator.mark_dirty_from_region(",
                },
            },
        ],
        internal_methods: &[
            InternalMethod {
                method: "evaluate_nodes",
                justification: "the authored LayerStack tree-walk recursion StackEvaluator drives itself; never an external entry (the single-authority shape check requires it to exist)",
            },
            InternalMethod {
                method: "evaluate_suffix",
                justification: "resumes a CPU suffix eval from a GPU checkpoint at a given layer index; public API retained for CPU/GPU parity tests (terra-gpu-eval engine cfg(test) + parity_matrix), production hybrid resume flows through EvalWorker/rebuild_incremental since b82726c",
            },
        ],
        result_test: ResultTestEvidence {
            path: "crates/terra-cpu-eval/tests/tropical_island_workflow.rs",
            test_name: "tropical_island_evaluates_with_biome_content",
            seam_needle: "StackEvaluator::new",
            result_needle: "height.min_max",
        },
    },
    ApprovedSeam {
        name: "EvalWorker",
        definition: "crates/terra-cpu-eval/src/worker.rs",
        role: SeamRole::Orchestrator,
        justification: "background job transport whose worker thread owns and invokes StackEvaluator",
        entry_points: &[
            EntryPoint {
                method: "spawn",
                caller: SourceEvidence {
                    path: "crates/terra-app/src/app/mod.rs",
                    needle: "EvalWorker::spawn",
                },
            },
            EntryPoint {
                method: "submit",
                caller: SourceEvidence {
                    path: "crates/terra-app/src/app/eval.rs",
                    needle: "eval_worker.submit",
                },
            },
            EntryPoint {
                method: "try_recv_event",
                caller: SourceEvidence {
                    path: "crates/terra-app/src/app/lifecycle.rs",
                    needle: "eval_worker.try_recv_event(",
                },
            },
            EntryPoint {
                method: "stats",
                caller: SourceEvidence {
                    path: "crates/terra-app/src/app/eval.rs",
                    needle: "eval_worker.stats()",
                },
            },
            EntryPoint {
                method: "set_token",
                caller: SourceEvidence {
                    path: "crates/terra-app/src/app/eval.rs",
                    needle: "eval_worker.set_token(",
                },
            },
            EntryPoint {
                method: "restart",
                caller: SourceEvidence {
                    path: "crates/terra-app/src/app/eval.rs",
                    needle: "eval_worker.restart(",
                },
            },
        ],
        internal_methods: &[InternalMethod {
            method: "shutdown",
            justification: "invoked by Drop and restart to stop the worker thread; not an external entry",
        }],
        result_test: ResultTestEvidence {
            path: "crates/terra-cpu-eval/src/worker.rs",
            test_name: "worker_height_mask_uses_layer_input_not_previous_frame",
            seam_needle: "EvalWorker::spawn",
            result_needle: "r.height.get",
        },
    },
    ApprovedSeam {
        name: "compile_gpu_graph",
        definition: "crates/terra-gpu/src/graph.rs",
        role: SeamRole::Planner,
        justification: "GPU preview planner (B1-D6, #90): lowers the LayerStack into per-layer executable plans (kernel, dirty policy, halo) plus the cpu_from boundary. GpuTerrainEngine::evaluate indexes this plan directly and never re-derives a per-layer plan mid-walk",
        entry_points: &[EntryPoint {
            method: "compile_gpu_graph",
            caller: SourceEvidence {
                path: "crates/terra-gpu-eval/src/engine/compiled_plan.rs",
                needle: "compile_gpu_graph(",
            },
        }],
        internal_methods: &[],
        result_test: ResultTestEvidence {
            path: "crates/terra-gpu/tests/support_matrix.rs",
            test_name: "first_unsupported_configuration_owns_cpu_from",
            seam_needle: "compile_gpu_graph(",
            result_needle: "graph.plans",
        },
    },
];

/// Workspace-discovered candidates whose honest inventory entry is deferred to a
/// named follow-up. See `DeferredCandidate` and `deferred_candidates_are_honest`.
///
/// Empty since B1-D6 (#90): `compile_gpu_graph` graduated from a deferral to a real
/// `SeamRole::Planner` entry in `APPROVED_EXECUTION_SEAMS` once the engine began
/// consuming its plan (kernel, dirty policy, halo) instead of re-planning mid-walk.
const DEFERRED_CANDIDATES: &[DeferredCandidate] = &[];

const RETIRED_PATHS: &[&str] = &[
    // #159: CPU evaluation belongs to terra-cpu-eval; terra-core must not grow
    // a compatibility module or a second implementation.
    "crates/terra-core/src/eval",
    "crates/terra-core/src/terrain_eval",
    "crates/terra-core/src/fields/context.rs",
    "crates/terra-core/src/domain/pipeline.rs",
    "crates/terra-core/src/terrain/executor.rs",
    "crates/terra-core/src/terrain/work.rs",
    // B1-D7 (#91): the viewport-region types (`NormalizedRect`/`RegionSet`) existed
    // only to feed the write-only tile plan; restoring the module fails this guard.
    "crates/terra-core/src/terrain/region.rs",
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

/// Public eval entry points retired by B1-D5 (#89) as production-dead: named on
/// the eval surface but never called by production. Banned from *production*
/// source workspace-wide — this scan strips `cfg(test)`, so `dirty_suffix_ids`
/// living in a `#[cfg(test)]` module is fine; re-exposing any as a production
/// item fails. This is B1-D5's revert check: restoring one trips this list.
const RETIRED_B1_D5_SYMBOLS: &[&str] = &[
    "run_step",
    "EvalJob",
    "evaluate_final_height",
    "pin_baked",
    "dirty_suffix_ids",
];

/// The write-only terrain residency plan retired by B1-D7 (#91). The CPU pyramid
/// mirrored GPU residency into `TileRecord`s and a viewport tile plan that no
/// production reader consumed — residency is now GPU-authoritative (the atlas page
/// table, mirrored once into `TileResidencyCache`). Banned from *production* source
/// workspace-wide (this scan strips `cfg(test)`). This is B1-D7's revert check:
/// restoring any of these — the pyramid residency records, the `plan_resident_tiles`
/// screen-space-error planner, or the renderer's `update_visible_tile_plan` HUD
/// feed — trips this list. `NormalizedRect`/`RegionSet` are guarded by their file
/// in `RETIRED_PATHS`, not by name, so a genuinely new rect type stays possible.
const RETIRED_B1_D7_SYMBOLS: &[&str] = &[
    "TileRecord",
    "publish_resident",
    "remove_resident",
    "clear_residency",
    "best_resident_ancestor",
    "level_metrics",
    "plan_resident_tiles",
    "projected_error_px",
    "ViewportTilePlan",
    "ResidentTileSelection",
    "update_visible_tile_plan",
];

/// Retired from the *CPU* evaluator (`terra-cpu-eval/src/lib.rs`) only — not workspace-wide.
/// terra-gpu-eval's `GpuTerrainEngine` legitimately keeps a `last_graph`; rather than
/// bless that with a free-text exemption (which is where B1-D6 grew unnoticed),
/// the GPU graph compiler is discovered by the workspace scan and, since B1-D6
/// (#90), carries an honest `SeamRole::Planner` entry in `APPROVED_EXECUTION_SEAMS`.
const RETIRED_EVAL_SYMBOLS: &[&str] = &["last_graph", "compile_graph", "compile_eval_graph"];

/// Retired from the *GPU* engine (`engine.rs` facade plus its `engine/` module tree)
/// only. B1-D6 (#90)
/// made `compile_gpu_graph`'s plan the single planning authority: the engine walk
/// indexes `last_graph.plans` and must not re-derive per-layer support or kernels
/// mid-walk. Referencing either helper from engine production reintroduces the
/// second planner the fix removed — even if `gpu_plan_for_layer`'s visibility is
/// widened again. This is B1-D6's revert check.
const RETIRED_GPU_ENGINE_SYMBOLS: &[&str] = &["gpu_plan_for_layer", "layer_gpu_supported"];

#[test]
fn authority_inventory_is_honest() {
    let scan = Scan::workspace();
    let candidates = scan.execution_candidates();
    let actual: BTreeSet<(String, String)> = candidates
        .iter()
        .map(|candidate| (candidate.name.clone(), candidate.path.clone()))
        .collect();
    let approved: BTreeSet<(String, String)> = APPROVED_EXECUTION_SEAMS
        .iter()
        .map(|seam| (seam.name.to_string(), seam.definition.to_string()))
        .collect();
    // Deferred candidates are discovered but intentionally lack a full entry (it
    // lands with a named follow-up). They count as represented so the scan does
    // not fail on them; `deferred_candidates_are_honest` enforces their freshness.
    let represented: BTreeSet<(String, String)> = approved
        .iter()
        .cloned()
        .chain(
            DEFERRED_CANDIDATES
                .iter()
                .map(|deferred| (deferred.name.to_string(), deferred.path.to_string())),
        )
        .collect();

    let mut violations = Vec::new();
    for (name, path) in actual.difference(&represented) {
        let details = candidates
            .iter()
            .find(|candidate| candidate.name == *name && candidate.path == *path)
            .map(Candidate::location)
            .unwrap_or_else(|| format!("{path} ({name})"));
        violations.push(format!(
            "unapproved execution candidate {details}; delete it, route through StackEvaluator, \
             add a justified inventory entry with entry-point and result-test evidence, or (if a \
             named follow-up owns it) a DEFERRED_CANDIDATES entry"
        ));
    }
    for (name, path) in approved.difference(&actual) {
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
        validate_seam_entries(&scan, seam, &mut violations);
        validate_result_test_evidence(seam, &mut violations);
    }

    let authorities: Vec<_> = APPROVED_EXECUTION_SEAMS
        .iter()
        .filter(|seam| seam.role == SeamRole::Authority)
        .map(|seam| seam.name)
        .collect();
    if authorities != ["StackEvaluator"] {
        violations.push(format!(
            "the workspace must have exactly one CPU authority named StackEvaluator; inventory has {authorities:?}"
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
            for symbol in RETIRED_B1_D5_SYMBOLS {
                if tokens.contains(symbol) {
                    violations.push(format!(
                        "{}:{} reintroduces B1-D5 production-dead entry point `{symbol}`; it was \
                         retired as an inert eval seam (#89) and must stay out of production",
                        file.path,
                        line_index + 1
                    ));
                }
            }
            for symbol in RETIRED_B1_D7_SYMBOLS {
                if tokens.contains(symbol) {
                    violations.push(format!(
                        "{}:{} reintroduces the B1-D7 write-only residency plan symbol `{symbol}`; \
                         terrain residency is GPU-authoritative (atlas page table + \
                         TileResidencyCache) and the CPU plan was retired (#91)",
                        file.path,
                        line_index + 1
                    ));
                }
            }
        }
    }

    let eval_path = root.join("crates/terra-cpu-eval/src/lib.rs");
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

    let gpu_engine_root = root.join("crates/terra-gpu-eval/src/engine");
    let gpu_engine_facade = root.join("crates/terra-gpu-eval/src/engine.rs");
    let mut gpu_engine_files = source_files_under(&gpu_engine_root, &root);
    gpu_engine_files.push(SourceFile {
        path: relative_path(&root, &gpu_engine_facade),
        production: production_source(&read(&gpu_engine_facade)),
    });
    for file in gpu_engine_files {
        for symbol in RETIRED_GPU_ENGINE_SYMBOLS {
            if contains_ident(&file.production, symbol) {
                violations.push(format!(
                    "{} references `{symbol}`; the GPU engine must consume compile_gpu_graph's plan \
                     (last_graph.plans), not re-derive per-layer support or kernels (B1-D6 #90)",
                    file.path
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "retired evaluator generation returned:\n  {}",
        violations.join("\n  ")
    );
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

/// Every `DEFERRED_CANDIDATES` entry must really be discovered by the workspace
/// scan, carry a reason, and not double as an approved seam. When B1-D6 (#90)
/// reshapes or shrinks the GPU compiler, its deferral goes stale here and forces
/// the honest inventory entry (or its removal) — the deferral's revert check.
#[test]
fn deferred_candidates_are_honest() {
    let scan = Scan::workspace();
    let actual: BTreeSet<(String, String)> = scan
        .execution_candidates()
        .into_iter()
        .map(|candidate| (candidate.name, candidate.path))
        .collect();
    let approved: BTreeSet<&str> = APPROVED_EXECUTION_SEAMS
        .iter()
        .map(|seam| seam.name)
        .collect();

    let mut violations = Vec::new();
    let mut seen = BTreeSet::new();
    for deferred in DEFERRED_CANDIDATES {
        if !seen.insert((deferred.name, deferred.path)) {
            violations.push(format!(
                "DEFERRED_CANDIDATES lists `{}` ({}) more than once",
                deferred.name, deferred.path
            ));
        }
        if deferred.reason.trim().is_empty() {
            violations.push(format!(
                "DEFERRED_CANDIDATES entry `{}` has an empty reason",
                deferred.name
            ));
        }
        if approved.contains(deferred.name) {
            violations.push(format!(
                "`{}` is both approved and deferred; a seam with a real inventory entry needs no deferral",
                deferred.name
            ));
        }
        if !actual.contains(&(deferred.name.to_string(), deferred.path.to_string())) {
            violations.push(format!(
                "deferred candidate `{}` ({}) is no longer discovered; its follow-up landed or it \
                 was renamed — remove the deferral (and add a real inventory entry if it survives)",
                deferred.name, deferred.path
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "deferred-candidate list drifted:\n  {}",
        violations.join("\n  ")
    );
}

/// Revert check (b): an authority-shaped type in *any* crate — not just
/// terra-cpu-eval — is discovered, now that the scan is workspace-wide.
#[test]
fn authority_shaped_type_outside_cpu_eval_is_discovered() {
    let source = r#"
        pub struct RogueTileExecutor;
        impl RogueTileExecutor {
            pub fn execute(&self, stack: &LayerStack) {}
        }
    "#;
    let names: BTreeSet<_> = discover_candidates(
        "crates/terra-render/src/rogue.rs",
        &production_source(source),
    )
    .into_iter()
    .map(|candidate| candidate.name)
    .collect();
    assert!(
        names.contains("RogueTileExecutor"),
        "a *Executor owning a LayerStack method must be discovered regardless of crate"
    );
}

/// Revert check (a): entry-point liveness flags a seam method whose production
/// caller vanished. `validate_entry_caller` reports a fabricated (absent) needle
/// against a real production file as a lost-caller violation.
#[test]
fn entry_point_liveness_detects_a_missing_caller() {
    let seam = ApprovedSeam {
        name: "StackEvaluator",
        definition: "crates/terra-cpu-eval/src/lib.rs",
        role: SeamRole::Authority,
        justification: "fixture",
        entry_points: &[EntryPoint {
            method: "rebuild_all",
            caller: SourceEvidence {
                path: "crates/terra-io/src/lib.rs",
                needle: "evaluator.this_entry_was_removed(",
            },
        }],
        internal_methods: &[],
        result_test: ResultTestEvidence {
            path: "",
            test_name: "",
            seam_needle: "",
            result_needle: "",
        },
    };
    let mut violations = Vec::new();
    validate_entry_caller(&seam, &seam.entry_points[0], &mut violations);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("lost its last production caller")),
        "an absent entry needle must be reported as a lost caller, got: {violations:?}"
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

    // A planner is a free function, not a type: require a `pub fn` of the seam's
    // name that consumes a `LayerStack`, and forbid it from owning layer dispatch
    // (only StackEvaluator may own a ProcessorRegistry).
    if seam.role == SeamRole::Planner {
        let Some((offset, _)) = find_function_item(&file.production, seam.name) else {
            violations.push(format!(
                "planner seam `{}` must expose `pub fn {}` in {}",
                seam.name, seam.name, seam.definition
            ));
            return;
        };
        let item = extract_item(&file.production, offset).unwrap_or_default();
        if !public_fn_names(&file.production).contains(seam.name) {
            violations.push(format!(
                "planner seam `{}` must be a `pub fn`, not a private or `pub(crate)` function",
                seam.name
            ));
        }
        if !contains_ident(&item, "LayerStack") {
            violations.push(format!(
                "planner `{}` must compile a LayerStack into an execution plan",
                seam.name
            ));
        }
        if contains_ident(&item, "ProcessorRegistry") {
            violations.push(format!(
                "planner `{}` owns or invokes ProcessorRegistry directly; only StackEvaluator may own layer dispatch",
                seam.name
            ));
        }
        return;
    }

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
        // Planners are free functions, validated and returned above before any
        // struct definition is required.
        SeamRole::Planner => unreachable!("planner seams return before struct-shape checks"),
    }
}

/// Prove each entry method has a live external call site, that internal methods
/// are justified, and that entry + internal declarations account for *every*
/// `pub fn` the seam exposes (none hidden, none stale).
fn validate_seam_entries(scan: &Scan, seam: &ApprovedSeam, violations: &mut Vec<String>) {
    // No method may be declared twice, across either list.
    let mut declared: BTreeSet<&str> = BTreeSet::new();
    for method in seam
        .entry_points
        .iter()
        .map(|entry| entry.method)
        .chain(seam.internal_methods.iter().map(|internal| internal.method))
    {
        if !declared.insert(method) {
            violations.push(format!(
                "seam `{}` declares method `{}` more than once across entry_points/internal_methods",
                seam.name, method
            ));
        }
    }

    for internal in seam.internal_methods {
        if internal.justification.trim().is_empty() {
            violations.push(format!(
                "seam `{}` internal method `{}` has an empty justification",
                seam.name, internal.method
            ));
        }
    }

    for entry in seam.entry_points {
        validate_entry_caller(seam, entry, violations);
    }

    // A planner is a single free function, not a type with an impl surface, so
    // there are no sibling `pub fn`s to reconcile. A second graph compiler landing
    // in the module is caught instead by `discover_candidates` (any new
    // compile*graph function becomes an unrepresented candidate).
    if seam.role == SeamRole::Planner {
        return;
    }

    // Completeness: reconcile the declared surface against the seam's real
    // `pub fn`s. `validate_seam_shape` already reported a missing/ambiguous
    // definition, so bail quietly here rather than double-reporting.
    let Some(file) = scan.files.iter().find(|file| file.path == seam.definition) else {
        return;
    };
    let matches: Vec<_> = public_structs(&file.production)
        .into_iter()
        .filter(|definition| definition.name == seam.name)
        .collect();
    let [definition] = matches.as_slice() else {
        return;
    };
    let context = associated_type_source(&file.production, definition);
    let public_fns = public_fn_names(&context);

    for method in &public_fns {
        if !declared.contains(method.as_str()) {
            violations.push(format!(
                "seam `{}` exposes `pub fn {}` with no entry_points/internal_methods declaration; \
                 add it as an entry point with a live caller, or justify it as internal",
                seam.name, method
            ));
        }
    }
    for method in &declared {
        if !public_fns.contains(*method) {
            violations.push(format!(
                "seam `{}` declares method `{}` that is no longer a `pub fn` on the seam (renamed or removed)",
                seam.name, method
            ));
        }
    }
}

/// One entry point's production caller must be a real, live call site: below
/// some `crates/*/src` (never a test tree), outside the seam's own defining
/// file, and present in that file's production source (comments, strings, and
/// `cfg(test)` stripped). A missing needle is a seam method that lost its last
/// production caller — B1-G2 revert check (a).
fn validate_entry_caller(seam: &ApprovedSeam, entry: &EntryPoint, violations: &mut Vec<String>) {
    let evidence = &entry.caller;
    let label = format!("{}::{}", seam.name, entry.method);
    if !evidence.path.starts_with("crates/")
        || !evidence.path.contains("/src/")
        || evidence.path.contains("/tests/")
    {
        violations.push(format!(
            "`{label}` entry-point caller must point below crates/*/src: {}",
            evidence.path
        ));
        return;
    }
    if evidence.path == seam.definition {
        violations.push(format!(
            "`{label}` entry-point caller must be outside the seam's defining module"
        ));
    }
    if evidence.needle.trim().is_empty() {
        violations.push(format!("`{label}` entry-point caller has an empty needle"));
        return;
    }
    let path = workspace_root().join(evidence.path);
    if !path.is_file() {
        violations.push(format!(
            "`{label}` entry-point caller {} does not exist",
            path.display()
        ));
        return;
    }
    let source = production_source(&read(&path));
    if !source.contains(evidence.needle) {
        violations.push(format!(
            "`{label}` entry-point evidence is stale: {} no longer contains `{}` outside \
             tests/comments/strings — the seam method lost its last production caller",
            evidence.path, evidence.needle
        ));
    }
}

/// Names of inherent `pub fn`s declared in `source` (a seam's struct item plus
/// its `impl` blocks). `pub(crate)`/`pub(super)` are intentionally excluded —
/// only the genuinely public surface must be accounted for; trait-impl methods
/// (`fn default`, `fn drop`) are not `pub fn` and so never appear.
fn public_fn_names(source: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for line in source.lines() {
        let trimmed = line.trim_start();
        let rest = trimmed
            .strip_prefix("pub fn ")
            .or_else(|| trimmed.strip_prefix("pub async fn "))
            .or_else(|| trimmed.strip_prefix("pub unsafe fn "));
        if let Some(rest) = rest {
            if let Some(name) = identifiers(rest).first() {
                names.insert((*name).to_string());
            }
        }
    }
    names
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
    fn workspace() -> Self {
        let root = workspace_root();
        Self {
            files: source_files_under(&root.join("crates"), &root),
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
