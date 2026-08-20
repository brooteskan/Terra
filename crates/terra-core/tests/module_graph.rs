//! terra-core purity + module-graph ratchet (audit A3-G2).
//!
//! Station A3 (#1) has two invariants that held only by discipline: terra-core
//! must not depend on `wgpu` or a UI crate, and its top-level module graph must
//! stay a DAG. A2 landed `terra-gui/tests/purity.rs` and B2 landed
//! `terra-core/tests/dead_seams.rs` as cargo-test guards; this is the equivalent
//! for A3. G2 (#85) closes the fail-open gaps in the first cut of this guard:
//! renamed/target-specific Cargo dependencies, crate-root re-export escapes,
//! unclassified modules, and acyclic-but-upward edges all now fail closed.
//!
//! Six rules over `crates/terra-core/src/**/*.rs` and resolved Cargo metadata:
//!
//! - **Rule 1** (`terra_core_is_pure`): every resolved dependency of terra-core
//!   — by its *real* package name, from `cargo metadata` — must sit in
//!   [`ALLOWED_DEPS`] for its kind (normal/dev/build), and none may be
//!   target-specific. Renames (`gpu = { package = "wgpu" }`) and
//!   `[target.'cfg(..)'.dependencies]` tables cannot hide a crate, because the
//!   package name and target come from cargo, not a hand-rolled TOML scan.
//!   Stripped source must additionally contain no `wgpu::` / `use wgpu` /
//!   `terra_gui` / `terra_render` / `terra_app` token.
//! - **Rule 2** (`no_root_qualified_paths`): production code must reach a sibling
//!   module through `crate::<module>::…`, never through a crate-root re-export
//!   (`crate::Heightfield`) or a `super::…` chain that climbs out to the crate
//!   root. Both would hide a real module→module edge from the graph below.
//! - **Rule 3** (`every_module_is_classified`): the keys of [`MODULE_DEPENDENCIES`]
//!   must equal the set of `pub mod` declarations in `lib.rs` exactly, and every
//!   listed target must be a real module. A new `pub mod` fails the build until
//!   it is classified — the partition is exhaustive by construction.
//! - **Rule 4** (`module_edges_match_allowlist`): the production cross-module
//!   edges scanned from source must equal the flattened [`MODULE_DEPENDENCIES`]
//!   allowlist exactly. Two-sided: a new edge (even an acyclic upward one) fails
//!   immediately, and a fix that drops an edge must delete its allowlist entry in
//!   the same commit — the list only shrinks, consciously.
//! - **Rule 5** (`allowlist_is_a_dag`): the allowlist edges must themselves form
//!   a DAG. Combined with Rule 4 this makes a cycle impossible to reintroduce,
//!   even by editing the allowlist.
//! - **Rule 6** (`purity_allowlist_excludes_forbidden`): a careless edit to
//!   [`ALLOWED_DEPS`] cannot itself admit `wgpu`/`winit`/`egui`/`terra-*`, save
//!   the pure leaf siblings named in [`ALLOWED_SIBLING_DEPS`] (`terra-jobs`).
//!
//! Kept deliberately dumb — a hand-written source lexer and `serde_json` over
//! cargo's own output, no `syn`/`regex` — so it grows no new dependencies
//! (`serde_json` is already a terra-core dependency) and cannot rot, like
//! `purity.rs` and `dead_seams.rs`. The scanning conventions:
//!
//! - The lexer ([`strip_source`]) blanks line comments (`//`), nested block
//!   comments (`/* /* */ */`), string literals (`"…"`, escapes honoured), raw
//!   strings (`r"…"`, `r#"…"#`, `br"…"`), and char literals (`'x'`, `'\''`) while
//!   leaving lifetimes (`&'a T`) intact. A `crate::layer` inside any of those is
//!   not an edge, and braces inside them never perturb the `#[cfg(test)]` tracker.
//! - **Production only.** `#[cfg(test)]`-attributed modules/items are skipped, so
//!   a test-only `use crate::foo` is not an edge. `cfg(all(test, …))` and other
//!   compound forms count as production (the conservative, fail-closed choice).
//! - Cross-module edges are read solely from `crate::<module>::…` paths (in `use`
//!   statements and inline), including grouped and nested `crate::{ … }` imports.
//!   Rule 2 guarantees no production edge can hide behind a root re-export, so
//!   this is now a closed model of the crate's real module graph. Macro-generated
//!   `crate::` paths are out of scope (terra-core defines no such macro).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// Allowed Cargo dependencies of terra-core, by section. Exact, two-sided: a
/// dependency not listed here fails Rule 1, and a listed name that Cargo.toml no
/// longer declares fails too — the allowlist mirrors the manifest and only
/// changes by conscious edit. Names are *resolved package names* (what
/// `cargo metadata` reports), so a `{ package = "wgpu" }` rename cannot pass.
const ALLOWED_DEPS: &[(&str, &[&str])] = &[
    (
        "dependencies",
        &[
            "glam",
            "image",
            "log",
            "profiling",
            "rayon",
            "serde",
            "serde_json",
            "terra-jobs",
            "thiserror",
            "uuid",
        ],
    ),
    ("build-dependencies", &[]),
    ("dev-dependencies", &["approx"]),
];

/// The only `terra-*` sibling crates terra-core is consciously allowed to depend
/// on: pure leaf primitives that carry no GPU/UI/domain code. `terra-jobs` (issue
/// #101) is the cancellation + parallel-fill primitive and depends only on
/// `rayon`. Every other `terra-*` crate stays forbidden by prefix, and listing a
/// name here still requires a matching [`ALLOWED_DEPS`] entry (Rule 1).
const ALLOWED_SIBLING_DEPS: &[&str] = &["terra-jobs"];

/// Crate names (exact) that terra-core must never take as any dependency.
/// `terra-*` crates are rejected by prefix in addition to these. Enforced both
/// against resolved dependencies (Rule 1) and against [`ALLOWED_DEPS`] itself
/// (Rule 6), so the allowlist can never be edited to admit one.
const FORBIDDEN_DEPS: &[&str] = &["wgpu", "winit", "egui"];

/// Tokens whose presence in stripped source betrays a GPU/UI leak.
const FORBIDDEN_SOURCE_TOKENS: &[&str] = &[
    "wgpu::",
    "use wgpu",
    "terra_gui",
    "terra_render",
    "terra_app",
];

/// The exhaustive production module-dependency graph of terra-core: for every
/// `pub mod` in `lib.rs`, the exact set of sibling modules it references through
/// `crate::<module>::…` in production code. This is the whole ratchet — Rules 3,
/// 4 and 5 read it. Regenerate the block from a failing Rule 4 (its message
/// prints a paste-ready replacement) and review new edges by eye before pasting.
const MODULE_DEPENDENCIES: &[(&str, &[&str])] = &[
    (
        "analyze",
        &[
            "filter_params",
            "geology",
            "geomorph",
            "heightfield",
            "hydro",
            "mask",
            "mask_types",
            "material_schema",
            "noise",
            "quality",
            "spatial_kernels",
        ],
    ),
    (
        "authoring",
        &[
            "analyze",
            "field_data",
            "heightfield",
            "hydro",
            "landscape_evolution",
            "mask",
        ],
    ),
    ("biome_definition", &["layer", "mask_ir", "mask_types"]),
    ("biome_paint", &["ids", "mask"]),
    (
        "climate",
        &["heightfield", "mask", "material_schema", "spatial_kernels"],
    ),
    (
        "command",
        &[
            "authoring",
            "deps",
            "field_data",
            "layer",
            "mask",
            "operation_placement",
            "raster",
            "terrain_plan",
        ],
    ),
    (
        "contextual_create",
        &[
            "authoring",
            "biome_paint",
            "command",
            "document",
            "layer",
            "matter_sim",
            "operation_placement",
            "shape_object",
            "simulation_scenario",
            "world_rules",
        ],
    ),
    ("deps", &["layer", "mask"]),
    (
        "document",
        &[
            "analyze",
            "biome_definition",
            "biome_paint",
            "command",
            "deps",
            "domain",
            "heightfield",
            "landscape_blueprint",
            "layer",
            "mask",
            "rebuild_state",
            "shape_object",
            "simulation_scenario",
            "sparse_paint",
            "world_rules",
        ],
    ),
    (
        "domain",
        &[
            "biome_definition",
            "biome_paint",
            "landscape_blueprint",
            "layer",
        ],
    ),
    (
        "eval",
        &[
            "analyze",
            "authoring",
            "climate",
            "field_data",
            "fields",
            "generators",
            "heightfield",
            "hydro",
            "invalidation",
            "landscape_blueprint",
            "landscape_evolution",
            "layer",
            "mask",
            "quality",
            "surface",
            "tiling",
            "volumetric",
        ],
    ),
    (
        "field_data",
        &["geology", "heightfield", "mask_field", "spatial_kernels"],
    ),
    (
        "fields",
        &[
            "field_data",
            "geology",
            "heightfield",
            "mask_field",
            "material_schema",
        ],
    ),
    ("filter_params", &["invalidation", "noise"]),
    (
        "generators",
        &[
            "analyze",
            "filter_params",
            "geology",
            "geomorph",
            "heightfield",
            "hydro",
            "mask",
            "material_schema",
            "noise",
            "raster",
        ],
    ),
    ("geology", &["noise"]),
    (
        "geomorph",
        &["heightfield", "mask", "noise", "spatial_kernels"],
    ),
    ("heightfield", &[]),
    (
        "hydro",
        &[
            "geology",
            "geomorph",
            "heightfield",
            "mask",
            "mask_types",
            "material_schema",
        ],
    ),
    ("ids", &[]),
    ("invalidation", &[]),
    ("landscape_blueprint", &[]),
    (
        "landscape_evolution",
        &[
            "analyze",
            "field_data",
            "geomorph",
            "heightfield",
            "hydro",
            "mask",
            "noise",
        ],
    ),
    (
        "landscape_style",
        &[
            "analyze",
            "authoring",
            "generators",
            "hydro",
            "landscape_evolution",
            "material_schema",
        ],
    ),
    (
        "layer",
        &[
            "analyze",
            "authoring",
            "biome_paint",
            "field_data",
            "generators",
            "hydro",
            "ids",
            "invalidation",
            "landscape_blueprint",
            "landscape_evolution",
            "mask",
            "material_schema",
            "noise",
            "raster",
            "scatter",
            "volumetric",
        ],
    ),
    (
        "mask",
        &[
            "invalidation",
            "mask_execution",
            "mask_field",
            "mask_ir",
            "mask_types",
            "spatial_kernels",
        ],
    ),
    (
        "mask_execution",
        &[
            "heightfield",
            "ids",
            "mask_field",
            "mask_ir",
            "mask_types",
            "noise",
            "spatial_kernels",
        ],
    ),
    ("mask_field", &["heightfield", "simd_ops"]),
    (
        "mask_ir",
        &["ids", "mask_field", "mask_types", "raster", "simd_ops"],
    ),
    ("mask_types", &["ids"]),
    ("material_schema", &["geology", "mask", "mask_types"]),
    (
        "matter_sim",
        &["domain", "field_data", "simulation_scenario"],
    ),
    ("noise", &[]),
    ("operation_placement", &["layer", "mask"]),
    ("quality", &[]),
    ("raster", &[]),
    (
        "realism_benchmark",
        &[
            "analyze",
            "document",
            "eval",
            "heightfield",
            "landscape_style",
            "layer",
            "world_archetype",
        ],
    ),
    (
        "rebuild_feedback",
        &[
            "deps",
            "document",
            "domain",
            "layer",
            "rebuild_state",
            "simulation_scenario",
        ],
    ),
    (
        "rebuild_state",
        &[
            "deps",
            "domain",
            "layer",
            "simulation_scenario",
            "world_rules",
        ],
    ),
    ("scatter", &["heightfield", "mask", "spatial_kernels"]),
    ("shader_progress", &[]),
    ("shape_history", &["authoring", "layer"]),
    ("shape_object", &["authoring", "ids"]),
    ("simd_ops", &[]),
    (
        "simulation_scenario",
        &["biome_definition", "domain", "field_data", "layer", "mask"],
    ),
    ("sparse_paint", &["biome_paint", "ids"]),
    ("spatial_kernels", &["heightfield", "mask_field"]),
    (
        "surface",
        &[
            "climate",
            "fields",
            "geology",
            "heightfield",
            "mask",
            "material_schema",
            "scatter",
            "spatial_kernels",
        ],
    ),
    ("terrain", &["field_data", "heightfield", "layer"]),
    (
        "terrain_plan",
        &[
            "deps",
            "field_data",
            "ids",
            "invalidation",
            "layer",
            "mask",
            "tiling",
        ],
    ),
    ("terrain_recipe", &["layer"]),
    (
        "test_fixtures",
        &["document", "heightfield", "layer", "mask"],
    ),
    ("tiling", &["heightfield", "invalidation", "layer"]),
    ("volumetric", &["heightfield", "mask_field", "noise"]),
    (
        "world_archetype",
        &[
            "authoring",
            "biome_definition",
            "biome_paint",
            "document",
            "heightfield",
            "landscape_blueprint",
            "landscape_style",
            "layer",
            "mask",
            "shape_object",
            "sparse_paint",
        ],
    ),
    (
        "world_rules",
        &[
            "biome_definition",
            "domain",
            "heightfield",
            "landscape_blueprint",
            "layer",
            "mask",
        ],
    ),
];

// ===========================================================================
// Rule 1 — purity (resolved Cargo metadata + source tokens)
// ===========================================================================

#[test]
fn terra_core_is_pure() {
    let mut violations = Vec::new();

    let json = cargo_metadata_json();
    let deps = parse_dependencies(&json).expect("parse `cargo metadata` output");
    violations.extend(purity_violations(&deps));
    violations.extend(source_token_violations());

    assert!(
        violations.is_empty(),
        "terra-core purity guard failed:\n  {}",
        violations.join("\n  "),
    );
}

// ===========================================================================
// Rule 2 — no crate-root escape hides a module edge
// ===========================================================================

#[test]
fn no_root_qualified_paths() {
    let modules = module_set();
    let mut violations = Vec::new();

    for file in &scan().files {
        if file.module.is_none() {
            continue; // lib.rs / main.rs own no module
        }
        let rel = file.rel.display().to_string();
        let prod = production_lines(&file.stripped);
        violations.extend(root_qualified_violations(&rel, &prod, &modules));
        violations.extend(super_escape_violations(&rel, file.depth, &prod));
    }

    assert!(
        violations.is_empty(),
        "root-escape guard failed — production code must use `crate::<module>::…`:\n  {}",
        violations.join("\n  "),
    );
}

// ===========================================================================
// Rule 3 — every module is classified
// ===========================================================================

#[test]
fn every_module_is_classified() {
    let modules = module_set();
    let violations = classification_violations(&modules, MODULE_DEPENDENCIES);
    assert!(
        violations.is_empty(),
        "module-classification guard failed — MODULE_DEPENDENCIES must partition every \
         `pub mod` in lib.rs:\n  {}",
        violations.join("\n  "),
    );
}

// ===========================================================================
// Rule 4 — scanned edges match the allowlist exactly
// ===========================================================================

#[test]
fn module_edges_match_allowlist() {
    let modules = module_set();
    let scanned = scanned_edges();
    let allow = allowlist_edges();

    let extra: Vec<_> = scanned.difference(&allow).cloned().collect();
    let missing: Vec<_> = allow.difference(&scanned).cloned().collect();

    let mut msg = String::new();
    if !extra.is_empty() {
        msg.push_str("\nNEW edges — add them to MODULE_DEPENDENCIES (review each first):\n");
        for (a, b) in &extra {
            msg.push_str(&format!("    {a} -> {b}\n"));
        }
    }
    if !missing.is_empty() {
        msg.push_str("\nStale allowlist edges — a change removed these; delete the entries:\n");
        for (a, b) in &missing {
            msg.push_str(&format!("    {a} -> {b}\n"));
        }
    }
    if !extra.is_empty() || !missing.is_empty() {
        msg.push_str("\nPaste-ready MODULE_DEPENDENCIES (replaces the constant wholesale):\n");
        msg.push_str(&render_allowlist(&modules, &scanned));
    }

    assert!(
        extra.is_empty() && missing.is_empty(),
        "module-edge ratchet drift ({} scanned, {} allowlisted):{}",
        scanned.len(),
        allow.len(),
        msg,
    );
}

// ===========================================================================
// Rule 5 — the allowlist is a DAG
// ===========================================================================

#[test]
fn allowlist_is_a_dag() {
    let modules = module_set();
    let edges = allowlist_edges();
    let violations = cycle_violations(&modules, &edges);
    assert!(
        violations.is_empty(),
        "allowlist-DAG guard failed — MODULE_DEPENDENCIES encodes a cycle:\n  {}",
        violations.join("\n  "),
    );
}

// ===========================================================================
// Rule 6 — the dependency allowlist cannot itself admit a forbidden crate
// ===========================================================================

#[test]
fn purity_allowlist_excludes_forbidden() {
    let mut violations = Vec::new();
    for (section, names) in ALLOWED_DEPS {
        for name in *names {
            if FORBIDDEN_DEPS.contains(name)
                || (name.starts_with("terra-") && !ALLOWED_SIBLING_DEPS.contains(name))
            {
                violations.push(format!(
                    "ALLOWED_DEPS[{section}] lists forbidden crate `{name}`"
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "purity allowlist is self-inconsistent:\n  {}",
        violations.join("\n  "),
    );
}

// ===========================================================================
// Rule 1 core — dependency purity over resolved metadata
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DepKind {
    Normal,
    Dev,
    Build,
}

impl DepKind {
    fn section(self) -> &'static str {
        match self {
            DepKind::Normal => "dependencies",
            DepKind::Dev => "dev-dependencies",
            DepKind::Build => "build-dependencies",
        }
    }
}

/// A resolved dependency of terra-core: its *real* package name (post-rename),
/// its kind, and its target predicate (`Some` iff platform-specific).
#[derive(Debug, Clone, PartialEq, Eq)]
struct DepRecord {
    kind: DepKind,
    name: String,
    target: Option<String>,
}

fn allowed_for(section: &str) -> &'static [&'static str] {
    ALLOWED_DEPS
        .iter()
        .find(|(s, _)| *s == section)
        .map(|(_, names)| *names)
        .unwrap_or(&[])
}

/// Every way a dependency can violate purity: it names a forbidden crate, is
/// target-specific, is absent from the allowlist, or the allowlist names it but
/// the manifest no longer declares it.
fn purity_violations(deps: &[DepRecord]) -> Vec<String> {
    let mut v = Vec::new();

    for d in deps {
        let section = d.kind.section();
        if FORBIDDEN_DEPS.contains(&d.name.as_str())
            || (d.name.starts_with("terra-") && !ALLOWED_SIBLING_DEPS.contains(&d.name.as_str()))
        {
            v.push(format!(
                "[{section}] resolves to forbidden dependency `{}`; terra-core must stay free of \
                 GPU/UI and sibling terra-* crates (a rename cannot hide it)",
                d.name
            ));
        }
        if let Some(target) = &d.target {
            v.push(format!(
                "[{section}] dependency `{}` is target-specific (`{target}`); terra-core is \
                 platform-independent and a target table must not smuggle in a dependency",
                d.name
            ));
        }
        if !allowed_for(section).contains(&d.name.as_str()) {
            v.push(format!(
                "[{section}] dependency `{}` is not in the terra-core allowlist; add it \
                 consciously to ALLOWED_DEPS if it is genuinely intended",
                d.name
            ));
        }
    }

    for (section, names) in ALLOWED_DEPS {
        for name in *names {
            let declared = deps
                .iter()
                .any(|d| d.kind.section() == *section && d.name == *name);
            if !declared {
                v.push(format!(
                    "[{section}] allowlist names `{name}` but Cargo.toml no longer declares it; \
                     delete the stale ALLOWED_DEPS entry"
                ));
            }
        }
    }

    v
}

/// Run `cargo metadata --no-deps` in the terra-core manifest directory. Fails
/// loudly (never silently skips) so the guard stays fail-closed.
fn cargo_metadata_json() -> String {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let out = Command::new(cargo)
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(manifest_dir())
        .output()
        .expect("run `cargo metadata`");
    assert!(
        out.status.success(),
        "`cargo metadata` failed: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    String::from_utf8(out.stdout).expect("`cargo metadata` output is UTF-8")
}

/// Resolved dependencies of the terra-core package parsed from `cargo metadata`
/// JSON. `kind` is `null` (normal) / `"dev"` / `"build"`; `name` is the real
/// package name (renames live in a separate `rename` field we ignore); `target`
/// is the `cfg(..)`/triple predicate for target-specific tables.
fn parse_dependencies(json: &str) -> Result<Vec<DepRecord>, String> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid metadata JSON: {e}"))?;
    let packages = value
        .get("packages")
        .and_then(|p| p.as_array())
        .ok_or("metadata has no `packages` array")?;
    let pkg = packages
        .iter()
        .find(|p| p.get("name").and_then(|n| n.as_str()) == Some("terra-core"))
        .ok_or("metadata has no `terra-core` package")?;
    let deps = pkg
        .get("dependencies")
        .and_then(|d| d.as_array())
        .ok_or("terra-core package has no `dependencies` array")?;

    let mut out = Vec::new();
    for dep in deps {
        let name = dep
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or("dependency has no `name`")?
            .to_string();
        let kind = match dep.get("kind").and_then(|k| k.as_str()) {
            None => DepKind::Normal, // JSON null → normal dependency
            Some("dev") => DepKind::Dev,
            Some("build") => DepKind::Build,
            Some(other) => return Err(format!("unknown dependency kind `{other}` for `{name}`")),
        };
        let target = dep
            .get("target")
            .and_then(|t| t.as_str())
            .map(|s| s.to_string());
        out.push(DepRecord { kind, name, target });
    }
    Ok(out)
}

/// Stripped-source occurrences of a forbidden GPU/UI token.
fn source_token_violations() -> Vec<String> {
    let mut v = Vec::new();
    for file in &scan().files {
        for (line_no, code) in &file.stripped {
            for tok in FORBIDDEN_SOURCE_TOKENS {
                if code.contains(tok) {
                    v.push(format!(
                        "{}:{} contains forbidden token `{tok}`",
                        file.rel.display(),
                        line_no
                    ));
                }
            }
        }
    }
    v
}

// ===========================================================================
// Rule 2 core — root-qualified and super-escape detection
// ===========================================================================

/// Every production `crate::…` on these lines whose first segment is not a real
/// module — a crate-root re-export (`crate::Heightfield`), a glob (`crate::*`),
/// or a group member that re-exports the root. Each hides a real edge.
fn root_qualified_violations(
    rel: &str,
    prod: &[(usize, String)],
    modules: &BTreeSet<String>,
) -> Vec<String> {
    let mut v = Vec::new();
    for (line_no, code) in prod {
        for occ in crate_paths(code) {
            match occ {
                CratePath::Single(id) => {
                    if id.is_empty() {
                        continue; // `crate::` with no following ident (e.g. macro noise)
                    }
                    if !modules.contains(&id) {
                        v.push(format!(
                            "{rel}:{line_no}: root-qualified path `crate::{id}` — reach the \
                             defining module with `crate::<module>::…` instead of a lib.rs \
                             re-export",
                        ));
                    }
                }
                CratePath::Glob => v.push(format!(
                    "{rel}:{line_no}: `crate::*` glob re-export hides every edge it pulls in; \
                     import `crate::<module>::…` explicitly",
                )),
                CratePath::Group(members) => {
                    for id in members {
                        if id == "*" {
                            v.push(format!(
                                "{rel}:{line_no}: `crate::{{ …, * }}` glob member hides edges",
                            ));
                        } else if !id.is_empty() && !modules.contains(&id) {
                            v.push(format!(
                                "{rel}:{line_no}: root-qualified group member `crate::{{ …, {id} }}` \
                                 — import `crate::<module>::{id}` instead",
                            ));
                        }
                    }
                }
                CratePath::Unclosed => v.push(format!(
                    "{rel}:{line_no}: multi-line `crate::{{ … }}` group is not supported by the \
                     guard — keep the group on one line or use `crate::<module>::…`",
                )),
            }
        }
    }
    v
}

/// Production `super::` chains that climb out of the file's own top-level module
/// to the crate root. A file at module depth `D` (e.g. `src/a/b.rs` → 2) may use
/// at most `D-1` stacked `super::`; `D` or more reaches the crate root, which is
/// the same root-namespace escape Rule 2 forbids for `crate::`.
fn super_escape_violations(rel: &str, depth: usize, prod: &[(usize, String)]) -> Vec<String> {
    let mut v = Vec::new();
    for (line_no, code) in prod {
        let run = max_super_run(code);
        if run > 0 && run >= depth {
            v.push(format!(
                "{rel}:{line_no}: `{}` climbs to the crate root (module depth {depth}); reference \
                 the sibling module with `crate::<module>::…`",
                "super::".repeat(run),
            ));
        }
    }
    v
}

/// Longest run of consecutive `super::` segments starting at a token boundary.
fn max_super_run(code: &str) -> usize {
    const S: &str = "super::";
    let bytes = code.as_bytes();
    let mut best = 0;
    let mut i = 0;
    while let Some(rel) = code[i..].find(S) {
        let start = i + rel;
        if start > 0 && is_ident_byte(bytes[start - 1]) {
            i = start + S.len();
            continue;
        }
        let mut run = 1;
        let mut p = start + S.len();
        while code[p..].starts_with(S) {
            run += 1;
            p += S.len();
        }
        best = best.max(run);
        i = p;
    }
    best
}

// ===========================================================================
// Rule 3 core — exhaustive classification
// ===========================================================================

fn classification_violations(
    modules: &BTreeSet<String>,
    allowlist: &[(&str, &[&str])],
) -> Vec<String> {
    let mut v = Vec::new();
    let keys: BTreeSet<String> = allowlist.iter().map(|(m, _)| m.to_string()).collect();

    for m in modules {
        if !keys.contains(m) {
            v.push(format!(
                "module `{m}` is declared in lib.rs but absent from MODULE_DEPENDENCIES; \
                 classify it (with `&[]` if it references no sibling)"
            ));
        }
    }
    for k in &keys {
        if !modules.contains(k) {
            v.push(format!(
                "MODULE_DEPENDENCIES names `{k}`, which is not a `pub mod` in lib.rs; remove it"
            ));
        }
    }
    for (m, targets) in allowlist {
        for t in *targets {
            if !modules.contains(*t) {
                v.push(format!(
                    "MODULE_DEPENDENCIES entry `{m}` lists unknown target `{t}`"
                ));
            }
        }
    }
    v
}

// ===========================================================================
// Rule 4/5 core — allowlist edges, DAG check, rendering
// ===========================================================================

fn allowlist_edges() -> BTreeSet<(String, String)> {
    MODULE_DEPENDENCIES
        .iter()
        .flat_map(|(m, targets)| targets.iter().map(move |t| (m.to_string(), t.to_string())))
        .collect()
}

/// SCC over an edge set; any component with more than one node is a cycle.
fn cycle_violations(modules: &BTreeSet<String>, edges: &BTreeSet<(String, String)>) -> Vec<String> {
    let mut nodes: BTreeSet<String> = modules.clone();
    for (a, b) in edges {
        nodes.insert(a.clone());
        nodes.insert(b.clone());
    }
    let (comp, comp_size) = strongly_connected(&nodes, edges);
    let mut by_comp: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (m, c) in &comp {
        if comp_size.get(c).copied().unwrap_or(0) > 1 {
            by_comp.entry(*c).or_default().push(m.clone());
        }
    }
    by_comp
        .into_values()
        .map(|members| format!("cycle among modules: {}", members.join(", ")))
        .collect()
}

/// Render `MODULE_DEPENDENCIES` from a scanned edge set: one entry per module,
/// targets sorted, ready to paste over the constant.
fn render_allowlist(modules: &BTreeSet<String>, edges: &BTreeSet<(String, String)>) -> String {
    let mut by_source: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for m in modules {
        by_source.entry(m.as_str()).or_default();
    }
    for (a, b) in edges {
        by_source.entry(a.as_str()).or_default().push(b.as_str());
    }
    let mut out = String::from("const MODULE_DEPENDENCIES: &[(&str, &[&str])] = &[\n");
    for (m, targets) in &by_source {
        if targets.is_empty() {
            out.push_str(&format!("    (\"{m}\", &[]),\n"));
        } else {
            let list = targets
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("    (\"{m}\", &[{list}]),\n"));
        }
    }
    out.push_str("];\n");
    out
}

// ===========================================================================
// Graph construction (scanned edges + SCC)
// ===========================================================================

fn scanned_edges() -> BTreeSet<(String, String)> {
    static EDGES: OnceLock<BTreeSet<(String, String)>> = OnceLock::new();
    EDGES.get_or_init(build_edges).clone()
}

fn build_edges() -> BTreeSet<(String, String)> {
    let modules = module_set();
    let mut edges = BTreeSet::new();
    for file in &scan().files {
        let Some(src) = file.module.as_deref() else {
            continue;
        };
        if !modules.contains(src) {
            continue;
        }
        for (_, code) in production_lines(&file.stripped) {
            for tgt in extract_targets(&code, &modules) {
                if tgt != src {
                    edges.insert((src.to_string(), tgt));
                }
            }
        }
    }
    edges
}

fn module_set() -> BTreeSet<String> {
    static MODULES: OnceLock<BTreeSet<String>> = OnceLock::new();
    MODULES
        .get_or_init(|| {
            let lib = scan()
                .files
                .iter()
                .find(|f| f.rel == Path::new("src").join("lib.rs"))
                .expect("terra-core has src/lib.rs");
            let mut mods = BTreeSet::new();
            for (_, code) in &lib.stripped {
                if let Some(name) = pub_mod_name(code) {
                    mods.insert(name);
                }
            }
            mods
        })
        .clone()
}

/// Name declared by a `pub mod X;` line, if any.
fn pub_mod_name(code: &str) -> Option<String> {
    let toks: Vec<&str> = code.split_whitespace().collect();
    match toks.as_slice() {
        ["pub", "mod", name, ..] => {
            let name = leading_ident(name);
            (!name.is_empty()).then(|| name.to_string())
        }
        _ => None,
    }
}

/// One parsed `crate::…` occurrence.
enum CratePath {
    Single(String),
    Group(Vec<String>),
    Glob,
    Unclosed,
}

/// Every `crate::…` occurrence named on `code` (whole-word `crate`), parsed into
/// its leading segment(s). Groups are brace-matched (nested groups honoured).
fn crate_paths(code: &str) -> Vec<CratePath> {
    const NEEDLE: &str = "crate::";
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    let mut search = 0;
    while let Some(rel) = code[search..].find(NEEDLE) {
        let idx = search + rel;
        search = idx + NEEDLE.len();
        if idx > 0 && is_ident_byte(bytes[idx - 1]) {
            continue; // `xcrate::` — not the `crate` keyword
        }
        let rest = code[idx + NEEDLE.len()..].trim_start();
        if let Some(after) = rest.strip_prefix('{') {
            match match_group(after) {
                Some(members) => out.push(CratePath::Group(members)),
                None => out.push(CratePath::Unclosed),
            }
        } else if rest.starts_with('*') {
            out.push(CratePath::Glob);
        } else {
            out.push(CratePath::Single(leading_ident(rest).to_string()));
        }
    }
    out
}

/// Given the text just after `crate::{`, brace-match to the closing `}` and
/// return each top-level member's leading identifier. `None` if unterminated on
/// the line.
fn match_group(after: &str) -> Option<Vec<String>> {
    let mut depth = 1i32;
    let mut end = None;
    for (k, ch) in after.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(k);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end?;
    let inner = &after[..end];
    let mut members = Vec::new();
    let mut d = 0i32;
    let mut start = 0;
    for (k, ch) in inner.char_indices() {
        match ch {
            '{' => d += 1,
            '}' => d -= 1,
            ',' if d == 0 => {
                members.push(member_ident(&inner[start..k]));
                start = k + 1;
            }
            _ => {}
        }
    }
    members.push(member_ident(&inner[start..]));
    Some(members.into_iter().filter(|s| !s.is_empty()).collect())
}

/// Leading identifier (or `*`) of one group member, trimmed.
fn member_ident(member: &str) -> String {
    let member = member.trim();
    if member.starts_with('*') {
        "*".to_string()
    } else {
        leading_ident(member).to_string()
    }
}

/// Every `crate::<module>::…` target named on `code`, groups included. Only
/// names in `modules` are returned (so a root re-export contributes nothing —
/// Rule 2 is what forbids it).
fn extract_targets(code: &str, modules: &BTreeSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    for occ in crate_paths(code) {
        match occ {
            CratePath::Single(id) => {
                if modules.contains(&id) {
                    out.push(id);
                }
            }
            CratePath::Group(members) => {
                for id in members {
                    if modules.contains(&id) {
                        out.push(id);
                    }
                }
            }
            CratePath::Glob | CratePath::Unclosed => {}
        }
    }
    out
}

/// Kosaraju SCC: returns each node's component id and the size of every
/// component. Nodes with no edges each get their own singleton component.
fn strongly_connected(
    nodes: &BTreeSet<String>,
    edges: &BTreeSet<(String, String)>,
) -> (BTreeMap<String, usize>, BTreeMap<usize, usize>) {
    let names: Vec<&str> = nodes.iter().map(String::as_str).collect();
    let index: BTreeMap<&str, usize> = names.iter().enumerate().map(|(i, &m)| (m, i)).collect();
    let n = names.len();

    let mut adj = vec![Vec::new(); n];
    let mut radj = vec![Vec::new(); n];
    for (a, b) in edges {
        let (Some(&u), Some(&v)) = (index.get(a.as_str()), index.get(b.as_str())) else {
            continue;
        };
        adj[u].push(v);
        radj[v].push(u);
    }

    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for s in 0..n {
        if visited[s] {
            continue;
        }
        visited[s] = true;
        let mut stack = vec![(s, 0usize)];
        while let Some(&(node, next)) = stack.last() {
            if next < adj[node].len() {
                stack.last_mut().unwrap().1 += 1;
                let nx = adj[node][next];
                if !visited[nx] {
                    visited[nx] = true;
                    stack.push((nx, 0));
                }
            } else {
                order.push(node);
                stack.pop();
            }
        }
    }

    let mut comp_id = vec![usize::MAX; n];
    let mut next_comp = 0;
    for &s in order.iter().rev() {
        if comp_id[s] != usize::MAX {
            continue;
        }
        comp_id[s] = next_comp;
        let mut stack = vec![s];
        while let Some(node) = stack.pop() {
            for &nx in &radj[node] {
                if comp_id[nx] == usize::MAX {
                    comp_id[nx] = next_comp;
                    stack.push(nx);
                }
            }
        }
        next_comp += 1;
    }

    let comp: BTreeMap<String, usize> = names
        .iter()
        .enumerate()
        .map(|(i, &m)| (m.to_string(), comp_id[i]))
        .collect();
    let mut comp_size: BTreeMap<usize, usize> = BTreeMap::new();
    for &c in &comp_id {
        *comp_size.entry(c).or_insert(0) += 1;
    }
    (comp, comp_size)
}

// ===========================================================================
// Source scanning
// ===========================================================================

/// One source file, comment- and string-stripped. `module` is its top-level
/// module (`None` for `lib.rs`); `depth` is its module-path depth.
struct SrcFile {
    rel: PathBuf,
    module: Option<String>,
    depth: usize,
    stripped: Vec<(usize, String)>,
}

struct Scan {
    files: Vec<SrcFile>,
}

fn scan() -> &'static Scan {
    static SCAN: OnceLock<Scan> = OnceLock::new();
    SCAN.get_or_init(|| {
        let src = manifest_dir().join("src");
        let mut abs_paths = Vec::new();
        collect_rs(&src, &mut abs_paths);
        abs_paths.sort();

        let mut files = Vec::new();
        for abs in abs_paths {
            let rel = abs
                .strip_prefix(manifest_dir())
                .unwrap_or(&abs)
                .to_path_buf();
            let module = module_of(&rel);
            let depth = module_depth(&rel);
            let text = fs::read_to_string(&abs).unwrap_or_default();
            let stripped = strip_source(&text);
            files.push(SrcFile {
                rel,
                module,
                depth,
                stripped,
            });
        }
        Scan { files }
    })
}

fn manifest_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Top-level module owning a file at `rel` (e.g. `src/layer/kinds/noise.rs` →
/// `layer`). `lib.rs`/`main.rs` own no module.
fn module_of(rel: &Path) -> Option<String> {
    let mut comps = rel.components().map(|c| c.as_os_str().to_string_lossy());
    let first = comps.next()?;
    if first != "src" {
        return None;
    }
    let second = comps.next()?;
    if comps.next().is_none() {
        match second.strip_suffix(".rs") {
            Some("lib") | Some("main") | None => None,
            Some(name) => Some(name.to_string()),
        }
    } else {
        Some(second.to_string())
    }
}

/// Module-path depth of a file: `src/a.rs` → 1, `src/a/b.rs` → 2,
/// `src/a/mod.rs` → 1 (mod.rs *is* its directory's module), `lib.rs` → 0.
fn module_depth(rel: &Path) -> usize {
    let comps: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if comps.first().map(String::as_str) != Some("src") {
        return 0;
    }
    let after = &comps[1..];
    let Some(last) = after.last() else {
        return 0;
    };
    match last.as_str() {
        "lib.rs" | "main.rs" => 0,
        "mod.rs" => after.len() - 1,
        _ => after.len(),
    }
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// The subset of `stripped` lines that are production code: `#[cfg(test)]`-
/// attributed modules and items are dropped (their bodies tracked by brace
/// depth). Returns the surviving `(line_no, code)` pairs in file order.
fn production_lines(stripped: &[(usize, String)]) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut depth: i32 = 0;
    let mut pending_cfg_test = false;
    let mut skip_base: Option<i32> = None;

    for (line_no, code) in stripped {
        let delta = brace_delta(code);

        if let Some(base) = skip_base {
            depth += delta;
            if depth <= base {
                skip_base = None;
            }
            continue;
        }

        let trimmed = code.trim();

        if trimmed.contains("#[cfg(test)]") {
            let base = depth;
            depth += delta;
            if depth > base {
                skip_base = Some(base); // block opens on this same line
            } else {
                pending_cfg_test = true; // the annotated item is on a later line
            }
            continue;
        }

        if pending_cfg_test {
            if trimmed.is_empty() || trimmed.starts_with("#[") {
                depth += delta;
                continue;
            }
            pending_cfg_test = false;
            let base = depth;
            depth += delta;
            if depth > base {
                skip_base = Some(base);
            }
            continue;
        }

        out.push((*line_no, code.clone()));
        depth += delta;
    }
    out
}

fn brace_delta(code: &str) -> i32 {
    let opens = code.bytes().filter(|&b| b == b'{').count() as i32;
    let closes = code.bytes().filter(|&b| b == b'}').count() as i32;
    opens - closes
}

/// Blank line comments, block comments (nested), string literals (normal and
/// raw), and char literals, leaving code and lifetimes intact. Returns one
/// `(line_no, stripped)` pair per source line.
fn strip_source(text: &str) -> Vec<(usize, String)> {
    if text.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();

    let mut block_depth: u32 = 0;
    let mut in_str = false;
    let mut str_escape = false;
    let mut raw_hashes: Option<usize> = None;
    let mut in_line_comment = false;

    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c == '\n' {
            lines.push(std::mem::take(&mut cur));
            in_line_comment = false;
            i += 1;
            continue;
        }
        if in_line_comment {
            cur.push(' ');
            i += 1;
            continue;
        }
        if block_depth > 0 {
            if c == '/' && chars.get(i + 1) == Some(&'*') {
                block_depth += 1;
                cur.push_str("  ");
                i += 2;
            } else if c == '*' && chars.get(i + 1) == Some(&'/') {
                block_depth -= 1;
                cur.push_str("  ");
                i += 2;
            } else {
                cur.push(' ');
                i += 1;
            }
            continue;
        }
        if let Some(h) = raw_hashes {
            if c == '"' && closing_hashes(&chars, i + 1, h) {
                for _ in 0..(1 + h) {
                    cur.push(' ');
                }
                i += 1 + h;
                raw_hashes = None;
            } else {
                cur.push(' ');
                i += 1;
            }
            continue;
        }
        if in_str {
            cur.push(' ');
            if str_escape {
                str_escape = false;
            } else if c == '\\' {
                str_escape = true;
            } else if c == '"' {
                in_str = false;
            }
            i += 1;
            continue;
        }

        // Normal mode.
        if c == '/' && chars.get(i + 1) == Some(&'/') {
            in_line_comment = true;
            cur.push_str("  ");
            i += 2;
            continue;
        }
        if c == '/' && chars.get(i + 1) == Some(&'*') {
            block_depth = 1;
            cur.push_str("  ");
            i += 2;
            continue;
        }
        if let Some((consumed, h)) = raw_string_opener(&chars, i) {
            for _ in 0..consumed {
                cur.push(' ');
            }
            raw_hashes = Some(h);
            i += consumed;
            continue;
        }
        if c == '"' {
            in_str = true;
            str_escape = false;
            cur.push(' ');
            i += 1;
            continue;
        }
        if c == '\'' {
            if let Some(len) = char_literal_len(&chars, i) {
                for _ in 0..len {
                    cur.push(' ');
                }
                i += len;
            } else {
                cur.push(c); // lifetime or label
                i += 1;
            }
            continue;
        }
        cur.push(c);
        i += 1;
    }
    lines.push(cur);
    if text.ends_with('\n') {
        lines.pop();
    }
    lines
        .into_iter()
        .enumerate()
        .map(|(k, s)| (k + 1, s))
        .collect()
}

/// True if `chars[start..start+h]` are all `#` (vacuously true for `h == 0`).
fn closing_hashes(chars: &[char], start: usize, h: usize) -> bool {
    (0..h).all(|k| chars.get(start + k) == Some(&'#'))
}

/// If a raw-string opener (`r"`, `r#"…`, `br"`, …) begins at `i` on a token
/// boundary, return `(chars_consumed_by_opener, hash_count)`.
fn raw_string_opener(chars: &[char], i: usize) -> Option<(usize, usize)> {
    if i > 0 && is_ident_char(chars[i - 1]) {
        return None;
    }
    let mut j = i;
    if chars.get(j) == Some(&'b') {
        j += 1;
    }
    if chars.get(j) != Some(&'r') {
        return None;
    }
    j += 1;
    let mut h = 0;
    while chars.get(j) == Some(&'#') {
        h += 1;
        j += 1;
    }
    if chars.get(j) == Some(&'"') {
        Some((j + 1 - i, h))
    } else {
        None
    }
}

/// If a char literal begins at `i` (`chars[i] == '\''`), return its total length
/// including both quotes. `None` for a lifetime/label (`'a`, `'static`).
fn char_literal_len(chars: &[char], i: usize) -> Option<usize> {
    let n = chars.len();
    if chars.get(i) != Some(&'\'') {
        return None;
    }
    let c1 = *chars.get(i + 1)?;
    if c1 == '\n' {
        return None;
    }
    if c1 == '\\' {
        // Escape: skip the escaped item, then require a closing quote.
        let after_esc = if chars.get(i + 2) == Some(&'u') && chars.get(i + 3) == Some(&'{') {
            let mut k = i + 4;
            while k < n && chars[k] != '}' {
                if chars[k] == '\n' {
                    return None;
                }
                k += 1;
            }
            if k >= n {
                return None;
            }
            k + 1
        } else if chars.get(i + 2).is_some() {
            i + 3
        } else {
            return None;
        };
        if chars.get(after_esc) == Some(&'\'') {
            return Some(after_esc - i + 1);
        }
        return None;
    }
    if c1 == '\'' {
        return None; // empty `''` is not a literal
    }
    if chars.get(i + 2) == Some(&'\'') {
        Some(3) // 'x'
    } else {
        None // lifetime
    }
}

/// Leading run of identifier characters (`A-Za-z0-9_`, plus `-` for crate
/// names) from the start of `s`.
fn leading_ident(s: &str) -> &str {
    let end = s
        .char_indices()
        .find(|(_, c)| !(c.is_ascii_alphanumeric() || *c == '_' || *c == '-'))
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    &s[..end]
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

// ===========================================================================
// Fixtures — prove each guard fails for the mutation it is meant to catch.
// ===========================================================================

#[cfg(test)]
mod fixtures {
    use super::*;

    fn modset(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn prod(lines: &[&str]) -> Vec<(usize, String)> {
        lines
            .iter()
            .enumerate()
            .map(|(i, s)| (i + 1, s.to_string()))
            .collect()
    }

    fn edgeset(pairs: &[(&str, &str)]) -> BTreeSet<(String, String)> {
        pairs
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    // --- Mutation 1: a renamed wgpu (`gpu = { package = "wgpu" }`). -----------
    #[test]
    fn renamed_wgpu_is_caught() {
        let json = r#"{"packages":[{"name":"terra-core","dependencies":[
            {"name":"wgpu","kind":null,"rename":"gpu","target":null}
        ]}]}"#;
        let deps = parse_dependencies(json).unwrap();
        assert_eq!(deps[0].name, "wgpu"); // resolved name, not the `gpu` alias
        let v = purity_violations(&deps);
        assert!(
            v.iter().any(|m| m.contains("forbidden dependency `wgpu`")),
            "renamed wgpu must be rejected: {v:?}"
        );
    }

    // --- Mutation 2: a target-specific UI dependency. -------------------------
    #[test]
    fn target_specific_ui_dep_is_caught() {
        let json = r#"{"packages":[{"name":"terra-core","dependencies":[
            {"name":"winit","kind":null,"rename":null,"target":"cfg(windows)"}
        ]}]}"#;
        let deps = parse_dependencies(json).unwrap();
        let v = purity_violations(&deps);
        assert!(
            v.iter().any(|m| m.contains("target-specific")),
            "target-specific dep must be rejected: {v:?}"
        );
        assert!(v.iter().any(|m| m.contains("forbidden dependency `winit`")));
    }

    // --- Mutation 3a: a root-re-exported edge (crate::TerrainDocument). --------
    #[test]
    fn root_reexport_edge_is_caught() {
        let modules = modset(&["document", "heightfield", "eval"]);
        let lines = prod(&["    let doc: crate::TerrainDocument = load();"]);
        let v = root_qualified_violations("src/eval/mod.rs", &lines, &modules);
        assert!(
            v.iter().any(|m| m.contains("crate::TerrainDocument")),
            "root re-export must be rejected: {v:?}"
        );
        // The submodule-qualified form is accepted.
        let ok = prod(&["    let doc: crate::document::TerrainDocument = load();"]);
        assert!(root_qualified_violations("src/eval/mod.rs", &ok, &modules).is_empty());
    }

    // --- Mutation 3b: a cyclic pair in the allowlist. -------------------------
    #[test]
    fn allowlist_cycle_is_caught() {
        let modules = modset(&["a", "b"]);
        let edges = edgeset(&[("a", "b"), ("b", "a")]);
        let v = cycle_violations(&modules, &edges);
        assert!(!v.is_empty(), "a↔b cycle must be rejected: {v:?}");
        assert!(cycle_violations(&modules, &edgeset(&[("a", "b")])).is_empty());
    }

    // --- Mutation 4: an unclassified module. ----------------------------------
    #[test]
    fn unclassified_module_is_caught() {
        let modules = modset(&["kept", "added"]);
        let allowlist: &[(&str, &[&str])] = &[("kept", &[])];
        let v = classification_violations(&modules, allowlist);
        assert!(
            v.iter().any(|m| m.contains("`added`")),
            "unclassified module must be rejected: {v:?}"
        );
    }

    // --- Mutation 5: an acyclic upward edge not in the allowlist. --------------
    #[test]
    fn acyclic_new_edge_is_caught() {
        let scanned = edgeset(&[("low", "high"), ("low", "mid")]);
        let allow = edgeset(&[("low", "mid")]);
        let extra: Vec<_> = scanned.difference(&allow).cloned().collect();
        assert_eq!(extra, vec![("low".to_string(), "high".to_string())]);
    }

    // --- super:: escapes to the crate root. -----------------------------------
    #[test]
    fn super_to_root_is_caught_but_in_module_is_ok() {
        // depth-2 file: one super:: stays in-module (ok), two reaches root (bad).
        let ok = prod(&["    use super::sibling::Thing;"]);
        assert!(super_escape_violations("src/a/b.rs", 2, &ok).is_empty());

        let bad = prod(&["    use super::super::RootThing;"]);
        let v = super_escape_violations("src/a/b.rs", 2, &bad);
        assert!(
            !v.is_empty(),
            "super::super to root must be rejected: {v:?}"
        );

        // depth-3 file: super::super lands in the top-level module (ok).
        let deep = prod(&["    use super::super::filter_kernels::grad;"]);
        assert!(super_escape_violations("src/a/b/c.rs", 3, &deep).is_empty());
    }

    // --- Lexer: comments, strings, raws, chars, lifetimes. --------------------
    #[test]
    fn lexer_blanks_noncode_and_keeps_edges() {
        let src = "\
use crate::layer::Layer; // crate::mask hidden in a comment
let s = \"crate::noise not an edge\";
/* crate::hydro also hidden */ use crate::geology::Rock;
let c = '\\'';
fn f<'a>(x: &'a crate::terrain::Tile) {}
";
        let stripped = strip_source(src);
        let modules = modset(&["layer", "mask", "noise", "hydro", "geology", "terrain"]);
        let mut targets: Vec<String> = Vec::new();
        for (_, code) in &stripped {
            targets.extend(extract_targets(code, &modules));
        }
        targets.sort();
        targets.dedup();
        assert_eq!(targets, vec!["geology", "layer", "terrain"]);
    }

    #[test]
    fn lexer_nested_block_comment_and_raw_string() {
        let src = "\
/* outer /* inner crate::mask */ still comment crate::layer */ let x = 1;
let r = r#\"crate::noise \"# ; use crate::hydro::Flow;
";
        let stripped = strip_source(src);
        let modules = modset(&["mask", "layer", "noise", "hydro"]);
        let mut targets: Vec<String> = Vec::new();
        for (_, code) in &stripped {
            targets.extend(extract_targets(code, &modules));
        }
        targets.sort();
        assert_eq!(targets, vec!["hydro"], "only the real use survives");
    }

    #[test]
    fn grouped_and_nested_imports_extract_all_modules() {
        let modules = modset(&["layer", "mask", "noise", "ids"]);
        let flat = extract_targets("use crate::{layer::A, mask::B};", &modules);
        assert_eq!(sorted(flat), vec!["layer", "mask"]);
        let nested = extract_targets("use crate::{layer::{A, B}, ids};", &modules);
        assert_eq!(sorted(nested), vec!["ids", "layer"]);
    }

    #[test]
    fn aliased_import_still_counts() {
        let modules = modset(&["heightfield"]);
        let t = extract_targets("use crate::heightfield::Heightfield as HF;", &modules);
        assert_eq!(t, vec!["heightfield"]);
    }

    #[test]
    fn cfg_test_block_is_not_production() {
        let stripped = strip_source(
            "\
use crate::layer::A;
#[cfg(test)]
mod tests {
    use crate::mask::B;
}
",
        );
        let prod = production_lines(&stripped);
        let modules = modset(&["layer", "mask"]);
        let mut targets: Vec<String> = Vec::new();
        for (_, code) in &prod {
            targets.extend(extract_targets(code, &modules));
        }
        assert_eq!(targets, vec!["layer"], "the cfg(test) edge is excluded");
    }

    #[test]
    fn module_depth_is_computed_from_path() {
        assert_eq!(module_depth(Path::new("src/analyze.rs")), 1);
        assert_eq!(module_depth(Path::new("src/layer/stack.rs")), 2);
        assert_eq!(module_depth(Path::new("src/generators/mod.rs")), 1);
        assert_eq!(
            module_depth(Path::new("src/generators/geology/terrace.rs")),
            3
        );
        assert_eq!(module_depth(Path::new("src/lib.rs")), 0);
    }

    fn sorted(mut v: Vec<String>) -> Vec<String> {
        v.sort();
        v.dedup();
        v
    }
}
