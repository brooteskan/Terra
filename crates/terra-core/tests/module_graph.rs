//! terra-core purity + module-graph ratchet (audit A3-G1).
//!
//! Station A3 (#1) has two invariants that held only by discipline: terra-core
//! must not depend on `wgpu` or a UI crate, and its top-level module graph must
//! stop being one big cycle. A2 landed `terra-gui/tests/purity.rs` and B2 landed
//! `terra-core/tests/dead_seams.rs` as cargo-test guards; this is the equivalent
//! for A3. Three rules, all scanned over `crates/terra-core/src/**/*.rs`:
//!
//! - **Rule 1** (`terra_core_is_pure`): `Cargo.toml`'s `[dependencies]` /
//!   `[build-dependencies]` must not name `wgpu`, `winit`, `egui`, or any
//!   `terra-*` crate; stripped source must contain no `wgpu::` / `use wgpu` /
//!   `terra_gui` / `terra_render` / `terra_app` token.
//! - **Rule 2** (`module_graph_cycles_match_allowlist`): the set of production
//!   cross-module edges that participate in a cycle must equal [`CYCLIC_EDGES`]
//!   exactly. Two-sided: a new cyclic edge fails immediately, and a fix that
//!   removes an edge must delete its allowlist entry in the same commit — the
//!   list only shrinks, consciously.
//! - **Rule 3** (`clean_modules_stay_acyclic`): the modules currently outside
//!   every cycle ([`CLEAN_MODULES`]) must stay outside — the knot cannot recruit
//!   them.
//!
//! Kept deliberately dumb — line scanning, no `syn`/`regex` — so it grows no
//! dependencies and cannot rot, exactly like `purity.rs` and `dead_seams.rs`.
//! The scanning conventions and their consequences:
//!
//! - Line comments (`//`, `///`, `//!`) and the contents of `"…"` string
//!   literals are blanked before scanning, so a `crate::layer` mentioned in a
//!   doc-comment or string is not counted as an edge, and braces inside strings
//!   do not perturb the `#[cfg(test)]` block tracker. Char literals are left
//!   intact (blanking `'…'` would eat lifetimes like `&'a T`); block comments
//!   (`/* … */`) are treated as code — both tolerated tripwire gaps.
//! - **Production only.** `#[cfg(test)]`-attributed modules/items are skipped, so
//!   a test-only `use crate::foo` is not an edge. This matches the audit's
//!   headline inventory (the 29-module production SCC; `#[cfg(test)]` edges were
//!   tracked separately).
//! - Cross-module edges are read solely from `crate::<module>::…` paths (in
//!   `use` statements and inline). Verified sufficient at this commit: no
//!   production `super::`-to-root chains exist (every `super::` in a top-level
//!   file is `use super::*` inside a test module), and the only bare
//!   `use crate::{…}` group is test-only. Crate-root-qualified references
//!   (`crate::LayerKind`, via the lib.rs convenience surface) are intentionally
//!   *not* attributed to their defining module; the audit's mechanism is the
//!   submodule-qualified path the crate's own rules prefer.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Production cross-module edges `(from, to)` that participate in a cycle at this
/// commit — i.e. `from` and `to` are in the same strongly-connected component.
/// Exact set (Rule 2). Seeded to today's graph; each A3 fix deletes the edges it
/// removes in the same commit. Sorted for readable diffs.
const CYCLIC_EDGES: &[(&str, &str)] = &[
    ("analyze", "fields"),
    ("analyze", "generators"),
    ("analyze", "geomorph"),
    ("analyze", "hydro"),
    ("analyze", "layer"),
    ("analyze", "mask"),
    ("authoring", "analyze"),
    ("authoring", "fields"),
    ("authoring", "hydro"),
    ("authoring", "landscape_evolution"),
    ("authoring", "layer"),
    ("authoring", "mask"),
    ("biome_definition", "layer"),
    ("biome_definition", "mask"),
    ("biome_paint", "mask"),
    ("climate", "analyze"),
    ("climate", "layer"),
    ("climate", "mask"),
    ("document", "rebuild_feedback"),
    ("eval", "analyze"),
    ("eval", "authoring"),
    ("eval", "climate"),
    ("eval", "fields"),
    ("eval", "generators"),
    ("eval", "hydro"),
    ("eval", "landscape_evolution"),
    ("eval", "layer"),
    ("eval", "mask"),
    ("eval", "surface"),
    ("eval", "volumetric"),
    ("fields", "analyze"),
    ("fields", "generators"),
    ("fields", "layer"),
    ("fields", "mask"),
    ("generators", "analyze"),
    ("generators", "eval"),
    ("generators", "geomorph"),
    ("generators", "hydro"),
    ("generators", "layer"),
    ("generators", "mask"),
    ("geomorph", "analyze"),
    ("geomorph", "mask"),
    ("hydro", "fields"),
    ("hydro", "geomorph"),
    ("hydro", "layer"),
    ("hydro", "mask"),
    ("landscape_evolution", "analyze"),
    ("landscape_evolution", "fields"),
    ("landscape_evolution", "geomorph"),
    ("landscape_evolution", "hydro"),
    ("landscape_evolution", "layer"),
    ("landscape_evolution", "mask"),
    ("layer", "authoring"),
    ("layer", "biome_paint"),
    ("layer", "climate"),
    ("layer", "fields"),
    ("layer", "landscape_evolution"),
    ("layer", "mask"),
    ("layer", "operation_placement"),
    ("layer", "tiling"),
    ("mask", "analyze"),
    ("mask", "biome_definition"),
    ("operation_placement", "layer"),
    ("operation_placement", "mask"),
    ("rebuild_feedback", "document"),
    ("scatter", "analyze"),
    ("scatter", "layer"),
    ("scatter", "mask"),
    ("surface", "analyze"),
    ("surface", "climate"),
    ("surface", "fields"),
    ("surface", "layer"),
    ("surface", "mask"),
    ("surface", "scatter"),
    ("tiling", "layer"),
    ("volumetric", "layer"),
    ("volumetric", "mask"),
];

/// Top-level modules that are outside every cycle at this commit and must stay
/// that way (Rule 3). The newest module families live here; the guard keeps the
/// knot from recruiting them.
const CLEAN_MODULES: &[&str] = &[
    "command",
    "contextual_create",
    "deps",
    "domain",
    "heightfield",
    "ids",
    "landscape_blueprint",
    "landscape_style",
    "matter_sim",
    "noise",
    "quality",
    "realism_benchmark",
    "shape_history",
    "shape_object",
    "simd_ops",
    "simulation_scenario",
    "sparse_paint",
    "terrain",
    "terrain_recipe",
    "world_archetype",
    "world_rules",
];

/// Crate names (exact) that terra-core must never take as a normal or build
/// dependency. `terra-*` crates are rejected by prefix in addition to these.
const FORBIDDEN_DEPS: &[&str] = &["wgpu", "winit", "egui"];

/// Tokens whose presence in stripped source betrays a GPU/UI leak.
const FORBIDDEN_SOURCE_TOKENS: &[&str] = &[
    "wgpu::",
    "use wgpu",
    "terra_gui",
    "terra_render",
    "terra_app",
];

// ===========================================================================
// Rule 1 — purity
// ===========================================================================

#[test]
fn terra_core_is_pure() {
    let mut violations = Vec::new();

    let toml_path = manifest_dir().join("Cargo.toml");
    let toml = fs::read_to_string(&toml_path).expect("terra-core Cargo.toml is readable");
    for (section, name) in dependency_names(&toml) {
        let forbidden = FORBIDDEN_DEPS.contains(&name.as_str()) || name.starts_with("terra-");
        if forbidden {
            violations.push(format!(
                "Cargo.toml [{section}] declares forbidden dependency `{name}`; terra-core must \
                 stay free of GPU/UI and sibling terra-* crates"
            ));
        }
    }

    for file in &scan().files {
        for (line_no, code) in &file.stripped {
            for tok in FORBIDDEN_SOURCE_TOKENS {
                if code.contains(tok) {
                    violations.push(format!(
                        "{}:{} contains forbidden token `{tok}`",
                        file.rel.display(),
                        line_no
                    ));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "terra-core purity guard failed:\n  {}",
        violations.join("\n  "),
    );
}

// ===========================================================================
// Rule 2 — module-graph ratchet
// ===========================================================================

#[test]
fn module_graph_cycles_match_allowlist() {
    let g = graph();
    let cyclic = g.cyclic_edges();
    let allow: BTreeSet<(String, String)> = CYCLIC_EDGES
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();

    let extra: Vec<_> = cyclic.difference(&allow).collect();
    let missing: Vec<_> = allow.difference(&cyclic).collect();

    let mut msg = String::new();
    if !extra.is_empty() {
        msg.push_str(
            "\nNEW cyclic edges — break them, or (if genuinely intended) add to CYCLIC_EDGES:\n",
        );
        for (a, b) in &extra {
            msg.push_str(&format!("    (\"{a}\", \"{b}\"),\n"));
        }
    }
    if !missing.is_empty() {
        msg.push_str("\nStale CYCLIC_EDGES entries — a fix removed these; delete the entries:\n");
        for (a, b) in &missing {
            msg.push_str(&format!("    (\"{a}\", \"{b}\"),\n"));
        }
    }

    assert!(
        extra.is_empty() && missing.is_empty(),
        "module-graph ratchet drift ({} cyclic edges now, {} allowlisted):{}",
        cyclic.len(),
        allow.len(),
        msg,
    );
}

// ===========================================================================
// Rule 3 — the clean set stays clean
// ===========================================================================

#[test]
fn clean_modules_stay_acyclic() {
    let g = graph();
    let mut violations = Vec::new();

    for &m in CLEAN_MODULES {
        if !g.modules.contains(m) {
            violations.push(format!("CLEAN_MODULES names unknown module `{m}`"));
        } else if g.is_cyclic(m) {
            violations.push(format!(
                "module `{m}` was outside every cycle and has now joined one; \
                 CLEAN_MODULES must never regress"
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "clean-set guard failed:\n  {}",
        violations.join("\n  "),
    );
}

// ===========================================================================
// Graph construction
// ===========================================================================

struct Graph {
    modules: BTreeSet<String>,
    edges: BTreeSet<(String, String)>,
    /// SCC id per module.
    comp: BTreeMap<String, usize>,
    /// Node count per SCC id (a module is cyclic iff its component has > 1 node).
    comp_size: BTreeMap<usize, usize>,
}

impl Graph {
    fn is_cyclic(&self, m: &str) -> bool {
        self.comp
            .get(m)
            .and_then(|c| self.comp_size.get(c))
            .is_some_and(|&size| size > 1)
    }

    fn cyclic_edges(&self) -> BTreeSet<(String, String)> {
        self.edges
            .iter()
            .filter(|(a, b)| self.comp[a] == self.comp[b])
            .cloned()
            .collect()
    }
}

fn graph() -> &'static Graph {
    static GRAPH: OnceLock<Graph> = OnceLock::new();
    GRAPH.get_or_init(build_graph)
}

fn build_graph() -> Graph {
    let modules = module_set();

    let mut edges = BTreeSet::new();
    for file in &scan().files {
        let Some(src) = file.module.as_deref() else {
            continue;
        };
        if !modules.contains(src) {
            continue;
        }
        for code in production_lines(&file.stripped) {
            for tgt in extract_targets(&code, &modules) {
                if tgt != src {
                    edges.insert((src.to_string(), tgt));
                }
            }
        }
    }

    let (comp, comp_size) = strongly_connected(&modules, &edges);
    Graph {
        modules,
        edges,
        comp,
        comp_size,
    }
}

/// The top-level module names, read from `lib.rs`'s `pub mod X;` declarations.
fn module_set() -> BTreeSet<String> {
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

/// Every `crate::<module>::…` target named on `code`, including the modules
/// inside a `crate::{ … }` group. Only names in `modules` are returned.
fn extract_targets(code: &str, modules: &BTreeSet<String>) -> Vec<String> {
    const NEEDLE: &str = "crate::";
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    let mut search = 0;
    while let Some(rel) = code[search..].find(NEEDLE) {
        let idx = search + rel;
        search = idx + NEEDLE.len();
        // `crate` must be a whole word (reject `xcrate::`).
        if idx > 0 && is_ident_byte(bytes[idx - 1]) {
            continue;
        }
        let rest = code[idx + NEEDLE.len()..].trim_start();
        if let Some(group) = rest.strip_prefix('{') {
            let end = group.find('}').unwrap_or(group.len());
            for item in group[..end].split(',') {
                let id = leading_ident(item.trim_start());
                if modules.contains(id) {
                    out.push(id.to_string());
                }
            }
        } else {
            let id = leading_ident(rest);
            if modules.contains(id) {
                out.push(id.to_string());
            }
        }
    }
    out
}

/// Kosaraju SCC: returns each module's component id and the size of every
/// component. Modules with no edges each get their own singleton component.
fn strongly_connected(
    modules: &BTreeSet<String>,
    edges: &BTreeSet<(String, String)>,
) -> (BTreeMap<String, usize>, BTreeMap<usize, usize>) {
    let names: Vec<&str> = modules.iter().map(String::as_str).collect();
    let index: BTreeMap<&str, usize> = names.iter().enumerate().map(|(i, &m)| (m, i)).collect();
    let n = names.len();

    let mut adj = vec![Vec::new(); n];
    let mut radj = vec![Vec::new(); n];
    for (a, b) in edges {
        let (u, v) = (index[a.as_str()], index[b.as_str()]);
        adj[u].push(v);
        radj[v].push(u);
    }

    // Pass 1: iterative DFS, push nodes in order of finish time.
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

    // Pass 2: DFS the transpose in reverse finish order; each tree is one SCC.
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
/// module (`None` for `lib.rs`).
struct SrcFile {
    rel: PathBuf,
    module: Option<String>,
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
            let text = fs::read_to_string(&abs).unwrap_or_default();
            let stripped = text
                .lines()
                .enumerate()
                .map(|(i, raw)| (i + 1, strip_comment_and_strings(raw)))
                .collect();
            files.push(SrcFile {
                rel,
                module,
                stripped,
            });
        }
        Scan { files }
    })
}

fn manifest_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Top-level module owning a file at `rel` (relative to the crate root, e.g.
/// `src/layer/kinds/noise.rs` → `layer`). `lib.rs`/`main.rs` own no module.
fn module_of(rel: &Path) -> Option<String> {
    let mut comps = rel.components().map(|c| c.as_os_str().to_string_lossy());
    let first = comps.next()?;
    if first != "src" {
        return None;
    }
    let second = comps.next()?;
    if comps.next().is_none() {
        // A file directly in `src/`.
        match second.strip_suffix(".rs") {
            Some("lib") | Some("main") | None => None,
            Some(name) => Some(name.to_string()),
        }
    } else {
        // A file under `src/<module>/…`.
        Some(second.to_string())
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
/// depth). Returns just the code strings, in file order.
fn production_lines(stripped: &[(usize, String)]) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth: i32 = 0;
    let mut pending_cfg_test = false;
    let mut skip_base: Option<i32> = None;

    for (_, code) in stripped {
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
            // Skip blank lines and stacked attributes between the marker and its
            // item, so `#[cfg(test)]\n mod tests {` is caught.
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

        out.push(code.clone());
        depth += delta;
    }
    out
}

fn brace_delta(code: &str) -> i32 {
    let opens = code.bytes().filter(|&b| b == b'{').count() as i32;
    let closes = code.bytes().filter(|&b| b == b'}').count() as i32;
    opens - closes
}

/// Blank line comments (`//…`) and the contents of `"…"` string literals,
/// leaving code (and char literals / block comments) intact. Escapes inside
/// strings are honoured so `"\""` does not reopen the string.
fn strip_comment_and_strings(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut in_str = false;
    while let Some(c) = chars.next() {
        if in_str {
            out.push(' ');
            if c == '\\' {
                if chars.next().is_some() {
                    out.push(' ');
                }
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '/' if chars.peek() == Some(&'/') => break,
            '"' => {
                in_str = true;
                out.push(' ');
            }
            _ => out.push(c),
        }
    }
    out
}

/// Section (`dependencies` / `build-dependencies`) and crate name of every
/// dependency declared in `toml`, covering both `name = …` rows and
/// `[dependencies.name]` sub-tables. Deliberately tiny — enough for the guard.
fn dependency_names(toml: &str) -> Vec<(String, String)> {
    const SECTIONS: &[&str] = &["dependencies", "build-dependencies"];
    let mut out = Vec::new();
    let mut active: Option<String> = None;

    for raw in toml.lines() {
        let line = raw.trim();
        if let Some(header) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            active = None;
            for sec in SECTIONS {
                if header == *sec {
                    active = Some((*sec).to_string());
                } else if let Some(rest) = header.strip_prefix(&format!("{sec}.")) {
                    // `[dependencies.wgpu]` — the name is the sub-table.
                    out.push(((*sec).to_string(), leading_ident(rest).to_string()));
                }
            }
            continue;
        }
        let Some(section) = &active else { continue };
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let name = leading_ident(line);
        if !name.is_empty() {
            out.push((section.clone(), name.to_string()));
        }
    }
    out
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
