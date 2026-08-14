//! Dead-seam scan: a named-but-inert public abstraction fails `cargo test`.
//! (audit B2-G1)
//!
//! Rust's dead-code lint is silent for `pub` items in a library crate, so a
//! public trait nobody implements or a type alias nobody uses compiles clean
//! forever. Station B2 (#1) proposes the standing rule *"a named abstraction
//! with zero implementors fails the audit"*; this test is the mechanism behind
//! it. Three rules, all scanned over `crates/*/src/**/*.rs`:
//!
//! - **Rule 1** (`public_traits_have_implementors`): every `pub trait X` must
//!   have at least one occurrence of `X` outside its own definition and outside
//!   `use` statements — an `impl … X for …`, a `dyn X`, or a generic bound all
//!   qualify. A `pub use` re-export does not.
//! - **Rule 2** (`public_type_aliases_have_consumers`): every `pub type X = …;`
//!   must likewise be named somewhere that is neither its definition nor a
//!   re-export line.
//! - **Rule 3** (`converged_kernels_stay_single`): the routing/erosion kernels
//!   that B2-D3/D4 collapsed to one lineage must stay single — each name in
//!   `SINGLE_KERNELS` must be defined *exactly once*. Two definitions is a
//!   re-fork; zero means it was renamed or removed and the list here is stale.
//!
//! An intentional forward declaration is not forbidden — it goes in
//! `ALLOWED_INERT` with a justification, making the exception visible in review.
//! `allowlist_is_honest` keeps that list from rotting: a blank justification
//! fails, and so does a stale entry (one whose item gained a consumer or
//! vanished), forcing the allowlist to shrink back on its own.
//!
//! Kept deliberately dumb — line scanning, no `syn` — so it grows no
//! dependencies and cannot rot, exactly like terra-gui's `purity.rs` guard.
//! The consequences, all matching that guard's philosophy of erring toward a
//! reviewed guard edit rather than a silent pass:
//!
//! - Line comments (`//`, `///`, `//!`) are stripped, so a doc-comment mention
//!   of a trait is correctly *not* counted as keeping it alive.
//! - Block comments (`/* … */`) and string literals are treated as code. Per
//!   #26 this is acceptable for a tripwire: a false *positive* goes to
//!   `ALLOWED_INERT` with a reason; a name kept "alive" only by a block-comment
//!   mention is a tolerated tripwire gap, not a silent architectural hole.
//! - `use`-statement lines (including multi-line `pub use foo::{ … };` blocks)
//!   are blanked, so re-exports never count as life. The scan assumes the
//!   one-`use`-statement-per-line form the workspace uses today.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Intentional forward declarations excused from Rules 1 and 2, as
/// `(identifier, justification)`. Empty today — the workspace has no inert
/// `pub trait`/`pub type`. Any entry needs a real reason (an empty one fails),
/// and `allowlist_is_honest` deletes stale entries by failing until they go.
const ALLOWED_INERT: &[(&str, &str)] = &[];

/// Kernel signatures that converged to a single lineage in B2-D3/D4 (flow
/// routing on the geomorph `FlowGraph`, the shared stream-power increment).
/// Each must have exactly one `fn NAME(` across `crates/*/src`.
///
/// `accumulate_drainage_area_d8` is the #27 lean flat-D8 accumulator: a second
/// *consumer* of `flow_direction_d8`, not a re-fork of it. Pinning it single
/// enforces #27's revert-check — there is exactly one flat D8 accumulation body,
/// never a copy that could drift from the general path it must stay bit-identical
/// to.
const SINGLE_KERNELS: &[&str] = &[
    "flow_direction_d8",
    "flow_direction_dinfinity",
    "build_flow_graph",
    "priority_flood_fill",
    "accumulate_drainage_area",
    "accumulate_drainage_area_d8",
    "accumulate_discharge",
    "spe_increment",
];

#[test]
fn public_traits_have_implementors() {
    let mut violations = Vec::new();
    for def in scan().trait_defs() {
        if scan().is_alive(&def) {
            continue;
        }
        match allowlist_reason(&def.name) {
            Some(reason) if !reason.trim().is_empty() => {} // intentional, and justified
            Some(_) => violations.push(format!(
                "pub trait `{}` ({}) is inert and its ALLOWED_INERT entry has an empty \
                 justification; give a reason or delete the trait",
                def.name,
                def.location(),
            )),
            None => violations.push(format!(
                "pub trait `{}` ({}) has no implementor, `dyn`, or bound consumer anywhere in \
                 crates/*/src; delete it, or add it to ALLOWED_INERT with a justification",
                def.name,
                def.location(),
            )),
        }
    }
    assert!(
        violations.is_empty(),
        "dead-seam scan found inert pub traits:\n  {}",
        violations.join("\n  "),
    );
}

#[test]
fn public_type_aliases_have_consumers() {
    let mut violations = Vec::new();
    for def in scan().type_defs() {
        if scan().is_alive(&def) {
            continue;
        }
        match allowlist_reason(&def.name) {
            Some(reason) if !reason.trim().is_empty() => {}
            Some(_) => violations.push(format!(
                "pub type `{}` ({}) is inert and its ALLOWED_INERT entry has an empty \
                 justification; give a reason or delete the alias",
                def.name,
                def.location(),
            )),
            None => violations.push(format!(
                "pub type `{}` ({}) is never named outside its definition and re-exports in \
                 crates/*/src; delete it, or add it to ALLOWED_INERT with a justification",
                def.name,
                def.location(),
            )),
        }
    }
    assert!(
        violations.is_empty(),
        "dead-seam scan found inert pub type aliases:\n  {}",
        violations.join("\n  "),
    );
}

#[test]
fn converged_kernels_stay_single() {
    let mut violations = Vec::new();
    for &name in SINGLE_KERNELS {
        let count = scan().kernel_def_count(name);
        if count != 1 {
            violations.push(format!(
                "kernel `fn {name}(` is defined {count} time(s) across crates/*/src; exactly one \
                 is required (>1 re-forks a converged lineage; 0 means it was renamed/removed and \
                 SINGLE_KERNELS is stale)"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "dead-seam scan found drifted kernel lineages:\n  {}",
        violations.join("\n  "),
    );
}

#[test]
fn allowlist_is_honest() {
    // Names that would actually fail Rule 1 or 2 right now. Bind the def lists
    // to locals so the `&str` set can borrow their owned names.
    let trait_defs = scan().trait_defs();
    let type_defs = scan().type_defs();
    let inert: BTreeSet<&str> = trait_defs
        .iter()
        .chain(type_defs.iter())
        .filter(|d| !scan().is_alive(d))
        .map(|d| d.name.as_str())
        .collect();

    let mut violations = Vec::new();
    let mut seen = BTreeSet::new();
    for (name, reason) in ALLOWED_INERT {
        if !seen.insert(*name) {
            violations.push(format!("ALLOWED_INERT lists `{name}` more than once"));
        }
        if reason.trim().is_empty() {
            violations.push(format!(
                "ALLOWED_INERT entry `{name}` has an empty justification; state why the inert \
                 declaration is intentional"
            ));
        }
        if !inert.contains(*name) {
            violations.push(format!(
                "ALLOWED_INERT entry `{name}` is stale: it is no longer an inert pub trait/type \
                 (it gained a consumer, or was removed); delete the entry"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "dead-seam allowlist drifted from the tree:\n  {}",
        violations.join("\n  "),
    );
}

/// Justification recorded for `name` in `ALLOWED_INERT`, or `None` if absent.
fn allowlist_reason(name: &str) -> Option<&'static str> {
    ALLOWED_INERT
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, reason)| *reason)
}

/// A `pub trait`/`pub type` declaration and where it lives (path relative to the
/// workspace root).
struct Def {
    name: String,
    path: PathBuf,
    line: usize,
}

impl Def {
    fn location(&self) -> String {
        format!("{}:{}", self.path.display(), self.line)
    }
}

/// One source file, comment-stripped and with `use` lines blanked, ready to
/// scan. `lines` holds `(1-based line number, filtered code)`.
struct SrcFile {
    path: PathBuf,
    lines: Vec<(usize, String)>,
}

/// The whole `crates/*/src` tree, filtered once and shared across the tests.
struct Scan {
    files: Vec<SrcFile>,
}

fn scan() -> &'static Scan {
    static SCAN: OnceLock<Scan> = OnceLock::new();
    SCAN.get_or_init(Scan::gather)
}

impl Scan {
    fn gather() -> Self {
        let root = workspace_root();
        let mut files = Vec::new();
        for abs in crate_src_files(&root) {
            let rel = abs.strip_prefix(&root).unwrap_or(&abs).to_path_buf();
            files.push(filter_file(&abs, rel));
        }
        Scan { files }
    }

    fn trait_defs(&self) -> Vec<Def> {
        self.defs_where(public_trait_name)
    }

    fn type_defs(&self) -> Vec<Def> {
        self.defs_where(public_type_name)
    }

    fn defs_where(&self, extract: fn(&str) -> Option<String>) -> Vec<Def> {
        let mut defs = Vec::new();
        for file in &self.files {
            for (line_no, code) in &file.lines {
                if let Some(name) = extract(code) {
                    defs.push(Def {
                        name,
                        path: file.path.clone(),
                        line: *line_no,
                    });
                }
            }
        }
        defs
    }

    /// Whether `def`'s name is named anywhere in the tree other than on its own
    /// definition line (and `use` lines, already blanked during filtering).
    fn is_alive(&self, def: &Def) -> bool {
        for file in &self.files {
            for (line_no, code) in &file.lines {
                let is_def_line = file.path == def.path && *line_no == def.line;
                if is_def_line {
                    continue;
                }
                if count_ident_occurrences(code, &def.name) > 0 {
                    return true;
                }
            }
        }
        false
    }

    fn kernel_def_count(&self, name: &str) -> usize {
        let mut count = 0;
        for file in &self.files {
            for (_, code) in &file.lines {
                if is_fn_def_of(code, name) {
                    count += 1;
                }
            }
        }
        count
    }
}

/// Workspace root: `crates/terra-core/../..`.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Every `.rs` file under `crates/*/src`, sorted for deterministic reporting.
fn crate_src_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let crates_dir = root.join("crates");
    let entries = fs::read_dir(&crates_dir).expect("crates/ directory is readable");
    for entry in entries.flatten() {
        let src = entry.path().join("src");
        if src.is_dir() {
            collect_rs(&src, &mut out);
        }
    }
    out.sort();
    out
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

/// Read a file and produce its filtered lines: line comments removed, and every
/// `use` statement line (single- or multi-line) blanked so re-exports never
/// count as evidence of life.
fn filter_file(abs: &Path, rel: PathBuf) -> SrcFile {
    let text = fs::read_to_string(abs).unwrap_or_default();
    let mut lines = Vec::new();
    let mut in_use = false;
    for (i, raw) in text.lines().enumerate() {
        let mut code = strip_line_comment(raw).to_string();
        if in_use {
            if code.contains(';') {
                in_use = false;
            }
            code.clear();
        } else if starts_use_statement(&code) {
            if !code.contains(';') {
                in_use = true;
            }
            code.clear();
        }
        lines.push((i + 1, code));
    }
    SrcFile { path: rel, lines }
}

/// Name of a `pub trait X` (or `pub unsafe trait X`) declared on this line, if
/// any. `pub(crate)`/`pub(super)` are intentionally excluded — rustc's
/// dead-code lint already covers non-`pub` items; this guard closes the gap for
/// genuinely public ones.
fn public_trait_name(code: &str) -> Option<String> {
    let toks: Vec<&str> = code.split_whitespace().collect();
    let name_tok: &str = match toks.as_slice() {
        ["pub", "trait", name, ..] => name,
        ["pub", "unsafe", "trait", name, ..] => name,
        _ => return None,
    };
    let name = leading_ident(name_tok);
    (!name.is_empty()).then(|| name.to_string())
}

/// Name of a `pub type X = …;` declared on this line, if any (`pub(crate)`
/// excluded, as above).
fn public_type_name(code: &str) -> Option<String> {
    let toks: Vec<&str> = code.split_whitespace().collect();
    let name_tok: &str = match toks.as_slice() {
        ["pub", "type", name, ..] => name,
        _ => return None,
    };
    let name = leading_ident(name_tok);
    (!name.is_empty()).then(|| name.to_string())
}

/// Whether `code` defines `fn NAME(` (or a generic `fn NAME<…>`) as a whole
/// word. Call sites lack the `fn` keyword, so they never match.
fn is_fn_def_of(code: &str, name: &str) -> bool {
    let bytes = code.as_bytes();
    let mut start = 0;
    while let Some(rel) = code[start..].find("fn") {
        let idx = start + rel;
        start = idx + 2;
        let before_ok = idx == 0 || !is_ident_byte(bytes[idx - 1]);
        let after_fn = idx + 2;
        let after_ok = after_fn >= bytes.len() || !is_ident_byte(bytes[after_fn]);
        if !(before_ok && after_ok) {
            continue;
        }
        if let Some(rest) = code[after_fn..].trim_start().strip_prefix(name) {
            let name_boundary = rest
                .chars()
                .next()
                .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'));
            let shape = {
                let r = rest.trim_start();
                r.starts_with('(') || r.starts_with('<')
            };
            if name_boundary && shape {
                return true;
            }
        }
    }
    false
}

/// Count of whole-word occurrences of `ident` in `haystack`.
fn count_ident_occurrences(haystack: &str, ident: &str) -> usize {
    let bytes = haystack.as_bytes();
    let len = ident.len();
    let mut count = 0;
    let mut start = 0;
    while let Some(rel) = haystack[start..].find(ident) {
        let idx = start + rel;
        let before_ok = idx == 0 || !is_ident_byte(bytes[idx - 1]);
        let after = idx + len;
        let after_ok = after >= bytes.len() || !is_ident_byte(bytes[after]);
        if before_ok && after_ok {
            count += 1;
        }
        start = idx + len;
    }
    count
}

/// Whether `code` (already trimmed of comments) begins a `use` statement,
/// including `pub use` and `pub(crate) use` re-exports.
fn starts_use_statement(code: &str) -> bool {
    let s = strip_pub_visibility(code.trim_start());
    s == "use" || s.starts_with("use ") || s.starts_with("use\t") || s.starts_with("use::")
}

/// Strip a leading `pub` or `pub(...)` visibility marker, returning the rest
/// trimmed. Leaves the input untouched if `pub` is only a prefix of a longer
/// identifier (e.g. `public`).
fn strip_pub_visibility(s: &str) -> &str {
    let rest = match s.strip_prefix("pub") {
        Some(rest) => rest,
        None => return s,
    };
    match rest.chars().next() {
        Some('(') => match rest.find(')') {
            Some(close) => rest[close + 1..].trim_start(),
            None => rest.trim_start(),
        },
        Some(c) if c.is_whitespace() => rest.trim_start(),
        _ => s,
    }
}

/// Leading run of identifier characters from the start of `s` (so `X<T>` and
/// `X:` both yield `X`).
fn leading_ident(s: &str) -> &str {
    let end = s
        .char_indices()
        .find(|(_, c)| !(c.is_ascii_alphanumeric() || *c == '_'))
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    &s[..end]
}

/// Everything before the first `//`, Rust's line-comment marker (also handles
/// `///` and `//!` doc comments).
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    }
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
