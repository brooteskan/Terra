//! Purity guard: terra-gui depends on no sibling `terra-*` crate, and imports
//! none. (audit A2-G1)
//!
//! terra-gui is a reusable, app-agnostic wgpu UI toolkit. The editor's whole
//! separation of concerns rests on this crate never entangling with
//! terra-core/terra-render. The crate graph enforces edges that *exist*, but
//! nothing guards the manifest — the one file that can *create* the edge. These
//! two std-only tests are that guard, and ride plain `cargo test`.
//!
//! - `dependencies_match_allowlist` pins `Cargo.toml`'s dependency tables to an
//!   exact set. Adding any crate — sibling or third-party — fails the test until
//!   the allowlist here is edited in the same diff, making the boundary change
//!   reviewable instead of silent.
//! - `no_sibling_crate_imports` scans `src/` and `tests/` for a sibling import
//!   path, belt-and-braces for cfg-gated code the manifest test cannot see.
//!
//! Kept deliberately dumb — line scanning, no `syn` — so it grows no
//! dependencies of its own and cannot rot. Two consequences follow, both erring
//! toward a false failure (which forces a reviewed guard edit — the safe
//! direction) rather than a false pass: a forbidden token inside a string
//! literal or a `/* */` block comment is treated as real, and the manifest
//! parser assumes the one-entry-per-line dependency form the crate uses today.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// Exact set of crates terra-gui's `[dependencies]` may name.
fn expected_dependencies() -> BTreeSet<String> {
    [
        "wgpu",
        "bytemuck",
        "glam",
        "log",
        "fontdue",
        "lucide-icons",
        "serde",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Crates terra-gui's `[dev-dependencies]` may name. Both are dev-only and so
/// cannot create a runtime edge: `terra-test-gpu` is the sanctioned wgpu+pollster
/// headless harness, `serde_json` backs the `LayoutPrefs` schema snapshot test.
fn allowed_dev_dependencies() -> BTreeSet<String> {
    ["terra-test-gpu", "serde_json"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

#[test]
fn dependencies_match_allowlist() {
    let manifest = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
        .expect("terra-gui Cargo.toml is readable");

    let mut deps = BTreeSet::new();
    let mut dev_deps = BTreeSet::new();
    let mut violations: Vec<String> = Vec::new();
    let mut section = Section::Other;

    for raw in manifest.lines() {
        let line = strip_toml_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = section_header(line) {
            section = classify_section(name, &mut violations);
            continue;
        }
        match section {
            Section::Dependencies => {
                if let Some(dep) = dependency_name(line) {
                    deps.insert(dep);
                }
            }
            Section::DevDependencies => {
                if let Some(dep) = dependency_name(line) {
                    dev_deps.insert(dep);
                }
            }
            Section::Other => {}
        }
    }

    let expected = expected_dependencies();
    for extra in deps.difference(&expected) {
        violations.push(format!(
            "[dependencies] declares `{extra}`, which is not on terra-gui's allowlist; \
             if the boundary review approves it, add it to `expected_dependencies` here"
        ));
    }
    for missing in expected.difference(&deps) {
        violations.push(format!(
            "[dependencies] no longer declares `{missing}`; update `expected_dependencies` \
             here to match"
        ));
    }

    let allowed_dev = allowed_dev_dependencies();
    for dep in dev_deps.difference(&allowed_dev) {
        violations.push(format!(
            "[dev-dependencies] declares `{dep}`, which is not sanctioned; add it to \
             `allowed_dev_dependencies` here only if it cannot create a runtime edge"
        ));
    }

    assert!(
        violations.is_empty(),
        "terra-gui dependency manifest drifted from the purity allowlist:\n  {}",
        violations.join("\n  ")
    );
}

#[test]
fn no_sibling_crate_imports() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut violations: Vec<String> = Vec::new();

    // `terra_test_gpu` is a dev-only dependency, sanctioned in tests but not src.
    scan_dir(&root.join("src"), false, &mut violations);
    scan_dir(&root.join("tests"), true, &mut violations);

    assert!(
        violations.is_empty(),
        "terra-gui source reaches for a sibling terra-* crate:\n  {}",
        violations.join("\n  ")
    );
}

enum Section {
    Dependencies,
    DevDependencies,
    Other,
}

/// Classify a manifest section header. Only the bare `[dependencies]` and
/// `[dev-dependencies]` inline tables are sanctioned; every other section whose
/// name carries `dependencies` (build deps, a `[target.'cfg(..)'.dependencies]`
/// table, a dotted `[dependencies.foo]` entry) is a violation on sight — those
/// are the ways a forbidden edge could hide from a header-literal scan.
fn classify_section(name: &str, violations: &mut Vec<String>) -> Section {
    if name == "dependencies" {
        Section::Dependencies
    } else if name == "dev-dependencies" {
        Section::DevDependencies
    } else if name.contains("dependencies") {
        violations.push(format!(
            "manifest declares dependencies through `[{name}]`; only the bare \
             [dependencies] and [dev-dependencies] tables are sanctioned"
        ));
        Section::Other
    } else {
        Section::Other
    }
}

/// Inner name of a `[section]` header line, or `None` if the line is not one.
fn section_header(line: &str) -> Option<&str> {
    if line.starts_with('[') && line.ends_with(']') {
        Some(line.trim_matches(|c| c == '[' || c == ']').trim())
    } else {
        None
    }
}

/// Dependency name from a table line: the token before the first `=`, `.`, or
/// whitespace (so both `wgpu = { .. }` and `wgpu.workspace = true` yield `wgpu`).
fn dependency_name(line: &str) -> Option<String> {
    if !line.contains('=') {
        return None;
    }
    let name: String = line
        .chars()
        .take_while(|&c| c != '=' && c != '.' && !c.is_whitespace())
        .collect();
    let name = name.trim_matches('"').trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// Everything before the first `#`, TOML's line-comment marker.
fn strip_toml_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

/// Recurse `dir`, scanning every `.rs` file. A missing directory scans as empty.
fn scan_dir(dir: &Path, allow_test_gpu: bool, out: &mut Vec<String>) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_dir(&path, allow_test_gpu, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let text = fs::read_to_string(&path).unwrap_or_default();
            for (i, line) in text.lines().enumerate() {
                if let Some(token) = forbidden_token(line, allow_test_gpu) {
                    out.push(format!("{}:{}: `{token}`", path.display(), i + 1));
                }
            }
        }
    }
}

/// First sibling-crate import token on `line`, if any. The needle is assembled
/// at compile time so this scanner's own source carries no literal match to trip
/// over; the crate's own `terra_gui`, and `terra_test_gpu` under `tests/`, pass.
fn forbidden_token(line: &str, allow_test_gpu: bool) -> Option<String> {
    let code = strip_line_comment(line);
    let needle = concat!("terra", "_");
    let bytes = code.as_bytes();

    let mut start = 0;
    while let Some(rel) = code[start..].find(needle) {
        let idx = start + rel;
        let at_boundary = idx == 0 || !is_ident_byte(bytes[idx - 1]);
        if at_boundary {
            let mut end = idx;
            while end < bytes.len() && is_ident_byte(bytes[end]) {
                end += 1;
            }
            let ident = &code[idx..end];
            if !ident_allowed(ident, allow_test_gpu) {
                return Some(ident.to_string());
            }
        }
        start = idx + needle.len();
    }
    None
}

fn ident_allowed(ident: &str, allow_test_gpu: bool) -> bool {
    ident == "terra_gui" || (allow_test_gpu && ident == "terra_test_gpu")
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Everything before the first `//`, Rust's line-comment marker.
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    }
}
