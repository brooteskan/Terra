//! Architecture-documentation drift guard for issue #45.
//!
//! `docs/architecture.md` is the authoritative workspace inventory. This test
//! keeps its crate table aligned with the root manifest and rejects retired
//! UI-stack terminology in documents that present current implementation
//! guidance. The parser is deliberately std-only and limited to the small
//! Markdown/TOML shapes owned by this repository.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const ACTIVE_ARCHITECTURE_DOCS: &[&str] = &[
    "README.md",
    "CONTRIBUTING.md",
    "docs/architecture.md",
    "docs/roadmap.md",
    "docs/perf.md",
];

const RETIRED_ACTIVE_PHRASES: &[&str] = &[
    "terra-ui",
    "egui chrome",
    "egui panels",
    "\u{2192} egui",
    "-> egui",
];

#[test]
fn active_architecture_docs_match_the_workspace() {
    let root = workspace_root();
    let manifest = read(&root.join("Cargo.toml"));
    let workspace_crates = workspace_members(&manifest);
    assert!(
        !workspace_crates.is_empty(),
        "architecture docs guard could not find [workspace].members in Cargo.toml"
    );

    let architecture = read(&root.join("docs/architecture.md"));
    let documented_crates = architecture_crate_table(&architecture);
    assert_eq!(
        documented_crates, workspace_crates,
        "docs/architecture.md workspace table drifted from Cargo.toml\n  missing from docs: {}\n  not workspace members: {}",
        joined_difference(&workspace_crates, &documented_crates),
        joined_difference(&documented_crates, &workspace_crates),
    );

    let mut violations = Vec::new();
    for relative in ACTIVE_ARCHITECTURE_DOCS {
        let source = read(&root.join(relative));
        let lower = source.to_lowercase();
        for phrase in RETIRED_ACTIVE_PHRASES {
            if lower.contains(phrase) {
                violations.push(format!("{relative}: retired active phrase `{phrase}`"));
            }
        }

        for named_crate in backticked_terra_crates(&source) {
            if !workspace_crates.contains(&named_crate) {
                violations.push(format!(
                    "{relative}: names nonexistent workspace crate `{named_crate}`"
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "active architecture documentation drifted:\n  {}\n\nHistorical material belongs in a clearly labeled snapshot; current structure belongs in docs/architecture.md.",
        violations.join("\n  ")
    );
}

#[test]
fn performance_migration_is_clearly_historical() {
    let source = read(&workspace_root().join("docs/perf_migration.md"));
    let lower = source.to_lowercase();
    assert!(
        lower.contains("historical migration snapshot")
            && lower.contains("not current architecture guidance"),
        "docs/perf_migration.md must identify itself as a historical snapshot, not current guidance"
    );
    assert!(
        source.contains("[architecture.md](architecture.md)"),
        "docs/perf_migration.md must link to the authoritative architecture document"
    );
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("terra-app is two directories below the workspace root")
        .to_path_buf()
}

fn read(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn workspace_members(manifest: &str) -> BTreeSet<String> {
    let mut members = BTreeSet::new();
    let mut in_workspace = false;
    let mut in_members = false;

    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_workspace = trimmed == "[workspace]";
            in_members = false;
            continue;
        }
        if !in_workspace {
            continue;
        }
        if trimmed.starts_with("members") && trimmed.contains('[') {
            in_members = true;
        }
        if in_members {
            if let Some(path) = first_quoted_value(trimmed) {
                if let Some(crate_name) = path.strip_prefix("crates/") {
                    members.insert(crate_name.to_owned());
                }
            }
            if trimmed.contains(']') {
                in_members = false;
            }
        }
    }
    members
}

fn architecture_crate_table(markdown: &str) -> BTreeSet<String> {
    let mut crates = BTreeSet::new();
    let mut in_section = false;

    for line in markdown.lines() {
        if line.trim() == "## Workspace crates" {
            in_section = true;
            continue;
        }
        if in_section && line.starts_with("## ") {
            break;
        }
        if !in_section {
            continue;
        }
        let Some(first_cell) = line
            .strip_prefix('|')
            .and_then(|rest| rest.split('|').next())
        else {
            continue;
        };
        let cell = first_cell.trim();
        if let Some(name) = cell
            .strip_prefix('`')
            .and_then(|value| value.strip_suffix('`'))
        {
            if name.starts_with("terra-") {
                crates.insert(name.to_owned());
            }
        }
    }
    crates
}

fn backticked_terra_crates(markdown: &str) -> BTreeSet<String> {
    let mut crates = BTreeSet::new();
    let mut rest = markdown;
    while let Some(open) = rest.find('`') {
        rest = &rest[open + 1..];
        let Some(close) = rest.find('`') else {
            break;
        };
        let code = &rest[..close];
        rest = &rest[close + 1..];
        if code.starts_with("terra-") {
            let name = code
                .split(|ch: char| ch == ':' || ch == '/' || ch == '\\' || ch.is_whitespace())
                .next()
                .unwrap_or(code);
            crates.insert(name.to_owned());
        }
    }
    crates
}

fn first_quoted_value(line: &str) -> Option<&str> {
    let start = line.find('"')? + 1;
    let end = start + line[start..].find('"')?;
    Some(&line[start..end])
}

fn joined_difference(left: &BTreeSet<String>, right: &BTreeSet<String>) -> String {
    let values: Vec<_> = left.difference(right).cloned().collect();
    if values.is_empty() {
        "(none)".to_owned()
    } else {
        values.join(", ")
    }
}

#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn member_parser_stops_at_the_end_of_the_array() {
        let manifest = r#"
[workspace]
members = [
    "crates/terra-core",
    "crates/terra-app",
]
default-members = ["crates/terra-app"]

[workspace.dependencies]
terra-extra = { path = "crates/terra-extra" }
"#;
        assert_eq!(
            workspace_members(manifest),
            BTreeSet::from(["terra-app".to_owned(), "terra-core".to_owned()])
        );
    }

    #[test]
    fn crate_reference_parser_ignores_non_crate_code() {
        let markdown = "`terra-core::LayerKind` `terra-app/tests/foo.rs` `terrain`";
        assert_eq!(
            backticked_terra_crates(markdown),
            BTreeSet::from(["terra-app".to_owned(), "terra-core".to_owned()])
        );
    }
}
