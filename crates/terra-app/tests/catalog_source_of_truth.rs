//! D2 source-of-truth ratchet for the live layer catalog (audit D2-D1).
//!
//! Layer defaults and metadata belong to `LayerTypeRegistry`; live UI routes and
//! curated presets belong to `ui/tool_catalog.rs`. The retired Add Layer module
//! duplicated both catalogs despite having no production caller. This std-only
//! guard prevents that API and its test-only catalog from returning.

use std::fs;
use std::path::{Path, PathBuf};

const RETIRED_IDENTIFIERS: &[&str] = &[
    "AddLayerEntry",
    "OrganisationKind",
    "all_add_layer_entries",
    "add_layer_menu",
    "create_layer_by_type_id",
];

#[test]
fn retired_add_layer_catalog_stays_deleted() {
    let ui_root = manifest_dir().join("src/ui");
    let retired_module = ui_root.join("add_layer_menu.rs");
    assert!(
        !retired_module.exists(),
        "D2 catalog drift: {} must not return; use tool_catalog and LayerTypeRegistry",
        relative_path(&retired_module)
    );

    let mut violations = Vec::new();
    for path in rust_files(&ui_root) {
        let source = fs::read_to_string(&path).expect("UI source is readable");
        let identifiers = identifiers_outside_comments_and_literals(&source);
        for retired in RETIRED_IDENTIFIERS {
            if identifiers.iter().any(|identifier| identifier == retired) {
                violations.push(format!("{}: retired `{retired}` API", relative_path(&path)));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "D2 catalog drift: the retired parallel Add Layer catalog returned:\n  {}\n\n\
         Put live UI routes and curated presets in src/ui/tool_catalog.rs; keep type metadata \
         and defaults in terra_core::layer::LayerTypeRegistry.",
        violations.join("\n  ")
    );
}

fn manifest_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn relative_path(path: &Path) -> String {
    path.strip_prefix(manifest_dir())
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn rust_files(root: &Path) -> Vec<PathBuf> {
    fn collect(dir: &Path, files: &mut Vec<PathBuf>) {
        let entries = fs::read_dir(dir)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", dir.display()));
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect(&path, files);
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                files.push(path);
            }
        }
    }

    let mut files = Vec::new();
    collect(root, &mut files);
    files.sort();
    files
}

/// Extract Rust-like identifiers while skipping comments and string/char literals.
/// Test-only code is intentionally included: the retired catalog is forbidden there too.
fn identifiers_outside_comments_and_literals(source: &str) -> Vec<String> {
    let bytes = source.as_bytes();
    let mut identifiers = Vec::new();
    let mut index = 0;

    while index < bytes.len() {
        if bytes[index..].starts_with(b"//") {
            index = bytes[index..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |offset| index + offset + 1);
        } else if bytes[index..].starts_with(b"/*") {
            index = skip_block_comment(bytes, index);
        } else if let Some(end) = skip_raw_string(bytes, index) {
            index = end;
        } else if bytes[index] == b'"' {
            index = skip_quoted(bytes, index, b'"');
        } else if bytes[index] == b'\'' && looks_like_char_literal(bytes, index) {
            index = skip_quoted(bytes, index, b'\'');
        } else if is_identifier_start(bytes[index]) {
            let start = index;
            index += 1;
            while index < bytes.len() && is_identifier_continue(bytes[index]) {
                index += 1;
            }
            identifiers.push(source[start..index].to_owned());
        } else {
            index += 1;
        }
    }

    identifiers
}

fn skip_block_comment(bytes: &[u8], start: usize) -> usize {
    let mut index = start + 2;
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
    index
}

fn skip_raw_string(bytes: &[u8], start: usize) -> Option<usize> {
    let mut index = start;
    if bytes.get(index) == Some(&b'b') {
        index += 1;
    }
    if bytes.get(index) != Some(&b'r') {
        return None;
    }
    index += 1;
    let hash_start = index;
    while bytes.get(index) == Some(&b'#') {
        index += 1;
    }
    if bytes.get(index) != Some(&b'"') {
        return None;
    }
    let hashes = index - hash_start;
    index += 1;
    while index < bytes.len() {
        if bytes[index] == b'"'
            && bytes
                .get(index + 1..index + 1 + hashes)
                .is_some_and(|tail| tail.iter().all(|byte| *byte == b'#'))
        {
            return Some(index + 1 + hashes);
        }
        index += 1;
    }
    Some(bytes.len())
}

fn skip_quoted(bytes: &[u8], start: usize, quote: u8) -> usize {
    let mut index = start + 1;
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            index = (index + 2).min(bytes.len());
        } else if bytes[index] == quote {
            return index + 1;
        } else {
            index += 1;
        }
    }
    bytes.len()
}

fn looks_like_char_literal(bytes: &[u8], index: usize) -> bool {
    match bytes.get(index + 1) {
        Some(b'\\') => bytes
            .get(index + 3..)
            .is_some_and(|tail| tail.contains(&b'\'')),
        Some(_) => bytes.get(index + 2) == Some(&b'\''),
        None => false,
    }
}

fn is_identifier_start(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphabetic()
}

fn is_identifier_continue(byte: u8) -> bool {
    is_identifier_start(byte) || byte.is_ascii_digit()
}

#[test]
fn scanner_rejects_legacy_catalog_declarations_but_ignores_prose() {
    let source = r#"
        // AddLayerEntry in a comment is harmless.
        const NOTE: &str = "all_add_layer_entries in prose";
        struct AddLayerEntry;
        fn all_add_layer_entries() -> Vec<AddLayerEntry> { Vec::new() }
    "#;
    let identifiers = identifiers_outside_comments_and_literals(source);
    assert!(identifiers
        .iter()
        .any(|identifier| identifier == "AddLayerEntry"));
    assert!(identifiers
        .iter()
        .any(|identifier| identifier == "all_add_layer_entries"));

    let prose = r#"// AddLayerEntry\nconst NOTE: &str = "all_add_layer_entries";"#;
    let prose_identifiers = identifiers_outside_comments_and_literals(prose);
    assert!(!prose_identifiers
        .iter()
        .any(|identifier| RETIRED_IDENTIFIERS.contains(&identifier.as_str())));
}
