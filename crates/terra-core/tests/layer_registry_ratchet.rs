//! D2 layer-registry completeness ratchet.
//!
//! Rust cannot reflect over enum variants, so this test uses a deliberately
//! small, dependency-free source scanner. Runtime assertions then verify the
//! metadata and factories exposed by `LayerTypeRegistry`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use terra_core::layer::{LayerKind, LayerTypeRegistry, WorkflowStage};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Token {
    text: String,
    line: usize,
}

#[test]
fn layer_kind_variants_and_builtin_factories_are_exactly_equal() {
    let enum_path = manifest_dir().join("src/layer/kinds/mod.rs");
    let registry_path = manifest_dir().join("src/layer/registry.rs");
    let enum_source = read(&enum_path);
    let registry_source = read(&registry_path);

    let variants = enum_variants(&tokenize(&sanitize_rust(&enum_source)), "LayerKind");
    let factories = registry_factory_variants(&tokenize(&sanitize_rust(&registry_source)));
    let factory_set: BTreeSet<_> = factories.keys().cloned().collect();

    let missing: Vec<_> = variants.difference(&factory_set).cloned().collect();
    let unknown: Vec<_> = factory_set.difference(&variants).cloned().collect();
    let duplicates: Vec<_> = factories
        .iter()
        .filter(|(_, lines)| lines.len() > 1)
        .map(|(variant, lines)| format!("{variant} at lines {lines:?}"))
        .collect();

    let mut problems = Vec::new();
    if !missing.is_empty() {
        problems.push(format!("  missing registrations: {}", missing.join(", ")));
    }
    if !unknown.is_empty() {
        problems.push(format!(
            "  unknown factory variants: {}",
            unknown.join(", ")
        ));
    }
    if !duplicates.is_empty() {
        problems.push(format!("  duplicate factories: {}", duplicates.join("; ")));
    }

    assert!(
        problems.is_empty(),
        "D2 layer registry completeness failed:\n{}",
        problems.join("\n")
    );
}

#[test]
fn builtin_registry_metadata_matches_every_factory() {
    let registry_source = read(&manifest_dir().join("src/layer/registry.rs"));
    let factories = registry_factory_variants(&tokenize(&sanitize_rust(&registry_source)));
    let registry = LayerTypeRegistry::builtin();
    let mut ids = BTreeSet::new();
    let mut runtime_variants = BTreeSet::new();

    for meta in registry.all() {
        assert!(
            ids.insert(meta.type_id),
            "duplicate type id: {}",
            meta.type_id
        );
        let created = registry
            .create(meta.type_id)
            .unwrap_or_else(|| panic!("registered type {} has no factory", meta.type_id));
        let kind = &created.kind;
        let variant = debug_variant_name(kind);
        assert!(
            factories.contains_key(&variant),
            "factory for {} created unregistered variant {}",
            meta.type_id,
            variant
        );
        assert!(
            runtime_variants.insert(variant.clone()),
            "multiple registry entries create LayerKind::{variant}"
        );
        assert_eq!(meta.type_id, kind.type_id(), "type id for {variant}");
        assert_eq!(
            meta.display_name,
            kind.type_display_name(),
            "display name for {variant}"
        );
        assert_eq!(
            meta.workflow_stage,
            kind.workflow_stage(),
            "workflow stage for {variant}"
        );
        assert_eq!(
            created.common.name, meta.display_name,
            "factory name for {variant}"
        );
    }

    assert_eq!(runtime_variants, factories.keys().cloned().collect());
    assert_eq!(
        registry.get("stamp_2d").unwrap().workflow_stage,
        WorkflowStage::Foundation
    );
    assert_eq!(
        registry.get("stamp_3d").unwrap().workflow_stage,
        WorkflowStage::Foundation
    );
    assert_eq!(
        registry.get("coastal").unwrap().workflow_stage,
        WorkflowStage::Simulation
    );
}

fn debug_variant_name(kind: &LayerKind) -> String {
    format!("{kind:?}")
        .split(['(', '{'])
        .next()
        .expect("LayerKind Debug output has a variant name")
        .trim()
        .to_owned()
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn enum_variants(tokens: &[Token], enum_name: &str) -> BTreeSet<String> {
    let enum_index = tokens
        .windows(2)
        .position(|pair| pair[0].text == "enum" && pair[1].text == enum_name)
        .unwrap_or_else(|| panic!("could not find enum {enum_name}"));
    let open = tokens[enum_index + 2..]
        .iter()
        .position(|token| token.text == "{")
        .map(|offset| enum_index + 2 + offset)
        .expect("enum has an opening brace");
    let close = matching(tokens, open, "{", "}");

    let mut variants = BTreeSet::new();
    let (mut parens, mut brackets, mut braces) = (0usize, 0usize, 0usize);
    let mut expect_variant = true;
    for token in &tokens[open + 1..close] {
        match token.text.as_str() {
            "(" => parens += 1,
            ")" => parens = parens.saturating_sub(1),
            "[" => brackets += 1,
            "]" => brackets = brackets.saturating_sub(1),
            "{" => braces += 1,
            "}" => braces = braces.saturating_sub(1),
            "," if parens == 0 && brackets == 0 && braces == 0 => expect_variant = true,
            text if expect_variant
                && parens == 0
                && brackets == 0
                && braces == 0
                && is_identifier(text) =>
            {
                variants.insert(text.to_owned());
                expect_variant = false;
            }
            _ => {}
        }
    }
    variants
}

fn registry_factory_variants(tokens: &[Token]) -> BTreeMap<String, Vec<usize>> {
    let body = function_body(tokens, "register_all");
    let mut factories: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut index = body.start;
    while index + 2 < body.end {
        if tokens[index].text == "entry"
            && tokens[index + 1].text == "!"
            && tokens[index + 2].text == "("
        {
            let close = matching(tokens, index + 2, "(", ")");
            let matches: Vec<_> = (index + 3..close.saturating_sub(3))
                .filter(|&at| {
                    tokens[at].text == "LayerKind"
                        && tokens[at + 1].text == ":"
                        && tokens[at + 2].text == ":"
                        && is_identifier(&tokens[at + 3].text)
                })
                .collect();
            assert_eq!(
                matches.len(),
                1,
                "entry! at registry.rs:{} must contain exactly one LayerKind factory",
                tokens[index].line
            );
            let at = matches[0];
            factories
                .entry(tokens[at + 3].text.clone())
                .or_default()
                .push(tokens[at].line);
            index = close + 1;
        } else {
            index += 1;
        }
    }
    factories
}

fn function_body(tokens: &[Token], name: &str) -> std::ops::Range<usize> {
    let function = tokens
        .windows(2)
        .position(|pair| pair[0].text == "fn" && pair[1].text == name)
        .unwrap_or_else(|| panic!("could not find function {name}"));
    let open = tokens[function + 2..]
        .iter()
        .position(|token| token.text == "{")
        .map(|offset| function + 2 + offset)
        .expect("function has an opening brace");
    open + 1..matching(tokens, open, "{", "}")
}

fn matching(tokens: &[Token], open: usize, opener: &str, closer: &str) -> usize {
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate().skip(open) {
        if token.text == opener {
            depth += 1;
        } else if token.text == closer {
            depth -= 1;
            if depth == 0 {
                return index;
            }
        }
    }
    panic!("unclosed delimiter {opener} at line {}", tokens[open].line);
}

fn tokenize(source: &str) -> Vec<Token> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let (mut index, mut line) = (0usize, 1usize);
    while index < bytes.len() {
        if bytes[index] == b'\n' {
            line += 1;
            index += 1;
        } else if bytes[index] == b'_' || bytes[index].is_ascii_alphabetic() {
            let start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index] == b'_' || bytes[index].is_ascii_alphanumeric())
            {
                index += 1;
            }
            tokens.push(Token {
                text: source[start..index].to_owned(),
                line,
            });
        } else if !bytes[index].is_ascii_whitespace() {
            tokens.push(Token {
                text: (bytes[index] as char).to_string(),
                line,
            });
            index += 1;
        } else {
            index += 1;
        }
    }
    tokens
}

fn is_identifier(text: &str) -> bool {
    text.as_bytes()
        .first()
        .is_some_and(|byte| *byte == b'_' || byte.is_ascii_alphabetic())
        && text
            .as_bytes()
            .iter()
            .all(|byte| *byte == b'_' || byte.is_ascii_alphanumeric())
}

/// Blank comments and literal contents while preserving byte width and lines.
fn sanitize_rust(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = bytes.to_vec();
    let mut index = 0usize;
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
            let mut depth = 1usize;
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
        } else if let Some(end) = raw_string_end(bytes, index) {
            blank(&mut out, index, end);
            index = end;
        } else if bytes[index] == b'"'
            || (bytes[index] == b'\'' && looks_like_char_literal(bytes, index))
        {
            let quote = bytes[index];
            let start = index;
            index += 1;
            while index < bytes.len() {
                if bytes[index] == b'\\' {
                    index = (index + 2).min(bytes.len());
                } else if bytes[index] == quote {
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

fn looks_like_char_literal(bytes: &[u8], start: usize) -> bool {
    match bytes.get(start + 1) {
        Some(b'\\') => bytes.get(start + 3) == Some(&b'\''),
        Some(_) => bytes.get(start + 2) == Some(&b'\''),
        None => false,
    }
}

fn raw_string_end(bytes: &[u8], start: usize) -> Option<usize> {
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

fn blank(bytes: &mut [u8], start: usize, end: usize) {
    for byte in &mut bytes[start..end] {
        if *byte != b'\n' && *byte != b'\r' {
            *byte = b' ';
        }
    }
}

#[test]
fn scanner_handles_variant_shapes_attributes_and_fake_source_text() {
    let source = r###"
        enum LayerKind {
            #[serde(rename = "fake")]
            Tuple(Value),
            Unit,
            Struct { value: usize },
            // CommentedOut(Value),
        }
        const FAKE: &str = "StringVariant(Value)";
        fn register_all() {
            macro_rules! entry { ($factory:expr) => {}; }
            entry!(|| LayerKind::Tuple(Default::default()));
            entry!(|| LayerKind::Unit);
            entry!(|| LayerKind::Struct { value: 0 });
            let _ = r#"LayerKind::NotAFactory"#;
        }
    "###;
    let tokens = tokenize(&sanitize_rust(source));
    assert_eq!(
        enum_variants(&tokens, "LayerKind"),
        ["Struct", "Tuple", "Unit"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    );
    assert_eq!(
        registry_factory_variants(&tokens)
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>(),
        ["Struct", "Tuple", "Unit"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    );
}
