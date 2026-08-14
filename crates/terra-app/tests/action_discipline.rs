//! A4 shell/action discipline ratchet.
//!
//! Production UI may observe `TerrainDocument`, but it must emit `PanelAction`
//! values instead of mutating the document. Every action variant must then have
//! exactly one production handler under `app/actions`. This source-level guard
//! is deliberately std-only and runs with ordinary `cargo test`.
//!
//! The scanner ignores comments, literals, and `#[cfg(test)]` items. Any future
//! exception should be an exact, reasoned entry beside the relevant assertion;
//! never exempt an entire file or directory.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
struct Token {
    text: String,
    line: usize,
}

#[test]
fn production_ui_cannot_mutate_terrain_document() {
    let root = manifest_dir().join("src/ui");
    let mut violations = Vec::new();

    for path in rust_files(&root) {
        let source = production_source(&fs::read_to_string(&path).expect("UI source is readable"));
        let tokens = tokenize(&source);
        let relative = relative_path(&path);

        for line in mutable_terrain_document_lines(&tokens) {
            violations.push(format!(
                "  {relative}:{line}: mutable TerrainDocument reference"
            ));
        }
        for (line, binding) in direct_document_write_lines(&tokens) {
            violations.push(format!(
                "  {relative}:{line}: direct write rooted at `{binding}`"
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "A4 UI mutation boundary failed:\n{}\n\nRoute changes through \
         FrameUiOutput.actions and handle them under src/app/actions.",
        violations.join("\n")
    );
}

#[test]
fn panel_action_variants_have_exactly_one_production_handler() {
    let action_path = manifest_dir().join("src/ui/actions/mod.rs");
    let action_source = production_source(
        &fs::read_to_string(&action_path).expect("PanelAction source is readable"),
    );
    let declared = panel_action_variants(&tokenize(&action_source));

    let mut handled: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in rust_files(&manifest_dir().join("src/app/actions")) {
        let source = production_source(
            &fs::read_to_string(&path).expect("action handler source is readable"),
        );
        for (variant, line) in panel_action_handlers(&tokenize(&source)) {
            handled
                .entry(variant)
                .or_default()
                .push(format!("{}:{line}", relative_path(&path)));
        }
    }

    let missing: Vec<_> = declared
        .iter()
        .filter(|variant| !handled.contains_key(*variant))
        .cloned()
        .collect();
    let undeclared: Vec<_> = handled
        .keys()
        .filter(|variant| !declared.contains(*variant))
        .cloned()
        .collect();
    let duplicates: Vec<_> = handled
        .iter()
        .filter(|(_, locations)| locations.len() > 1)
        .map(|(variant, locations)| format!("{variant} [{}]", locations.join(", ")))
        .collect();

    let mut problems = Vec::new();
    if !missing.is_empty() {
        problems.push(format!("  missing handlers: {}", missing.join(", ")));
    }
    if !undeclared.is_empty() {
        problems.push(format!(
            "  handlers for undeclared variants: {}",
            undeclared.join(", ")
        ));
    }
    if !duplicates.is_empty() {
        problems.push(format!("  duplicate handlers: {}", duplicates.join("; ")));
    }

    assert!(
        problems.is_empty(),
        "A4 PanelAction dispatch drift:\n{}",
        problems.join("\n")
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
    fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = fs::read_dir(dir).unwrap_or_else(|error| {
            panic!("failed to read {}: {error}", dir.display());
        });
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect(&path, out);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }

    let mut files = Vec::new();
    collect(root, &mut files);
    files.sort();
    files
}

/// Blank comments and literal contents while preserving newlines and byte width.
fn sanitize_rust(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = bytes.to_vec();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i..].starts_with(b"//") {
            let end = bytes[i..]
                .iter()
                .position(|&byte| byte == b'\n')
                .map_or(bytes.len(), |offset| i + offset);
            blank(&mut out, i, end);
            i = end;
        } else if bytes[i..].starts_with(b"/*") {
            let start = i;
            i += 2;
            let mut depth = 1;
            while i < bytes.len() && depth > 0 {
                if bytes[i..].starts_with(b"/*") {
                    depth += 1;
                    i += 2;
                } else if bytes[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            blank(&mut out, start, i);
        } else if let Some((quote, hashes)) = raw_string_start(bytes, i) {
            let start = i;
            i = quote + 1;
            while i < bytes.len() {
                if bytes[i] == b'"'
                    && bytes
                        .get(i + 1..i + 1 + hashes)
                        .is_some_and(|tail| tail.iter().all(|&byte| byte == b'#'))
                {
                    i += 1 + hashes;
                    break;
                }
                i += 1;
            }
            blank(&mut out, start, i);
        } else if let Some(quote) = ordinary_string_start(bytes, i) {
            let start = i;
            i = quote + 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i = (i + 2).min(bytes.len());
                } else if bytes[i] == b'"' {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
            blank(&mut out, start, i);
        } else if bytes[i] == b'\'' && looks_like_char_literal(bytes, i) {
            let start = i;
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i = (i + 2).min(bytes.len());
                } else if bytes[i] == b'\'' {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
            blank(&mut out, start, i);
        } else {
            i += 1;
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

fn raw_string_start(bytes: &[u8], i: usize) -> Option<(usize, usize)> {
    let mut cursor = i;
    if bytes.get(cursor) == Some(&b'b') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'r') {
        return None;
    }
    cursor += 1;
    let hash_start = cursor;
    while bytes.get(cursor) == Some(&b'#') {
        cursor += 1;
    }
    (bytes.get(cursor) == Some(&b'"')).then_some((cursor, cursor - hash_start))
}

fn ordinary_string_start(bytes: &[u8], i: usize) -> Option<usize> {
    match (bytes.get(i), bytes.get(i + 1)) {
        (Some(b'"'), _) => Some(i),
        (Some(b'b'), Some(b'"')) | (Some(b'c'), Some(b'"')) => Some(i + 1),
        _ => None,
    }
}

fn looks_like_char_literal(bytes: &[u8], i: usize) -> bool {
    let Some(&next) = bytes.get(i + 1) else {
        return false;
    };
    if next == b'\\' {
        bytes.get(i + 3..).is_some_and(|tail| tail.contains(&b'\''))
    } else {
        bytes.get(i + 2) == Some(&b'\'')
    }
}

/// Remove cfg(test)-attributed items after sanitizing, preserving line numbers.
fn production_source(source: &str) -> String {
    let sanitized = sanitize_rust(source);
    let mut out = String::with_capacity(sanitized.len());
    let mut depth: i32 = 0;
    let mut pending_cfg_test = false;
    let mut skip_base: Option<i32> = None;

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

        if trimmed.contains("#[cfg(test)]") {
            push_blank_line(&mut out, line);
            let base = depth;
            depth += delta;
            if depth > base {
                skip_base = Some(base);
            } else if trimmed
                .split_once("#[cfg(test)]")
                .is_some_and(|(_, rest)| rest.trim().is_empty())
            {
                pending_cfg_test = true;
            }
            continue;
        }

        if pending_cfg_test {
            push_blank_line(&mut out, line);
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

        out.push_str(line);
        depth += delta;
    }
    out
}

fn push_blank_line(out: &mut String, line: &str) {
    for byte in line.bytes() {
        out.push(if byte == b'\n' { '\n' } else { ' ' });
    }
}

fn brace_delta(line: &str) -> i32 {
    line.bytes().filter(|&byte| byte == b'{').count() as i32
        - line.bytes().filter(|&byte| byte == b'}').count() as i32
}

fn tokenize(source: &str) -> Vec<Token> {
    const THREE_CHAR: &[&str] = &["<<=", ">>="];
    const TWO_CHAR: &[&str] = &[
        "::", "=>", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<",
        ">>",
    ];

    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    let mut line = 1;
    while i < bytes.len() {
        if bytes[i] == b'\n' {
            line += 1;
            i += 1;
        } else if !bytes[i].is_ascii() {
            // Source audit vocabulary is ASCII. Skip a complete Unicode scalar
            // (including an optional UTF-8 BOM) without slicing inside it.
            i += source[i..]
                .chars()
                .next()
                .expect("index is on a char boundary")
                .len_utf8();
        } else if bytes[i].is_ascii_whitespace() {
            i += 1;
        } else if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_continue(bytes[i]) {
                i += 1;
            }
            tokens.push(Token {
                text: source[start..i].to_string(),
                line,
            });
        } else {
            let mut matched = None;
            for operator in THREE_CHAR {
                if source[i..].starts_with(operator) {
                    matched = Some(*operator);
                    break;
                }
            }
            for operator in TWO_CHAR {
                if matched.is_none() && source[i..].starts_with(operator) {
                    matched = Some(*operator);
                    break;
                }
            }
            let text = matched.unwrap_or_else(|| &source[i..i + 1]);
            tokens.push(Token {
                text: text.to_string(),
                line,
            });
            i += text.len();
        }
    }
    tokens
}

fn is_ident_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn is_ident_continue(byte: u8) -> bool {
    is_ident_start(byte) || byte.is_ascii_digit()
}

fn mutable_terrain_document_lines(tokens: &[Token]) -> BTreeSet<usize> {
    let mut lines = BTreeSet::new();
    for (index, token) in tokens.iter().enumerate() {
        if token.text != "&" {
            continue;
        }
        let end = (index + 12).min(tokens.len());
        let window = &tokens[index + 1..end];
        let Some(mut_pos) = window.iter().position(|token| token.text == "mut") else {
            continue;
        };
        if window[mut_pos + 1..]
            .iter()
            .take_while(|token| !matches!(token.text.as_str(), "," | ")" | "=" | ";"))
            .any(|token| token.text == "TerrainDocument")
        {
            lines.insert(token.line);
        }
    }
    lines
}

fn direct_document_write_lines(tokens: &[Token]) -> BTreeSet<(usize, String)> {
    let mut bindings = BTreeSet::from(["doc".to_string(), "document".to_string()]);
    for index in 0..tokens.len().saturating_sub(2) {
        if !is_identifier(&tokens[index].text) || tokens[index + 1].text != ":" {
            continue;
        }
        let end = (index + 18).min(tokens.len());
        if tokens[index + 2..end]
            .iter()
            .take_while(|token| !matches!(token.text.as_str(), "," | ")" | "=" | ";"))
            .any(|token| token.text == "TerrainDocument")
        {
            bindings.insert(tokens[index].text.clone());
        }
    }

    let assignments = [
        "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=",
    ];
    let mut writes = BTreeSet::new();
    for (index, token) in tokens.iter().enumerate() {
        if !bindings.contains(&token.text)
            || tokens.get(index + 1).map(|t| t.text.as_str()) != Some(".")
        {
            continue;
        }
        let mut cursor = index + 2;
        let mut saw_member = false;
        let mut bracket_depth = 0_i32;
        while cursor < tokens.len() {
            match tokens[cursor].text.as_str() {
                text if bracket_depth > 0 => match text {
                    "[" => bracket_depth += 1,
                    "]" => bracket_depth -= 1,
                    _ => {}
                },
                text if is_identifier(text) => saw_member = true,
                "." => {}
                "[" if saw_member => bracket_depth = 1,
                text if assignments.contains(&text) && saw_member => {
                    writes.insert((token.line, token.text.clone()));
                    break;
                }
                ";" | "," | "{" | "}" | "(" | ")" | "=>" => break,
                _ => break,
            }
            cursor += 1;
        }
    }
    writes
}

fn panel_action_variants(tokens: &[Token]) -> BTreeSet<String> {
    let enum_index = tokens
        .windows(2)
        .position(|pair| pair[0].text == "enum" && pair[1].text == "PanelAction")
        .expect("PanelAction enum exists");
    let open = tokens[enum_index..]
        .iter()
        .position(|token| token.text == "{")
        .map(|offset| enum_index + offset)
        .expect("PanelAction enum has a body");

    let mut variants = BTreeSet::new();
    let mut depth = 1_i32;
    let mut expect_variant = true;
    for token in &tokens[open + 1..] {
        match token.text.as_str() {
            "{" | "(" | "[" => depth += 1,
            "}" | ")" | "]" => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            "," if depth == 1 => expect_variant = true,
            text if depth == 1 && expect_variant && is_identifier(text) => {
                variants.insert(text.to_string());
                expect_variant = false;
            }
            _ => {}
        }
    }
    variants
}

fn panel_action_handlers(tokens: &[Token]) -> Vec<(String, usize)> {
    let mut handlers = Vec::new();
    for index in 0..tokens.len().saturating_sub(2) {
        if tokens[index].text != "PanelAction"
            || tokens[index + 1].text != "::"
            || !is_identifier(&tokens[index + 2].text)
        {
            continue;
        }
        if is_match_arm(tokens, index + 3) || is_let_pattern(tokens, index, index + 3) {
            handlers.push((tokens[index + 2].text.clone(), tokens[index].line));
        }
    }
    handlers
}

fn is_match_arm(tokens: &[Token], start: usize) -> bool {
    let mut depth = 0_i32;
    for token in &tokens[start..] {
        match token.text.as_str() {
            "{" | "(" | "[" => depth += 1,
            "}" | ")" | "]" if depth > 0 => depth -= 1,
            "=>" if depth == 0 => return true,
            ";" | "," | "=" if depth == 0 => return false,
            _ => {}
        }
    }
    false
}

fn is_let_pattern(tokens: &[Token], occurrence: usize, start: usize) -> bool {
    let boundary = tokens[..occurrence]
        .iter()
        .rposition(|token| matches!(token.text.as_str(), ";" | "{" | "}" | "=>"))
        .map_or(0, |index| index + 1);
    if !tokens[boundary..occurrence]
        .iter()
        .any(|token| token.text == "let")
    {
        return false;
    }

    let mut depth = 0_i32;
    let mut saw_equals = false;
    for token in &tokens[start..] {
        match token.text.as_str() {
            "{" | "(" | "[" => depth += 1,
            "}" | ")" | "]" if depth > 0 => depth -= 1,
            "=" if depth == 0 => saw_equals = true,
            "else" if depth == 0 && saw_equals => return true,
            ";" if depth == 0 => return saw_equals,
            _ => {}
        }
    }
    false
}

fn is_identifier(text: &str) -> bool {
    text.as_bytes()
        .first()
        .is_some_and(|byte| is_ident_start(*byte))
        && text.as_bytes().iter().all(|byte| is_ident_continue(*byte))
}

#[cfg(test)]
mod scanner_tests {
    use super::*;

    #[test]
    fn sanitizing_ignores_comments_and_literals() {
        let source = r###"
            // PanelAction::Fake => doc.value = 1;
            /* nested /* PanelAction::AlsoFake */ doc.value += 1; */
            const TEXT: &str = r#"PanelAction::StringFake => doc.value = 2"#;
            fn production(doc: &TerrainDocument) { let _ = doc.value == 2; }
        "###;
        let tokens = tokenize(&production_source(source));
        assert!(panel_action_handlers(&tokens).is_empty());
        assert!(mutable_terrain_document_lines(&tokens).is_empty());
        assert!(direct_document_write_lines(&tokens).is_empty());
    }

    #[test]
    fn production_filter_excludes_test_only_handlers() {
        let source = r#"
            fn production(action: PanelAction) {
                match action { PanelAction::Live => {} }
            }
            #[cfg(test)]
            mod tests {
                fn helper(action: PanelAction) {
                    match action { PanelAction::TestOnly => {} }
                }
            }
            #[cfg(test)] fn one_line(action: PanelAction) { match action { PanelAction::AlsoTestOnly => {} } }
        "#;
        let handlers = panel_action_handlers(&tokenize(&production_source(source)));
        assert_eq!(
            handlers
                .iter()
                .map(|pair| pair.0.as_str())
                .collect::<Vec<_>>(),
            ["Live"]
        );
    }

    #[test]
    fn ui_boundary_detects_mutability_and_writes_but_not_comparisons() {
        let source = r#"
            fn good(doc: &TerrainDocument) { let _ = doc.selected == None; }
            fn bad(project: &mut terra_core::document::TerrainDocument) {
                project.preview_resolution = 4;
                project.level_steps.max_level += 1;
                project.values[0] <<= 1;
            }
        "#;
        let tokens = tokenize(&production_source(source));
        assert_eq!(mutable_terrain_document_lines(&tokens), BTreeSet::from([3]));
        assert_eq!(
            direct_document_write_lines(&tokens),
            BTreeSet::from([
                (4, "project".into()),
                (5, "project".into()),
                (6, "project".into()),
            ])
        );
    }

    #[test]
    fn enum_extraction_handles_all_variant_shapes() {
        let tokens = tokenize("enum PanelAction { Unit, Tuple(u32), Struct { value: u32 } }");
        assert_eq!(
            panel_action_variants(&tokens),
            BTreeSet::from(["Struct".into(), "Tuple".into(), "Unit".into()])
        );
    }

    #[test]
    fn handlers_count_patterns_not_constructors() {
        let source = r#"
            fn route(action: PanelAction) {
                match action { PanelAction::Matched { value } => use_it(value) }
                let PanelAction::LetElse(value) = action else { return; };
                let _constructed = PanelAction::Constructed;
                match other { make(PanelAction::NestedConstructor), PanelAction::Second => {} }
            }
        "#;
        let handlers = panel_action_handlers(&tokenize(&production_source(source)));
        assert_eq!(
            handlers
                .iter()
                .map(|pair| pair.0.as_str())
                .collect::<Vec<_>>(),
            ["Matched", "LetElse", "Second"]
        );
    }
}
