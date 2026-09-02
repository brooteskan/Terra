//! E1-G1 single-door ratchet (#87).
//!
//! Edits that advance the terrain output revision must retire GPU tile residency
//! in lockstep, or a page from the prior revision keeps rendering -- the E1-C2
//! defect (#86). That lockstep lives in exactly one place:
//! `TerraApp::advance_output_revision`, which bumps the runtime revision *and*
//! calls `retire_streamed_residency`. Any edit path that instead reaches for the
//! raw `self.terrain_runtime.advance_output_revision()` bypasses the GPU-side
//! retirement, and stale pages render again (the one-line regression #86 fixed in
//! `app/actions/mod.rs`).
//!
//! This source-level guard pins that door shut: the raw runtime bump may appear
//! exactly once in production `terra-app` source -- inside the wrapper. It is
//! deliberately std-only and runs under ordinary `cargo test` (no GPU adapter
//! needed), so it protects future edit paths the GPU lifecycle tests never
//! construct.
//!
//! The scanner ignores comments, string/char literals, and `#[cfg(test)]` items,
//! reusing the sanitize approach proven in `action_discipline.rs`. Any future
//! exception should be an exact, reasoned change to this ratchet; never exempt an
//! entire file or directory.

use std::fs;
use std::path::{Path, PathBuf};

/// The raw runtime revision bump. Production code must reach it only through
/// `TerraApp::advance_output_revision`; edit sites call that wrapper instead.
const RAW_BUMP: &str = "terrain_runtime.advance_output_revision(";

/// The one file allowed to contain the raw bump: the wrapper's own definition.
const WRAPPER_FILE: &str = "src/app/eval.rs";

#[test]
fn runtime_revision_bump_is_reached_only_through_the_wrapper() {
    let mut sites = Vec::new();
    for path in rust_files(&manifest_dir().join("src")) {
        let source =
            production_source(&fs::read_to_string(&path).expect("terra-app source is readable"));
        for (offset, _) in source.match_indices(RAW_BUMP) {
            let line = source[..offset]
                .bytes()
                .filter(|&byte| byte == b'\n')
                .count()
                + 1;
            sites.push(format!("{}:{line}", relative_path(&path)));
        }
    }

    let stray: Vec<_> = sites
        .iter()
        .filter(|site| !site.starts_with(&format!("{WRAPPER_FILE}:")))
        .cloned()
        .collect();

    assert!(
        stray.is_empty(),
        "E1-G1: `{RAW_BUMP}` may be called only inside \
         TerraApp::advance_output_revision ({WRAPPER_FILE}), which pairs the runtime \
         revision bump with retire_streamed_residency so GPU tile residency is \
         invalidated in lockstep (see #86/#87). Direct call(s) found at:\n  {}\n\
         Route the edit through `self.advance_output_revision()` instead.",
        stray.join("\n  ")
    );
    assert_eq!(
        sites.len(),
        1,
        "expected exactly one raw runtime bump -- the wrapper body in {WRAPPER_FILE} -- \
         but found {}: {sites:?}. If the wrapper was refactored, update this ratchet.",
        sites.len()
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
