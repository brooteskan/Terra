//! D2 registry, catalog-exposure, and executable-command ratchets.
//!
//! These tests use public runtime registries where possible and a small
//! dependency-free source scan only for the two production wiring seams that
//! runtime reflection cannot prove.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use terra_app::ui::{
    all_workspace_definitions, commands, format_shortcuts, quick_add_entries, resolve_shortcut,
    BindingVisibility, CommandCategory, CommandId, ShortcutChord, ShortcutModifiers, ToolAction,
    WorkspaceId,
};
use terra_core::layer::LayerTypeRegistry;
use winit::keyboard::KeyCode;

/// Narrow exceptions for types that remain creatable but intentionally have no
/// live route. Empty by design; add only an exact type id and a reviewed reason.
const CREATABLE_UI_EXEMPTIONS: &[(&str, &str)] = &[];

const RETIRED_VISIBLE_COMMAND_IDS: &[&str] = &[
    "mode.sculpt",
    "mode.generate",
    "mode.erosion",
    "mode.masks",
    "mode.paint",
    "mode.biomes",
    "mode.scatter",
    "workspace.all_tools",
];

#[test]
fn every_creatable_layer_type_has_exactly_one_live_default_route() {
    let registry = LayerTypeRegistry::builtin();
    let tools = quick_add_entries();
    let exemptions: BTreeMap<_, _> = CREATABLE_UI_EXEMPTIONS.iter().copied().collect();
    assert_eq!(
        exemptions.len(),
        CREATABLE_UI_EXEMPTIONS.len(),
        "duplicate type id in CREATABLE_UI_EXEMPTIONS"
    );

    let mut routes: HashMap<&str, Vec<_>> = HashMap::new();
    for tool in &tools {
        let Some(type_id) = tool.registry_type_id else {
            continue;
        };
        let meta = registry.get(type_id).unwrap_or_else(|| {
            panic!(
                "tool {} references unknown registry type {type_id}",
                tool.id
            )
        });
        assert!(
            meta.capabilities.user_creatable,
            "tool {} exposes non-creatable registry type {type_id}",
            tool.id
        );
        routes.entry(type_id).or_default().push(tool);
    }

    let mut missing = Vec::new();
    let mut duplicate = Vec::new();
    for meta in registry.creatable() {
        let route_count = routes.get(meta.type_id).map_or(0, Vec::len);
        if let Some(reason) = exemptions.get(meta.type_id) {
            assert!(
                !reason.trim().is_empty(),
                "empty exemption reason for {}",
                meta.type_id
            );
            assert_eq!(
                route_count, 0,
                "stale exemption for {}: a live route now exists",
                meta.type_id
            );
        } else if route_count == 0 {
            missing.push(meta.type_id);
        } else if route_count != 1 {
            duplicate.push(format!("{} ({route_count} routes)", meta.type_id));
        }
    }

    for &(type_id, _) in CREATABLE_UI_EXEMPTIONS {
        let meta = registry
            .get(type_id)
            .unwrap_or_else(|| panic!("exemption references unknown registry type {type_id}"));
        assert!(
            meta.capabilities.user_creatable,
            "exemption for {type_id} is obsolete because the type is not user-creatable"
        );
    }

    assert!(
        missing.is_empty() && duplicate.is_empty(),
        "D2 catalog exposure failed:\n  missing routes: {}\n  duplicate routes: {}",
        missing.join(", "),
        duplicate.join(", ")
    );
}

#[test]
fn canonical_layer_routes_derive_metadata_and_factory_defaults() {
    let registry = LayerTypeRegistry::builtin();
    for tool in quick_add_entries()
        .into_iter()
        .filter(|tool| tool.registry_type_id.is_some())
    {
        let type_id = tool.registry_type_id.unwrap();
        let meta = registry
            .get(type_id)
            .expect("canonical route metadata exists");
        let expected = registry
            .create(type_id)
            .expect("canonical route factory exists");
        assert_eq!(tool.label, meta.display_name, "label for {type_id}");
        assert_eq!(
            tool.description, meta.description,
            "description for {type_id}"
        );
        let ToolAction::AddLayer { name, kind } = tool.action else {
            panic!("canonical route {} does not add a layer", tool.id);
        };
        assert_eq!(name, meta.display_name, "created name for {type_id}");
        assert_eq!(kind.type_id(), meta.type_id, "created type for {type_id}");
        assert_eq!(
            serde_json::to_value(kind).expect("serialize catalog kind"),
            serde_json::to_value(expected.kind).expect("serialize registry kind"),
            "factory defaults for {type_id}"
        );
    }
}

#[test]
fn visible_commands_are_unique_and_workspace_commands_are_one_to_one() {
    let visible = commands();
    let mut ids = HashSet::new();
    let mut chords = HashSet::new();
    for command in &visible {
        assert!(
            ids.insert(command.id),
            "duplicate visible command id: {}",
            command.id
        );
        assert!(
            !RETIRED_VISIBLE_COMMAND_IDS.contains(&command.id),
            "deprecated compatibility alias is visible: {}",
            command.id
        );
        for binding in command.bindings {
            assert!(
                chords.insert(binding.chord),
                "duplicate executable shortcut {:?} on {}",
                binding.chord,
                command.id
            );
        }
    }

    let definitions: Vec<_> = all_workspace_definitions()
        .iter()
        .filter(|definition| definition.id.command_id().is_some())
        .collect();
    let canonical_ids: BTreeSet<_> = definitions
        .iter()
        .map(|definition| definition.id.command_id().unwrap())
        .collect();
    let mode_commands: Vec<_> = visible
        .iter()
        .filter(|command| command.category == CommandCategory::Mode)
        .collect();

    for definition in &definitions {
        let id = definition.id.command_id().unwrap();
        let matching: Vec<_> = visible.iter().filter(|command| command.id == id).collect();
        assert_eq!(
            matching.len(),
            1,
            "workspace {:?} must have one visible command",
            definition.id
        );
        let command = matching[0];
        assert_eq!(
            command.name,
            definition.command_name.unwrap(),
            "name for {id}"
        );
        assert_eq!(command.icon, Some(definition.icon), "icon for {id}");
        assert_eq!(command.category, CommandCategory::Mode, "category for {id}");
        let digit = definition
            .id
            .digit_shortcut()
            .expect("visible workspace has a digit");
        assert_eq!(command.bindings.len(), 1, "binding count for {id}");
        assert_eq!(
            command.bindings[0].chord,
            digit_chord(digit),
            "binding for {id}"
        );
    }

    let visible_mode_ids: BTreeSet<_> = mode_commands.iter().map(|command| command.id).collect();
    assert_eq!(
        visible_mode_ids, canonical_ids,
        "visible Mode commands must be exactly the command-visible workspaces"
    );
    assert_eq!(mode_commands.len(), definitions.len());
    assert!(WorkspaceId::AllTools.command_id().is_none());
}

#[test]
fn every_displayed_shortcut_is_executable_and_required_bindings_resolve() {
    for command in commands() {
        for binding in command
            .bindings
            .iter()
            .filter(|binding| binding.visibility == BindingVisibility::Displayed)
        {
            assert_eq!(
                resolve_shortcut(binding.chord),
                Some(command.id),
                "displayed shortcut {:?} for {} is not executable",
                binding.chord,
                command.id
            );
        }
        assert!(
            !format_shortcuts(command.bindings).contains("Unknown"),
            "command {} contains an unformattable displayed shortcut",
            command.id
        );
    }

    assert_resolves(KeyCode::KeyF, none(), CommandId::FRAME_TERRAIN);
    assert_resolves(KeyCode::KeyP, ctrl(), CommandId::OPEN_COMMAND_PALETTE);
    assert_resolves(KeyCode::KeyL, ctrl(), CommandId::OPEN_QUICK_ADD);
    assert_resolves(KeyCode::Insert, none(), CommandId::OPEN_QUICK_ADD);
    assert_resolves(KeyCode::KeyS, ctrl(), CommandId::SAVE);
    assert_resolves(KeyCode::KeyS, ctrl_shift(), CommandId::SAVE_AS);
    assert_resolves(KeyCode::KeyZ, ctrl(), CommandId::UNDO);
    assert_resolves(KeyCode::KeyZ, none(), CommandId::UNDO);
    for (key, modifiers) in [
        (KeyCode::KeyY, ctrl()),
        (KeyCode::KeyZ, ctrl_shift()),
        (KeyCode::KeyZ, shift()),
        (KeyCode::KeyY, none()),
    ] {
        assert_resolves(key, modifiers, CommandId::REDO);
    }
    for definition in all_workspace_definitions()
        .iter()
        .filter(|definition| definition.id.command_id().is_some())
    {
        assert_eq!(
            resolve_shortcut(digit_chord(definition.id.digit_shortcut().unwrap())),
            definition.id.command_id()
        );
    }

    let quick_add = commands()
        .into_iter()
        .find(|command| command.id == CommandId::OPEN_QUICK_ADD)
        .unwrap();
    assert_eq!(format_shortcuts(quick_add.bindings), "Ctrl+L / Insert");
}

#[test]
fn keyboard_and_palette_remain_wired_to_executable_bindings() {
    let lifecycle = identifiers(&read(&manifest_dir().join("src/app/lifecycle.rs")));
    assert!(
        identifiers_in_order(
            &lifecycle,
            &["resolve_shortcut_for_input", "dispatch_command"]
        ),
        "D2 shortcut wiring failed: KeyboardInput must resolve a binding and dispatch its command"
    );

    let palette = identifiers(&read(&manifest_dir().join("src/ui/command_palette.rs")));
    assert!(
        identifiers_in_order(&palette, &["format_shortcuts", "command", "bindings"]),
        "D2 palette wiring failed: shortcut labels must be formatted from command.bindings"
    );

    let mut violations = Vec::new();
    for root in [
        manifest_dir().join("src/app"),
        manifest_dir().join("src/ui"),
    ] {
        for path in rust_files(&root) {
            if identifiers(&read(&path))
                .iter()
                .any(|identifier| identifier == "default_shortcut")
            {
                violations.push(relative_path(&path));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "D2 shortcut wiring failed: display-only default_shortcut metadata returned in {}",
        violations.join(", ")
    );
}

fn assert_resolves(key: KeyCode, modifiers: ShortcutModifiers, expected: &str) {
    assert_eq!(
        resolve_shortcut(ShortcutChord::new(key, modifiers)),
        Some(expected)
    );
}

fn digit_chord(digit: u8) -> ShortcutChord {
    let key = match digit {
        1 => KeyCode::Digit1,
        2 => KeyCode::Digit2,
        3 => KeyCode::Digit3,
        4 => KeyCode::Digit4,
        5 => KeyCode::Digit5,
        6 => KeyCode::Digit6,
        7 => KeyCode::Digit7,
        8 => KeyCode::Digit8,
        _ => panic!("unsupported workspace digit {digit}"),
    };
    ShortcutChord::new(key, none())
}

fn none() -> ShortcutModifiers {
    ShortcutModifiers::default()
}

fn ctrl() -> ShortcutModifiers {
    ShortcutModifiers {
        ctrl: true,
        ..none()
    }
}

fn shift() -> ShortcutModifiers {
    ShortcutModifiers {
        shift: true,
        ..none()
    }
}

fn ctrl_shift() -> ShortcutModifiers {
    ShortcutModifiers {
        ctrl: true,
        shift: true,
        ..none()
    }
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn relative_path(path: &Path) -> String {
    path.strip_prefix(manifest_dir())
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn rust_files(root: &Path) -> Vec<PathBuf> {
    fn collect(root: &Path, files: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(root)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", root.display()))
            .flatten()
        {
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

fn identifiers_in_order(haystack: &[String], needles: &[&str]) -> bool {
    let mut next = 0usize;
    for identifier in haystack {
        if needles.get(next).is_some_and(|needle| identifier == needle) {
            next += 1;
            if next == needles.len() {
                return true;
            }
        }
    }
    false
}

/// Extract identifiers while ignoring comments and string/character literals.
fn identifiers(source: &str) -> Vec<String> {
    let bytes = source.as_bytes();
    let mut found = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index..].starts_with(b"//") {
            index = bytes[index..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |offset| index + offset + 1);
        } else if bytes[index..].starts_with(b"/*") {
            index = skip_block_comment(bytes, index);
        } else if let Some(end) = raw_string_end(bytes, index) {
            index = end;
        } else if bytes[index] == b'"'
            || (bytes[index] == b'\'' && looks_like_char_literal(bytes, index))
        {
            index = skip_quoted(bytes, index, bytes[index]);
        } else if bytes[index] == b'_' || bytes[index].is_ascii_alphabetic() {
            let start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index] == b'_' || bytes[index].is_ascii_alphanumeric())
            {
                index += 1;
            }
            found.push(source[start..index].to_owned());
        } else {
            index += 1;
        }
    }
    found
}

fn looks_like_char_literal(bytes: &[u8], start: usize) -> bool {
    match bytes.get(start + 1) {
        Some(b'\\') => bytes.get(start + 3) == Some(&b'\''),
        Some(_) => bytes.get(start + 2) == Some(&b'\''),
        None => false,
    }
}

fn skip_block_comment(bytes: &[u8], start: usize) -> usize {
    let mut index = start + 2;
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
    index
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

#[test]
fn source_scanner_ignores_comments_and_literals() {
    let source = r###"
        // default_shortcut resolve_shortcut_for_input
        const NOTE: &str = "default_shortcut dispatch_command";
        let _ = r#"format_shortcuts command bindings"#;
        resolve_shortcut_for_input(chord, false);
        dispatch_command(command);
    "###;
    let found = identifiers(source);
    assert!(!found
        .iter()
        .any(|identifier| identifier == "default_shortcut"));
    assert!(identifiers_in_order(
        &found,
        &["resolve_shortcut_for_input", "dispatch_command"]
    ));
    assert!(!identifiers_in_order(
        &found,
        &["format_shortcuts", "command", "bindings"]
    ));
}
