//! Central definition of editor commands and their default shortcuts.

use crate::ui::workspace::{all_workspace_definitions, WorkspaceId};
use terra_gui::Icon;
use winit::keyboard::KeyCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandCategory {
    File,
    Edit,
    View,
    Layer,
    Mode,
    Viewport,
    Tools,
    /// Developer / analysis overlays (not shown in default artist chrome).
    Debug,
}

impl CommandCategory {
    pub const fn label(self) -> &'static str {
        match self {
            Self::File => "File",
            Self::Edit => "Edit",
            Self::View => "View",
            Self::Layer => "Layer",
            Self::Mode => "Mode",
            Self::Viewport => "Viewport",
            Self::Tools => "Tools",
            Self::Debug => "Debug",
        }
    }
}

/// Stable IDs shared by keyboard dispatch and the command palette.
pub struct CommandId;

impl CommandId {
    pub const ADD_MOUNTAIN: &'static str = "layer.add_mountain";
    pub const ADD_HYDRAULIC_EROSION: &'static str = "layer.add_hydraulic_erosion";
    pub const FRAME_TERRAIN: &'static str = "viewport.frame_terrain";
    pub const TOP_VIEW: &'static str = "viewport.top_view";
    pub const FRAME_SELECTION: &'static str = "viewport.frame_selection";
    pub const TOGGLE_HEIGHT_VIEW: &'static str = "view.toggle_height";
    pub const TOGGLE_WIREFRAME: &'static str = "view.toggle_wireframe";
    pub const OPEN_QUICK_ADD: &'static str = "tools.quick_add";
    pub const OPEN_COMMAND_PALETTE: &'static str = "tools.command_palette";
    pub const UNDO: &'static str = "edit.undo";
    pub const REDO: &'static str = "edit.redo";
    pub const EXPORT: &'static str = "file.export";
    pub const SAVE: &'static str = "file.save";
    pub const SAVE_AS: &'static str = "file.save_as";
    pub const NEW_PROJECT: &'static str = "file.new";
    pub const OPEN_PROJECT: &'static str = "file.open";
    pub const CLOSE_PROJECT: &'static str = "file.close";
    pub const BAKE_SELECTED: &'static str = "layer.bake_selected";
    pub const OPEN_PROFILER: &'static str = "view.profiler";
    pub const TOGGLE_HISTORY: &'static str = "view.history";
    pub const TOGGLE_PIPELINE: &'static str = "view.pipeline";
    pub const CLEAR_GEOMORPH_DEBUG: &'static str = "debug.geomorph.clear";
}

#[derive(Debug, Clone, Copy)]
pub struct CommandDef {
    pub id: &'static str,
    pub name: &'static str,
    pub category: CommandCategory,
    pub bindings: &'static [ShortcutBinding],
    pub icon: Option<Icon>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ShortcutModifiers {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub super_key: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShortcutChord {
    pub key: KeyCode,
    pub modifiers: ShortcutModifiers,
}
impl ShortcutChord {
    pub const fn new(key: KeyCode, modifiers: ShortcutModifiers) -> Self {
        Self { key, modifiers }
    }
    pub fn is_unmodified_text_key(self) -> bool {
        self.modifiers == ShortcutModifiers::default()
            && matches!(
                self.key,
                KeyCode::Digit0
                    | KeyCode::Digit1
                    | KeyCode::Digit2
                    | KeyCode::Digit3
                    | KeyCode::Digit4
                    | KeyCode::Digit5
                    | KeyCode::Digit6
                    | KeyCode::Digit7
                    | KeyCode::Digit8
                    | KeyCode::Digit9
                    | KeyCode::KeyA
                    | KeyCode::KeyB
                    | KeyCode::KeyC
                    | KeyCode::KeyD
                    | KeyCode::KeyE
                    | KeyCode::KeyF
                    | KeyCode::KeyG
                    | KeyCode::KeyH
                    | KeyCode::KeyI
                    | KeyCode::KeyJ
                    | KeyCode::KeyK
                    | KeyCode::KeyL
                    | KeyCode::KeyM
                    | KeyCode::KeyN
                    | KeyCode::KeyO
                    | KeyCode::KeyP
                    | KeyCode::KeyQ
                    | KeyCode::KeyR
                    | KeyCode::KeyS
                    | KeyCode::KeyT
                    | KeyCode::KeyU
                    | KeyCode::KeyV
                    | KeyCode::KeyW
                    | KeyCode::KeyX
                    | KeyCode::KeyY
                    | KeyCode::KeyZ
            )
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingVisibility {
    Displayed,
    LegacyAlias,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortcutBinding {
    pub chord: ShortcutChord,
    pub visibility: BindingVisibility,
}
impl ShortcutBinding {
    pub const fn displayed(key: KeyCode, modifiers: ShortcutModifiers) -> Self {
        Self {
            chord: ShortcutChord::new(key, modifiers),
            visibility: BindingVisibility::Displayed,
        }
    }
    pub const fn alias(key: KeyCode, modifiers: ShortcutModifiers) -> Self {
        Self {
            chord: ShortcutChord::new(key, modifiers),
            visibility: BindingVisibility::LegacyAlias,
        }
    }
}
const NONE: ShortcutModifiers = ShortcutModifiers {
    ctrl: false,
    shift: false,
    alt: false,
    super_key: false,
};
const CTRL: ShortcutModifiers = ShortcutModifiers { ctrl: true, ..NONE };
const SHIFT: ShortcutModifiers = ShortcutModifiers {
    shift: true,
    ..NONE
};
const CTRL_SHIFT: ShortcutModifiers = ShortcutModifiers {
    ctrl: true,
    shift: true,
    ..NONE
};
const D1: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::Digit1, NONE)];
const D2: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::Digit2, NONE)];
const D3: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::Digit3, NONE)];
const D4: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::Digit4, NONE)];
const D5: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::Digit5, NONE)];
const D6: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::Digit6, NONE)];
const D7: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::Digit7, NONE)];
const D8: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::Digit8, NONE)];
const FRAME: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::KeyF, NONE)];
const QUICK_ADD: &[ShortcutBinding] = &[
    ShortcutBinding::displayed(KeyCode::KeyL, CTRL),
    ShortcutBinding::displayed(KeyCode::Insert, NONE),
];
const PALETTE: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::KeyP, CTRL)];
const UNDO: &[ShortcutBinding] = &[
    ShortcutBinding::displayed(KeyCode::KeyZ, CTRL),
    ShortcutBinding::alias(KeyCode::KeyZ, NONE),
];
const REDO: &[ShortcutBinding] = &[
    ShortcutBinding::displayed(KeyCode::KeyY, CTRL),
    ShortcutBinding::alias(KeyCode::KeyZ, CTRL_SHIFT),
    ShortcutBinding::alias(KeyCode::KeyZ, SHIFT),
    ShortcutBinding::alias(KeyCode::KeyY, NONE),
];
const NEW: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::KeyN, CTRL)];
const OPEN: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::KeyO, CTRL)];
const SAVE: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::KeyS, CTRL)];
const SAVE_AS: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::KeyS, CTRL_SHIFT)];
const CLOSE: &[ShortcutBinding] = &[ShortcutBinding::displayed(KeyCode::KeyW, CTRL)];
fn digit_bindings(digit: u8) -> &'static [ShortcutBinding] {
    match digit {
        1 => D1,
        2 => D2,
        3 => D3,
        4 => D4,
        5 => D5,
        6 => D6,
        7 => D7,
        8 => D8,
        _ => &[],
    }
}

const LEGACY_WORKSPACE_COMMAND_ALIASES: &[(&str, WorkspaceId)] = &[
    ("mode.sculpt", WorkspaceId::Sculpt),
    ("mode.generate", WorkspaceId::Sculpt),
    ("mode.erosion", WorkspaceId::Simulation),
    ("mode.masks", WorkspaceId::Rules),
    ("mode.paint", WorkspaceId::Surface),
    ("mode.biomes", WorkspaceId::Biomes),
    ("mode.scatter", WorkspaceId::Objects),
    ("workspace.all_tools", WorkspaceId::AllTools),
];

/// Resolve canonical and legacy workspace command ids without exposing aliases
/// in the visible command list.
pub(crate) fn resolve_workspace_command(id: &str) -> Option<WorkspaceId> {
    all_workspace_definitions()
        .iter()
        .find(|definition| definition.id.command_id() == Some(id))
        .map(|definition| definition.id)
        .or_else(|| {
            LEGACY_WORKSPACE_COMMAND_ALIASES
                .iter()
                .find_map(|(alias, workspace)| (*alias == id).then_some(*workspace))
        })
}

pub fn resolve_shortcut(chord: ShortcutChord) -> Option<&'static str> {
    commands()
        .into_iter()
        .find_map(|c| c.bindings.iter().any(|b| b.chord == chord).then_some(c.id))
}
pub fn resolve_shortcut_for_input(chord: ShortcutChord, text_input: bool) -> Option<&'static str> {
    if text_input && chord.is_unmodified_text_key() {
        None
    } else {
        resolve_shortcut(chord)
    }
}
pub fn format_shortcuts(bindings: &[ShortcutBinding]) -> String {
    bindings
        .iter()
        .filter(|b| b.visibility == BindingVisibility::Displayed)
        .map(|b| format_chord(b.chord))
        .collect::<Vec<_>>()
        .join(" / ")
}
fn format_chord(chord: ShortcutChord) -> String {
    let mut p = Vec::new();
    if chord.modifiers.ctrl {
        p.push("Ctrl");
    }
    if chord.modifiers.alt {
        p.push("Alt");
    }
    if chord.modifiers.shift {
        p.push("Shift");
    }
    if chord.modifiers.super_key {
        p.push("Super");
    }
    p.push(match chord.key {
        KeyCode::Digit1 => "1",
        KeyCode::Digit2 => "2",
        KeyCode::Digit3 => "3",
        KeyCode::Digit4 => "4",
        KeyCode::Digit5 => "5",
        KeyCode::Digit6 => "6",
        KeyCode::Digit7 => "7",
        KeyCode::Digit8 => "8",
        KeyCode::KeyF => "F",
        KeyCode::KeyL => "L",
        KeyCode::KeyN => "N",
        KeyCode::KeyO => "O",
        KeyCode::KeyP => "P",
        KeyCode::KeyS => "S",
        KeyCode::KeyW => "W",
        KeyCode::KeyY => "Y",
        KeyCode::KeyZ => "Z",
        KeyCode::Insert => "Insert",
        _ => "Unknown",
    });
    p.join("+")
}

/// Built-in commands presented by the command palette.
pub fn commands() -> Vec<CommandDef> {
    use CommandCategory::*;
    let mut out = vec![
        CommandDef {
            id: CommandId::ADD_MOUNTAIN,
            name: "Add Mountain",
            category: Layer,
            bindings: &[],
            icon: Some(Icon::Mountain),
        },
        CommandDef {
            id: CommandId::ADD_HYDRAULIC_EROSION,
            name: "Add Hydraulic Erosion",
            category: Layer,
            bindings: &[],
            icon: Some(Icon::Droplets),
        },
    ];

    for definition in all_workspace_definitions() {
        let (Some(id), Some(name)) = (definition.id.command_id(), definition.command_name) else {
            continue;
        };
        out.push(CommandDef {
            id,
            name,
            category: Mode,
            bindings: definition
                .id
                .digit_shortcut()
                .map(digit_bindings)
                .unwrap_or(&[]),
            icon: Some(definition.icon),
        });
    }

    out.extend([
        CommandDef {
            id: CommandId::FRAME_TERRAIN,
            name: "Frame Terrain",
            category: Viewport,
            bindings: FRAME,
            icon: Some(Icon::Camera),
        },
        CommandDef {
            id: CommandId::TOP_VIEW,
            name: "Top View",
            category: Viewport,
            bindings: &[],
            icon: Some(Icon::Camera),
        },
        CommandDef {
            id: CommandId::FRAME_SELECTION,
            name: "Frame Selection",
            category: Viewport,
            bindings: &[],
            icon: Some(Icon::Camera),
        },
        CommandDef {
            id: CommandId::TOGGLE_HEIGHT_VIEW,
            name: "Toggle Height View",
            category: View,
            bindings: &[],
            icon: Some(Icon::Image),
        },
        CommandDef {
            id: CommandId::TOGGLE_WIREFRAME,
            name: "Toggle Wireframe",
            category: Viewport,
            bindings: &[],
            icon: Some(Icon::Grid3x3),
        },
        CommandDef {
            id: CommandId::OPEN_QUICK_ADD,
            name: "Open Quick Add",
            category: Tools,
            bindings: QUICK_ADD,
            icon: Some(Icon::Plus),
        },
        CommandDef {
            id: CommandId::OPEN_COMMAND_PALETTE,
            name: "Open Command Palette",
            category: Tools,
            bindings: PALETTE,
            icon: Some(Icon::ScrollText),
        },
        CommandDef {
            id: CommandId::UNDO,
            name: "Undo",
            category: Edit,
            bindings: UNDO,
            icon: Some(Icon::Undo2),
        },
        CommandDef {
            id: CommandId::REDO,
            name: "Redo",
            category: Edit,
            bindings: REDO,
            icon: Some(Icon::Redo2),
        },
        CommandDef {
            id: CommandId::NEW_PROJECT,
            name: "New Project",
            category: File,
            bindings: NEW,
            icon: Some(Icon::Plus),
        },
        CommandDef {
            id: CommandId::OPEN_PROJECT,
            name: "Open Project",
            category: File,
            bindings: OPEN,
            icon: Some(Icon::FolderOpen),
        },
        CommandDef {
            id: CommandId::SAVE,
            name: "Save Project",
            category: File,
            bindings: SAVE,
            icon: Some(Icon::Save),
        },
        CommandDef {
            id: CommandId::SAVE_AS,
            name: "Save Project As",
            category: File,
            bindings: SAVE_AS,
            icon: Some(Icon::Save),
        },
        CommandDef {
            id: CommandId::CLOSE_PROJECT,
            name: "Close Project",
            category: File,
            bindings: CLOSE,
            icon: Some(Icon::X),
        },
        CommandDef {
            id: CommandId::EXPORT,
            name: "Export",
            category: File,
            bindings: &[],
            icon: Some(Icon::Download),
        },
        CommandDef {
            id: CommandId::BAKE_SELECTED,
            name: "Bake Selected",
            category: Layer,
            bindings: &[],
            icon: Some(Icon::Layers),
        },
        CommandDef {
            id: CommandId::OPEN_PROFILER,
            name: "Open Profiler",
            category: View,
            bindings: &[],
            icon: Some(Icon::Gauge),
        },
        CommandDef {
            id: CommandId::TOGGLE_HISTORY,
            name: "Toggle History",
            category: View,
            bindings: &[],
            icon: Some(Icon::ScrollText),
        },
        CommandDef {
            id: CommandId::TOGGLE_PIPELINE,
            name: "Toggle Terrain Recipe",
            category: View,
            bindings: &[],
            icon: Some(Icon::Activity),
        },
        CommandDef {
            id: CommandId::CLEAR_GEOMORPH_DEBUG,
            name: "Clear Geomorph Debug Overlay",
            category: Debug,
            bindings: &[],
            icon: Some(Icon::EyeOff),
        },
    ]);

    for field in terra_core::GeomorphDebugField::ALL {
        out.push(CommandDef {
            id: field.command_id(),
            name: field.label(),
            category: Debug,
            bindings: &[],
            icon: Some(Icon::Activity),
        });
    }
    out
}

/// Case-insensitive substring or ordered-subsequence matching.
pub fn fuzzy_match(query: &str, text: &str) -> bool {
    let query = query.trim().to_lowercase();
    let text = text.to_lowercase();
    if query.is_empty() || text.contains(&query) {
        return true;
    }
    let mut chars = text.chars();
    query
        .chars()
        .all(|needle| chars.by_ref().any(|hay| hay == needle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn fuzzy_match_handles_subsequences_and_case() {
        assert!(fuzzy_match("hyd ero", "Add Hydraulic Erosion"));
        assert!(fuzzy_match("MOUN", "Add Mountain"));
        assert!(!fuzzy_match("xyz", "Add Mountain"));
    }

    #[test]
    fn command_list_is_populated() {
        assert!(!commands().is_empty());
    }

    #[test]
    fn command_ids_and_executable_shortcuts_are_unique() {
        let commands = commands();
        let mut ids = HashSet::new();
        let mut shortcuts = HashSet::new();

        for command in &commands {
            assert!(
                ids.insert(command.id),
                "duplicate command id: {}",
                command.id
            );
            for binding in command.bindings {
                assert!(
                    shortcuts.insert(binding.chord),
                    "duplicate executable shortcut: {:?}",
                    binding.chord
                );
            }
        }
    }

    #[test]
    fn required_shortcuts_are_executable_and_display_is_derived() {
        let c = |key, modifiers| ShortcutChord::new(key, modifiers);
        assert_eq!(
            resolve_shortcut(c(KeyCode::KeyF, NONE)),
            Some(CommandId::FRAME_TERRAIN)
        );
        assert_eq!(
            resolve_shortcut(c(KeyCode::KeyP, CTRL)),
            Some(CommandId::OPEN_COMMAND_PALETTE)
        );
        assert_eq!(
            resolve_shortcut(c(KeyCode::KeyL, CTRL)),
            Some(CommandId::OPEN_QUICK_ADD)
        );
        assert_eq!(
            resolve_shortcut(c(KeyCode::Insert, NONE)),
            Some(CommandId::OPEN_QUICK_ADD)
        );
        assert_eq!(
            resolve_shortcut(c(KeyCode::KeyS, CTRL)),
            Some(CommandId::SAVE)
        );
        assert_eq!(
            resolve_shortcut(c(KeyCode::KeyS, CTRL_SHIFT)),
            Some(CommandId::SAVE_AS)
        );
        assert_eq!(
            resolve_shortcut(c(KeyCode::KeyZ, CTRL)),
            Some(CommandId::UNDO)
        );
        assert_eq!(
            resolve_shortcut(c(KeyCode::KeyZ, NONE)),
            Some(CommandId::UNDO)
        );
        for redo in [
            c(KeyCode::KeyY, CTRL),
            c(KeyCode::KeyZ, CTRL_SHIFT),
            c(KeyCode::KeyZ, SHIFT),
            c(KeyCode::KeyY, NONE),
        ] {
            assert_eq!(resolve_shortcut(redo), Some(CommandId::REDO));
        }
        for command in commands() {
            for binding in command
                .bindings
                .iter()
                .filter(|b| b.visibility == BindingVisibility::Displayed)
            {
                assert_eq!(resolve_shortcut(binding.chord), Some(command.id));
            }
            assert!(!format_shortcuts(command.bindings).contains("Unknown"));
        }
        let quick = commands()
            .into_iter()
            .find(|c| c.id == CommandId::OPEN_QUICK_ADD)
            .unwrap();
        assert_eq!(format_shortcuts(quick.bindings), "Ctrl+L / Insert");
    }

    #[test]
    fn modifiers_and_text_input_are_respected() {
        let ctrl_alt = ShortcutModifiers {
            ctrl: true,
            alt: true,
            ..NONE
        };
        assert_eq!(
            resolve_shortcut(ShortcutChord::new(KeyCode::KeyS, ctrl_alt)),
            None
        );
        assert_eq!(
            resolve_shortcut(ShortcutChord::new(KeyCode::Digit1, CTRL)),
            None
        );
        assert_eq!(
            resolve_shortcut_for_input(ShortcutChord::new(KeyCode::KeyF, NONE), true),
            None
        );
        assert_eq!(
            resolve_shortcut_for_input(ShortcutChord::new(KeyCode::Digit1, NONE), true),
            None
        );
        assert_eq!(
            resolve_shortcut_for_input(ShortcutChord::new(KeyCode::KeyP, CTRL), true),
            Some(CommandId::OPEN_COMMAND_PALETTE)
        );
        assert_eq!(
            resolve_shortcut_for_input(ShortcutChord::new(KeyCode::Insert, NONE), true),
            Some(CommandId::OPEN_QUICK_ADD)
        );
    }

    #[test]
    fn visible_workspace_commands_come_from_workspace_metadata() {
        let commands = commands();
        let visible_definitions: Vec<_> = all_workspace_definitions()
            .iter()
            .filter(|definition| definition.id.command_id().is_some())
            .collect();

        for definition in &visible_definitions {
            let id = definition.id.command_id().unwrap();
            let command = commands
                .iter()
                .find(|command| command.id == id)
                .unwrap_or_else(|| panic!("missing workspace command: {id}"));

            assert_eq!(command.name, definition.command_name.unwrap());
            assert_eq!(command.icon, Some(definition.icon));
            let digit = definition.id.digit_shortcut().unwrap();
            assert_eq!(command.bindings, digit_bindings(digit));
            assert_eq!(resolve_shortcut(command.bindings[0].chord), Some(id));
            assert_eq!(command.category, CommandCategory::Mode);
        }

        let mut targets = HashSet::new();
        let visible_workspace_commands: Vec<_> = commands
            .iter()
            .filter_map(|command| {
                resolve_workspace_command(command.id).map(|workspace| (command, workspace))
            })
            .collect();
        for (command, workspace) in &visible_workspace_commands {
            assert!(
                targets.insert(*workspace),
                "duplicate visible workspace target {:?} from command {}",
                workspace,
                command.id
            );
        }

        assert_eq!(visible_workspace_commands.len(), visible_definitions.len());
        assert_eq!(targets.len(), 8);
    }

    #[test]
    fn legacy_workspace_aliases_resolve_but_are_not_visible() {
        let commands = commands();

        for &(alias, expected) in LEGACY_WORKSPACE_COMMAND_ALIASES {
            assert_eq!(resolve_workspace_command(alias), Some(expected));
            assert!(
                commands.iter().all(|command| command.id != alias),
                "legacy alias is visible: {alias}"
            );
        }

        assert_eq!(
            resolve_workspace_command("workspace.all_tools"),
            Some(WorkspaceId::AllTools)
        );
        assert!(WorkspaceId::AllTools.command_id().is_none());
    }
}
