//! Viewport / hierarchy contextual Create-here menus.

use crate::ui::style::{self, FONT_SCALE, TYPE_BODY, TYPE_LABEL};
use terra_core::contextual_create::{
    available_kinds, propose_owner, CreateContext, CreateKind, CreateOwner, CreateWorkspace,
};
use terra_core::document::TerrainDocument;
use terra_gui::{GuiContext, Id, Rect};

use crate::ui::actions::PanelAction;
use crate::ui::{UiState, WorkspaceId};

/// Open viewport right-click create menu.
#[derive(Debug, Clone)]
pub struct ViewportContextMenu {
    pub x: f32,
    pub y: f32,
    pub uv: Option<(f32, f32)>,
    /// Forced owner when opened from hierarchy with a known scope.
    pub locked_owner: Option<CreateOwner>,
    /// When set, show owner picker for this kind instead of the kind list.
    pub picking_owner_for: Option<CreateKind>,
    pub owner_override: Option<CreateOwner>,
}

impl Default for ViewportContextMenu {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            uv: None,
            locked_owner: None,
            picking_owner_for: None,
            owner_override: None,
        }
    }
}

pub fn workspace_to_create(id: WorkspaceId) -> CreateWorkspace {
    match id {
        WorkspaceId::World => CreateWorkspace::World,
        WorkspaceId::Sculpt => CreateWorkspace::Sculpt,
        WorkspaceId::Biomes => CreateWorkspace::Biomes,
        WorkspaceId::Develop => CreateWorkspace::Develop,
        WorkspaceId::Rules => CreateWorkspace::Rules,
        WorkspaceId::Simulation => CreateWorkspace::Simulation,
        WorkspaceId::Surface => CreateWorkspace::Surface,
        WorkspaceId::Objects => CreateWorkspace::Objects,
        WorkspaceId::AllTools => CreateWorkspace::AllTools,
    }
}

pub fn create_to_workspace(id: CreateWorkspace) -> WorkspaceId {
    match id {
        CreateWorkspace::World => WorkspaceId::World,
        CreateWorkspace::Sculpt => WorkspaceId::Sculpt,
        CreateWorkspace::Biomes => WorkspaceId::Biomes,
        CreateWorkspace::Develop => WorkspaceId::Develop,
        CreateWorkspace::Rules => WorkspaceId::Rules,
        CreateWorkspace::Simulation => WorkspaceId::Simulation,
        CreateWorkspace::Surface => WorkspaceId::Surface,
        CreateWorkspace::Objects => WorkspaceId::Objects,
        CreateWorkspace::AllTools => WorkspaceId::AllTools,
    }
}

pub fn build_create_context(
    doc: &TerrainDocument,
    ui: &UiState,
    menu: &ViewportContextMenu,
) -> CreateContext {
    CreateContext::from_document(
        doc,
        workspace_to_create(ui.active_workspace),
        ui.auto_switch_workspace_on_create,
    )
    .with_cursor(menu.uv)
}

/// Draw the viewport / hierarchy Create-here overlay. Returns actions to apply.
pub fn draw_viewport_context_menu(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
) -> Vec<PanelAction> {
    let Some(menu) = ui_state.viewport_context_menu.clone() else {
        return Vec::new();
    };
    let mut actions = Vec::new();
    let ctx = build_create_context(doc, ui_state, &menu);

    ui.with_menu_input(|ui| {
        if let Some(kind) = menu.picking_owner_for {
            draw_owner_picker(ui, doc, ui_state, &menu, &ctx, kind, &mut actions);
        } else {
            draw_kind_list(ui, doc, ui_state, &menu, &ctx, &mut actions);
        }
    });
    actions
}

fn draw_kind_list(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    menu: &ViewportContextMenu,
    ctx: &CreateContext,
    actions: &mut Vec<PanelAction>,
) {
    let kinds = available_kinds(ctx);
    let row_h = 22.0;
    let header_h = 28.0;
    let owner_h = 36.0;
    let w = 200.0;
    let h = header_h + owner_h + kinds.len() as f32 * row_h + 12.0;
    let menu_rect = Rect::from_pos_size(
        menu.x.min(ui.screen_w - w - 4.0).max(4.0),
        menu.y.min(ui.screen_h - h - 4.0).max(4.0),
        w,
        h,
    );

    ui.begin_overlay();
    ui.panel_rounded(menu_rect, style::POPUP_BG, style::RADIUS_SM);
    ui.state.set_hot(Id::new("__viewport_create_ctx"));

    let mut iy = menu_rect.min_y + 6.0;
    ui.label_at(
        menu_rect.min_x + 10.0,
        iy,
        "Create here",
        style::TEXT_DIM,
        FONT_SCALE * TYPE_LABEL,
    );
    iy += header_h - 6.0;

    // Preview owner for the hovered kind (falls back to first available).
    let mut preview_kind = kinds.first().copied().unwrap_or(CreateKind::SculptLayer);
    let owner_y = iy;
    iy += owner_h - 10.0;

    let mut hovered_kind: Option<CreateKind> = None;
    let list_start_y = iy;
    for (i, kind) in kinds.iter().enumerate() {
        let row = Rect::from_pos_size(
            menu_rect.min_x + 4.0,
            list_start_y + i as f32 * row_h,
            w - 8.0,
            row_h - 2.0,
        );
        if ui.pointer_in(row) {
            hovered_kind = Some(*kind);
        }
    }
    if let Some(k) = hovered_kind {
        preview_kind = k;
    }
    let proposal = if let Some(o) = menu.locked_owner.or(menu.owner_override) {
        terra_core::contextual_create::override_owner(doc, preview_kind, o)
            .unwrap_or_else(|_| propose_owner(doc, preview_kind, ctx))
    } else {
        propose_owner(doc, preview_kind, ctx)
    };
    ui.label_at(
        menu_rect.min_x + 10.0,
        owner_y,
        &format!("Create {} in: {}", preview_kind.label(), proposal.label),
        style::TEXT,
        FONT_SCALE * TYPE_LABEL,
    );

    for kind in &kinds {
        let row = Rect::from_pos_size(menu_rect.min_x + 4.0, iy, w - 8.0, row_h - 2.0);
        let hovered = ui.pointer_in(row);
        if hovered {
            ui.panel_rounded(row, style::ROW_HOVER, style::RADIUS_SM);
        }
        ui.label_at(
            row.min_x + 8.0,
            row.min_y + 4.0,
            kind.label(),
            style::TEXT,
            FONT_SCALE * TYPE_BODY,
        );
        if hovered && ui.input.primary_released {
            let prop = if let Some(o) = menu.locked_owner.or(menu.owner_override) {
                terra_core::contextual_create::override_owner(doc, *kind, o)
                    .unwrap_or_else(|_| propose_owner(doc, *kind, ctx))
            } else {
                propose_owner(doc, *kind, ctx)
            };
            if prop.ambiguous || !prop.owner.is_resolved() {
                if let Some(m) = ui_state.viewport_context_menu.as_mut() {
                    m.picking_owner_for = Some(*kind);
                }
            } else {
                actions.push(PanelAction::ContextualCreate {
                    kind: *kind,
                    owner: Some(prop.owner),
                    uv: menu.uv,
                });
                ui_state.viewport_context_menu = None;
            }
        }
        iy += row_h;
    }

    if (ui.input.primary_pressed || ui.input.secondary_pressed) && !ui.pointer_in(menu_rect) {
        ui_state.viewport_context_menu = None;
    }
    ui.end_overlay();
}

fn draw_owner_picker(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    menu: &ViewportContextMenu,
    ctx: &CreateContext,
    kind: CreateKind,
    actions: &mut Vec<PanelAction>,
) {
    let proposal = propose_owner(doc, kind, ctx);
    let alts = &proposal.alternatives;
    let row_h = 22.0;
    let w = 220.0;
    let h = 48.0 + alts.len() as f32 * row_h + 8.0;
    let menu_rect = Rect::from_pos_size(
        menu.x.min(ui.screen_w - w - 4.0).max(4.0),
        menu.y.min(ui.screen_h - h - 4.0).max(4.0),
        w,
        h,
    );

    ui.begin_overlay();
    ui.panel_rounded(menu_rect, style::POPUP_BG, style::RADIUS_SM);
    ui.state.set_hot(Id::new("__viewport_create_owner"));

    let mut iy = menu_rect.min_y + 6.0;
    ui.label_at(
        menu_rect.min_x + 10.0,
        iy,
        &format!("Create {} in:", kind.label()),
        style::TEXT_DIM,
        FONT_SCALE * TYPE_LABEL,
    );
    iy += 24.0;

    for (owner, label) in alts {
        if matches!(owner, CreateOwner::Unspecified) {
            continue;
        }
        let row = Rect::from_pos_size(menu_rect.min_x + 4.0, iy, w - 8.0, row_h - 2.0);
        let hovered = ui.pointer_in(row);
        if hovered {
            ui.panel_rounded(row, style::ROW_HOVER, style::RADIUS_SM);
        }
        ui.label_at(
            row.min_x + 8.0,
            row.min_y + 4.0,
            label,
            style::TEXT,
            FONT_SCALE * TYPE_BODY,
        );
        if hovered && ui.input.primary_released {
            actions.push(PanelAction::ContextualCreate {
                kind,
                owner: Some(*owner),
                uv: menu.uv,
            });
            ui_state.viewport_context_menu = None;
        }
        iy += row_h;
    }

    if (ui.input.primary_pressed || ui.input.secondary_pressed) && !ui.pointer_in(menu_rect) {
        ui_state.viewport_context_menu = None;
    }
    ui.end_overlay();
}

/// Hierarchy Add submenu items for a known owner (Region / Biome / Global).
pub fn hierarchy_add_actions(
    doc: &TerrainDocument,
    ui_state: &UiState,
    owner: CreateOwner,
) -> Vec<(CreateKind, String)> {
    let mut ctx = CreateContext::from_document(
        doc,
        workspace_to_create(ui_state.active_workspace),
        ui_state.auto_switch_workspace_on_create,
    );
    match owner {
        CreateOwner::Biome(id) => {
            ctx.active_biome = Some(id);
        }
        _ => {}
    }
    available_kinds(&ctx)
        .into_iter()
        .filter(|k| terra_core::contextual_create::owner_compatible(*k, owner))
        .map(|k| {
            let prop = terra_core::contextual_create::override_owner(doc, k, owner)
                .unwrap_or_else(|_| propose_owner(doc, k, &ctx));
            (k, format!("{} â†’ {}", k.label(), prop.label))
        })
        .collect()
}
