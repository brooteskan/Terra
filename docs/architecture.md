# Terra Architecture

This is the authoritative description of Terra's current workspace boundaries and
editor frame composition. Historical migration notes may explain how the current
design was reached, but they do not override this document.

## Workspace crates

| Crate | Responsibility |
|-------|----------------|
| `terra-core` | Domain model: heightfields, layer stack, masks, biomes, CPU evaluation, and editor commands |
| `terra-gpu` | GPU compute for supported terrain generators, filters, and simulations |
| `terra-render` | wgpu terrain viewport, clipmaps, camera, lighting, and terrain render pass |
| `terra-gui` | Reusable, domain-neutral immediate-mode wgpu UI toolkit and design system |
| `terra-io` | Project JSON, import/export, GeoTIFF bridge, and export packages |
| `terra-app` | Application shell: winit event loop, editor panels and tools, `PanelAction` dispatch, and renderer integration |
| `terra-test-gpu` | Non-published headless GPU harness used by render and UI tests |

`terra-core` must stay free of `wgpu` and UI crates. `terra-gui` must stay free
of `terra-core` and other domain types. `terra-render` and `terra-gui` do not
depend on one another; `terra-app` owns both and integrates them.

## Editor action boundary

Editor panels and tools live in `terra-app` and are drawn with `terra-gui`. UI
code may observe `TerrainDocument`, but document changes are emitted as
`PanelAction` values in `FrameUiOutput`. The app shell dispatches those actions
through `TerraApp::apply_actions`, whose domain handlers own mutation, dirty
propagation, rebuild scheduling, history, and other application side effects.

This keeps `terra-gui` reusable and prevents the editor presentation layer from
becoming a second mutation path or catalog of domain truth.

## Frame composition

`terra-app` composes terrain and UI into one swapchain frame:

1. `TerrainRenderer::render_terrain` acquires the surface texture, submits the
   terrain pass, and returns the frame without presenting it.
2. `terra-app` creates a view of that same surface texture and builds the editor
   UI for the frame.
3. `terra_gui::GuiRenderer::render` submits one UI pass using `LoadOp::Load`, so
   the UI overlays rather than clears the terrain result.
4. `terra-app` presents the surface texture once, after both submissions.

The frame-seam tests in `terra-gui` and `terra-app` guard the load and
composition behavior.

## User model vs internal model

**User-facing:** ordered layer stack (World Creator style). Groups nest; no node editor.

**Internal:** each layer kind is evaluated by `ProcessorRegistry::evaluate` (a `match` on `LayerKind`), producing a height contribution blended onto the accumulator. Masks and aux maps (erosion, flow, wetness) are cacheable artifacts. Dirty propagation invalidates from the earliest edited layer upward.

## Evaluation

```
H_0 = 0
for layer L in bottom→top:
  if disabled: continue
  G = processor(L, H_{i-1})
  M = composite masks(L)
  H_i = mix(H_{i-1}, blend(H_{i-1}, G), opacity * M)
```

Phase 1 used full rebuilds. Incremental rebuild uses `LayerCache` + `mark_dirty_from`. Progressive preview walks `Draft → Medium → Full` via `EvalScheduler`.

## Tiles & ghosts

Default tile 256² with halo 2. Halos are refreshed from neighbors before stencil reads. Phase 9 tile scheduler processes dirty tiles + neighbors to avoid seams.

## Undo

`EditorCommand` records stack/parameter deltas only — never full height textures.
