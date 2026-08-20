# Terra Architecture

This is the authoritative description of Terra's current workspace boundaries and
editor frame composition. Historical migration notes may explain how the current
design was reached, but they do not override this document.

## Workspace crates

| Crate | Responsibility |
|-------|----------------|
| `terra-core` | Domain model: heightfields, layer stack, masks, biomes, CPU evaluation, and editor commands |
| `terra-jobs` | Cancellation primitive (`CancelToken`) and cancellable parallel-fill helpers shared by CPU eval; a leaf crate below `terra-core` |
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

**Internal:** the authored `LayerStack` is projected into a backend-neutral
`CompiledTerrainPlan`. The plan describes ordered operations, logical height/mask/aux
fields, spatial reach, and stable authored provenance without owning layer payloads,
CPU heightfields, or GPU resources. CPU and GPU backends realize that plan into their
own physical resources and execution work. During the staged migration, the existing
CPU evaluator and flat GPU planner remain active until their plan consumers land.

The ownership boundary is deliberate:

- `TerrainDocument` owns authored layers, masks, Base samples, and stroke history.
- `CompiledTerrainPlan` owns runtime-derived semantic descriptors and provenance.
- CPU/GPU evaluators own physical fields, caches, pipelines, and textures.
- Per-frame scheduling owns transient execution state rather than authored identity.

Plan-local operation and field IDs are valid only for one structural revision. Stable
cross-plan identity comes from authored layer/group/output IDs. Content edits such as
brush dabs advance output freshness and dirty regions without changing the plan's
structural revision.

Before a plan becomes executable, validation builds immutable def-use metadata,
checks field kinds and production order, rejects dependency cycles, and derives
operation/field liveness plus logical field lifetimes. Provenance is bidirectional:
authored layers and groups map to their operation spans and fields, named outputs map
to fields and publisher operations, and operations map back to authored owners.

`TerrainPlanCache` owns the authored structural revision and the last successfully
compiled plan. Add/remove/reorder, enable/solo, dependency-placement, and operation
shape changes advance that revision once per command batch. Parameter edits, Base or
stroke content, resolution changes, backend-resource preferences, and view-only edits
reuse the compatible structure. A failed candidate leaves the previous backend output
available for presentation, but the stale plan is rejected at the execution gate.

Plan invalidation starts from stable authored provenance and walks the live def-use
subgraph in operation order. Local scopes accumulate `Reach` halos; observed global
auxiliary inputs use `AuxReach` to escalate only at the first operation that requires
a full field. Unobserved auxiliary branches therefore do not globalize an otherwise
height-only suffix. Cache counters expose compiles, hits, patched/reached operations,
and full-field escalation reasons for tests and profiling.

## Evaluation

```
H_0 = 0
for layer L in bottom→top:
  if disabled: continue
  G = processor(L, H_{i-1})
  M = composite masks(L)
  H_i = mix(H_{i-1}, blend(H_{i-1}, G), opacity * M)
```

The target evaluation boundary is:

```text
LayerStack (authored source)
    -> CompiledTerrainPlan (logical dataflow + provenance)
    -> CPU/GPU backend realization
    -> final height and auxiliary outputs
```

The plan IR and recursive `LayerStack` compiler are present. The compiler preserves
bottom-to-top order, pass-through folders, private `CopyInput` / `EmptyHeight` group
fields, group composites, point-of-use mask operations, explicit auxiliary merges,
and authored provenance. Solo filtering and selected/cross-tree fields remain explicit
compile diagnostics until their dedicated compiler phases land. Production GPU
realization and consumption land in subsequent phases. Until then, incremental CPU
rebuild continues to use `LayerCache` + `mark_dirty_from`, and the existing GPU planner
continues to serve flat stacks.
Progressive preview walks `Draft → Medium → Full`: the app advances the quality ladder
held on `EvalScheduler` and runs each authoritative CPU pass on the background
`EvalWorker`.

## Tiles & ghosts

Default tile 256² with halo 2. Halos are refreshed from neighbors before stencil reads. Phase 9 tile scheduler processes dirty tiles + neighbors to avoid seams.

## Undo

`EditorCommand` records stack/parameter deltas only — never full height textures.
