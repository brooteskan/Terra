# Terra Architecture

This is the authoritative description of Terra's current workspace boundaries and
editor frame composition. Historical migration notes may explain how the current
design was reached, but they do not override this document.

## Workspace crates

| Crate | Responsibility |
|-------|----------------|
| `terra-world` | Fixed-origin coordinates, signed tile addresses, LODs, sample/world transforms, and bounded/infinite topology contracts |
| `terra-core` | Backend-neutral domain model: heightfields, layer stack, masks, biomes, and editor commands |
| `terra-cpu-eval` | Stateful CPU terrain evaluation, caches, scheduling, workers, timing, and output lifecycle |
| `terra-jobs` | Cancellation primitive (`CancelToken`) and cancellable parallel-fill helpers shared by core algorithms and CPU evaluation |
| `terra-gpu` | Reusable GPU kernels, capability descriptions, compiled-plan resources, derivatives, and tile caching |
| `terra-gpu-eval` | Stateful GPU terrain evaluation, refinement jobs, timing, submission, and output lifecycle |
| `terra-render` | wgpu terrain viewport, clipmaps, camera, lighting, and terrain render pass |
| `terra-gui` | Reusable, domain-neutral immediate-mode wgpu UI toolkit and design system |
| `terra-io` | Project JSON, import/export, GeoTIFF bridge, and export packages |
| `terra-app` | Application shell: winit event loop, editor panels and tools, `PanelAction` dispatch, and renderer integration |
| `terra-test-gpu` | Non-published headless GPU harness used by render and UI tests |

`terra-world` depends only on backend-independent serialization support and must
stay free of domain content, evaluators, GPU, renderer, IO, and app crates.
`terra-core` must stay free of evaluator, `wgpu`, and UI crates. `terra-cpu-eval`
depends on `terra-core`, never the reverse. `terra-gpu-eval` depends on
`terra-gpu`, never the reverse. `terra-gui` must stay free
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

Terra's editor loop is event-driven. OS input is accumulated promptly and is
sealed into an immutable snapshot at the next `about_to_wait` scheduling
boundary. A logical frame then advances through application/tool update,
required interactive work, a presentation request, and optional refinement.
Logical frames are demand-driven scheduling and diagnostic lifecycles, not a
fixed-timestep simulation and not a promise of a particular refresh rate.

`LogicalFrameCoordinator` is the single owner of frame demand, generation-
stamped evaluation/refinement deadlines, presentation demand, and the next
`Wait`/`WaitUntil`/`Poll` decision. Window callbacks only capture bounded state:
ordered input, the latest coalesced resize, lifecycle notifications, and UI
output. UI output is applied during the next logical frame, before required
terrain work. Surface preparation and renderer uploads also occur before the
presentation request; `RedrawRequested` only composes and presents the already
prepared terrain and UI.

Logical frame IDs and terrain edit generations are intentionally distinct.
Several logical frames may observe the same generation; an edit may advance the
generation during application update and supersede work submitted by an older
frame. Ordered pointer samples and button/modifier transitions are never
coalesced. Input recorded after a snapshot is sealed is retained for a new
logical frame.

Required Draft work has priority over optional Medium/Full refinement. A frame
budget hook decides only whether optional work may start; it does not cancel or
preempt GPU commands after submission. With no input, animation, background
completion, or scheduled terrain work, the winit loop returns to
`ControlFlow::Wait`.

Resize and recoverable surface errors become coalesced logical-frame work.
Focus/capture loss appends cancellation to the ordered input stream. Device loss
and shutdown invalidate CPU and GPU work, abandon the active frame, clear queued
presentation and application work, and exit through the event-loop boundary.
Every detailed trace phase must carry a real logical-frame identity; rejected
orphan trace attempts are counted in the profiler diagnostics.

The surface presentation path remains last-complete: pending or stale-generation
terrain candidates do not replace the renderer's current complete textures.
The swapchain composition inside a requested presentation is:

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
own physical resources and execution work. The GPU backend executes this plan as its
ordering, field-wiring, group, and invalidation authority; the legacy flat graph is
retained only as a compatibility diagnostic and layer-kernel capability adapter.

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
to fields and publisher operations, and operations map back to authored owners. The
plan also records every authored node's solo selection state, including ancestor paths
and excluded nodes that intentionally own no operations, so diagnostics remain tied to
stable authoring identities.

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

`terra-gpu::compiled_plan` realizes the validated logical fields as R32Float wgpu
textures. Final, named-output, root, reusable layer candidates, and composite/aux
checkpoints are persistent; masks and isolated-group private slices are transient.
Transient allocation uses the plan's inclusive operation lifetimes, so storage is
reused only when `previous.last_operation < next.first_operation`. Inputs and outputs
used by the same operation therefore never alias, nested private groups remain
isolated, and disjoint sibling groups can reuse compatible backing storage.

Backend materialization is distinct from semantic dirtiness: when a selected dirty
operation reads transient scratch, the GPU layout walks backward to reconstruct that
scratch and stops at persistent checkpoints. Resource realizations are keyed by plan
structure, extent, scalar format, and device generation. Resolution/resource changes
re-realize the compatible semantic plan rather than recompiling it. Execution may be
recorded into a staged resource set and committed only after successful validation and
submission; dropping a failed candidate leaves the active last-good textures intact.

The field-addressable backend implements zero/copy/selected seeds, supported Constant/
Height/Slope and named-output distributions, standard group blends, biome CopyInput
height-delta composition, and independently live masked auxiliary publication.
Unsupported distribution nodes, field-backed masks whose producer has no GPU auxiliary
publication, and dynamic parameter reductions fall back at their exact consumer.
`terra-gpu-eval::GpuTerrainEngine` accepts a revision-matching compiled plan and dispatches these
operations in plan order. Scoped groups no longer trigger a categorical tree fallback;
unsupported layer configurations or not-yet-resident auxiliary dependencies report the
precise plan operation and authored owner through provenance. Recording uses staged
resources so a stale plan or failed candidate cannot replace the last-good output.

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
fields, selected field/output seeds, retained named publications, point-of-use mask
operations, per-field auxiliary merges, parameter-binding edges, and authored
provenance. Solo is compiled as a sibling-level tree selection: when a
sibling list contains a solo descendant, only participating paths are lowered, while
ancestor containers remain present and disabled participants emit no work. Missing,
disabled/excluded, unavailable, duplicate, and cyclic cross-references are explicit
compile diagnostics. Incremental GPU work is selected from plan invalidation and persistent
field checkpoints; incremental CPU rebuild continues to use `LayerCache` +
`mark_dirty_from`.
Progressive preview walks `Draft → Medium → Full`. Required interactive Draft work
keeps the immediate evaluator path. Supported optional Medium/Full work is represented
by an app-owned, generation-stamped refinement job and advances after required work by
one existing safe boundary per logical frame: one resource allocation, one compiled
plan operation submission, or the final presentation copy. Unsupported work retains
the authoritative background `EvalWorker` fallback.

Refinement realizes an isolated candidate resource set. The scheduler checks generation
freshness before every unit and immediately before publication; supersession drops all
unsubmitted cursor work. Submitted GPU work cannot be cancelled, so the engine retains
its completion fence even after the owning job is abandoned. No later refinement unit
may submit until that fence resolves, bounding optional submission depth globally at
one while leaving required Draft submissions free to enter the queue. The last complete
renderer texture remains visible until the candidate's final fence resolves and the
current generation atomically commits resources, graph metadata, quality, and
presentation. Frame traces correlate job lifecycle, progress, submission depth,
supersession, completion, and publication with frame, generation, and evaluation IDs.

## Terrain residency and demand planning

The terrain atlas page-table system is the shader-visible residency authority. Bounded
worlds use a dense virtual directory and `TerrainPyramid`'s stable metadata index to map
level/tile coordinates to physical atlas slots in O(1). Infinite worlds use a
fixed-capacity open-addressed directory containing the full signed 64-bit X/Z address.
Each physical row validates slot generation, virtual coordinates, extent, halo,
publication frame, and the complete content identity. Resident-ancestor search is
bounded by topology depth rather than atlas capacity.
`terra-core::TileResidencyCache` is the one mutable CPU policy
mirror: it owns the byte budget, LRU and pin policy, virtual keys, and generation-checked
handles. `terra-gpu::GpuTileAtlas` owns that cache and translates its insertions,
evictions, explicit removals, and clears directly into both GPU table layers. Dense and
sparse directories are shader-visible projections, not second CPU residency databases.
The sparse directory is rebuilt transiently from the cache on mutation. Reported counts are
derived from this path and are tested against valid page-table rows.

`terra-world` owns the fixed-origin `f64` CPU coordinate system, signed `i64` tile
addresses, validated finest-first LODs, sample/world transforms, and both bounded and
infinite topology contracts. The infinite topology exposes only direct addressing and
finite rectangle queries: it has no total dimensions, root tile, dense metadata index,
or complete-world iterator. Content identity remains in `terra-core` and composes a
world tile address with layer and field identity.

`TerrainPyramid` is the bounded adapter over `terra-world::BoundedTopology` for a complete
ceil-halving resolution hierarchy. It preserves the legacy coarse-first bounded level
index, partial edge tiles, normalized-footprint parent/child coverage, and stable dense
metadata indices; it owns no resident tiles, page handles, or copy of the page table.
`terra-gpu::GpuHeightPyramid` materializes a
completed GPU result into immutable R32Float level textures and an output-identity-stamped
dense geometric-error buffer. Those errors describe content, not residency.

`TerrainDemandPlanner` walks that immutable hierarchy from coarse to fine using
conservative frustum culling and projected screen-space error. The compact error buffer
is transferred asynchronously once per accepted pyramid identity; height textures remain
GPU-resident. Planner state contains only prior refinement decisions for hysteresis, and
its deterministic, bounded output is consumed by the app's tile-upload seam. The planner
neither queries nor mutates residency. The consumer may skip exact current pages by
querying `GpuTileAtlas`'s authoritative cache before publication. See
[Terrain camera demand](algorithms/terrain_demand.md).

For Infinite projects the planner instead derives a finite signed window at the
configured coarsest LOD from camera centre and horizon. Mandatory coarse coverage is
inserted before visible SSE refinement, each refinement is closed over its Euclidean
ancestor chain, and node/tile budgets cap every plan. Fixed-origin bounds are translated
relative to the eye before `f32` frustum projection. Error policy is O(LOD), with room
for certified sparse tile envelopes; there is no dense global metadata array. Replanning
replaces hysteresis state rather than accumulating travelled addresses.

`TerrainTileWorkScheduler` is the bounded consumer between demand and atlas
publication. Its request key includes field/level/tile, compiled-plan revision, and
output revision; its full content stamp also carries document generation and immutable
content revision. Replanning reprioritizes retained requests without resetting age.
Required coarse/ancestor coverage gates optional refinement, and budgeted dequeue has
both age promotion and a one-item oversized escape. Scheduler records are demand and
lease state only: exact-current checks always query `GpuTileAtlas`, and publication
revalidates the full content stamp before cache or page-table mutation. See
[Terrain tile work scheduling](algorithms/terrain_work_scheduling.md).

Eligible height requests can now execute the compiled plan over an isolated
tile-plus-halo domain. The backend-neutral domain separates the atlas publication
halo from cumulative operation reach and retains the full-level integer sample
lattice for deterministic world-space sampling. Tile evaluators own and recycle
tile-sized plan/scratch resources independently of the complete-field engine.
No atlas slot is allocated until the GPU suffix completes and its scheduler lease
and complete content stamp are revalidated. `Reach::Full`, observed global
auxiliary dependencies, and kernels without a domain-coordinate contract defer
explicitly to the existing complete-field pyramid. See
[Compiled tile-domain GPU evaluation](algorithms/compiled_tile_evaluation.md).

Infinite projects add a stricter admission layer over the same compiled plan and
`Reach` vocabulary. Every live operation declares whether it can execute directly
over sparse tiles; the dependency walk accumulates finite halos and rejects full
reach, basin coupling, consumed global auxiliaries, bounded authored data, and
unclassified work with an authored-owner diagnostic. Unlike bounded execution,
Infinite admission never selects the complete-field checkpoint fallback. Catalog,
inspector, and scheduling use this shared result. See
[Infinite-world spatial capabilities](algorithms/infinite_spatial_capabilities.md).

Admitted Infinite height work is realized over a signed tile-plus-halo domain by
the CPU tile evaluator or the compiled GPU tile producer. Neither path flattens
the world address into an unsigned level rectangle. The GPU uses split-origin
coordinates and performs backend preflight before allocation; its packed result
removes operation guards without finite-world edge clamping. OpenSimplex and
Worley use the CPU realization until matching GPU contracts exist.

Infinite demand keys feed that same tile-domain contract and are reconciled into a
capacity-bounded scheduler. Current residents are touched, obsolete queued and in-flight
work is cancelled, and coarse/ancestor protection moves with demand. GPU-compatible
plans publish tile-domain results directly; other admitted graphs publish packed CPU
results. Both paths allocate only after live content validation and update the sparse
directory from the authoritative cache. Renderer lookup remains camera-relative work.

Output edits advance `TerrainRuntime::output_revision` through the app's single revision
boundary, which drops old pyramid content and clears pending uploads, the cache and page
table, and renderer streaming state. Document reset performs the same retirement while
preserving reusable atlas resources. GPU pyramid tiles are packed directly into the atlas
with regenerated, world-clamped halos; publication rejects a content revision that differs
from the live revision before mutating cache state. Every valid row is stamped with its
document, plan, output, and content revisions, and the shader rejects any mismatch even
if app-side invalidation were missed. Slot generations independently prevent an old
virtual mapping or CPU handle from resolving after reuse. Root coverage is published and
pinned before GPU-pyramid streaming is enabled; finer pages blend spatially and temporally
from their best current resident ancestor. See
[GPU terrain pyramids](algorithms/terrain_pyramid.md) for sampling and error conventions.

Commit `339a837` removed an earlier CPU `TerrainPyramid` residency map and viewport tile
plan because they duplicated GPU residency, published placeholder geometric error, had
no production consumer, and performed an O(world tiles) editor-frame probe. That cleanup
does not prohibit the concepts involved. Screen-space-error demand, coarse-first
refinement, viewport regions, and resident-ancestor fallback may return under these
constraints:

- planner output must change requested, uploaded, or rendered pages in production;
- demand and fallback query the atlas/cache authority rather than creating another
  mutable residency database;
- selected pages and fallbacks preserve output-revision and handle-generation checks;
- geometric error is measured or derived from real terrain data, never placeholder
  metadata; and
- work is bounded by visible/requested regions rather than scanning every world tile on
  every frame.

The source authority test discovers persistent tile-keyed stores by shape, not by names,
and requires any future demand planner to land with a production consumer and a
large-world bounded-work behavior test. Shader rendering tests separately prove that a
current child refines a resident root, unpublishing returns to the root, and pages from a
stale document or output never replace current ancestors.

Viewport residency is also distinct from streaming export. An export pipeline may
materialize and persist a complete deterministic pyramid at export quality, but it must
not reinterpret the viewport cache as a guaranteed complete artifact.

The production height-pyramid exporter supplies a deterministic complete demand set to
the same revision-aware scheduler and compiled GPU tile producer used by the viewport.
It persists fixed pages with the shared halo convention into immutable,
content-addressed generations and publishes only after complete validation. See
[Deterministic height-pyramid export](algorithms/terrain_streaming_export.md).

## Tiles & ghosts

Default tile 256² with halo 2. Halos are refreshed from neighbors before stencil reads. Phase 9 tile scheduler processes dirty tiles + neighbors to avoid seams.

## Undo

`EditorCommand` records stack/parameter deltas only — never full height textures.
