# Terrain camera demand

Viewport demand is a deterministic, residency-free description of which immutable
terrain tiles would improve the current camera view. `TerrainDemandPlanner` owns only
its previous refinement decisions for hysteresis. It does not contain atlas slots, page
handles, publication state, or a copy of `TileResidencyCache`.

## Inputs and error projection

The planner consumes the immutable `TerrainPyramid` addressing descriptor, its measured
per-child-tile world-space errors, the camera view-projection matrix and eye, physical
viewport dimensions, conservative terrain height bounds, and a pixel-error policy.
Errors are transferred asynchronously once per accepted GPU pyramid and stamped with the
same output, generation, plan-revision, and output-revision identity. Stale transfers are
discarded before planning.

The GPU values are local child-versus-parent deltas. Once after transfer, Terra builds a
conservative top-down envelope: each tile's local error plus the largest child envelope.
This triangle-inequality bound ensures a fine feature lost through several coarse levels
still causes traversal to reach the level where it appears. This is content preprocessing,
not a per-frame hierarchy scan.

For a visible child tile, the parent approximation error is projected with

`error_px = error_world * viewport_height / (2 * tan(fov_y / 2) * max(distance, near))`.

Distance is the shortest distance from the eye to the tile's conservative world AABB.
The child refines when this value exceeds the configured target. A previously refined
child remains selected until it falls below a lower coarsening threshold, preventing
threshold jitter from replacing the request set every frame.

## Hierarchy walk and bounds

Planning starts at the root and walks breadth-first. Each tile AABB is frustum-culled
before descendants are considered, so a narrow view never scans every finest-level tile.
Visible children are considered in projected-error descending, distance ascending, then
stable level/tile order. Both visited hierarchy nodes and emitted tiles have explicit
limits. Reaching either limit stops refinement and leaves the coarser representation in
the plan.

Every selected tile is closed over `covering_parent_tiles`. This is required for
non-power-of-two levels where one child footprint can intersect multiple parent pages.
The final output is unique and ordered coarse-to-fine, guaranteeing that the current
consumer and the later work scheduler can establish coverage before optional detail.

## Infinite sparse topology

Infinite planning is a separate traversal over shared projection, culling, ordering, and
budget primitives. It never invents a root tile or enumerates a complete level. A finite
half-open tile window at the configured coarsest LOD is derived directly from the camera
centre and horizon. That entire window is emitted as `CoarseCoverage` before refinement;
a plan is rejected if mandatory coverage cannot fit the tile and node limits.

Visible refinement is restricted to the preview radius. Signed children use the
`terra-world` Euclidean power-of-two hierarchy, and a refinement is admitted only when
its complete chain through the active coarsest window fits atomically. Planning AABBs
remain `f64` in fixed-origin X/Z until they are translated relative to the eye for
clip-space tests. Replanning replaces the previous Infinite hysteresis set, so travel
does not accumulate addresses or work.

Infinite geometric error is indexed by LOD rather than by a global tile array. The
initial model uses a conservative configured or certified height/error envelope; future
generated tiles may provide sparse certified envelopes without making visited area a
planner data structure.

## Production consumption

`terra-app` converts the current renderer camera into planner input after compact error
metadata becomes available. A changed plan replaces pending GPU-pyramid uploads; exact
current pages may be skipped only by querying the authoritative atlas cache. Planning
itself never changes cache or page-table state. The existing upload path remains the sole
publisher and repeats the live output-revision check before atlas mutation.

The revision-aware tile scheduler incrementally reconciles changed plans instead of
replacing a FIFO. It preserves age for retained demand, establishes required coarse
coverage before optional refinement, applies editor-state budgets, and rejects stale
content identities at atlas publication. See
[Terrain tile work scheduling](terrain_work_scheduling.md). Resident-ancestor shader
resolution remains separate renderer work: the plan is never passed to the renderer, and
the GPU page directory alone determines which exact or ancestor page is sampled.

Infinite plans already produce canonical signed `TerrainTileKey` values accepted by
`TerrainEvaluationDomain::for_infinite_tile`. The app reconciles those keys into bounded
work, touches demanded residents, and moves coarse/ancestor protection with the current
plan. A fixed-capacity open-addressed GPU directory stores full signed coordinates; it is
rebuilt from the authoritative residency cache on publication and eviction, so travelled
addresses do not accumulate in a second CPU map.
