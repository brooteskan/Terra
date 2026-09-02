# Terrain tile work scheduling

Camera demand is reconciled into a bounded `TerrainTileWorkScheduler`. The
scheduler stores requests and leases only. It never stores atlas slots, page
handles, resident flags, or a copy of the page table; `GpuTileAtlas` and its
`TileResidencyCache` remain the residency authority.

## Identity and reconciliation

Requests deduplicate by virtual field/level/tile plus compiled-plan and output
revision. A full content stamp additionally carries document generation and the
accepted content revision (the GPU output ID for an immutable pyramid). A change
to any stamp component drops queued work and cancels in-flight leases.

Replanning updates the class, projected error, distance, and visibility of
retained requests without resetting their first-seen age. Requests absent from
the current plan are cancelled. Before reconciliation and again before dispatch,
the app queries `GpuTileAtlas::is_current`; the scheduler never remembers that
answer as residency.

## Priority and starvation

Required coarse coverage and fallback ancestors form a strict lane ahead of
optional refinement. Within a lane, visible requests lead, then demand class,
coarser level, projected error, camera proximity, age, and stable tile order.
Once a live request reaches the maximum waiting age it is promoted ahead of
ordinary requests in its lane.

Dispatch sums estimated publication cost under the active editor refinement
budget and caps item count and in-flight work. When no item fits an otherwise
empty frame, one oversized live request may run and the overrun is reported.
This escape plus age promotion prevents stationary-view starvation.

## Publication and statistics

The production work sources are CPU height-tile upload, immutable GPU-pyramid
publication, and compiled tile-domain GPU evaluation. The compiled source stages
tile-sized plan resources and becomes publishable only after its complete suffix
finishes and the scheduler lease remains current. Unsupported or global plans
defer explicitly to the immutable pyramid. Atlas publication
compares the complete live content identity before cache or page-table mutation.
Payload commands precede the valid page-table write, and revision retirement
disables streaming and queues a page-table clear after older GPU work.

GPU-pyramid publication pins current root coverage and enables streaming only after that
coverage is complete. Optional refinement therefore cannot evict the terminal resident
ancestor; under insufficient capacity it waits rather than converting monolithic height
into the correctness fallback.

Snapshots report queued and in-flight counts, deduplication, reprioritization,
cancellation, stale drops, authoritative-cache skips, submissions, completions,
failures, budget overruns, estimated dispatched cost, and queue/completion
latency totals and maxima.

See [Compiled tile-domain GPU evaluation](compiled_tile_evaluation.md) for the
domain, reach, and atomic-publication contract.
