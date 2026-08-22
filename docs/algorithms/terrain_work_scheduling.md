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

The only work sources are the two production executors that exist today: CPU
height-tile upload and immutable GPU-pyramid publication. Atlas publication
compares the complete live content identity before cache or page-table mutation.
Payload commands precede the valid page-table write, and revision retirement
disables streaming and queues a page-table clear after older GPU work.

Snapshots report queued and in-flight counts, deduplication, reprioritization,
cancellation, stale drops, authoritative-cache skips, submissions, completions,
failures, budget overruns, estimated dispatched cost, and queue/completion
latency totals and maxima.
