# Progressive Rendering Strategy

Terra’s viewport progressive renderer targets **World Creator–style** interactive refinement: fast feedback while editing, then climbing quality while the scene is stable.

## Architecture (viewport backend rewrite)

Presentation is owned by a **frame graph** + **presentation backends**, not a single god-object frame path:

| Mode | Backend |
|------|---------|
| Fast / Raster | `RasterLit` (shadows + lit mesh / crack-free clipmap) |
| ProgressiveRayTraced / Final | `ProgressivePt` + `ProgressivePost` |

`TerrainRenderer` schedules:

`Begin → optional Shadow → PresentBackend → optional ProgressivePost → Overlays → timestamps`

Mode alone selects the backend — there is no dual progressive enable flag.

## Current approach

- **Heightfield compute path tracing** — rays intersect the displaced heightfield directly in WGSL. No BLAS/TLAS; no mesh BVH for terrain.
- **Progressive post** — consumes typed PT HDR + depth (`HdrFrame` / `GBufferViews`); no raster intermediate or optional `scene_override`.
- **RasterLit clipmap** — dense innermost ring near height-tex density; coarse rings punch Chebyshev holes so they never overdraw fine coverage; small worlds stay on a single dense grid.
- **Tile-stream height** — primary sample path with continuous monolithic fallback and generation-checked page-table rows.
- **Adaptive quality** — dynamic internal resolution, spp/bounce/denoise under GPU budget. Fake variance-from-sample-count is **not** on the hot path until GPU variance exists.
- **Editor refinement hysteresis** — `Interactive → Settling → Refining → Converged` driven by meaningful scene changes.

## Why HF compute PT

Terrain is a height function over a regular grid. Analytic ray–heightfield intersection is simpler and more stable than maintaining a triangle BVH that changes every sculpt stroke. Material and aux maps stay in the same sampling pipeline as Fast Lit.

## Future: hardware ray tracing

When HW RT is available, the same `PresentationBackend` + scene version contracts remain. A future path may build a coarse mesh BLAS for distant tiles while keeping HF PT for near-field edits. That migration must not regress interactive latency or break accumulation versioning.
