# Infinite Procedural World — Slice 1

Infinite Procedural World is an experimental project type that generates a
camera-centred set of sparse terrain tiles from one fixed world origin and seed.
It is intended for procedural traversal, including signed coordinates thousands
of kilometres from the origin. It is not an infinite authored canvas.

## Create and identify a project

From Project Home, choose **New Project**, select **Infinite Procedural World**,
adjust the settings, and choose **Create World**. The project chip identifies the
open world as `Infinite` and reports its finest spacing and horizon.

Saving writes the world recipe: project seed, fixed origin, topology and LOD
settings, preview distances, residency budgets, and the compatible layer stack.
Generated samples, resident pages, page tables, scheduler state, and caches are
not saved. Reopening regenerates demanded tiles from the recipe.

Existing projects without a world-type field continue to open as Bounded
Heightfield projects. Terra does not convert between the two project types.

## Settings

- **Seed** determines procedural content. The same seed, graph, absolute tile
  address, and revision produce the same backend result.
- **Origin X/Z** is the fixed authoritative `f64` world origin. Camera travel does
  not move it.
- **Finest spacing** is the metres-per-sample spacing at the finest LOD.
- **Tile size** is the power-of-two interior sample extent of a sparse page.
- **Max LOD** controls the coarsest available coverage ring.
- **Preview radius** is the nearby refinement region.
- **Horizon** is the camera-centred area for which coarse valid coverage is
  demanded. It must be at least the preview radius.
- **CPU/GPU budgets** bound transient CPU tile payloads and GPU resident pages.
  Each budget must hold at least one tile including its publication halo.

Distance travelled and previously visited area do not enlarge the configured
demand, scheduler, or residency limits. Open **View → Profiler** to inspect
`Infinite demand`, `Infinite work`, `Infinite residency`, and `Infinite CPU
payload`. A count exceeding its displayed limit is a contract violation worth
reporting with the persistent log from `%LOCALAPPDATA%\Terra\logs`.

## Supported operation set

Slice 1 directly supports Flat, Value/Perlin/OpenSimplex/Worley noise, fBm,
ridged noise, and bounded Blur, plus compatible seed, mask arithmetic,
layer/group blend, auxiliary composite, and publication steps. OpenSimplex and
Worley currently use the CPU tile backend; the other listed generators have a
GPU path where supported by the adapter.

The complete graph must be sparse-compatible. Quick Add and the tool catalog
leave incompatible operations visible but disabled with a reason. The inspector
reports the selected operation's contract and the current graph's compatibility.
If a full-field, basin-dependent, global-reduction, bounded-raster, or otherwise
unclassified dependency enters the graph, Terra rejects the graph before CPU or
GPU tile work is scheduled and keeps the last valid presentation.

## Presentation and troubleshooting

Coarse pages are required work and are published before optional fine
refinement. The renderer uses the finest current page available and falls back to
a current ancestor while a child is pending or evicted. Camera-relative
presentation preserves nearby precision at large absolute coordinates.

If terrain does not appear or stops refining:

1. Check the status line and inspector for an `Infinite graph unavailable`
   reason. Remove or disable the named incompatible operation.
2. Open the profiler. Demand tiles/nodes, live work, resident pages, and CPU
   payload should remain at or below their displayed limits.
3. Verify the horizon is at least the preview radius and both memory budgets can
   hold one halo-expanded page.
4. Check the persistent log for topology, demand-budget, evaluation, or upload
   rejection messages.

Slice 1 does not include complete-world export, sparse sculpting, paths, painted
masks, imported rasters, globally consistent erosion/drainage/climate/river
routing, project-type conversion, or fixed frame-time guarantees.

The backend contract and admission rules are specified in
[Infinite-world spatial capability contracts](algorithms/infinite_spatial_capabilities.md).
