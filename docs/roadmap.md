# Terra Roadmap (Phases 1–12)

Implemented incrementally in this repository. User-facing workflow remains a **layer stack** throughout.

| Phase | Status | Summary |
|------:|--------|---------|
| 1 | Done | Heightfield tiles, layer stack, viewport, custom `terra-gui` editor chrome |
| 2 | Done | Noise suite, fBm/ridged/warp, terrain shape generators |
| 3 | Done | Mask fields, ops, analysis (slope/aspect/curvature/AO/JFA), mask editor |
| 4 | Done | Dirty cache, incremental eval, progressive `EvalScheduler` |
| 5 | Done | Thermal erosion (CPU + GPU WGSL), erosion/deposition masks |
| 6 | Done | Hydraulic erosion (CPU reference + GPU shader layout) |
| 7 | Done | D8/D∞ flow, accumulation, watersheds, river carve |
| 8 | Done | Materials, biomes, Poisson vegetation scatter |
| 9 | Done | Tile scheduler, halo sync, seam metric |
| 10 | Done | Background export, PNG/RAW, GeoTIFF bridge |
| 11 | Done | Clipmap config, buffer pool, memory budget |
| 12 | Done | Command undo/redo, presets, content browser, export panel |

See [architecture.md](architecture.md) and `docs/algorithms/` for technical notes.
