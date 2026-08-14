# Real-time Terrain Pipeline — Historical Migration Snapshot

> [!NOTE]
> This document records the completed GPU/clipmap performance migration and is
> retained for historical rationale. It is not current architecture guidance.
> See [architecture.md](architecture.md) for authoritative crate boundaries and
> frame composition, and [perf.md](perf.md) for current performance guidance.

## Interactive path at the time of migration

```
UI edit → mark_dirty_from → debounce → GpuTerrainEngine (WGSL)
  → layer cache textures → ping-pong thermal/hydraulic
  → dirty tile bounds → region copy + region normals → swap
  → clipmap nested grids sample height (camera-centered LOD)
CPU fallback when stack has unsupported layers / masks
```

### Responsiveness (event loop)

- `ControlFlow::Wait` / `WaitUntil` — no busy-poll; redraw only on input, evaluation, or UI repaint
- Eval skipped while orbiting / dragging UI; hybrid CPU + GPU readback only on Full
- Draft/Medium skip sync mask readback (`Maintain::Wait`); camera not retargeted every present

## Incremental migration

### Wave A — Viewport GPU path (done)
- Static XZ grid, sample height texture in VS
- Upload `R32Float` height, GPU normals, double-buffered display

### Wave B — Interactive scheduling (done)
- Suffix dirty only, debounce + generation ID, progressive refine, profiler

### Wave C — GPU generators (done)
- WGSL noise/fBm/ridged, blend, sims; GPU→GPU present; CPU readback on Full only

### Wave D — Tiles + large worlds (done)
- Incremental ghost exchange (`refresh_halos_for`) + `TileScheduler` expand/sync
- Dirty `SampleRect` region present + region normal compute
- Geometry clipmaps: nested grids from `ClipmapConfig::for_world`, camera-snapped origins
- Profiler: clipmap levels + tile grid

## CPU vs GPU responsibilities

| CPU | GPU |
|-----|-----|
| Params, dirty graph, tile scheduler | Height displace + clipmap LOD |
| Incremental halo exchange | Noise/fBm, blend, blur, terrace |
| Fallback eval (masks, rivers, …) | Thermal / hydraulic ping-pong |
| Export / Full readback | Region upload + region normals |

## Wave C/D GPU-supported layers

`Flat`, `Ramp`, `NoiseValue`, `NoisePerlin`, `Fbm` (Value/Perlin), `Ridged`, `ThermalErosion`, `HydraulicErosion`, `Blur`, `Terrace`

## Verification

| Before | After |
|--------|--------|
| Single 384² grid over whole world | 4 nested clipmap levels, spacing from world size |
| Full-field halo via `to_dense` | Incremental `refresh_halos_for` on dirty tiles |
| Full normal dispatch every upload | Region-limited normal compute |
| Full texture copy always | Dirty-rect GPU present when available |

Run: `cargo run -p terra-app` — Profiler shows path, tiles, clipmap levels, µs.
