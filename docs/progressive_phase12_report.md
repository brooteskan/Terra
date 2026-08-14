# Viewport backend rewrite — status

Supersedes the Phase 1–12 “extend TerrainRenderer” milestone with the long-term presentation architecture.

## Architecture summary

| Layer | Responsibility |
|-------|----------------|
| `frame_graph` | Pass schedule + dense GPU timestamp resolve |
| `backends/raster_lit` | Lit present planning (single grid / crack-free clipmap) |
| `backends/progressive_pt` | Typed PT HDR/GBuffer outputs |
| `backends/progressive_post` | Temporal / à-trous / composite from explicit HDR+depth |
| `terra-app` redraw/eval | One mode sync; tile-stream enable with monolithic fallback |

## Mode → backend

- Fast / Raster → `RasterLit`
- ProgressiveRayTraced / Final → `ProgressivePt` + `ProgressivePost`

## Landed in this rewrite

- [x] FrameGraph + PresentationBackendId schedule
- [x] Dual progressive enable removed (mode alone arms post)
- [x] RasterLit high-res / clipmap with hole punch (no coarse overdraw of fine)
- [x] ProgressivePost typed HDR/GBuffer (no `scene_override`)
- [x] DepthExporter deleted (PT supplies float depth)
- [x] Adaptive sample-count variance stub gated off the hot path
- [x] Tile-stream primary height sample + monolithic miss fallback
- [x] Clipmap densest ring tracks height-tex spacing

## Verification

| Check | Command |
|-------|---------|
| Compile | `cargo check -p terra-render -p terra-app` |
| Render tests | `cargo test -p terra-render --lib` |
| Core tests | `cargo test -p terra-core --lib` |

## Remaining cleanup (non-blocking)

- Finish extracting pipeline objects out of `TerrainRenderer` into backend crates
- Real GPU variance readback for adaptive sampling
- Optional HW RT behind the same `PresentationBackend` trait
