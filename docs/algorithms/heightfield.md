# Heightfield storage

## Representation

- Regular grid DEM; heights in **world meters** (Y up).
- Sample centers at `(i + 0.5) * dx`, `(j + 0.5) * dz`.
- Tiles: interior `tile_size` (default 256) + `halo` ghost cells (default 2) on each edge.

## Ghost cells

Before stencil reads (blur, erosion, slope):

1. Flatten or read neighbor interiors.
2. Write overlapping samples into each tile’s halo (`refresh_halos`).
3. Out-of-domain ghosts use clamp-to-edge.

Halo width must be ≥ stencil radius for a single pass. Multi-iteration sims either refresh each iteration or use `halo ≥ radius * iterations_per_batch`.

The [CPU determinism contract](../../crates/terra-core/tests/cpu_determinism_contract.rs)
checks every ghost cell against its global clamped source, including both sides
of each axis, corners, every configured halo depth, and partial final tiles. It
also proves that a stale reverse copy is visible before synchronization.

## Assumptions

- No CRS in Phase 1–9; GeoTIFF import may lack georeferencing unless GDAL is used.
- Finite differences use clamped neighbors when halo is unavailable.
