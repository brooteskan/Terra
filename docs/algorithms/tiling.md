# Tiled processing

- Dirty tile + 8-neighbors marked for recompute (`TileScheduler::mark_tile_and_neighbors`).
- `expand(radius)` grows the dirty set for multi-iteration stencils.
- `Heightfield::refresh_halos_for` performs destination-oriented incremental ghost exchange without a full dense flatten.
- `sync_dirty` treats dirty tiles as changed interior sources, refreshes their bounded 8-neighbor destination ring (including diagonal corner ghosts), then measures seams on edges touching the dirty sources.
- Seam metrics compare both tiles' halos with their source interiors at every configured halo depth; all errors must be 0 after synchronization.
- Halo destinations outside the field are clipped, while ghost samples outside the DEM retain clamp-to-edge behavior.
- Blur/erosion tile-local reads use halo samples only (`map_tiles`).
- `map_tiles_batched` refreshes halos after every pass; `iters_per_batch` only
  groups scheduling/accounting. A wider halo alone cannot make skipped refreshes
  correct unless the stencil also evolves the expanded halo domain.
- `SampleRect` / `bounds_from_tiles` feed region GPU upload and normal recompute.

The executable
[CPU determinism contract](../../crates/terra-core/tests/cpu_determinism_contract.rs)
compares batched tiled stencils with an independent dense oracle and evaluates
the same authored stack with 16, 32, 256, and full-field tile sizes using exact
`f32::to_bits` equality.
