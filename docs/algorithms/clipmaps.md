# Geometry clipmaps

Losasso & Hoppe nested grids: each level doubles spacing; origin snaps to the spacing lattice around the camera XZ.

## Runtime (Wave D)

- `ClipmapConfig::for_world` picks `base_spacing` so the coarsest level spans the world extent.
- Viewport draws nested unit grids (coarse → fine) sampling the GPU height/normal textures.
- Per-level uniforms: origin, spacing, grid size; vertex shader maps UV → world XZ → height UV.
- Profiler reports active clipmap level count.

Bounded projects clamp snapped origins to their finite heightfield rectangle. Infinite
projects retain the same dyadic lattice but do not clamp: origins may be negative and
move seamlessly across either fixed-origin axis. Camera look, pan, fly, bookmarks, and
render preparation use the same topology-aware traversal policy.
