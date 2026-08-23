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

Infinite presentation keeps the camera target and eye in fixed-origin `f64` coordinates.
Each frame selects the finest-tile corner containing the camera target as a transient
render origin, subtracts it on the CPU, and sends only small `f32` local positions to the
GPU. Ring origins are still snapped in absolute `f64` space before that subtraction.
Changing this origin invalidates presentation history only; tile keys, procedural inputs,
content stamps, demand, and residency remain fixed-origin.

The Infinite fallback is not a world grid. It is a camera-centred grid covering the
configured horizon on the coarsest topology lattice. The vertex shader resolves the best
resident signed sparse page at each point and walks Euclidean ancestors when detail is
pending. Infinite rendering is enabled only after the current mandatory coarse coverage
is resident, so the finite monolithic height texture is never used as a terminal fallback.
