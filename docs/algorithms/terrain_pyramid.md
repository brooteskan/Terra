# GPU terrain pyramids

The final-output pyramid is a deterministic, GPU-resident content hierarchy. It is built
after Terra accepts a complete GPU height result; interactive camera demand and partial
global-operation evaluation are separate concerns.

## Levels and coordinates

The spatial math is implemented by `terra-world::BoundedTopology`; `TerrainPyramid`
is the existing bounded-world adapter and content-facing iterator. Shared `TileAddress`
values use signed `i64` coordinates and finest-first LODs (LOD 0 is finest). At the
bounded boundary, the adapter maps `bounded_level = max_level - lod` so the package,
dense metadata, GPU directory, and traversal order remain coarse to fine exactly as
before. Infinite topology uses the same address type without total world dimensions or
complete enumeration.

Level 0 is 2×2. Starting at the requested finest resolution, each parent dimension is
`ceil(child / 2)`, then the sequence is stored coarse to fine. Thus 512 produces
`2, 4, ..., 512`, while 1000 produces
`2, 4, 8, 16, 32, 63, 125, 250, 500, 1000`.

Every level spans the same world rectangle. A sample `(x, z)` represents the normalized
cell `[x / width, (x + 1) / width) × [z / height, (z + 1) / height)` and therefore the
same fractions of `world_size_x` and `world_size_z`. Parent/child tile coverage is the
inclusive range of tiles whose normalized cell footprints intersect. The mapping uses
integer floor/ceil arithmetic so non-power-of-two edges do not leave gaps.

Tiles use row-major `(tx, tz)` addressing. Interiors are `tile_size`² except at the right
and bottom edges, where they are clipped to the level dimensions. Content metadata uses
a stable dense index: levels coarse to fine, then tile rows and columns.

## GPU materialization

The accepted source texture is copied into an owned finest-level R32Float texture so its
identity cannot change beneath the hierarchy. Coarser samples are exact normalized-cell
area-weighted averages of the next finer level. This rule handles both power-of-two and
irregular ratios deterministically and preserves constant fields exactly. Production
materialization submits GPU work only. Height readback helpers exist solely for tests.
After materialization, camera-demand integration asynchronously transfers the compact
dense geometric-error buffer once for the immutable content identity; it never maps a
height texture and never blocks the interactive thread.

Each `GpuHeightPyramid` is immutable and stamped with output revision, GPU output ID,
evaluation generation, and plan revision. The app retains only the latest accepted
complete hierarchy.

## Geometric error

For every non-root child sample, the GPU reconstructs the parent at the child cell center
with clamped bilinear sampling and measures `abs(child - reconstructed_parent)`. Heights
are world-space height values, so the result is a world-space vertical error. An atomic
maximum records the conservative error for the child tile in the pyramid's dense metadata
buffer. Root entries remain zero. Non-finite results saturate to the largest finite f32;
all published errors are finite and non-negative.

## Atlas publication and halos

`GpuTileAtlas` remains the sole residency authority. Publishing a pyramid tile allocates
through its existing `TileResidencyCache`, keeps the page-table slot invalid while packing,
then marks the row valid after the payload submission. The pack shader copies the interior
and regenerates each halo directly from the complete level texture. Neighbor halos
therefore read the same source samples; terrain edges clamp to the outermost sample, and
unused texels in a partial physical page are zeroed deterministically.

Publication compares the immutable pyramid's output revision with the current runtime
revision before any cache or page-table mutation. Old content cannot be relabeled as a
current page.

## Resident-ancestor presentation

The atlas owns a dense virtual directory indexed by the same stable metadata order used
for geometric errors. The shader starts at the current source level and walks toward the
root, performing one direct lookup per level. A mapping resolves only when its physical
slot generation, coordinates, and complete content stamp match. Lookup cost is independent
of physical atlas capacity, and GPU page-table state—not camera demand—selects rendering.

Every level covers normalized UV `[0, 1]²`. Sampling maps UV to
`clamp(uv) * (resolution - 1)`, selects `floor(sample / tile_size)`, and filters wholly
inside that page using its level-local halo. Partial edge pages use their clipped extent;
world-edge halos clamp to the outermost level sample. A missing same-level neighbor causes
a short spatial blend to the best resident ancestor, while a newly published page also
morphs from that ancestor over a bounded frame interval. Root coverage is pinned before
GPU-pyramid streaming is enabled.

Monolithic page-miss sampling remains an explicit bounded-project/CPU migration mode and
an emergency diagnostic path. GPU-pyramid correctness requires current resident root
coverage, so its tests treat terminal fallback as a failure.

Camera-driven selection of these tiles is described in
[Terrain camera demand](terrain_demand.md).
