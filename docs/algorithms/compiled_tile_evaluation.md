# Compiled tile-domain GPU evaluation

Camera demand can be fulfilled directly from the backend-neutral
`CompiledTerrainPlan` without materializing a complete target-resolution field.
The first production slice admits Flat, world-coordinate NoiseValue/NoisePerlin,
bounded Blur, SculptBase, compiled masks, and layer/group composites. Other
kernels defer explicitly to the immutable complete-field pyramid until their
domain-coordinate contract is implemented; `Reach::Full` and observed
`AuxReach::Global` always defer.

## Domain and reach

`TerrainEvaluationDomain` carries the level and tile identity, exact interior,
complete content stamp, full-level world transform, publication halo, and the
additional guard required by the plan. The plan guard is resolved by walking
backward from final height through validated def-use metadata. Sequential local
reaches add, dependency branches merge by maximum, and operations remain in
compiled order.

For bounded projects the physical evaluation rectangle is clamped to the level
boundary. World positions are defined from integer global sample coordinates and
the full level dimensions, never from the tile texture dimensions. At world
edges the atlas pack pass clamps to the outermost evaluated sample.

For Infinite projects the authoritative rectangle is instead a signed `i64`
sample lattice produced by `InfiniteTopology`. Its publication and operation
halos are never clamped, including across coordinate zero. Local unsigned texture
coordinates are only a backend adapter: the signed evaluation origin is converted
to a fixed-origin `f64` world position before evaluation. GPU noise uniforms carry
that origin as high/low `f32` components so large tile addresses do not lose the
local increments between adjacent samples.

## Isolation, cancellation, and publication

`GpuCompiledTileProducer` owns evaluators independently of the complete-field
engine. A job evaluates into tile-sized plan and kernel scratch textures, submits
one completion fence, and exposes no atlas resource while in flight. Completed
or canceled evaluators are recycled only after their fence resolves.

The app revalidates the scheduler lease and complete `TerrainContentStamp` after
completion. Only then may `GpuTileAtlas` allocate a slot, pack the interior and
publication halo, and mark the page-table row valid. A stale job therefore cannot
evict current residency or publish a partially evaluated suffix.

Unsupported local kernels and all global work use the existing complete-field
GPU pyramid with an explicit diagnostic. Global checkpoint and basin strategies
belong to issue #177 and do not weaken this conservative boundary.

Infinite execution has no complete-field resource to defer to. The producer runs
Infinite admission and backend preflight before allocating textures or submitting
commands. Admitted GPU work evaluates the complete tile-plus-guard rectangle and
packs the publication halo without an edge clamp. A kernel that passes the shared
spatial contract but lacks a GPU implementation is sent to the CPU tile evaluator;
unsupported graphs remain rejected rather than being approximated per tile.
