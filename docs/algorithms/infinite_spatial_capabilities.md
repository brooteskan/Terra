# Infinite-world spatial capability contracts

Infinite projects evaluate an unbounded world as finite sparse tile requests. A
compiled operation may enter that scheduler only when its complete dependency
chain has a direct sparse evaluation contract.

## Operation contract

`Reach` remains the machine authority for sample support:

- `Localized { halo_samples: 0 }` is a coordinate-pure/per-sample operation.
- `Localized { halo_samples: n }` requires `n` guard samples on each side.
- `Full` observes a complete field and is never directly tile-evaluable.

`OperationSpatialContract` adds an Infinite strategy without recalculating that
reach. Slice 1 implements `Direct`; `RegionBaked` and `Hierarchical` are reserved
and reject explicitly until their execution strategies exist.

The initial direct kernel set is Flat, Value/Perlin/OpenSimplex/Worley noise,
fBm, ridged noise, and bounded Blur. Seed, mask arithmetic, layer/group blend,
auxiliary composite, and publication operations are direct when all of their
inputs are direct. Local invalidation alone is not proof of Infinite support:
authored bounded rasters and kernels without an absolute-coordinate contract
remain unavailable.

## Plan admission

`resolve_infinite_plan_domain` walks backward from the requested output through
the validated plan def-use graph. Sequential halos add with saturating arithmetic;
converging dependency branches retain their maximum. A rejection at any upstream
operation rejects the downstream output and retains the authored owner and a
stable reason code for UI diagnostics.

Full reach, basin coupling, consumed global auxiliary fields, global parameter
reductions, bounded authored data, and unclassified operations reject. Infinite
analysis never converts these blockers into the bounded complete-field checkpoint
strategy and never approximates them independently per tile.

Bounded projects continue to use `resolve_plan_execution_strategy`, including
immutable complete-field checkpoints. Spatial contracts are runtime-derived and
are not serialized into project documents.

## User-visible availability

The tool catalog and Quick Add derive availability from the same layer contract;
unsupported entries remain visible with their reason but cannot be activated.
The central action handler repeats the check so drag/drop and command-palette
paths cannot bypass the UI. The inspector shows both the selected layer contract
and the cached compatibility of the current compiled graph.

The app resolves and caches graph compatibility before creating evaluation work.
Rejected Infinite graphs enqueue no CPU, GPU, checkpoint, or tile work and retain
the last valid presentation.

## Backend realization

Both tile backends consume the same admitted plan and signed
`TerrainEvaluationDomain`. The CPU evaluator implements the initial direct set,
including OpenSimplex and Worley. The GPU evaluator currently implements Flat,
SculptBase, Value/Perlin noise, Value/Perlin fBm and ridged noise, and bounded
Blur; OpenSimplex and Worley are explicit CPU boundaries. GPU preflight happens
before command encoding or resource allocation.

The CPU and GPU paths form world coordinates from the signed sample lattice and
the topology's fixed `f64` origin. GPU shaders receive a split high/low origin and
add local coordinates after scaling, preserving nearby variation at very large
tile addresses. Regenerating a tile is deterministic within a backend, request
order is irrelevant, and adjacent tile evaluation rectangles agree in their
overlap. Packing removes only the cumulative operation guard and retains the
requested publication halo.
