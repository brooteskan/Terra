# Flow routing

A single routing implementation lives in `crate::geomorph` (`geomorph::routing`).
Every drainage consumer — the stream-power / river-carve oracles in
`crate::hydro`, `landscape_evolution::DrainageCache`, mass-wasting, terrain
amplification, and `analyze_terrain` — builds on the same `FlowGraph`, so there
is one flow-routing lineage, not two.

## FlowGraph

`build_flow_graph(&heightfield, model)` returns per-cell receivers, a
single-receiver D8 direction index per cell (`NO_FLOW` for sinks), donor lists, a
topological order (upstream → downstream), and a normalised direction mask.

## D8

Steepest strictly-downhill of the 8 neighbours. Exact ties break toward the
lowest direction index (fixed offset order, index 0 = +X). That rule makes
routing reproducible: a fixed `(terrain, model)` yields bit-identical `f32`
drainage products (pinned by `routing_is_deterministic` and, for the tie-break
itself, `geomorph::routing::tests::d8_deterministic_ties`). Cells with no lower
neighbour are marked `NO_FLOW` and retain their own accumulation rather than
routing arbitrarily.

## D∞ (Tarboton)

Partition flow between the two cells bounding the steepest downhill facet,
proportional to the facet geometry; fall back to the steeper bounding edge when
the gradient leaves the facet's angular wedge. `build_flow_graph` fills the D8
direction index and mask alongside the D∞ receivers.

## Accumulation

`accumulate_drainage_area(&graph, &precip)` walks the topological order once,
pushing each cell's weighted contribution to its receiver(s). Uniform `P = 1`
reproduces classic cell-count flow accumulation; a precipitation map or artist
weight yields discharge-weighted area.

## Depression fill

`geomorph::priority_flood_fill` (also reached as `hydro::fill_depressions`, which
delegates to it) raises enclosed pits to their lowest spill elevation via
Priority-Flood while preserving boundary outlets, so drainage is continuous
wherever a path to the terrain boundary exists.

## Rivers and stream order

Threshold accumulation → carve width and depth scaled by upstream accumulation;
banks use a Gaussian-like falloff, and increasing bank smoothing broadens the
soft shoulders. Two distinct stream-order notions coexist under distinct names:

- `hydro::stream_order_log2` — `1 + floor(log2(acc / threshold))`, the
  visualisation bucketing published as the `STREAM_ORDER` aux overlay.
- `geomorph::strahler_order` — true Horton–Strahler order on the donor graph,
  carried by `analyze_terrain`'s `StreamNetwork`.

## Pitfalls

- Flats remain no-flow cells after routing; depression fill removes only
  enclosed pits before river carving.
- Tile seams: recompute flow on the full field or with sufficient halo after
  height edits.

Landscape Evolution's executable fixed/open/outlet semantics and transport
material policies are documented by the
[simulation stability contract](simulation_stability.md).
