# Simulation stability contract

Terra's CPU erosion oracles share an executable numerical contract in
[`simulation_numerical_contract.rs`](../../crates/terra-core/tests/simulation_numerical_contract.rs).
It complements the qualitative landform fixtures; it does not replace them or
introduce a second evaluator.

The contract covers classical and layered Thermal, every Hydraulic transport
model with and without particles, Debris Flow, Stream Power Erosion, Fast and
Accurate Landscape Evolution, and an authored Hydraulic -> Debris -> Thermal
`StackEvaluator` sequence. The fields are deliberately small, deterministic,
and CPU-only.

## State invariants

- Height, every published aux field, every raw diagnostic, and every ledger
  value must remain finite. Sub-zero height is valid bathymetry.
- Water, suspended material, bedrock, debris, sediment, erosion, and deposition
  are physical quantities and must be non-negative.
- Normalized display masks remain in `[0, 1]`; raw physical fields such as
  Stream Power incision depth are not clamped to that range.
- Flat and epsilon-slope inputs cannot amplify into extreme finite relief.
- Zero effect is an identity or a bounded uplift-only limit, as documented by
  the operator. Full hardness/resistance prevents erosion and incision.

## Material ledgers

All sums are accumulated as `f64` over production `f32` fields. Conservative
checks use a scale-aware roundoff allowance:

```text
16 * f32::EPSILON *
    (cell_count * max_abs_terrain_height + moved_material)
```

This allowance scales with both terrain storage and actual transfers. It is
small enough to reject the historical visible debris-flow drift instead of
hiding it behind a broad absolute epsilon.

- Classical and layered Thermal are closed redistribution systems. Layered
  state additionally satisfies `surface = base + bedrock + debris + sediment`.
- Hydraulic uses a closed domain. Terrain plus final suspended sediment equals
  the initial terrain plus any initial suspended sediment. Rainfall creates
  water, not terrain material.
- Debris Flow publishes `DebrisFlowMassLedger`. Its current closed policy
  settles terminal load in-domain, so `in_flight_sum` and `exported_sum` are
  zero. A future open policy must report export rather than discard it.
- Stream Power deliberately removes terrain. The raw `spe_incision` depth is
  the removed-material ledger; it is not classified as conservative transport.

## Landscape Evolution boundaries

Routing outlets and elevation locks are separate concepts in both solvers:

- `Fixed`: the one-cell rim routes out and remains bit-identical to the original
  input through preprocessing, evolution, hillslope smoothing, and restoration.
- `OpenDrainage`: the rim terminates routing but is free to uplift and evolve.
- `SeaLevel`: cells at or below `base_level` are locked. The rim is a
  routing-only fallback, so above-sea edge cells may evolve.
- `OutletMask`: authored cells are outlets and locks. A missing mask follows
  `SeaLevel`; an empty supplied mask gets a routing-only rim fallback and does
  not invent locks.

## Cross-layer material state

The typed `AuxMaps` representation is authoritative between erosion layers:

- `BEDROCK_HEIGHT`: bedrock elevation;
- `DEBRIS_DEPTH`: coarse debris/talus thickness;
- `SEDIMENT_THICKNESS`: fine sediment/alluvium thickness.

Legacy sediment spellings are accepted while loading, but processors and cache
checkpoints emit the canonical keys. The contract verifies exact inventory
handoff on positive and sub-zero terrain and reconstructs the surface after the
cache/hash-map round trip.

Run the station contract with:

```text
cargo test -p terra-core --test simulation_numerical_contract
```
