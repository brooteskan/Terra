# Thermal erosion

## Model

Talus-angle slope limiting: material moves to 4-neighbors when height difference exceeds `tan(talus) * dx`.

Per cell with excess slopes `Δ_i`:

```
move = strength * 0.25 * ΣΔ_i
h -= move
neighbor_i += move * Δ_i / ΣΔ_i
```

## Stability

- `strength ∈ [0,1]`; large values can overshoot — prefer more iterations at lower strength.
- GPU kernel uses ping-pong buffers; current WGSL pass redistributes outbound mass on the source cell (neighbor deposit refined on CPU mask derivation).
- The CPU classical and layered paths are closed redistribution systems. Their
  finite-state, mass, hardness, and bathymetry invariants are executable in the
  [simulation stability contract](simulation_stability.md).

## Outputs

Normalized erosion / deposition masks for reuse in the mask system.
