# Masks & analysis

## Ops

Add, subtract, multiply, min, max, invert, clamp, levels, smoothstep, blur, remap — applied as a per-asset op list.

## Analysis

- Slope/aspect: central differences.
- Curvature: discrete Laplacian mapped to \[0,1\].
- Convexity/concavity: split curvature about 0.5.
- AO: neighbor rise heuristic (not screen-space ray AO).
- Distance: jump-flood (JFA) on binary seeds.

Simulation masks (flow, wetness, sediment, erosion, deposition) are schema-ready and filled by sim layers.
