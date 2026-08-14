# Hydraulic erosion

Adapted from Mei et al. (GPU shallow water) and Šťava et al. (interactive artist control).

## CPU reference (oracle)

State: height, water, sediment.

Each iteration:

1. Add rainfall.
2. Compute outflow toward lower 4-neighbors.
3. Erode/deposit from capacity `C = capacity * slope * flow`.
4. Transport water/sediment shares.
5. Evaporate water; accumulate wetness.

## GPU

WGSL kernel provides ping-pong height/water layout. Production path may call CPU oracle when adapter init fails; shaders are kept for RenderDoc iteration.

## Assumptions / limits

- No pipe cross-sections; 4-neighbor only (not full Mei pipe model).
- The CPU oracle uses a closed material boundary: terrain plus suspended
  sediment closes within the scale-aware f32 tolerance exercised by the
  [simulation stability contract](simulation_stability.md).
- Prefer lower timesteps / more iterations for stability.
