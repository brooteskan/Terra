# Fidelity notes (post M1–M5)

## CPU vs GPU

- **CPU** remains the export oracle (`StackEvaluator` / `terra-io` background export).
- **GPU** preview supports only configurations covered by the executable matrix below. Generator compositing supports Replace/Normal/Interpolate, Add, Subtract, Multiply, Min, Max, and Overlay. HeightBlend and smooth blend modes fall back to CPU rather than being substituted.
- **In-place GPU kernels** (Blur, Terrace, EffectFilter, Thermal, Hydraulic, and RiverCarve) are supported only with their exact default outer composite: full opacity, no layer mask, and Replace/Normal/Interpolate blending. Masked, partial-opacity, or otherwise blended configurations fall back to CPU because these kernels do not yet preserve and composite their entering height.
- **GPU masks** are limited to one Multiply entry referencing an existing, operation-free Constant, Height, or Slope asset.
- **CPU fallback**: Coastal currently requires CPU evaluation. Materials, Biomes, and Vegetation also require CPU evaluation because their observable auxiliary fields are not published by the GPU path.
- **Hybrid**: GPU preview may continue supported work speculatively above an unsupported layer when no readback is requested. A requested CPU checkpoint instead stops before the first unsupported layer, and its height is exactly the field entering `resume_cpu_from`. Prefixes that publish auxiliary fields or named outputs cannot be represented by height alone and conservatively restart the CPU evaluator from layer zero.

## Executable parity matrix

Errors compare complete GPU and CPU height fields. Normalized RMSE uses the maximum of CPU range, CPU RMS, and one metre as its scale. The table is checked against `terra_gpu::parity::FIDELITY_MATRIX_MARKDOWN` by the test suite.

<!-- BEGIN GENERATED GPU PARITY MATRIX -->
| Contract | Supported configuration | Max abs (m) | Normalized RMSE |
| --- | --- | ---: | ---: |
| `exact-height` | Flat, Ramp, SculptBase, exact blends, hybrid checkpoint | 0.001 | 0.00001 |
| `mask.simple` | One Constant, Height, or Slope Multiply entry without asset operations | 0.001 | 0.0001 |
| `noise.value` | Value noise with a 32-bit seed | 17.0 | 0.23 |
| `filter.blur` | Default outer composite; radius/iteration fixture | 2.1 | 0.0055 |
| `effect.smooth` | Smooth, default outer composite | 1.8 | 0.03 |
| `effect.inflate` | Inflate, default outer composite | 2.7 | 0.028 |
| `filter.terrace` | Default outer composite | 8.5 | 0.10 |
| `simulation.thermal` | Non-layered, constant hardness, no weathering extension | 3.2 | 0.05 |
| `simulation.hydraulic` | Base transport, no sources, particles, layers, or post-effects | 3.0 | 0.03 |
| `island.volcanic-high` | VolcanicHighIsland with a 32-bit seed | 220.0 | 0.10 |
<!-- END GENERATED GPU PARITY MATRIX -->

## Remaining intentional gaps

- GPU hydraulic omits full neighbor water/sediment gather (atomic-free preview).
- GPU value noise is a portable hash approximation, not bit-identical to CPU.
- Multi-entry distributions, non-Multiply combines, mask asset operations, missing assets, and Noise/Curvature mask sources fall back to CPU.
- Perlin, fBm, ridged, domain-warp, dunes, mountains, canyons, mesa, volcano, uplift, plateau, non-volcanic island archetypes, RiverCarve, and unratcheted EffectFilter variants fall back to CPU. They may return only after gaining a named full-field contract.
- Seeds with non-zero upper 32 bits fall back instead of being truncated.
- Layered/source-driven/multilevel thermal and particle/layered/source-driven hydraulic configurations fall back to CPU.
- Materials/biomes are ID masks + procedural viewport palette, not a PBR asset library.
- Clipmaps are nested full grids (no skirts/morphing yet).
