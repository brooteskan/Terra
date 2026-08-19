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
| `authoring.sculpt-strokes` | Per-sample stroke kinds, Smooth/Pinch/Coastline, Flatten, and distance stamps, supported blend/mask | 0.001 | 0.00001 |
| `mask.simple` | One Constant, Height, or Slope Multiply entry without asset operations | 0.001 | 0.0001 |
| `noise.value` | Value noise with a 32-bit seed | 17.0 | 0.23 |
| `noise.perlin` | Perlin, 1-12 octaves, reproducible 32-bit seed stream | 0.001 | 0.00001 |
| `noise.fbm.value` | Value fBm, 1-12 octaves, reproducible 32-bit seed stream | 0.001 | 0.00001 |
| `noise.fbm.perlin` | Perlin fBm, 1-12 octaves, reproducible 32-bit seed stream | 0.001 | 0.00001 |
| `noise.ridged.value` | Value ridged MF, 1-12 octaves, reproducible 32-bit seed stream | 0.001 | 0.00001 |
| `noise.ridged.perlin` | Perlin ridged MF, 1-12 octaves, reproducible 32-bit seed stream | 0.001 | 0.00001 |
| `noise.domain-warp` | Perlin domain warp, 1-12 octaves, reproducible 32-bit seed stream | 0.001 | 0.00001 |
| `filter.blur` | Default outer composite; radius/iteration fixture | 2.1 | 0.0055 |
| `effect.smooth` | Smooth, default outer composite | 1.8 | 0.03 |
| `effect.inflate` | Inflate, default outer composite | 2.7 | 0.028 |
| `effect.denoise` | Denoise (bilateral), default outer composite | 2.2 | 0.038 |
| `effect.add-set` | Add and Set pointwise remaps, default outer composite | 0.001 | 0.00001 |
| `effect.deflate` | Amount-limited greyscale erosion, default outer composite | 0.001 | 0.00001 |
| `effect.curve` | Exact entering-field range reduction, default outer composite | 0.001 | 0.00001 |
| `effect.cutoff` | Exact entering-field range reduction, default outer composite | 0.001 | 0.00001 |
| `effect.pointwise-procedural` | TerraceSimple, Shore, Blocks, ZeroEdge, Squeeze, Perlin-family noise, scatter, Hexagons, and authored absolute border/flatten targets | 0.001 | 0.00001 |
| `effect.spatial` | DirectionalBlur, AngleBlur, Balloon, Crater, TerraceSteep, and radius-one SpikeRemoval | 0.02 | 0.0002 |
| `effect.warp` | Swirl and Distortion with a reproducible 32-bit seed | 0.05 | 0.001 |
| `filter.terrace` | Default outer composite | 8.5 | 0.10 |
| `simulation.thermal` | Non-layered, constant hardness, no weathering extension | 3.2 | 0.05 |
| `simulation.hydraulic` | Base transport, no sources, particles, layers, or post-effects | 3.0 | 0.03 |
| `simulation.river-carve.d8` | D8 routing, no guide mask, bounded bank radius | 0.001 | 0.00001 |
| `simulation.river-carve.d-infinity` | D-infinity authored mode approximated by D8 preview, no guide mask, bounded bank radius | 6.0 | 0.02 |
| `shape.mountains` | Mountains with reproducible 32-bit seed streams | 10.0 | 0.005 |
| `shape.dunes` | Default transport controls, 2-4 octaves, reproducible 32-bit seed stream | 36.0 | 0.78 |
| `shape.canyons` | Canyons with a 32-bit seed | 0.001 | 0.00001 |
| `shape.mesa` | Mesa with a 32-bit seed | 0.003 | 0.00001 |
| `shape.volcano` | Volcano with a 32-bit seed | 0.004 | 0.00001 |
| `shape.uplift` | Uplift with reproducible 32-bit seed streams | 0.1 | 0.00002 |
| `shape.plateau` | Pointwise input remap | 0.001 | 0.00001 |
| `island.archipelago` | Archipelago with reproducible 32-bit seed streams | 0.03 | 0.000005 |
| `island.atoll` | Atoll with reproducible 32-bit seed streams | 0.001 | 0.00001 |
| `island.volcanic-high` | VolcanicHighIsland with a 32-bit seed | 220.0 | 0.10 |
<!-- END GENERATED GPU PARITY MATRIX -->

## Remaining intentional gaps

- GPU hydraulic omits full neighbor water/sediment gather (atomic-free preview).
- GPU value noise is a portable hash approximation, not bit-identical to CPU.
- Multi-entry distributions, non-Multiply combines, mask asset operations, missing assets, and Noise/Curvature mask sources fall back to CPU.
- OpenSimplex/Worley generators, OpenSimplex fBm/ridged configurations, non-default dune transport controls, and the EffectFilter variants not named in the matrix fall back to CPU. They may return only after gaining a named parity contract.
- RiverCarve preview uses iterative D8 routing for both authored routing modes and omits guide-mask bias; guided configurations fall back to CPU. D8 is exact-height class on the monotone drainage fixture, while authored D-infinity has its own bounded approximation contract. CPU priority-fill routing and auxiliary flow/accumulation/wetness fields remain authoritative for export and downstream auxiliary consumers.
- Seeds/derived octave streams outside 32 bits and noise configurations above 12 octaves fall back instead of being truncated.
- Layered/source-driven/multilevel thermal and particle/layered/source-driven hydraulic configurations fall back to CPU.
- SculptStrokes previews the per-sample stroke kinds (Raise/Lower/Ridge/Valley/Inflate/Terrace/Noise, the distance stamps, and the aux-only kinds) plus Smooth, Pinch, and Coastline, whose base-3x3 pull the stamp kernel reads directly from the layer input (Pinch at a 1.25 overdrive; Coastline a lower-and-blend toward that mean under a weight gate), and Flatten, whose footprint mean a reduce/resolve pair measures against the running field before the stamp path applies it (the stroke run is segmented at each Flatten). Every stroke kind now previews on the GPU. It is a height-only preview — the CPU eval remains authoritative for the protection/uplift/hardness/sediment/edit-region aux, so a stroke layer resumes on CPU whenever an enabled downstream layer consumes that aux.
- Materials/biomes are ID masks + procedural viewport palette, not a PBR asset library.
- Clipmaps are nested full grids (no skirts/morphing yet).
