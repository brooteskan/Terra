# Issues #148 and #150 — Untitled6 tree-aware GPU brushing

## Production-shaped regression fixture

The shared `terra_core::test_fixtures::untitled6_document` builder now models
the saved document topology that exposed #150:

- `Base` is a `SculptBase` with a 512x512 embedded sample payload, independent
  of the requested evaluation resolution.
- `Terrain / Semantic Sculpt` is a direct `SculptStrokes` layer.
- `Biomes` contains the isolated `Default biome / Filters / Volcano` tree,
  followed by empty isolated `Water`, `Beach`, `Grassland`, and `Rock` biome
  siblings.

Each empty biome's compiled `CompositeGroup` intentionally reads its seed twice
(`private_seed == child_output`). The resource validator permits this read/read
alias while continuing to reject read/write and write/write aliases.

## Automated acceptance coverage

- Focused validator tests cover the allowed read/read case and both rejected
  write hazards.
- A compiled-plan resource test builds the production topology, verifies all
  four empty-biome duplicate inputs, and verifies their output allocation is
  distinct.
- Raise and Pinch gestures run against both Base and Semantic Sculpt through the
  production-shaped tree.
- Three rapidly appended Draft dabs remain on the bounded compiled GPU plan,
  with no fallback, deferred operation, or readback.
- Medium and Full refinement complete on the GPU after the Draft gesture.
- Full-quality settled results are checked against a fresh CPU oracle under the
  named `UNTITLED6_INTERACTION` tolerance (0.02 m maximum absolute error and
  1e-5 normalized RMSE).
- The app regression resolves the selected Base through
  `ensure_shape_history_target`, sends actual `PanelAction::PaintSculptStamp`
  actions through `apply_actions`, and presents via the renderer.
- The app worker snapshot confirms no CPU job was submitted, started, completed,
  cancelled, failed, or published during Draft, Medium, or Full evaluation.
- Warm dabs do not recompile the structural plan, rebuild dependencies, or walk
  the authored tree.

## Release timing record

Measured 2026-08-20 with the ignored release test
`engine::smoke_tests::untitled6_release_timing_probe_2048_4096`.

Adapter: AMD Radeon RX 7900 XTX, DX12, driver 32.0.31035.1003, vendor 4098,
device 29772.

| Resolution | Cold complete | Warm interactive | Warm upload | Warm logical workgroups | Warm dispatch | Publish | Reuse | Defer | Readback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048² | 134.086 ms | 3.503 ms | 7,056 B | 864 | 24 | 5 | 1 | 0 | 0 B |
| 4096² | 477.886 ms | 4.745 ms | 26,896 B | 2,904 | 24 | 5 | 1 | 0 | 0 B |

The cold evaluations also completed fully on the GPU with zero readback. Cold
plan compile times were 230 us at 2048² and 155 us at 4096². Each warm edit was
a cache hit with two patched operations and no additional plan compile,
authored-tree walk, or dependency build. These adapter-specific wall times wait
for submitted GPU work and are recorded for comparison rather than asserted in
CI.
