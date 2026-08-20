# Issue #148 — Untitled6 tree-aware GPU brushing

## Regression fixture

The shared `terra_core::test_fixtures::untitled6_document` builder owns the
representative document and stable handles used across the core, GPU, and app
tests:

- `Base` (`SculptBase`)
- `Terrain / Semantic Sculpt / SculptStrokes`
- `Biomes / Default biome` (isolated `CopyInput`) `/ Filters / Volcano`

It also exposes empty-isolated and masked/non-default-composite variants. All
three variants compile through the same structural-plan authority.

## Automated acceptance coverage

- Raise and Pinch on Base and the existing SculptStrokes layer.
- Three rapidly appended dabs evaluated as one warm generation.
- Visible app presentation during the interactive-local evaluation.
- No structural-plan compile, authored tree walk, dependency build, CPU job
  submission, or GPU readback caused by the supported tree.
- Per-operation incoming/output scope and concrete texel region, with
  dispatched/skipped/reused/deferred disposition.
- Bounded upload and logical-workgroup counts, including reuse of the
  input-independent Volcano contribution.
- Full-field settled GPU result against a fresh CPU oracle under the named
  `UNTITLED6_INTERACTION` tolerance (0.02 m maximum absolute error and 1e-5
  normalized RMSE).
- Existing latest-generation-wins and deferred-FullField tests remain the
  mouse-up/stale-publication and downstream-global-policy ratchets.

The profiler overlay reports generation/path and precise fallback diagnostic,
first-visible and settled latency, cumulative plan compile time/count, authored
walk/dependency counts, GPU operation disposition/workgroups/upload/readback,
and CPU submit/cancel/complete/publish counters.

## Release timing record

Measured 2026-08-20 with the ignored release test
`engine::smoke_tests::untitled6_release_timing_probe_2048_4096`.

Adapter: AMD Radeon RX 7900 XTX, DX12, vendor 4098, device 29772.

| Resolution | Cold complete | Warm interactive | Upload | Logical workgroups | Dispatch | Publish | Skip | Reuse | Defer | Readback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048² | 144.032 ms | 8.083 ms | 7,104 B | 432 | 12 | 5 | 0 | 1 | 0 | 0 B |
| 4096² | 490.371 ms | 20.474 ms | 26,944 B | 1,452 | 12 | 5 | 0 | 1 | 0 | 0 B |

These are adapter-specific wall times after waiting for submitted GPU work.
They are recorded for comparison and are not asserted in CI.
