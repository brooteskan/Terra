# Issue #152 — Base stroke/refinement latency

## Implemented path

The Base brush now uses the logical-frame coordinator as the scheduling authority.
Input snapshots and required interactive work run before optional refinement. Stroke
release only finalizes state, records the release, arms an 80 ms idle grace period,
and requests later work. Optional work is rejected while input is pending, before
the grace deadline, or when its generation no longer matches the current edit.
The last complete generation remains presented until a newer candidate is accepted.

After a fully valid Full-quality Base result exists, a bounded follow-up stroke keeps
that realization resident and evaluates the changed region at Full quality. This
avoids the former Draft-to-Full resource transition on every stroke. Other edit kinds,
unbounded edits, CPU-fallback stacks, and cold projects retain the normal Draft-first
policy.

## Trace and diagnostics

The bounded in-memory Base trace correlates logical frame, edit generation, and
evaluation ID across OS receipt, snapshot sealing, tool update, release/follow-up
press, plan acquisition, evaluation request/submit, candidate accept or stale reject,
presentation request/surface present, and delayed GPU completion.

The profiler reports rolling input-to-visible, release-to-next-press, and
release-to-refined p50/p95/max.
Each submitted evaluation carries quality, intent, resolution, cold/warm state,
selected operations, dirty texels, workgroups, uploads/readback, scratch allocation
counts, and CPU resource-preparation, capability-preflight, encoding, and queue-submit
spans. Over-budget host frames are attributed to their active logical-frame phase.
Whole-evaluation and whole-presentation GPU timestamps use asynchronous map callbacks
and `Maintain::Poll`; the interactive path never calls `Maintain::Wait`.

## Reference benchmark

Measured 2026-08-20 with the ignored release probe
`engine::smoke_tests::untitled6_release_timing_probe_2048_4096` on AMD Radeon RX
7900 XTX, DX12, driver 32.0.31035.1003. Each baseline sample deliberately switches
Draft then Full to reproduce the old per-stroke transition. Each resident sample is
a bounded Base edit against an already-valid Full realization. GPU completion waits
exist only in this benchmark.

| Resolution | Samples | Draft→Full p50 | Draft→Full p95 | Resident Full p50 | Resident Full p95 | p95 reduction | Resident max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048² | 10 / 20 | 102.067 ms | 130.871 ms | 2.852 ms | 4.148 ms | 96.8% | 4.701 ms |
| 4096² | 10 / 20 | 536.568 ms | 613.579 ms | 6.021 ms | 6.557 ms | 98.9% | 6.595 ms |

At 4096² the cold trace attributed 101.900 ms to resource preparation and
382.343 ms to command encoding, versus 0 us and roughly 3 ms respectively on the
warm regional result. The final warm sample touched 7,140 of 16,777,216 texels,
uploaded 27,224 bytes, reused one contribution, and performed no readback.

Compiled mask evaluation also retains its four resolution-keyed scratch textures.
The GPU regression observes four allocations on the first masked evaluation and zero
allocations plus one reuse on the warm evaluation, while checking the result against
the CPU oracle.

## Boundary for #154

Cold Full evaluation remains monolithic once encoded and submitted (524 ms in this
run, with a 613.579 ms transition maximum). A cold project, resolution change, or
unbounded/global invalidation can therefore still delay later GPU work. Splitting that
preparation/submission across frames, and superseding not-yet-submitted chunks, is the
resumable GPU-job work tracked by #154; #152 does not claim cancellation of submitted
command buffers.

## Verification

- App regression: production-shaped Untitled6 Base gestures stay GPU-only, preserve
  pixel-equivalent shared/regional presentation, and a follow-up bounded stroke keeps
  Full resident with a warm regional evaluation.
- GPU regression: constant/height/slope masks match the CPU oracle and prove scratch
  allocation reuse.
- Logical-frame regressions: input denies optional work, budgets gate optional starts,
  follow-up input receives a new frame, and latency samples close only on the matching
  generation.
