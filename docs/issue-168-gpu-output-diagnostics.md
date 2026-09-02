# GPU output identity and regional integrity diagnostics (#168)

Issue #168 adds evidence collection around the GPU evaluation-to-presentation seam. It is
shadow instrumentation: diagnostics are recorded after the existing evaluation and
presentation decisions, and do not accept, refuse, retain, reschedule, or alter a terrain
candidate.

## Correlated identity chain

Every GPU result carries a `GpuTerrainOutputIdentity` from final-field selection through the
renderer. The identity includes the logical frame, edit generation, evaluation, plan revision,
requested and actual quality, intent, selected and expected-final fields, physical allocation,
plan-resource incarnation, device/output-resource incarnation and ping/pong slot, extent,
whole/patch coverage and expected patch base, completeness, invalidation kind, output ID, and
last-write submission serial/completion state.

The existing fixed-capacity frame trace attaches that identity to evaluation selection and to
accepted or refused presentation records. Presentation records add the requested and actual
mode (`Shared`, `FullCopy`, `RegionalCopy`, or `CpuUpload`), requested and actual rectangle,
candidate decision, renderer baseline before and after, local-slot coherence and epoch, and the
last full-baseline generation. Per-evaluation GPU statistics provide cold/warm state, dirty
texels, and dispatched, reused, deferred, published, materialized, and skipped operation counts.

The compact trace is always retained for the preceding 2,048 events. Detailed durations remain
controlled by the profiler or `TERRA_FRAME_TRACE=verbose`. The first detected violation is
latched, logged once with expected and actual identities, and followed by the retained preceding
trace. Later symptoms cannot replace the first-cause code.

Stable transition diagnostic strings are:

- `non_final_current_output`
- `regional_without_complete_baseline`
- `plan_revision_mismatch`
- `device_generation_mismatch`
- `resource_incarnation_mismatch`
- `source_renderer_extent_mismatch`
- `regional_base_mismatch`
- `source_generation_stale`
- `source_generation_superseded`
- `partial_source_incomplete`
- `height_normal_lineage_mismatch`
- `outside_dirty_region_changed`

Stable candidate-decision strings are `accepted`, `refused_stale_generation`, and
`refused_no_output`.

The local-baseline checks apply only to an actual regional copy. A shared presentation or a
full-copy promotion replaces the complete visible source and does not consume the renderer's
prior local baseline.

## Asynchronous integrity probe

For a regional copy, a compute pass compares deterministic samples outside the actual copied
rectangle with the samples retained from the preceding presentation. The GPU reduces the result
to four words: failure, maximum delta, encoded first failing sample, and compared count. Four
readback slots are mapped asynchronously and advanced with `wgpu::Maintain::Poll`; the
interactive path contains no `Maintain::Wait` and performs no full-field readback. If all slots
are busy, the result readback is skipped rather than blocking.

`TERRA_GPU_INTEGRITY_PROBE` controls the sample count:

- unset: 64 samples in debug-assertion builds, disabled in release builds;
- `off` or `0`: disabled;
- a positive integer: that many samples, clamped to 256.

With the probe disabled, presentation creates no probe texture view, bind group, command
encoder, submission, or readback. Enabling it creates one fixed baseline buffer, one 16-byte
reduction buffer, one uniform buffer, and four 16-byte staging buffers; each presentation uses
one small compute dispatch, and only regional presentations compare against the prior samples.

## Measured disabled-path overhead

Measured on 2026-08-21 in an optimized Windows release test with 1,000,000 compact events:

```text
event_bytes=1048
capacity=2048
baseline_ns_per_event=45.0
total_ns_per_event=60.4
incremental_ns_per_event=15.4
```

The ring's event payload is therefore bounded to 2,146,304 bytes (about 2.05 MiB), excluding
small container bookkeeping. The release-default GPU probe cost is zero. Reproduce the CPU
measurement with:

```text
cargo test -p terra-app --release --lib compact_trace_overhead_probe -- --ignored --nocapture
```

The timing is a local micro-measurement rather than a cross-machine performance guarantee; the
fixed capacity and allocation-free steady state are structural bounds.

## Regression and fault coverage

- Renderer invariant tests cover each stable code, shared/full replacement semantics, and
  regional baseline, plan, extent, generation, resource, completeness, and lineage faults.
- Renderer integration tests cover shared to regional and regional to regional transitions and
  deliberately corrupt a deterministic outside-region point to verify asynchronous detection.
- Engine tests verify warm patches chain their expected base while physical allocation and
  resource incarnation remain stable, and verify reset/cold realization advances incarnation.
- The app stress test performs 72 separately evaluated and presented Raise dabs, crosses the
  8/16/32/64 growth boundaries, inserts complete refinement publications, and asserts Full
  quality, full resolution, GPU residency, and no transition false positive after every dab.
- Frame-trace fault injection verifies a wrong-field transition remains the first diagnostic
  even when a later pixel-probe failure is reported.

These diagnostics intentionally do not change publication behavior, terrain-plan compilation,
dirty propagation, quality policy, or scheduling. Enforcement and last-good retention remain
outside issue #168.
