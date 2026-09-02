# Issue #155 Frame-Coordinator Consolidation Audit

This audit records the scheduling entry points after consolidating the editor on
`LogicalFrameCoordinator`. It is an implementation record for issue #155; the
current architectural contract remains `architecture.md` and
`progressive_invariants.md`.

## Callback and entry-point audit

| Entry point | Previous behavior | Consolidated behavior | Status / exception |
|---|---|---|---|
| Keyboard, pointer, wheel, modifiers | Callback-local flags plus redraw requests | Append ordered `InputEvent` values and request `Input` work | Migrated |
| Focus loss, cursor/capture loss | Ad-hoc cancellation/redraw | Append cancellation after already-recorded input; repeated loss is idempotent | Migrated |
| `WindowEvent::Resized` | Resize renderer and redraw in callback | Retain only the latest physical size and request `Resize` work | Migrated |
| `SurfaceError::Lost` / `Outdated` | Reconfigure and redraw from presentation callback | Record a recovery request; reconfigure during the next logical frame | Migrated |
| Device-loss callback | No event-loop-owned cleanup path | Send `RuntimeEvent::DeviceLost`; request controlled shutdown and invalidate work | Migrated |
| Window close / UI close | Immediate or scattered exit behavior | Request coordinator shutdown; cleanup occurs at `about_to_wait` | Migrated |
| UI `FrameUiOutput` and domain actions | Apply after rendering inside `RedrawRequested` | Queue output and apply it during application update in the next logical frame | Migrated |
| Project/home/template actions | Direct redraw/rebuild paths | Request typed coordinator work; evaluation begins only in a logical frame | Migrated |
| Structural immediate rebuild | Direct evaluation from the initiating path | Queue generation-stamped required evaluation with a ready deadline | Migrated |
| Debounced interactive evaluation | App-owned timer checks | `InteractiveEvaluation` deadline owned by the coordinator | Migrated |
| Deferred full-field work | App-owned settle timer | `DeferredFullField` and `FullFieldRefinement` generation deadlines | Migrated |
| Optional Medium/Full refinement | App-owned `*_not_before` timers | `OptionalRefinement` deadline plus coordinator priority/budget gate | Migrated |
| Background completion and animation | Scattered wake/redraw decisions | Typed `Completion` / `Animation` frame requests and coordinator wake policy | Migrated |
| `RedrawRequested` | Terrain preparation, UI action dispatch, recovery, redraw loops | Compose terrain + UI, present once, queue any new UI output | Migrated; presentation callback is intentionally bounded |
| Initial/failure splash and renderer handoff in `resumed` / boot polling | Requests redraw before a logical coordinator frame can exist | Retained as two bootstrap-only calls; the completed renderer is prepared before its handoff redraw | Justified exception: renderer/coordinator application state is not yet fully constructed |
| Coordinator presentation adapter | Converts prepared logical-frame demand to winit | The sole post-bootstrap `Window::request_redraw` call | Justified boundary adapter |
| Evaluation helpers used by unit tests | Directly step evaluation under a test harness | Compiled only for tests | Justified test seam; no production entry point |

All production calls to `run_eval_step_with_intent` now originate in
`about_to_wait` after a logical frame has begun. All production renderer
resize/reconfigure calls occur in that same logical-frame boundary, except the
initial renderer sizing performed while completing boot.

## Coordinator-owned policy

Frame requests are coalesced by edit generation and promoted by priority:
shutdown, input, resize, UI actions, required evaluation, completion/surface
recovery, animation, then optional refinement. An interactive request therefore
cannot be displaced by optional work. The coordinator owns four generation-
stamped deadline classes: interactive evaluation, deferred full-field evaluation,
optional refinement, and full-field refinement. The earliest current deadline
feeds the event loop's `WaitUntil`; absent animation, completion, or a deadline,
the loop returns to `Wait`.

Input accumulation has one ownership transfer: `seal` produces an immutable
snapshot and `into_events` consumes it. Events arriving after sealing remain in
the accumulator for a follow-up frame. Resize deliberately coalesces to the latest
size because intermediate sizes have no semantic ordering requirement.

## Publication, recovery, and diagnostics

Refinement publication retains the #154 rule: only a complete candidate whose
generation is still current may replace last-good resources. Superseded submitted
GPU work keeps its completion fence, while unsubmitted work is abandoned.

Recoverable surface loss requests reconfiguration. Device loss and shutdown bump
the generation/token, invalidate the CPU worker, supersede GPU refinement, clear
deferred evaluation, input, UI, uploads, resize/recovery, and presentation, then
abort the active frame before exiting. This prevents a frame or refinement job
from remaining logically in flight.

Detailed tracing is enabled by the profiler or `TERRA_FRAME_TRACE=verbose`.
Summary latency accounting remains available without retaining detailed events.
Detailed events without a real `FrameIdentity` are rejected and counted as trace
orphans; the profiler displays that count.

## Enforcement and regression coverage

- `logical_frame_discipline.rs` rejects evaluation/resize/redraw work in window
  callbacks, rejects application work in `RedrawRequested`, and enforces the
  direct-redraw allowlist.
- Coordinator unit tests cover priority promotion, stale generation deadlines,
  wake selection, presentation identity, and shutdown cleanup.
- Input tests cover post-seal retention, ordered edges, exactly-once consumption,
  and capture-loss ordering.
- Lifecycle tests cover resize coalescing, repeated focus/capture cancellation,
  device-loss shutdown, and shutdown cleanup.
- The scheduling-discipline integration test locks input/generation checks both
  before optional refinement advances and immediately before publication; #154
  engine tests retain one-unit stepping and transactional publication coverage.
- Trace tests cover generation-correlated visible latency, refinement progress,
  bounded storage, and orphan rejection/counting.
