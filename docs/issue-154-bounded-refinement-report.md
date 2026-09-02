# Issue #154 — bounded, supersedable refinement

Implemented on the `wozzits` branch against the #152 bottlenecks and the #153 logical
frame scheduler.

## Runtime design

Optional GPU Medium/Full evaluation is now an explicit scheduler-owned job containing
its origin frame, edit generation, evaluation trace ID, target quality, immutable
authored inputs, isolated candidate resources, compiled-operation cursor, completion
fence, progress counters, and publication state.

Each scheduler advance performs at most one existing safe unit:

1. create one physical plan texture;
2. encode and submit one live `CompiledTerrainPlan` operation; or
3. encode and submit the final presentation copy.

Every submission installs an asynchronous `Queue::on_submitted_work_done` fence. The
job will not encode its next operation until that fence resolves. If input supersedes
the job, unsubmitted work and candidate resources are dropped. An outstanding fence is
retained by the engine, because submitted GPU work is not cancellable, and blocks later
optional submissions until it resolves. The documented global optional queue-depth
bound is therefore **one refinement submission** across both live and superseded jobs.
There are no synchronous waits or height readbacks in the app's interactive path.

## Scheduling and publication

Required input sealing, application updates, interactive Draft evaluation, and the
presentation request remain ahead of optional refinement. The existing 8 ms logical
frame host-budget gate decides whether another safe unit may start. Generation and
pending-input checks run before each advance and again immediately before publication.

Before partial execution begins, the renderer retains a local copy of the last complete
height result. The candidate plan resources and compute graph remain private until the
final copy fence resolves. Publication then commits the resource candidate, graph,
quality, generation/evaluation identity, profiling state, and shared renderer view as
one current result. A stale, failed, resized/reset, or otherwise abandoned job cannot
replace the last complete presentation. Unsupported plan operations use the existing
background CPU refinement path.

## Instrumentation

Frame-correlated tracing now records refinement creation, unit progress, submission
queued/completed, supersession, candidate completion, publication, and failure. Each
record carries the logical frame and edit generation plus evaluation/job IDs, target
quality, completed/total units, per-unit host duration, elapsed job time where relevant,
and the global optional submission depth.

## Verification

- Headless-GPU regression coverage compares resumable Medium output with the existing
  complete evaluator on the Untitled6 production topology.
- The same test proves multiple submissions, a maximum optional depth of one, and no
  plan-resource or graph commit before the final fence/publication.
- Trace tests pin frame/generation/evaluation/job correlation, progress, and depth.
- Existing app logical-frame, compositing, terrain correctness, and integration tests
  remain unchanged and pass.

The #152 release timing probe remains the hardware-dependent performance authority.
This change does not invent new timing claims: it changes queueing and safe-boundary
scheduling while preserving Draft/Medium/Full definitions and compiler semantics.
