# Progressive Renderer Invariants

Rules that must hold on every shipping build. Violations are release blockers.

## NEVER

- NEVER perform terrain evaluation or other substantial editor work directly in
  a winit input callback; record input and request a logical frame.
- NEVER coalesce ordered pointer samples, button edges, modifier transitions, or
  cancellation across logical input snapshots.
- NEVER reset progressive accumulation on UI chrome hover, panel focus, or menu open alone.
- NEVER block the viewport on CPU terrain evaluation — last-good GPU textures stay visible.
- NEVER treat mouse-button-down without a scene change as meaningful interaction for refinement.
- NEVER drop GPU height/normal textures synchronously on resize — retire after N frames.
- NEVER share tile cache pages across different layer identities without a generation handle.
- NEVER mark basin-coupled edits (SPE, amplify, river network) as strictly local invalidation.
- NEVER couple terrain `PreviewQuality` and path-tracing sample budgets in a single knob.
- NEVER `Maintain::Wait` on GPU timestamp readback on the interactive path.
- NEVER publish a Medium/Full candidate before its final submission fence resolves
  and its edit generation is revalidated.
- NEVER allow more than one optional refinement submission to remain in flight,
  including a fence retained from a superseded job.

## ALWAYS

- ALWAYS consume sealed input and required interactive Draft work before optional
  Medium/Full refinement.
- ALWAYS retain input received after snapshot sealing for a follow-up logical frame.
- ALWAYS return the event loop to `ControlFlow::Wait` when no logical frame work remains.
- ALWAYS bump the correct scene version counter for each invalidation reason.
- ALWAYS call `notify_invalidation` when terrain, materials, geometry, lighting, or viewport mode changes.
- ALWAYS drive `EditorRefinementState` from meaningful scene changes (camera, terrain, lighting, edits).
- ALWAYS set `spp_this_frame = 0` in `Converged` refinement state for progressive PT modes.
- ALWAYS reactivate adaptive sampling tiles on accumulation reset.
- ALWAYS keep Fast / Raster modes available without progressive overhead.
- ALWAYS map viewport mode to exactly one presentation backend (no dual progressive enable flags).
- ALWAYS preserve temporal history when only lighting or atmosphere changed (signature policy).
- ALWAYS replace renderer height/normal slots and evaluator working textures with the 8×8
  resident baseline on document reset; existing size checks restore the next project dimensions.
- ALWAYS document measured GPU pass timings in profiling reports before claiming performance targets.
- ALWAYS fall back to the monolithic height texture on tile-stream page misses.
- ALWAYS abandon unsubmitted refinement work when input advances the edit generation;
  an already-submitted unit is non-cancellable and its fence remains tracked.
- ALWAYS advance optional GPU refinement by one safe unit only after required Draft
  work and the logical-frame optional-work budget gate.
