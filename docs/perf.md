# Performance notes

- Prefer draft preview (256/512) while scrubbing; refine to full asynchronously.
- Cap interactive mesh upload (decimate to ≤512 edge).
- `terra-gpu::BufferPool` reuses STORAGE buffers across sim passes.
- `MemoryBudget` gates large allocations.
- Instrument with `profiling` scopes (`eval_step`, `upload_heightfield`, `rebuild_*`).
- Use RenderDoc on WGSL thermal/hydraulic dispatches; Tracy when `profiling` tracy feature enabled.
