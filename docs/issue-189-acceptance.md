# Issue #189 acceptance coverage

This matrix maps the Infinite Procedural World Slice 1 acceptance criteria from
issues #181 and #189 to frame-timing-independent automated checks or an explicit
manual desktop check. The named tests are the contract; this document does not
replace their assertions.

| Acceptance contract | Coverage |
|---|---|
| Create, identify, save, close, and reopen Infinite projects | Automated: `infinite_world_round_trips_all_authoritative_settings`, `infinite_project_save_close_and_reopen_preserves_identity_without_tiles`. Manual: QA section 13 verifies Project Home creation and the project chip. |
| Legacy projects remain bounded | Automated: `legacy_world_fields_migrate_to_bounded_heightfield`; the workspace bounded suites remain required. |
| Positive/negative traversal thousands of kilometres from the fixed origin | Automated: `infinite_planning_work_is_independent_of_distance_from_fixed_origin`, `infinite_repeated_long_traversal_keeps_planner_state_within_the_current_epoch`, `cpu_only_generators_and_large_signed_addresses_are_supported`, `infinite_sparse_page_renders_at_large_signed_coordinates`. |
| Deterministic eviction and regeneration | Automated: `negative_positive_tiles_regenerate_and_share_zero_halo_samples`, `infinite_gpu_tiles_are_order_independent_regenerable_and_unclamped`, `sparse_eviction_protects_current_coarse_page_and_regenerates`. |
| Zero-halo and bounded-halo seams | Automated: `negative_positive_tiles_regenerate_and_share_zero_halo_samples`, `bounded_blur_uses_declared_guard_and_crops_without_seams`, `infinite_gpu_blur_uses_guard_and_has_no_publication_seam`. |
| Unsupported full-field/basin/global work is visible and cannot run per tile | Automated: CPU/GPU `unsupported_graph_rejects_before_evaluation` / `infinite_gpu_rejects_unsupported_work_before_dispatch`, app `unsupported_infinite_graph_is_rejected_before_scheduling`, plus action/UI availability tests. Manual: QA section 13 checks the displayed owner/reason. |
| Coarse coverage remains while fine work is pending | Automated: `required_in_flight_work_gates_optional_refinement`, `resident_child_refines_and_unpublish_returns_to_current_root`, `infinite_default_horizon_draws_complete_multiring_coverage`, and the production app publication test. |
| Planner, evaluation, work, residency, and memory stay bounded by configuration | Automated: `infinite_repeated_long_traversal_keeps_planner_state_within_the_current_epoch`, `repeated_long_distance_replacement_never_accumulates_work_history`, sparse-atlas capacity tests, and profiler counter assertions. Manual: QA section 13 observes the counters during traversal. |
| Old document/plan/output/content results cannot publish | Automated: `every_content_identity_revision_rejects_an_old_completion`, `stale_packed_result_cannot_allocate_residency`, `stale_page_revision_falls_back_to_monolithic_height`. |
| Adjacent tiles agree at supported LODs and through supported halos | Automated by the CPU/GPU seam tests above using absolute shared samples and overlapping evaluation guards. |
| Camera-relative presentation is stable at large coordinates | Automated: `infinite_sparse_page_renders_at_large_signed_coordinates` and the app production render comparison. Manual: QA section 13 exercises continuous navigation. |
| Bounded creation, evaluation, demand, residency, rendering, and save/load remain supported | Automated by `cargo test --workspace`; bounded demand, tile-domain, renderer, document migration, project reset, and residency tests run unchanged. |
| Save files contain recipe/settings, not generated terrain | Automated: `infinite_project_save_close_and_reopen_preserves_identity_without_tiles` parses the JSON and recursively rejects runtime payload keys. |

## Required CI commands

CPU contract CI runs the Infinite core planner/scheduler, CPU sparse evaluation,
project I/O, and bounded document regressions. GPU parity CI runs sparse
residency, Infinite GPU tile evaluation, tile-stream rendering, and the app's
Infinite unit scenarios on the software Vulkan adapter. The normal workspace
suite remains the final bounded regression gate:

```text
cargo test --workspace
```

Export testing, sparse authoring, throughput targets, and expanding the
operation set are intentionally outside #189.
