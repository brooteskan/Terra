# Changelog

All notable changes to Terra will be documented in this file.

## [Unreleased]

### Fixed
- Progressive Draft-to-Full evaluation now resamples carried auxiliary mask fields to the active grid, preventing boundary panics in masked CPU filters such as Crater.
- Evaluation panics are reported as recoverable layer failures, unexpected worker disconnects restart the worker, and the editor keeps the last-good terrain visible with a persistent Retry status.
- Creating a filter or reopening a project with populated biome sections now expands the relevant `Biomes -> biome -> Filters` hierarchy paths, keeping filters discoverable in Layers even when another layer was saved as selected.
- Long inspector drop-downs, including the Effect Filter type picker, are now viewport-bounded and mouse-wheel scrollable instead of rendering rows beyond the screen and selecting an unintended filter.

## [0.1.0-beta.1] - 2026-08-14

First beta binary release candidate for Windows.

### Fixed
- Default material and biome height ranges now save as finite JSON values, and projects previously written with `null` open bounds load and normalize on the next save.
- Incremental dirty-tile synchronization now refreshes neighboring edge and corner ghosts, and seam metrics detect stale halos in both directions at every configured depth.
- Layered hydraulic, debris-flow, and thermal stacks now preserve distinct canonical debris and fine-sediment inventories across aux and cache handoffs.

### Added
- Phase 11 geological integration: shared typed AuxMaps (bedrock / sediment / soil /
  lithology), `ScaleBand::MultiScale` + `LayerKind::scale_band`, field invalidation
  graph with iterative guard, `TerrainStatistics`, `LandscapeStyle` presets, and
  realism benchmark worlds (Alpine, Desert Mesa, Badlands, Old/Young Mountains,
  Dune Field, Coastal)
- Rebuilt New World templates from scratch via `WorldTemplate` + shared
  cause→effect process chain (materials → evolution → hydro → meso → detail)
- MIT licensing and CONTRIBUTING guide
- Open-source packaging pass (README accuracy, user-facing docs for workflow / creating terrain)
- Drainage-conditioned multi-scale terrain amplification (`analyze::amplify_terrain` /
  Geomorphic Detail): macro/meso/micro metre bands, cascaded flow-aligned patterns,
  outputs for fine flow / micro-channel / ridge breakup / fine erosion

### Changed
- Split `terra-app` shell into lifecycle / eval / project / paint / shapes / actions modules
- Split UI `PanelAction` into `terra-ui::actions`; inspector and hierarchy live under module directories
- Split `LayerKind` param families under `terra-core::layer::kinds`
- Split `TerrainDocument` into document / migrate / session modules
- Shared geometry helpers in `terra-core::mask::geom`
- Layer type registry documented as catalog source of truth; tool catalog points contributors there
- Geomorphic Detail upgraded from isotropic-aligned polish to anti-soup structured amplification

### Removed
- Retired the unused terra-core evaluation graph, operator adapter, and duplicate GPU-support
  metadata; CPU evaluation remains owned by `StackEvaluator` and GPU planning by `terra-gpu`.
- Retired the unconsumed `TerrainContext` bridge and parallel `terrain_eval` state, derived-cache,
  diagnostics, mode, determinism, and tiling scaffolding; live field helpers remain under
  `fields`, `heightfield`, and `layer`.
- Retired the unexecuted per-layer terrain work scheduler and vector tile executor; viewport
  residency now records only payload-backed tiles produced by the live final-output upload path.
- Retired the delegating `TerrainPipelineExecutor` and fictional ten-stage rebuild diagnostics;
  final and export CPU evaluation now call the authored-tree `StackEvaluator` directly.
