# Editor overview

How the Terra shell is laid out and where common actions live. UI details may change while the product is unfinished.

## Shell

| Area | Role |
|------|------|
| **Menu bar** | File / Edit / View, project chip, Export |
| **TOOLS rail** | Workspaces: Sculpt, Biomes, Filters, Mask, Simulation, Surface, Objects |
| **Left tools palette** | Catalog of tools for the active workspace |
| **Center viewport** | 3D view, overlays, tool-mode bar (Move / Sculpt / Mask / Biome / …) |
| **Right hierarchy + inspector** | Terrain stack and selected-layer properties |
| **Bottom dock** | Preview resolution / quality / backend, build progress, cancel |

Workspaces only change emphasis and available tools. The full hierarchy stays editable.

## Floating windows

Useful panels include Mask Editor, Recipes / Pipeline, Export, 2D Preview, Profiler, History, Bookmarks, Quick Add, and the Command Palette. Open them from View or the command palette as needed.

## Inspector

The inspector shows the selected layer or group:

- Shared chrome: name, enable, lock, solo, opacity, blend, distribution
- Kind-specific parameters (Simple vs Advanced where available)
- **Apply Where** for ops inside biomes
- **Resolution** shows the selected source's stored/imported detail, the active
  Draft/Medium/Full/Export evaluation grid, and whether Terra resamples,
  rasterizes, generates, processes, or simulates at that grid size. A larger
  evaluation grid does not add source detail to a smaller fixed raster.

Prefer changing blend and distribution before inventing extra layers when a single contribution should simply be weaker or more localized.

## File actions

- **New / Open / Save / Save As / Close** — File menu and Project Home
- **Recents** — Project Home
- Dirty confirmation appears before discarding unsaved work

Projects are JSON documents. See [Creating terrain](creating-terrain.md) for a first-session walkthrough.

## Preview and builds

Terra evaluates progressively. The viewport may lag or show approximate results while a build is running. Use the bottom dock status and **View → Profiler** when diagnosing sluggish frames.

If a build fails or remains incomplete, check Terra's persistent log. On Windows
it is stored under `%LOCALAPPDATA%\Terra\logs` as a timestamped
`log-YYYY-MM-DD_HH-MM-SS-mmm.log` file; the exact path is also printed during
startup. Terra keeps the six most recent launch logs. Set `RUST_LOG=debug` before
launch for more detail. If file logging is unavailable, Terra continues with
console-only diagnostics.

## Export

Start Export writes a deterministic, content-addressed multiresolution R32F height package
through the same compiled GPU tile producer used by viewport streaming. The v1 package includes
measured geometric errors and seam metadata; materials, vegetation, and engine adapters are not
yet part of this streaming slice. Unsupported compiled operations fail explicitly.
