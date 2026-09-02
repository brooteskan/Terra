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

Open **File > Export...**, the command palette's Export command, or the toolbar's **EXPORT**
button. Choose PNG, Unsigned 16bit grayscale TIFF, R16, or R8, then check the fields to write.
Height, Slope, and Curvature are available from the terrain; additional scalar fields depend
on the active layers and their output settings. No field is required. Export is disabled
when nothing is selected or an export is already running.

Check **Open after export**, beneath the export directory controls, to open the completed
export's directory in your system file manager. It defaults to unchecked and can be changed while exporting.
Failed or cancelled exports do not open a folder.

The Resolution dropdown offers 256, 512, 1024, 2048, 4096, and 8192. Older custom
resolutions round up to the next listed size (clamped to this range) when the dialog opens.

The background exporter evaluates a snapshot of the current document at the chosen export
resolution using the existing CPU export evaluator. It writes one file per selected field
(for example, `height.png` or `wetness.r16`) plus `export_metadata.json` in the same directory.
Metadata is always included, even when Height is unchecked; it has no checkbox or toggle.
No LOD package is written.
PNG and TIFF are unsigned 16-bit grayscale. R16 is unsigned little-endian 16-bit raw;
R8 is unsigned 8-bit raw. Rows use increasing Z with X varying fastest, without a vertical flip.
Height is normalized over its evaluated minimum/maximum. Scalar maps already within 0–1
retain that range; other scalar maps are normalized over their evaluated minimum/maximum.
These integer formats quantize values. The versioned JSON metadata records the exported grid's
width/height, world dimensions in metres, `dx`/`dz` sample spacing, Y-up coordinates,
cell-centre sampling, row/column orientation, format, sample type, and raw R16 byte order.
Its `fields` list contains only this export's selected files, with field IDs, filenames,
original `min`/`max`, and the conversion `original_value = value_offset + stored_sample * value_scale`.
For Height, the recovered values are metres; other fields retain their original scalar units.
The conversion also covers masks that preserve 0–1 weights and constant-height maps.

Each successful export replaces `export_metadata.json`; older unselected field files in the
directory are not deleted. The metadata lists only the files from the latest export. Metadata
is published after those files, and a metadata write failure makes the export fail. A failed
or cancelled overwrite can leave partial field output, but not an old metadata file describing
newly changed data.

The custom height-pyramid writer remains available programmatically; it is not an option
in this field export dialog.
