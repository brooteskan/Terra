# Deterministic height-pyramid export

Terra's streaming height export changes only the source of terrain demand and
the publication consumer. Camera demand and complete export demand both pass
through `TerrainTileWorkScheduler`, `TerrainEvaluationDomain`, and
`GpuCompiledTileProducer`. Viewport work publishes current pages to
`GpuTileAtlas`; export work asynchronously maps the completed page and writes an
immutable package. Export never treats viewport residency as complete coverage.

## Traversal and revision identity

`TerrainPyramid::height_tiles` is the canonical order: levels coarse to fine,
then tile rows and columns. Export reconciles one complete level at a time and
does not request a child level until every parent-level evaluation, readback,
and payload write has completed. Every request carries one immutable document,
plan, output, and content stamp derived from the export snapshot. A stale or
unsupported compiled plan fails explicitly; the streaming path never switches
to an independent CPU terrain evaluation.

Global and basin-coupled prefixes use the compiled producer's immutable
complete-field checkpoint. A checkpoint is built once per level, plan boundary,
quality, and content stamp, then reused by all bounded suffix tiles in that
level. Cancellation retires submitted jobs without publishing mixed revisions.

## Package v1

The current generation is named by `height-pyramid.current` and stored under
`packages/<content-id>/`. A generation is immutable and contains `manifest.json`
plus content-addressed R32F payloads. The manifest contains no clock time,
absolute path, or random identifier. The package content ID is BLAKE3 over the
canonical manifest with an empty content-ID field; each payload has its own
BLAKE3 hash.

Payloads are little-endian IEEE-754 f32, row-major with X varying fastest. Each
is a fixed `(tile_size + 2 * halo)` square page. The interior begins at the halo
offset, partial right/bottom pages declare their valid extent, world-edge halos
clamp to the outermost sample, and unused page texels are zero. Samples represent
normalized cell centers and every level spans the same world rectangle in metres.

The writer stages a generation outside the visible package set, validates full
coverage and measured errors, writes and syncs its manifest, then publishes the
immutable content directory and current-generation pointer. Cancellation drops
only staging and leaves the previous generation readable.

## Geometric error and reading

For each non-root child sample, export reconstructs the immediate parent at the
child cell center with the same clamped bilinear convention as
`terrain_pyramid_error.wgsl`. The tile records the maximum absolute vertical
error in metres. `conservative_geometric_errors` then propagates descendant
error envelopes for screen-space selection. Root local error is zero.

`HeightPyramidPackage` validates schema, complete ordered coverage, extents,
payload sizes and hashes, errors, and package content identity. It can read an
exact page, reconstruct a cross-page region, or resolve the finest available
ancestor using normalized cell coordinates.

The byte-identical guarantee applies to unchanged content exported by the same
Terra build and GPU backend/device class. Cross-backend floating-point identity
is not claimed by v1.
