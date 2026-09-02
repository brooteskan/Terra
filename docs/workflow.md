# Workflow structure

Terra is a **layer-based** landscape editor. You compose height, coverage, materials, and simulations by stacking layers (and groups) rather than wiring nodes.

## Mental model

1. **Height** comes from Shape Layers and sculpt brushes, blended bottom → top.
2. **Where** a layer applies is controlled by **Distribution** (mask rules / paint) and, inside biomes, **Apply Where**.
3. **Biomes** package *what* lives where: filters, materials, objects, and local sims under a biome container.
4. **Simulation** and **World Rules** refine or condition the stack; they are optional authoring aids, not a separate product mode.

Evaluation walks the Terrain hierarchy from bottom to top. Disabling, soloing, or changing opacity/blend on a layer changes how much it contributes.

## Hierarchy (what you see)

Under the Terrain root, content is organized roughly as:

1. **Biomes** — biome containers and their sections
2. **Biome Layers** — painted placement / splat weights
3. **Shape Layers** — landforms, stamps, paths, heightmap imports, sculpt strokes
4. **Mask Layers** — mask authoring bound into the stack
5. **Simulation Layers** — world / global physical sims

Inside each **Biome**:

- **Filters** — terrain filters and most biome-owned effects
- **Materials** — surface / material rules (early / placeholder)
- **Objects** — scatter / vegetation / props (early / placeholder)
- **Local Simulations** — sand, fluid, debris-style locals

Category folders organize the UI and where **Add** routes new layers. They do not create a second evaluation pipeline.

## Layers

Each layer typically has:

| Property | Role |
|----------|------|
| Enabled / locked / solo | Visibility and edit safety |
| Opacity | Strength of the contribution |
| Blend mode | How height (or weights) combine with what’s below |
| Distribution | 0–1 coverage field (mask stack) |
| Kind | What the layer generates or transforms |

Common height blend modes include Replace, Add, Subtract, Multiply, Minimum, Maximum, Interpolate, Height Blend, and several smooth variants. Generators often default to **Add**; filters and sims often use **Replace**.

## Masks and Distribution

- **Mask assets** — named project masks (painted, height, slope, flow, climate channels, and so on) with optional paint buffers and post-ops.
- **Distribution** — the ordered coverage stack on a layer or group. This may reference mask assets or use distribution nodes (fill, noise, height, slope, paint, polygon, blur/invert/expand, All/Any groups, …).
- Layer contribution is gated by **opacity × distribution**.

Masks answer *where*; the layer kind answers *what*.

## Biomes: WHAT vs WHERE

| Piece | Role |
|-------|------|
| **Biome definition** | WHAT — name, color, packaged layers, placement rules, priority |
| **Biome group distribution** | WHERE — coverage of the biome container |
| **Biome Layers** | WHERE — painted / edited splat weights across biomes |
| **Apply Where** | WHERE — per-operation placement inside a biome (Entire Biome, height/slope range, near water, painted restriction, custom conditions, advanced mask, …) |

Child ops inherit the biome’s effective placement, then further restrict with their own Apply Where.

## Workspaces

Workspaces on the TOOLS rail change tools, hierarchy emphasis, and overlays. They are **task focus**, not wizard steps — switch freely.

| Workspace | Focus |
|-----------|--------|
| Sculpt | Height brushes, landforms, shapes |
| Biomes | Biome placement, paint, definitions |
| Filters | Terrain filters inside biomes |
| Mask | Mask paint / edit and coverage stacks |
| Simulation | Global sims and scenarios |
| Surface | Materials |
| Objects | Scatter / vegetation / props |

## Shape tools vs Shape Layers vs Shape objects

- **Brushes / editor tools** — raise, lower, smooth, flatten, terrace, stamps, paths, mask/biome paint, etc., usually targeting Base or a sculpt/shape layer.
- Sculpt brush cards normally remain available regardless of the current selection. The first dab creates a new **Semantic Sculpt** layer, while selecting an existing Semantic Sculpt layer continues editing it.
- **Foundation is the exception:** selecting it explicitly targets its raster, so brushes the Foundation cannot represent are dimmed and cannot be armed.
- **Shape Layers** — stack entries that generate or stamp height (procedural, stamps, polygon, path, heightmap import, sculpt strokes, …).
- **Shape objects** — world-design primitives (coastline, ridge, valley, river, plateau, …) that compile into authoring constraints.

## World Rules and Simulation Scenarios

- **World Rules** — condition-driven intentions such as “snow above 1200 m”: scope (entire world, selected biomes, exclusions, painted restriction), phase, and effects. They compile into placement/distribution rather than a separate solver UI.
- **Simulation Scenarios** — optional containers (domain, sources, passes, quality) that group Simulation Layers. Not required for everyday sculpting.

## Project document (high level)

A saved project is a `TerrainDocument`: layer stack, mask assets, biome library and biome layers, landscape blueprint, shape objects, world rules, optional simulation scenarios, and preview/export resolution settings.

See [Creating terrain](creating-terrain.md) for a practical pass through this model, and [Editor overview](editor.md) for chrome and file actions.
