<p align="center">
  <img src="assets/logo.png" alt="Terra" width="420" />
</p>

<p align="center">
  <img src="assets/preview.png" alt="Terra editor preview" width="800" />
</p>

# Terra

> [!CAUTION]
> ## Not ready for production
>
> **Terra is an early, unfinished work-in-progress.** Do not use it for shipping games, film pipelines, or any production workflow. Builds may crash, corrupt projects, or change behavior without warning. APIs, file formats, and UI are unstable.
>
> Large parts of the editor are incomplete or only sketched in:
>
> - **Real-time terrain visualization** — progressive Draft→Full preview with a hybrid GPU+CPU path exists, but live feedback still needs major refinement; expect laggy, approximate, or stale results
> - **Export pipeline** — a heightmap-oriented export package writer exists, but it is not production-ready and the main Export CTA is still blocked
> - **Objects** — vegetation scatter exists; general prop/object placement is largely placeholder
> - **Materials** — material IDs / strata / hardness and viewport tinting exist; this is not a full shading or surface-authoring pipeline
>
> Treat everything you see as experimental scaffolding. Star the repo, experiment locally, and contribute if you like — but do not depend on Terra for real work yet.

**Terra** aims to be a **free and open-source**, layer-based terrain and landscape creator — heavily inspired by [World Creator](https://www.world-creator.com/), with a familiar stack of layers, masks, and tools rather than a node-graph UI.

The goal is a real-time authoring environment where artists compose landforms, biomes, masks, and simulations on a progressive heightfield, then export results for games, film, and other pipelines.

## Vision

- **Layer-and-mask workflow** — paint, stack, and blend terrain like a digital landscape canvas
- **Free & open source** — MIT; anyone can use, study, and contribute
- **Artist-first** — shape tools, biomes, masks, and world rules without wiring a graph
- **Real-time preview** — progressive / multi-resolution evaluation with optional GPU compute

## Current status

Early development, but past empty scaffolding. A usable **layer-stack editor** is in place: heightfields, masks/distributions, biomes, shape tools, CPU evaluation, progressive Draft→Medium→Full preview, and a wgpu viewport with a hybrid GPU path for many generators and filters.

Still incomplete or experimental: export readiness/UX, materials shading, object/prop scattering beyond basic vegetation, volumetric overhangs/caves, and several simulations that are CPU-heavy or only partially on GPU. Product “Regions” were removed in favor of a single World Creator–style stack (biomes, masks, and world rules own placement).

See the disclaimer above before relying on any of this.

## Features (in progress)

- Layer stack with blend modes, groups, biome containers, masks, and distribution / placement rules
- Shape layers and brushes (procedural landforms, stamps, paths, polygons, sculpt base/strokes, heightmap import) plus shape objects
- Biome paint & placement, Apply Where, and world rules
- Progressive Draft→Full evaluation with hybrid GPU+CPU preview (`terra-gpu-eval` / `terra-render`)
- Geomorph, erosion, hydrology, and landscape-evolution operators (CPU foundations; GPU subset for common generators/filters and thermal/hydraulic)
- New World templates (`WorldTemplate`: Blank, Tropical Island, Alpine Range, Desert Mesa, River Valley, Badlands, Young/Old Mountains, Dune Field, Coastal)
- Project JSON save/load; heightmap-oriented export package via `terra-io` (not production-ready)
- Undo/redo for stack edits, paint strokes, world rules, and simulation scenarios
- Editor workspaces: Sculpt, Biomes, Filters, Mask, Simulation, Surface, Objects

## Stack

| Crate | Role |
|-------|------|
| `terra-world` | Fixed-origin coordinates, topology, bounds, tile addressing, and generic spatial indexing |
| `terra-authoring` | Persisted authored-feature identities, stores, mutations, and queries |
| `terra-core` | Backend-neutral documents, layers, terrain-plan semantics, evaluation inputs, and commands |
| `terra-cpu-eval` | Stateful CPU terrain evaluation, caches, scheduling, and worker runtime |
| `terra-gpu` | Reusable WGSL kernels, compiled-plan resources, derivatives, and tile caching |
| `terra-gpu-eval` | Stateful GPU terrain evaluation, refinement, timing, and output lifecycle |
| `terra-render` | wgpu 3D viewport / clipmaps / progressive presentation |
| `terra-gui` | Custom immediate-mode wgpu UI toolkit (not egui) |
| `terra-app` | Editor chrome, panels, tools; application binary (winit event loop + wiring) |
| `terra-io` | Project JSON, export package, limited grayscale GeoTIFF import |

Dependencies flow from `terra-world` through `terra-authoring` and `terra-core` to the
evaluators and application. `terra-core` must stay free of `wgpu` and UI crates;
`terra-gui` must stay free of domain types.

## Build & run

Requires a recent Rust toolchain (edition 2021) and a working wgpu/GPU driver.

```bash
cargo run -p terra-app
cargo test --workspace
```

Real-time viewport path: GPU height/normal textures + static grid displacement (no mesh rebuild). Enable **View → Profiler** for per-frame timings.

## Documentation

User-facing guides for authoring terrain. Treat these as orientation — the product is early and unfinished.

Terra uses a **layer stack** (World Creator–style), not a node graph. Workspaces change which tools are emphasized; they do not lock you into a fixed pipeline.

| Guide | What it covers |
|-------|----------------|
| [Workflow structure](docs/workflow.md) | Layer stack, biomes, masks, workspaces, and how pieces fit together |
| [Creating terrain](docs/creating-terrain.md) | New project → sculpt → biomes → filters → save |
| [Editor overview](docs/editor.md) | Shell layout, panels, and common actions |
| [Infinite Procedural World](docs/infinite-procedural-world.md) | Slice 1 settings, supported operations, limits, and troubleshooting |

## Contributing

Contributions are welcome as the project takes shape. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under the [MIT License](LICENSE).
