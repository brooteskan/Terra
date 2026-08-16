# Terra — Feature Testing Checklist

A walkthrough for a tester exercising **every** interactive feature in the editor. Keep notes terse.

**Terra is early WIP** — expect incomplete preview, export, materials, and objects. The goal is to flag two things per feature:

- **⚠️ Broken / obvious problem** — doesn't do what it says, crashes, no effect, wrong result.
- **👀 Works, but** — it functions, yet you'd like to see it demoed and have improvement ideas.

**How to mark each line:**

| Mark | Meaning |
|---|---|
| `[ ]` | Not tested yet |
| `[x]` ✅ | Tested, seems right |
| `[x]` ⚠️ | Problem (write what) |
| `[x]` 👀 | Works, want to see it / suggest improvement |

Reference: [Editor overview](editor.md) · [Workflow](workflow.md) · [Creating terrain](creating-terrain.md). Watch the persistent log at `%LOCALAPPDATA%\Terra\logs` if something misbehaves.

---

## 0. Startup, project, shell

- [ ] Launch (`cargo run -p terra-app`) — opens without crashing
- [ ] **Project Home** — New Project, Open, Recents
- [ ] **World Design templates** — Blank, Tropical Island, Alpine Range, Desert Mesa, River Valley, Badlands, Young Mountains, Old Mountains, Dune Field, Coastal
- [ ] Set **world size (m)** and **sea level** on new world
- [ ] **Save / Save As / Close** — writes JSON, `*` dirty marker appears, dirty-confirm on discard
- [ ] Re-open a saved project — loads back identically
- [ ] **Workspace rail** switches: World, Sculpt, Biomes, Filters, Mask, Simulation, Surface, Objects, Tools
- [ ] **Menu bar** — File / Edit / View, project chip, Export button
- [ ] **Bottom dock** — preview resolution / quality / backend, build progress, cancel
- [ ] **Right panel** — Hierarchy tree + Inspector for selected layer

---

## 1. Sculpt brushes

Select **Base** or a Shape Layer, then a brush. Test brush params (size, strength, falloff, spacing, flow) actually change the stroke.

- [ ] **Raise** (R) — plus Ctrl=lower, Shift=smooth modifiers
- [ ] **Lower** (L)
- [ ] **Smooth** (S)
- [ ] **Flatten** — flattens toward target height
- [ ] **Terrace** — quantizes into steps
- [ ] **Pinch** — pulls detail toward center
- [ ] **Inflate** — bulges outward
- [ ] **Erode Brush** — soft erosion
- [ ] **Noise Brush** — paints noise detail
- [ ] **Height Stamp** — stamps toward absolute height

**Stamp / path brushes**
- [ ] **Mountain Stamp**
- [ ] **Valley Stamp**
- [ ] **Plateau Stamp**
- [ ] **Crater Stamp**
- [ ] **Coastline Tool** — smooths along a coast path
- [ ] **River Path** — carves a river valley along a path

**Semantic brushes** (paint fields, not raw height)
- [ ] Ridge · [ ] Valley · [ ] Roughness · [ ] Uplift · [ ] Protect · [ ] Hardness · [ ] Sediment · [ ] River Constraint

**Brush parameters** — confirm each has visible effect
- [ ] Size / radius · [ ] Strength · [ ] Falloff (hardness) · [ ] Spacing · [ ] Flow

---

## 2. Shape layers & landform generators

Add via **Quick Add** / contextual create. Confirm blend mode, opacity, distribution in the inspector.

**Shape layer types**
- [ ] Sculpt Strokes · [ ] Stamp 2D · [ ] Stamp 3D · [ ] Procedural Shape · [ ] Polygon Height · [ ] Path · [ ] Import Heightmap

**Landform / noise generators**
- [ ] Flat · [ ] Ramp · [ ] Value Noise · [ ] Perlin · [ ] OpenSimplex · [ ] Worley · [ ] FBM · [ ] Ridged · [ ] Domain Warp
- [ ] Mountain · [ ] Volcano · [ ] Mesa · [ ] Island · [ ] Canyon · [ ] Dunes · [ ] Plateau · [ ] Voronoi
- [ ] Terrain Constraints · [ ] Gradient Reconstruct

**Shape editing tools**
- [ ] Edit Path (place control points) · [ ] Edit Polygon (place vertices) · [ ] Edit River Spring

---

## 3. Layer system (hierarchy + inspector)

- [ ] Add / delete / reorder / rename layers
- [ ] Enable / Lock / Solo toggles behave correctly
- [ ] **Opacity** slider changes contribution
- [ ] **Blend modes** — Replace, Add, Subtract, Multiply, Minimum, Maximum, Interpolate, Height Blend, smooth variants
- [ ] **Groups** — create group, group blend/opacity
- [ ] **Distribution** stack on a layer/group gates coverage
- [ ] **Bake Selected** — caches a layer
- [ ] Inspector **Simple vs Advanced** parameter toggle
- [ ] **Resolution** readout (stored/imported vs Draft/Medium/Full/Export grid)

---

## 4. Masks (Mask workspace)

Add each mask source; confirm its range/params drive the coverage and that the mask preview matches.

- [ ] **Height** (min/max)
- [ ] **Slope** (min/max degrees)
- [ ] **Curvature** (min/max)
- [ ] **Flow** (accumulation min/max)
- [ ] **Convexity** (ridges)
- [ ] **Concavity** (valleys)
- [ ] **Noise** (seed, frequency)
- [ ] **Distance** (from height threshold)
- [ ] **Painted** — arms Paint Mask tool, brush paints coverage
- [ ] **Combined Mask** — noise starter combinable with dist nodes
- [ ] **Mask stack** — combine modes + dist nodes (fill, blur, invert, expand, All/Any groups)
- [ ] Bind a mask to a layer/biome and confirm it restricts the effect

---

## 5. Filters (Filters workspace)

Add filters into a biome's Filters. Check **Apply Where**, opacity, and blend read as refinements.

- [ ] **General:** Geomorphic Detail · Add/Set · Border Blend · Flatten · Zero-Edge · Blur · Overhang Stamp · Local SDF
- [ ] **Design:** Curve · Cutoff · Height Map · Voronoi
- [ ] **Effect:** Angle Blur · Directional Blur · Balloon · Blocks · Crater · Deflate · Denoise · Smooth · Distortion · Hexagons · Inflate · Kuwahara · Ridged · Rugged · Scatter · Shore · Smooth Ridges · Squeeze · Strata · Swirl · Washed Off
- [ ] **Noise:** Billow · Gabor · Perlin · Phasor · Ridged · Simplex · Value · Voronoi · Wave · White
- [ ] **Arid:** Cliff Reinforce · Canyon · Chipped · Dunes · Rocky Cliffs · Rocky Hard · Rocky Layers · Rocky Plateaus · Rocky Sharp · Rocky Wide
- [ ] **Terrace:** Terrace · Irregular · Simple · Steep
- [ ] **Drift:** Angle Break · Wind
- [ ] **Basic Erosion:** Hydraulic · Ridged Flows · Rocky · Soft Flows
- [ ] **Advanced Erosion:** Hydraulic Sediment · Sediment Flows · Thin Flows · Wide Flows
- [ ] **Sediment:** Fill Soft · Mud · Talus

---

## 6. Biomes (Biomes workspace)

- [ ] **Create Biome** container (gets Filters / Materials / Objects / Local Sims sections)
- [ ] **Climate Biomes** layer
- [ ] Biome group **distribution** defines coverage
- [ ] Active-biome context — Add / paint targets follow it
- [ ] **Apply Where** — Entire Biome, height range, slope range, near water, painted restriction, custom conditions, advanced mask

**Biome paint tools**
- [ ] Paint Biome (B) · [ ] Erase Biome · [ ] Smooth Weights · [ ] Replace Biome · [ ] Normalize Weights · [ ] Flood Fill · [ ] Polygon Fill

---

## 7. Simulation (Simulation workspace)

Run each; watch bottom-dock status (Ready / Outdated / Running) and cancel behavior.

- [ ] Landscape Evolution · [ ] Multi-Scale Amplify · [ ] Hydrology Repair
- [ ] Hydraulic Erosion · [ ] Thermal Erosion · [ ] Debris Flow · [ ] Weathering · [ ] Stream Power
- [ ] Wind (sand) · [ ] Wind Carve · [ ] Sediment (fill soft) · [ ] Sediment Flows · [ ] Talus · [ ] Mud
- [ ] River Network · [ ] River (carve) · [ ] Lake · [ ] Waterfall · [ ] Coastal
- [ ] Fluid Simulation · [ ] Ecosystem Feedback
- [ ] **Simulation Scenario** container — domain, sources, passes, quality
- [ ] **World Rules** — e.g. "snow above 1200 m"; scope, phase, effects

---

## 8. Surface / Materials (Surface workspace) — *early / placeholder*

- [ ] Material (rule-based) · [ ] Colour Paint · [ ] Wetness · [ ] Snow · [ ] Rock · [ ] Grass
- [ ] Material rule params — min/max slope, min/max height, mask source, hardness, tint, roughness, metalness, albedo path, strata

---

## 9. Objects / Scatter (Objects workspace) — *early / placeholder*

- [ ] Trees (Vegetation) · [ ] Rocks · [ ] Grass · [ ] Debris · [ ] Custom Meshes
- [ ] Params — density, min distance, scale variation, rotation/yaw variation, seed
- [ ] Slope filtering · [ ] Height filtering · [ ] Coverage distribution

---

## 10. View, camera, rendering

**Shading / preview modes** — each renders without artifacts
- [ ] Lit · [ ] Unlit · [ ] Height · [ ] Slope · [ ] Curvature · [ ] Flow · [ ] Convexity · [ ] Concavity · [ ] Normals · [ ] Wireframe · [ ] Ambient Occlusion
- [ ] Material · [ ] Biome · [ ] Mask(s) · [ ] Vegetation Density · [ ] Water · [ ] Sediment · [ ] Hardness · [ ] Erosion · [ ] Deposition
- [ ] Stream Order · [ ] SPE Incision · [ ] Temperature · [ ] Rainfall · [ ] Snow · [ ] Soil Moisture · [ ] Overhang

**Lighting presets**
- [ ] Studio · [ ] Midday · [ ] Sunset · [ ] Overcast · [ ] High Contrast · [ ] Neutral · [ ] Progressive RT · [ ] custom lighting menu

**Overlays / display aids**
- [ ] Wireframe · [ ] Grid · [ ] Bounds · [ ] Water level · [ ] Contours

**Camera / navigation**
- [ ] Move (orbit/pan/zoom) · [ ] Frame Terrain · [ ] Top View · [ ] Frame Selection · [ ] Measure (world distance)
- [ ] Viewport mode bar — Terrain(Lit) / Height / Slope / Flow

---

## 11. Import / Export

**Import**
- [ ] Heightmap PNG (16-bit) · [ ] Heightmap RAW · [ ] GeoTIFF (grayscale) · [ ] Height Map filter source

**Export** *(not production-ready — expect the CTA blocked)*
- [ ] Export panel opens; choose directory
- [ ] Export resolution (512–8192) · [ ] Preview resolution (256–8192)
- [ ] Outputs: height.png (16-bit), height.r32 + meta, mask_*.png, splat.png + ids, color.png, normal.png, vegetation_instances.json, terrain_collision.obj, tile_manifest.json
- [ ] Toggles: include splat / splat IDs / collision mesh (+ stride)

---

## 12. Panels, palette, undo

- [ ] **Command Palette** (open, search, run a command)
- [ ] **Quick Add** search
- [ ] **Undo / Redo** — stack edits, paint strokes, world rules, sim scenarios
- [ ] **History panel**
- [ ] **Pipeline panel**
- [ ] **Profiler** (View → Profiler, per-frame timings)
- [ ] **Bookmarks**
- [ ] **2D Preview** window
- [ ] **Presets**
- [ ] **Contextual Create**
- [ ] **Terrain settings** — export/preview resolution, level steps / upsample levels

---

### Notes for the tester

- The authoritative list of clickable tools is `all_tools()` in `crates/terra-app/src/ui/tool_catalog.rs`; if something above isn't in the UI, note it — a few core variants are intentionally hidden.
- Prefer testing **Sculpt → Biomes → Filters → Masks** first (most complete), then Simulation, then Surface/Objects (most placeholder).
- For anything marked 👀, jot a one-line improvement idea next to it.
