# Terrain operation execution matrix

This is the production strategy matrix for compiled terrain operations. The
machine-checked authority is `LayerKind::intrinsic_reach`,
`LayerKind::aux_reach`, and `EffectFilterKind::spatial_dependency`; the
`execution_strategy_matrix` ratchet instantiates every registered layer and
every effect-filter subkind so a newly compiled operation cannot remain
unclassified.

| Operation semantics | Examples | Compiled reach | Tile strategy |
|---|---|---:|---|
| Per-sample generators and composites | Flat, Ramp, noise, masks, blend/group composite | `Localized(0)` | Ordinary local tile |
| Bounded neighborhood | Blur, bounded effect filters, sculpt reconciliation | `Localized(halo)` | Halo-expanded local tile |
| Full-field range or elliptic solve | Terrace, GradientReconstruct | `Full` | Immutable complete-field checkpoint |
| Basin/drainage coupling | RiverCarve, RiverNetwork, StreamPower, LandscapeEvolution, HydrologyRepair, GeomorphicDetail | `Full` / global aux | Immutable complete-field checkpoint |
| Whole-field transport/simulation | Thermal, Hydraulic, DebrisFlow, Dunes, Sand, Fluid, MultiScaleAmplify | `Full` | Immutable complete-field checkpoint |
| Whole-domain placement/distance | Biomes, Vegetation and global effect filters | `Full` / global aux | Immutable complete-field checkpoint |
| Local height with a consumed global auxiliary | Island or another local height producer whose global aux is live downstream | local height + `AuxReach::Global` | Immutable complete-field checkpoint at the auxiliary frontier |

The checkpoint is a half-open prefix cut advanced through the owning authored
span. All frontier fields are copied into dedicated immutable R32F textures.
Tile evaluators copy their level-local subrect from those textures and execute
only the remaining bounded suffix with its accumulated halo. Unsupported GPU
kernels or live auxiliary outputs fail checkpoint construction explicitly and
use the complete-field application fallback; they are never submitted as an
ordinary local tile.

Checkpoint identity includes the content stamp, level extent, plan boundary,
and quality. An upstream edit therefore creates a new checkpoint publication.
Old textures remain reference-owned by already-submitted work until its fence
retires, preventing cancellation or revision changes from mixing fields.
