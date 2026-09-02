//! Per-generator equivalence (#110): a `*_tiles` fill over a tile subset into a
//! seeded field must be bit-identical to the whole-field fill of the same
//! generator with the same params. One case per adopted generator; a world-index
//! bug in any tile entry (e.g. sampling from a local instead of a global index)
//! surfaces as a mismatch on a non-origin tile.

use terra_core::generators::{
    canyons, canyons_tiles, domain_warp_field, domain_warp_field_tiles, fbm_field, fbm_field_tiles,
    mesa, mesa_tiles, mountains, mountains_tiles, noise_field, noise_field_tiles, procedural_shape,
    procedural_shape_tiles, ridged_field, ridged_field_tiles, uplift, uplift_tiles,
    voronoi_regions, voronoi_regions_tiles, worley_field, worley_field_tiles, CanyonParams,
    DomainWarpParams, FbmParams, MesaParams, MountainParams, ProceduralGenerator,
    ProceduralShapeParams, UpliftParams, VoronoiParams,
};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics, TileId};
use terra_core::layer::PlateauParams;
use terra_core::noise::{FractalNoiseType, NoiseParams, WorleyParams};
use terra_core::CancelToken;

/// 72 is not a multiple of 16, so the last tile row/column is a partial edge
/// tile (5x5 grid, last tiles 8 samples wide) — the ragged-tile case.
fn metrics() -> HeightfieldMetrics {
    HeightfieldMetrics {
        width: 72,
        height: 72,
        world_size_x: 72.0,
        world_size_z: 72.0,
        tile_size: 16,
        halo: 2,
    }
}

/// A deterministic ~40% tile subset (fixed LCG, no `rand` dep), always including
/// a near-center interior tile and the far-corner partial tile.
fn scope(m: &HeightfieldMetrics) -> Vec<TileId> {
    let mut v = vec![
        TileId { tx: 2, tz: 1 },
        TileId {
            tx: m.tiles_x() - 1,
            tz: m.tiles_z() - 1,
        },
    ];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for tz in 0..m.tiles_z() {
        for tx in 0..m.tiles_x() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (state >> 33) % 5 < 2 {
                v.push(TileId { tx, tz });
            }
        }
    }
    v.sort_by_key(|t| (t.tz, t.tx));
    v.dedup();
    v
}

/// Assert `scoped` equals `whole` on every interior sample of the scope tiles,
/// and that the scope is not constant (else a world-index bug could hide).
fn assert_scope_matches(name: &str, whole: &Heightfield, scoped: &Heightfield, scope: &[TileId]) {
    let m = whole.metrics;
    let mut first: Option<u32> = None;
    let mut varies = false;
    for &id in scope {
        let tile = whole.tile(id).expect("scope tile exists");
        let (ox, oz) = tile.interior_origin(&m);
        for lz in 0..tile.interior_height {
            for lx in 0..tile.interior_width {
                let (i, j) = (ox + lx, oz + lz);
                let w = whole.get(i, j).to_bits();
                assert_eq!(
                    scoped.get(i, j).to_bits(),
                    w,
                    "{name}: tile {id:?} sample ({i},{j}) scoped != whole-field"
                );
                match first {
                    None => first = Some(w),
                    Some(f) => varies |= f != w,
                }
            }
        }
    }
    assert!(
        varies,
        "{name}: scope samples are all identical — the case has no teeth"
    );
}

/// Build whole + scoped fields for a `f(metrics, cancel, &P) -> Option<Heightfield>`
/// / `g(&mut hf, tiles, cancel, &P) -> bool` generator pair and compare them.
macro_rules! case {
    ($name:literal, $p:expr, $whole:expr, $tiles:expr) => {{
        let m = metrics();
        let p = $p;
        let cancel = CancelToken::never();
        let whole = $whole(m, &cancel, &p).expect(concat!($name, ": whole fill"));
        let sc = scope(&m);
        let mut scoped = Heightfield::zeros(m);
        assert!(
            $tiles(&mut scoped, &sc, &cancel, &p),
            concat!($name, ": tiles fill")
        );
        assert_scope_matches($name, &whole, &scoped, &sc);
    }};
}

/// A higher frequency than the 0.002 default sharpens the per-sample variation
/// so an index bug produces an obvious mismatch on a small world.
fn sharp_noise() -> NoiseParams {
    NoiseParams {
        frequency: 0.03,
        ..NoiseParams::default()
    }
}

#[test]
fn noise_field_tiles_matches_whole_field() {
    let m = metrics();
    let p = sharp_noise();
    let cancel = CancelToken::never();
    let kind = FractalNoiseType::Perlin;
    let whole = noise_field(m, &cancel, &p, kind).expect("noise: whole fill");
    let sc = scope(&m);
    let mut scoped = Heightfield::zeros(m);
    assert!(
        noise_field_tiles(&mut scoped, &sc, &cancel, &p, kind),
        "noise: tiles fill"
    );
    assert_scope_matches("noise", &whole, &scoped, &sc);
}

#[test]
fn worley_field_tiles_matches_whole_field() {
    case!(
        "worley",
        WorleyParams {
            base: sharp_noise(),
            ..WorleyParams::default()
        },
        worley_field,
        worley_field_tiles
    );
}

#[test]
fn fbm_field_tiles_matches_whole_field() {
    let mut p = FbmParams::default();
    p.base.frequency = 0.03;
    case!("fbm", p, fbm_field, fbm_field_tiles);
}

#[test]
fn ridged_field_tiles_matches_whole_field() {
    let mut p = FbmParams::default();
    p.base.frequency = 0.03;
    case!("ridged", p, ridged_field, ridged_field_tiles);
}

#[test]
fn domain_warp_field_tiles_matches_whole_field() {
    case!(
        "domain_warp",
        DomainWarpParams::default(),
        domain_warp_field,
        domain_warp_field_tiles
    );
}

#[test]
fn mesa_tiles_matches_whole_field() {
    case!("mesa", MesaParams::default(), mesa, mesa_tiles);
}

#[test]
fn mountains_tiles_matches_whole_field() {
    case!(
        "mountains",
        MountainParams::default(),
        mountains,
        mountains_tiles
    );
}

#[test]
fn uplift_tiles_matches_whole_field() {
    case!("uplift", UpliftParams::default(), uplift, uplift_tiles);
}

#[test]
fn canyons_tiles_matches_whole_field() {
    case!("canyons", CanyonParams::default(), canyons, canyons_tiles);
}

#[test]
fn voronoi_regions_tiles_matches_whole_field() {
    case!(
        "voronoi",
        VoronoiParams::default(),
        voronoi_regions,
        voronoi_regions_tiles
    );
}

#[test]
fn procedural_shape_tiles_matches_whole_field() {
    for generator in [
        ProceduralGenerator::Mountain,
        ProceduralGenerator::Hills,
        ProceduralGenerator::Plateau,
        ProceduralGenerator::Mesa,
        ProceduralGenerator::Volcano,
        ProceduralGenerator::Canyon,
        ProceduralGenerator::Noise,
    ] {
        let m = metrics();
        let mut p = ProceduralShapeParams {
            generator,
            ..ProceduralShapeParams::default()
        };
        // Sharpen the fBm feeding Hills/Plateau, and widen the plateau band so
        // its samples land in the sloped mid-branch rather than clamping flat —
        // otherwise the Plateau scope is constant and the case has no teeth.
        p.hills.base.frequency = 0.03;
        p.plateau = PlateauParams {
            low: -500.0,
            high: 500.0,
            soft: 10.0,
        };
        let cancel = CancelToken::never();
        let whole = procedural_shape(m, &cancel, &p).expect("proc: whole fill");
        let sc = scope(&m);
        let mut scoped = Heightfield::zeros(m);
        assert_eq!(
            procedural_shape_tiles(&mut scoped, &sc, &cancel, &p),
            Some(true),
            "proc {generator:?}: tile-supported"
        );
        assert_scope_matches(&format!("proc:{generator:?}"), &whole, &scoped, &sc);
    }
}

#[test]
fn procedural_shape_tiles_declines_unsupported_variants() {
    for generator in [ProceduralGenerator::Dunes, ProceduralGenerator::Crater] {
        let m = metrics();
        let p = ProceduralShapeParams {
            generator,
            ..ProceduralShapeParams::default()
        };
        let mut scoped = Heightfield::zeros(m);
        assert_eq!(
            procedural_shape_tiles(
                &mut scoped,
                &[TileId { tx: 0, tz: 0 }],
                &CancelToken::never(),
                &p
            ),
            None,
            "proc {generator:?}: no tile-sliced fill, caller must fall back"
        );
    }
}
