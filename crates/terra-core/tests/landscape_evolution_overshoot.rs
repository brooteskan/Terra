//! Regression coverage for the analytical `U / (K A^m)` ridge singularity.

use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::landscape_evolution::{
    EvolutionSolverMode, LandscapeEvolutionInput, LandscapeEvolutionOperator,
    LandscapeEvolutionParams,
};

const RES: u32 = 128;
const WORLD: f32 = 4000.0;
const PEAK: f32 = 628.0;

fn cone_island(rough: bool) -> Heightfield {
    fn hash(ix: i32, iy: i32) -> f32 {
        let mut h = (ix as u32).wrapping_mul(374_761_393) ^ (iy as u32).wrapping_mul(668_265_263);
        h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
        (h ^ (h >> 16)) as f32 / u32::MAX as f32
    }

    fn noise(x: f32, y: f32) -> f32 {
        let ix = x.floor() as i32;
        let iy = y.floor() as i32;
        let fx = x - ix as f32;
        let fy = y - iy as f32;
        let sx = fx * fx * (3.0 - 2.0 * fx);
        let sy = fy * fy * (3.0 - 2.0 * fy);
        let a = hash(ix, iy) + (hash(ix + 1, iy) - hash(ix, iy)) * sx;
        let b = hash(ix, iy + 1) + (hash(ix + 1, iy + 1) - hash(ix, iy + 1)) * sx;
        a + (b - a) * sy
    }

    let metrics = HeightfieldMetrics::new(RES, RES, WORLD, WORLD);
    let mut data = vec![0.0; (RES * RES) as usize];
    let center = (RES as f32 - 1.0) * 0.5;
    let shore_radius = center * 0.8;
    for j in 0..RES {
        for i in 0..RES {
            let radius = ((i as f32 - center).powi(2) + (j as f32 - center).powi(2)).sqrt();
            let envelope = (1.0 - radius / shore_radius).max(0.0);
            let detail = if rough {
                let mut value = 0.0;
                let mut amplitude = 0.5;
                let mut frequency = 4.0 / RES as f32;
                for _ in 0..5 {
                    value += amplitude * noise(i as f32 * frequency, j as f32 * frequency);
                    amplitude *= 0.5;
                    frequency *= 2.0;
                }
                0.55 + 0.45 * value
            } else {
                1.0
            };
            data[(j * RES + i) as usize] = PEAK * envelope * detail;
        }
    }
    Heightfield::from_dense(metrics, &data)
}

fn run(input: &Heightfield, params: &LandscapeEvolutionParams) -> Heightfield {
    LandscapeEvolutionOperator::new(params.clone())
        .evaluate(LandscapeEvolutionInput {
            elevation: input,
            painted_uplift: None,
            precipitation: None,
            erodibility: None,
            lithology_hardness: None,
            outlet_mask: None,
            protection: None,
        })
        .elevation
}

fn assert_within_uplift_bound(name: &str, input: &Heightfield, params: &LandscapeEvolutionParams) {
    let (_, input_max) = input.min_max();
    let (_, output_max) = run(input, params).min_max();
    let bound = input_max + params.peak_uplift_rate() * params.evolution_time();
    assert!(output_max.is_finite(), "{name}: non-finite elevation");
    assert!(
        output_max <= bound,
        "{name}: peak grew from {input_max:.1} m to {output_max:.1} m, above the physical bound {bound:.1} m"
    );
}

#[test]
fn resistant_rock_does_not_diverge() {
    let params = LandscapeEvolutionParams {
        solver: EvolutionSolverMode::Fast,
        geological_age: 0.8,
        erosion: 0.6,
        uplift: 0.8,
        terrain_resistance: 0.85,
        hillslope_diffusion: 0.0,
        ..LandscapeEvolutionParams::default()
    };
    assert_within_uplift_bound("resistant rock", &cone_island(false), &params);
}

#[test]
fn low_drainage_ridges_do_not_diverge_without_hillslope_companion() {
    let params = LandscapeEvolutionParams {
        solver: EvolutionSolverMode::Fast,
        geological_age: 1.0,
        erosion: 0.25,
        uplift: 0.65,
        rainfall: 0.5,
        drainage_scale: 0.2,
        hillslope_diffusion: 0.0,
        ..LandscapeEvolutionParams::default()
    };
    assert_within_uplift_bound("low-drainage cone", &cone_island(false), &params);
    assert_within_uplift_bound("low-drainage rough island", &cone_island(true), &params);
}

#[test]
fn tropical_island_parameters_stay_bounded_and_preserve_land() {
    let params = LandscapeEvolutionParams {
        solver: EvolutionSolverMode::Fast,
        iterations: 28,
        uplift_rate: 0.028,
        incision_k: 0.00042,
        area_exponent: 0.52,
        slope_exponent: 1.05,
        hillslope_diffusion: 0.22,
        talus_angle_deg: 33.0,
        sediment_transport: 0.48,
        constraint_preservation: 0.82,
        base_level: 0.0,
        use_dinfinity: true,
        geological_age: 0.55,
        rainfall: 1.8,
        erosion: 0.75,
        uplift: 0.65,
        river_incision: 0.7,
        drainage_scale: 0.7,
        ..LandscapeEvolutionParams::default()
    };
    let input = cone_island(true);
    let output = run(&input, &params);
    let (_, input_max) = input.min_max();
    let (_, output_max) = output.min_max();
    let bound = input_max + params.peak_uplift_rate() * params.evolution_time();
    assert!(
        output_max <= bound,
        "Tropical Island peak exceeded {bound:.1} m"
    );
    assert!(output_max > 50.0, "Tropical Island was erased");
}
