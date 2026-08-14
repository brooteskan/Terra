//! Artist and advanced parameters for physically-based landscape evolution.

use serde::{Deserialize, Serialize};

/// Solver fidelity / algorithm family.
///
/// - [`Fast`](Self::Fast): Tzathas-style analytical stream-power evaluation
///   (time is a continuous parameter; interactive iteration).
/// - [`Accurate`](Self::Accurate): Cordonnier / Braun–Willett style iterative
///   stream-power integration with coupled uplift each step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EvolutionSolverMode {
    #[default]
    Fast,
    Accurate,
}

/// How the tectonic uplift field is synthesised.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UpliftMode {
    /// Constant uplift rate over the domain (boundary cells forced to 0).
    Uniform,
    /// Smooth radial dome peaking at `(center_u, center_v)`.
    Radial,
    /// Linear orogenic belt (ridge corridor).
    LinearBelt,
    /// Use painted / aux `uplift_rate` field as-is.
    Painted,
    /// Low-frequency procedural uplift with geological-scale smoothness.
    Procedural,
    /// Prefer shape-compiled aux uplift; fall back to LinearBelt if absent.
    ShapeDerived,
}

impl Default for UpliftMode {
    fn default() -> Self {
        Self::ShapeDerived
    }
}

/// Drainage outlet / base-level boundary behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BoundaryMode {
    /// Cells at or below sea / base level are fixed; the rim remains a
    /// routing-only outlet so above-sea edge samples may evolve.
    #[default]
    SeaLevel,
    /// The one-cell domain rim is both a drainage outlet and elevation-locked.
    Fixed,
    /// The one-cell rim terminates drainage without locking its elevation.
    OpenDrainage,
    /// User outlet mask (`> 0.5` = fixed outlet). A missing mask falls back to
    /// SeaLevel; an empty supplied mask gets a routing-only rim fallback.
    OutletMask,
}

/// Physically-based landscape evolution controls.
///
/// Artist knobs map onto stream-power / uplift coefficients; Advanced exposes
/// the scientific parameters directly. Legacy fields (`iterations`,
/// `uplift_rate`, `incision_k`, …) remain for document compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandscapeEvolutionParams {
    // ── Artist controls ──────────────────────────────────────────────────
    /// Relative tectonic uplift amplitude `[0, 1+]`.
    #[serde(default = "default_artist_uplift")]
    pub uplift: f32,
    /// Relative fluvial erosion strength `[0, 1+]`.
    #[serde(default = "default_artist_erosion")]
    pub erosion: f32,
    /// Rainfall / precipitation multiplier (scales discharge).
    #[serde(default = "default_artist_rainfall")]
    pub rainfall: f32,
    /// Extra channel incision emphasis `[0, 1+]`.
    #[serde(default = "default_artist_incision")]
    pub river_incision: f32,
    /// Geological age `[0, 1]` → effective evolution time (not noise blend).
    #[serde(default = "default_geological_age")]
    pub geological_age: f32,
    /// Terrain resistance / hardness scale `[0, 1]`.
    #[serde(default)]
    pub terrain_resistance: f32,
    /// Drainage organisation scale `[0, 1]` (stream density / Hack-like area).
    #[serde(default = "default_drainage_scale")]
    pub drainage_scale: f32,

    // ── Mode selectors ───────────────────────────────────────────────────
    #[serde(default)]
    pub solver: EvolutionSolverMode,
    #[serde(default)]
    pub uplift_mode: UpliftMode,
    #[serde(default)]
    pub boundary: BoundaryMode,

    /// Uplift-field geometry (shared by Radial / LinearBelt / Procedural).
    #[serde(default = "half")]
    pub uplift_center_u: f32,
    #[serde(default = "half")]
    pub uplift_center_v: f32,
    /// Radial falloff / belt half-width in UV `[0.05, 0.8]`.
    #[serde(default = "default_uplift_falloff")]
    pub uplift_falloff: f32,
    /// Linear belt orientation in radians.
    #[serde(default)]
    pub uplift_angle_rad: f32,
    /// Procedural uplift seed.
    #[serde(default = "default_uplift_seed")]
    pub uplift_seed: u64,
    /// Soft geological-scale noise mix on uplift `[0, 0.5]`.
    #[serde(default = "default_uplift_noise")]
    pub uplift_noise: f32,

    // ── Advanced scientific ──────────────────────────────────────────────
    /// Stream-power erodibility \(K\).
    ///
    /// Acts on **world-metric** slope (drop per dx/dz metres) and rain-scaled
    /// discharge Q — see [`effective_k`](Self::effective_k) and the shared
    /// [`crate::hydro::spe_increment`]. **Not** numerically comparable with
    /// [`crate::layer::StreamPowerParams`]'s `k` (grid-relative slope, world-m²
    /// area); unifying them would retune saved projects and needs a versioned
    /// document migration (station D1).
    #[serde(default = "default_k")]
    pub k: f32,
    /// Area exponent \(m\) (Tzathas default ~0.4; classic ~0.5).
    #[serde(default = "default_m")]
    pub m: f32,
    /// Slope exponent \(n\) (analytical Fast uses \(n = 1\)).
    #[serde(default = "default_n")]
    pub n: f32,
    /// Maps geological age → years / effective time units.
    #[serde(default = "default_time_scale")]
    pub time_scale: f32,
    /// Accurate-mode integration step (years).
    #[serde(default = "default_dt")]
    pub dt: f32,
    /// Fast-mode fixed-point / multigrid passes.
    #[serde(default = "default_fixed_point_iters")]
    pub fixed_point_iters: u32,
    /// Optional deposition coefficient (transport-limited soft fill). 0 = off.
    #[serde(default)]
    pub deposition_coeff: f32,
    /// Hillslope diffusion strength (post / companion process).
    #[serde(default = "default_hillslope")]
    pub hillslope_diffusion: f32,
    #[serde(default = "default_talus")]
    pub talus_angle_deg: f32,
    /// How strongly protection / tectonic base is preserved.
    #[serde(default = "default_preservation")]
    pub constraint_preservation: f32,
    /// Absolute base level / sea level (metres).
    #[serde(default)]
    pub base_level: f32,
    #[serde(default)]
    pub use_dinfinity: bool,
    /// Recompute drainage when max |Δh| since last route exceeds this (m).
    #[serde(default = "default_drainage_invalidation")]
    pub drainage_invalidation_m: f32,
    /// Accurate-mode iteration budget (also used by quality caps).
    #[serde(default = "default_iterations")]
    pub iterations: u32,

    // ── Legacy aliases (document v4/v5 compatibility) ────────────────────
    /// Legacy uplift metres per evolution step (folded into `uplift`).
    #[serde(default = "default_legacy_uplift_rate")]
    pub uplift_rate: f32,
    /// Legacy incision \(K\) (folded into `k` when `k` is at default).
    #[serde(default = "default_legacy_incision_k")]
    pub incision_k: f32,
    #[serde(default = "default_m")]
    pub area_exponent: f32,
    #[serde(default = "default_n")]
    pub slope_exponent: f32,
    #[serde(default = "default_sediment")]
    pub sediment_transport: f32,
}

fn half() -> f32 {
    0.5
}
fn default_artist_uplift() -> f32 {
    0.55
}
fn default_artist_erosion() -> f32 {
    0.55
}
fn default_artist_rainfall() -> f32 {
    1.0
}
fn default_artist_incision() -> f32 {
    0.65
}
fn default_geological_age() -> f32 {
    0.45
}
fn default_drainage_scale() -> f32 {
    0.5
}
fn default_uplift_falloff() -> f32 {
    0.35
}
fn default_uplift_seed() -> u64 {
    42
}
fn default_uplift_noise() -> f32 {
    0.08
}
fn default_k() -> f32 {
    2e-5
}
fn default_m() -> f32 {
    0.4
}
fn default_n() -> f32 {
    1.0
}
fn default_time_scale() -> f32 {
    1.6e6
}
fn default_dt() -> f32 {
    2_500.0
}
fn default_fixed_point_iters() -> u32 {
    6
}
fn default_hillslope() -> f32 {
    0.18
}
fn default_talus() -> f32 {
    34.0
}
fn default_preservation() -> f32 {
    0.85
}
fn default_drainage_invalidation() -> f32 {
    2.5
}
fn default_iterations() -> u32 {
    36
}
fn default_legacy_uplift_rate() -> f32 {
    0.035
}
fn default_legacy_incision_k() -> f32 {
    0.0006
}
fn default_sediment() -> f32 {
    0.35
}

impl Default for LandscapeEvolutionParams {
    fn default() -> Self {
        Self {
            uplift: default_artist_uplift(),
            erosion: default_artist_erosion(),
            rainfall: default_artist_rainfall(),
            river_incision: default_artist_incision(),
            geological_age: default_geological_age(),
            terrain_resistance: 0.0,
            drainage_scale: default_drainage_scale(),
            solver: EvolutionSolverMode::Fast,
            uplift_mode: UpliftMode::ShapeDerived,
            boundary: BoundaryMode::SeaLevel,
            uplift_center_u: 0.5,
            uplift_center_v: 0.5,
            uplift_falloff: default_uplift_falloff(),
            uplift_angle_rad: 0.0,
            uplift_seed: default_uplift_seed(),
            uplift_noise: default_uplift_noise(),
            k: default_k(),
            m: default_m(),
            n: default_n(),
            time_scale: default_time_scale(),
            dt: default_dt(),
            fixed_point_iters: default_fixed_point_iters(),
            deposition_coeff: 0.0,
            hillslope_diffusion: default_hillslope(),
            talus_angle_deg: default_talus(),
            constraint_preservation: default_preservation(),
            base_level: 0.0,
            use_dinfinity: false,
            drainage_invalidation_m: default_drainage_invalidation(),
            iterations: default_iterations(),
            uplift_rate: default_legacy_uplift_rate(),
            incision_k: default_legacy_incision_k(),
            area_exponent: default_m(),
            slope_exponent: default_n(),
            sediment_transport: default_sediment(),
        }
    }
}

impl LandscapeEvolutionParams {
    /// Effective geological time \(t\) in years from age slider.
    pub fn evolution_time(&self) -> f32 {
        let age = self.geological_age.clamp(0.0, 1.0);
        // Ease-in so early ages keep uplift structure and immature drainage.
        let eased = age * age * (3.0 - 2.0 * age);
        (self.time_scale.max(1.0) * eased).max(0.0)
    }

    /// Effective stream-power \(K\) after artist scales.
    ///
    /// Artist erosion at or below zero is an exact no-incision limit. It must
    /// remain zero so solvers can select their finite no-advection path rather
    /// than approximating the limit with a tiny positive speed.
    pub fn effective_k(&self) -> f32 {
        let e = self.erosion.max(0.0);
        if e == 0.0 {
            return 0.0;
        }
        let base = if self.k > 0.0 {
            self.k
        } else {
            self.incision_k.max(1e-8)
        };
        let inc = 0.35 + 0.65 * self.river_incision.clamp(0.0, 2.0);
        // Interactive authoring boost: scientific K (~2e-5) is calibrated for
        // multi-Myr; artist erosion/incision scales bring visible organisation
        // into the age slider range without changing the SPE form.
        let interactive = 1.0 + 2.5 * e.min(1.5);
        (base * e * inc * interactive).max(0.0)
    }

    /// Effective area / slope exponents (prefer advanced; fall back to legacy).
    pub fn effective_m(&self) -> f32 {
        if (self.m - default_m()).abs() > 1e-6 {
            self.m.max(0.05)
        } else if (self.area_exponent - default_m()).abs() > 1e-6 {
            self.area_exponent.max(0.05)
        } else {
            self.m.max(0.05)
        }
    }

    pub fn effective_n(&self) -> f32 {
        if (self.n - default_n()).abs() > 1e-6 {
            self.n.max(0.05)
        } else if (self.slope_exponent - default_n()).abs() > 1e-6 {
            self.slope_exponent.max(0.05)
        } else {
            self.n.max(0.05)
        }
    }

    /// Peak uplift rate (m / year) after artist amplitude.
    ///
    /// Artist uplift at or below zero is an exact no-uplift limit, including
    /// for documents carrying the legacy `uplift_rate` alias.
    pub fn peak_uplift_rate(&self) -> f32 {
        let amp = self.uplift.max(0.0);
        if amp == 0.0 {
            return 0.0;
        }
        let legacy = self.uplift_rate.max(0.0);
        // Convert legacy m/step into a rate relative to dt when uplift slider is default-ish.
        let from_legacy = if self.dt > 0.0 {
            legacy / self.dt.max(1.0)
        } else {
            legacy * 1e-5
        };
        let scientific = 1e-3; // ~1 mm/yr reference (Tzathas Table 1)
        (scientific * amp + from_legacy * 0.25).max(0.0)
    }

    /// Accurate-mode step count derived from geological age + iterations budget.
    pub fn accurate_steps(&self) -> u32 {
        let age = self.geological_age.clamp(0.0, 1.0);
        let base = self.iterations.max(1);
        // Age scales effective steps; low age → few, high age → full budget.
        let scaled = (base as f32 * (0.15 + 0.85 * age)).round() as u32;
        scaled.max(1)
    }

    /// Fast-mode fixed-point iterations (quality may clamp further).
    pub fn fast_passes(&self) -> u32 {
        let age = self.geological_age.clamp(0.0, 1.0);
        let base = self.fixed_point_iters.max(1);
        // Slightly more passes as age increases (network reorganisation).
        (base as f32 * (0.7 + 0.5 * age)).round() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::LandscapeEvolutionParams;

    #[test]
    fn zero_artist_erosion_is_exact_no_incision() {
        let mut p = LandscapeEvolutionParams::default();
        p.erosion = 0.0;
        assert_eq!(p.effective_k(), 0.0);
    }

    #[test]
    fn zero_artist_uplift_overrides_legacy_default() {
        let mut p = LandscapeEvolutionParams::default();
        p.uplift = 0.0;
        assert!(
            p.uplift_rate > 0.0,
            "fixture must exercise the legacy alias"
        );
        assert_eq!(p.peak_uplift_rate(), 0.0);
    }
}
