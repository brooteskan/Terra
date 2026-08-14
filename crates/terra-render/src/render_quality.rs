//! Render quality configuration independent of terrain `PreviewQuality`.
//!
//! Terrain detail and path-tracing sample budgets must remain separable.

use terra_core::EditorRefinementState;

/// Artist-facing quality preset ceilings (Phase 7/12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualityPreset {
    Performance,
    #[default]
    Balanced,
    Quality,
}

impl QualityPreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::Performance => "Performance",
            Self::Balanced => "Balanced",
            Self::Quality => "Quality",
        }
    }

    pub const ALL: [Self; 3] = [Self::Performance, Self::Balanced, Self::Quality];
}

/// Viewport renderer presentation mode (Phase 12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewportRendererMode {
    Fast,
    /// Stable lit raster — default so the heightfield is always visible.
    #[default]
    Raster,
    ProgressiveRayTraced,
    Final,
}

impl ViewportRendererMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fast => "Fast",
            Self::Raster => "Raster",
            Self::ProgressiveRayTraced => "Progressive Ray Traced",
            Self::Final => "Final",
        }
    }

    pub const ALL: [Self; 4] = [
        Self::Fast,
        Self::Raster,
        Self::ProgressiveRayTraced,
        Self::Final,
    ];

    pub fn uses_progressive_path_tracer(self) -> bool {
        matches!(self, Self::ProgressiveRayTraced | Self::Final)
    }
}

/// Central render quality knobs uploaded as uniforms / used by the path tracer.
#[derive(Debug, Clone, Copy)]
pub struct RenderQualityConfig {
    pub preset: QualityPreset,
    pub mode: ViewportRendererMode,
    pub target_fps: f32,
    pub max_accumulated_spp: u32,
    pub interactive_spp: u32,
    pub settling_spp: u32,
    pub refining_spp: u32,
    pub max_bounces_interactive: u32,
    pub max_bounces_refining: u32,
    pub max_bounces_final: u32,
    pub direct_light_samples: u32,
    pub sun_angular_radius_rad: f32,
    pub denoise_enabled: bool,
    pub dynamic_resolution_enabled: bool,
    pub min_internal_scale: f32,
    pub max_internal_scale: f32,
    pub movement_history_cap: u32,
    pub stationary_history_cap: u32,
    pub depth_rel_tol: f32,
    pub normal_dot_min: f32,
    pub history_clamp_k: f32,
    pub atrous_iterations_interactive: u32,
    pub atrous_iterations_refining: u32,
    pub atrous_iterations_converged: u32,
    pub direct_luminance_clamp: f32,
    pub indirect_luminance_clamp: f32,
    pub terrain_pixel_error_interactive: f32,
    pub terrain_pixel_error_stationary: f32,
    pub min_samples_before_converge: u32,
    pub converge_fraction: f32,
}

impl Default for RenderQualityConfig {
    fn default() -> Self {
        Self::from_preset(QualityPreset::Balanced)
    }
}

impl RenderQualityConfig {
    pub fn from_preset(preset: QualityPreset) -> Self {
        let mut cfg = Self {
            preset,
            mode: ViewportRendererMode::Raster,
            target_fps: 60.0,
            max_accumulated_spp: 128,
            interactive_spp: 1,
            settling_spp: 1,
            refining_spp: 2,
            max_bounces_interactive: 1,
            max_bounces_refining: 3,
            max_bounces_final: 4,
            direct_light_samples: 1,
            sun_angular_radius_rad: 0.00465, // ~0.27°
            denoise_enabled: true,
            dynamic_resolution_enabled: true,
            min_internal_scale: 0.7,
            max_internal_scale: 1.0,
            movement_history_cap: 8,
            stationary_history_cap: 128,
            depth_rel_tol: 0.015,
            normal_dot_min: 0.92,
            history_clamp_k: 1.5,
            atrous_iterations_interactive: 2,
            atrous_iterations_refining: 4,
            atrous_iterations_converged: 1,
            direct_luminance_clamp: 32.0,
            indirect_luminance_clamp: 16.0,
            terrain_pixel_error_interactive: 4.0,
            terrain_pixel_error_stationary: 1.5,
            min_samples_before_converge: 8,
            converge_fraction: 0.92,
        };
        match preset {
            QualityPreset::Performance => {
                cfg.max_accumulated_spp = 32;
                cfg.refining_spp = 1;
                cfg.max_bounces_refining = 2;
                cfg.max_bounces_final = 2;
                cfg.min_internal_scale = 0.55;
                cfg.movement_history_cap = 4;
                cfg.atrous_iterations_interactive = 2;
                cfg.atrous_iterations_refining = 3;
            }
            QualityPreset::Balanced => {}
            QualityPreset::Quality => {
                cfg.mode = ViewportRendererMode::Final;
                cfg.max_accumulated_spp = 256;
                cfg.refining_spp = 2;
                cfg.max_bounces_refining = 4;
                cfg.max_bounces_final = 6;
                cfg.min_internal_scale = 0.85;
                cfg.atrous_iterations_converged = 0;
                cfg.terrain_pixel_error_stationary = 1.0;
            }
        }
        cfg
    }

    pub fn target_frame_ms(self) -> f32 {
        1000.0 / self.target_fps.max(1.0)
    }

    pub fn spp_for_state(self, state: EditorRefinementState) -> u32 {
        match state {
            EditorRefinementState::Interactive => self.interactive_spp,
            EditorRefinementState::Settling => self.settling_spp,
            EditorRefinementState::Refining => self.refining_spp,
            EditorRefinementState::Converged => 0,
            EditorRefinementState::Export => self.refining_spp.max(2),
        }
    }

    pub fn bounces_for_state(self, state: EditorRefinementState) -> u32 {
        match state {
            EditorRefinementState::Interactive | EditorRefinementState::Settling => {
                self.max_bounces_interactive
            }
            EditorRefinementState::Refining => self.max_bounces_refining,
            EditorRefinementState::Converged => self.max_bounces_refining,
            EditorRefinementState::Export => self.max_bounces_final,
        }
        .max(if matches!(self.mode, ViewportRendererMode::Final) {
            self.max_bounces_final.min(self.max_bounces_refining + 1)
        } else {
            0
        })
        .max(1)
    }

    pub fn atrous_iterations_for_state(self, state: EditorRefinementState) -> u32 {
        if !self.denoise_enabled {
            return 0;
        }
        match state {
            EditorRefinementState::Interactive | EditorRefinementState::Settling => {
                self.atrous_iterations_interactive
            }
            EditorRefinementState::Refining => self.atrous_iterations_refining,
            EditorRefinementState::Converged => self.atrous_iterations_converged,
            EditorRefinementState::Export => self.atrous_iterations_refining,
        }
    }

    pub fn history_cap_for_state(self, state: EditorRefinementState) -> u32 {
        match state {
            EditorRefinementState::Interactive | EditorRefinementState::Settling => {
                self.movement_history_cap
            }
            _ => self.stationary_history_cap.min(self.max_accumulated_spp),
        }
    }
}

/// Runtime quality state driven by interaction + frame-time budgeting (Phase 7).
#[derive(Debug, Clone)]
pub struct ViewportQualityManager {
    pub config: RenderQualityConfig,
    pub internal_scale: f32,
    smoothed_gpu_ms: f32,
    last_gpu_ms: f32,
    frames_over_budget: u32,
    frames_under_budget: u32,
    pub spp_this_frame: u32,
    pub bounce_count: u32,
    pub atrous_iterations: u32,
    pub history_cap: u32,
    pub terrain_compute_budget_us: u64,
    pub approx_rays_this_frame: u64,
    pub active_sampling_tiles: u32,
    pub reduced_sampling_tiles: u32,
    pub converged_sampling_tiles: u32,
    pub convergence_fraction: f32,
}

impl Default for ViewportQualityManager {
    fn default() -> Self {
        let config = RenderQualityConfig::default();
        Self {
            internal_scale: 1.0,
            smoothed_gpu_ms: config.target_frame_ms(),
            last_gpu_ms: 0.0,
            frames_over_budget: 0,
            frames_under_budget: 0,
            spp_this_frame: 1,
            bounce_count: 1,
            atrous_iterations: 2,
            history_cap: config.movement_history_cap,
            terrain_compute_budget_us: EditorRefinementState::Interactive.terrain_budget_us(),
            approx_rays_this_frame: 0,
            active_sampling_tiles: 0,
            reduced_sampling_tiles: 0,
            converged_sampling_tiles: 0,
            convergence_fraction: 0.0,
            config,
        }
    }
}

impl ViewportQualityManager {
    pub fn set_preset(&mut self, preset: QualityPreset) {
        let mode = self.config.mode;
        self.config = RenderQualityConfig::from_preset(preset);
        if !matches!(preset, QualityPreset::Quality) {
            self.config.mode = mode;
        }
        self.internal_scale = self.internal_scale.clamp(
            self.config.min_internal_scale,
            self.config.max_internal_scale,
        );
    }

    pub fn observe_gpu_frame_ms(&mut self, gpu_ms: f32) {
        self.last_gpu_ms = gpu_ms;
        if gpu_ms <= 0.0 {
            return;
        }
        // EMA — ignore isolated spikes by blending slowly.
        let alpha = 0.15;
        self.smoothed_gpu_ms = self.smoothed_gpu_ms * (1.0 - alpha) + gpu_ms * alpha;
    }

    pub fn smoothed_gpu_ms(&self) -> f32 {
        self.smoothed_gpu_ms
    }

    pub fn last_gpu_ms(&self) -> f32 {
        self.last_gpu_ms
    }

    pub fn target_gpu_ms(&self) -> f32 {
        self.config.target_frame_ms()
    }

    /// Update spp/bounces/denoise/scale from interaction state and smoothed GPU time.
    pub fn update_for_state(&mut self, state: EditorRefinementState) {
        self.spp_this_frame = self.config.spp_for_state(state);
        self.bounce_count = self.config.bounces_for_state(state);
        self.atrous_iterations = self.config.atrous_iterations_for_state(state);
        self.history_cap = self.config.history_cap_for_state(state);
        self.terrain_compute_budget_us = state.terrain_budget_us();

        if self.config.dynamic_resolution_enabled && self.config.mode.uses_progressive_path_tracer()
        {
            let target = self.target_gpu_ms();
            if self.smoothed_gpu_ms > target * 1.05 {
                self.frames_over_budget = self.frames_over_budget.saturating_add(1);
                self.frames_under_budget = 0;
            } else if self.smoothed_gpu_ms < target * 0.82 {
                self.frames_under_budget = self.frames_under_budget.saturating_add(1);
                self.frames_over_budget = 0;
            } else {
                self.frames_over_budget = 0;
                self.frames_under_budget = 0;
            }
            // Hysteresis: several frames before stepping ±0.05.
            if self.frames_over_budget >= 4 {
                self.internal_scale =
                    (self.internal_scale - 0.05).max(self.config.min_internal_scale);
                self.frames_over_budget = 0;
                self.degrade_for_budget();
            } else if self.frames_under_budget >= 8
                && matches!(
                    state,
                    EditorRefinementState::Settling
                        | EditorRefinementState::Refining
                        | EditorRefinementState::Converged
                )
            {
                self.internal_scale =
                    (self.internal_scale + 0.05).min(self.config.max_internal_scale);
                self.frames_under_budget = 0;
            }
        } else {
            self.internal_scale = self.config.max_internal_scale;
        }

        // Smooth scale toward full res while refining.
        if matches!(
            state,
            EditorRefinementState::Refining | EditorRefinementState::Converged
        ) && self.smoothed_gpu_ms < self.target_gpu_ms() * 0.9
        {
            self.internal_scale = (self.internal_scale + 0.02).min(self.config.max_internal_scale);
        }
    }

    fn degrade_for_budget(&mut self) {
        // Priority: spp beyond 1 → bounces → denoise iters → resolution (already stepped).
        if self.spp_this_frame > 1 {
            self.spp_this_frame = 1;
        } else if self.bounce_count > 1 {
            self.bounce_count -= 1;
        } else if self.atrous_iterations > 1 {
            self.atrous_iterations -= 1;
        }
    }
}
