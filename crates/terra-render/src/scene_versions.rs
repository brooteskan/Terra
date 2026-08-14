//! Scene generation counters and structured progressive invalidation.
//!
//! Different modifications bump different versions so temporal history and
//! terrain caches can stay valid when only lighting or camera moved.

use crate::camera::OrbitCamera;
use glam::Vec3;

/// Monotonic generation counters for selective invalidation.
#[derive(Debug, Clone, Copy, Default)]
pub struct SceneVersions {
    pub camera_version: u64,
    pub viewport_version: u64,
    pub terrain_version: u64,
    pub geometry_version: u64,
    pub material_version: u64,
    pub lighting_version: u64,
    pub atmosphere_version: u64,
}

impl SceneVersions {
    pub fn bump_camera(&mut self) {
        self.camera_version = self.camera_version.wrapping_add(1);
    }

    pub fn bump_viewport(&mut self) {
        self.viewport_version = self.viewport_version.wrapping_add(1);
    }

    pub fn bump_terrain(&mut self) {
        self.terrain_version = self.terrain_version.wrapping_add(1);
    }

    pub fn bump_geometry(&mut self) {
        self.geometry_version = self.geometry_version.wrapping_add(1);
    }

    pub fn bump_material(&mut self) {
        self.material_version = self.material_version.wrapping_add(1);
    }

    pub fn bump_lighting(&mut self) {
        self.lighting_version = self.lighting_version.wrapping_add(1);
    }

    pub fn bump_atmosphere(&mut self) {
        self.atmosphere_version = self.atmosphere_version.wrapping_add(1);
    }
}

/// Why progressive accumulation / history was invalidated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InvalidationReason {
    #[default]
    None,
    CameraCut,
    CameraMoved,
    ProjectionChanged,
    ViewportResized,
    TerrainChanged,
    GeometryChanged,
    MaterialChanged,
    LightingChanged,
    AtmosphereChanged,
    RenderModeChanged,
}

impl InvalidationReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::CameraCut => "CameraCut",
            Self::CameraMoved => "CameraMoved",
            Self::ProjectionChanged => "ProjectionChanged",
            Self::ViewportResized => "ViewportResized",
            Self::TerrainChanged => "TerrainChanged",
            Self::GeometryChanged => "GeometryChanged",
            Self::MaterialChanged => "MaterialChanged",
            Self::LightingChanged => "LightingChanged",
            Self::AtmosphereChanged => "AtmosphereChanged",
            Self::RenderModeChanged => "RenderModeChanged",
        }
    }

    /// Full accumulation reset is required for this reason (Phase 2 policy).
    pub fn resets_accumulation(self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn is_meaningful_interaction(self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn is_camera_related(self) -> bool {
        matches!(
            self,
            Self::CameraCut | Self::CameraMoved | Self::ProjectionChanged
        )
    }
}

/// World-scale aware thresholds so float jitter does not thrash history.
#[derive(Debug, Clone, Copy)]
pub struct CameraChangeThresholds {
    /// Absolute eye/target translation in metres.
    pub translation_m: f32,
    /// Yaw/pitch delta in radians.
    pub rotation_rad: f32,
    /// Vertical FOV delta in radians.
    pub fov_rad: f32,
    /// Relative near/far change.
    pub near_far_rel: f32,
    /// Aspect ratio absolute delta.
    pub aspect_abs: f32,
    /// Teleport/cut distance in metres.
    pub cut_translation_m: f32,
}

impl Default for CameraChangeThresholds {
    fn default() -> Self {
        Self {
            // ~5 cm — below typical sculpt brush / grid scale noise.
            translation_m: 0.05,
            rotation_rad: 0.0008,
            fov_rad: 0.0005,
            near_far_rel: 1e-4,
            aspect_abs: 1e-4,
            cut_translation_m: 50.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CameraSnapshot {
    pub eye: Vec3,
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
}

impl CameraSnapshot {
    pub fn from_camera(camera: &OrbitCamera, aspect: f32) -> Self {
        Self {
            eye: camera.eye(),
            target: camera.target,
            yaw: camera.yaw,
            pitch: camera.pitch,
            fov_y: camera.fov_y,
            near: camera.near,
            far: camera.far,
            aspect,
        }
    }

    pub fn detect_change(
        self,
        previous: Option<Self>,
        thresholds: &CameraChangeThresholds,
    ) -> Option<InvalidationReason> {
        let Some(prev) = previous else {
            return Some(InvalidationReason::CameraMoved);
        };
        let eye_delta = (self.eye - prev.eye).length();
        let target_delta = (self.target - prev.target).length();
        let translation = eye_delta.max(target_delta);
        if translation >= thresholds.cut_translation_m {
            return Some(InvalidationReason::CameraCut);
        }
        let rot = (self.yaw - prev.yaw)
            .abs()
            .max((self.pitch - prev.pitch).abs());
        let fov = (self.fov_y - prev.fov_y).abs();
        let near_rel = ((self.near - prev.near).abs() / prev.near.max(1e-3))
            .max((self.far - prev.far).abs() / prev.far.max(1e-3));
        let aspect = (self.aspect - prev.aspect).abs();

        if fov > thresholds.fov_rad
            || near_rel > thresholds.near_far_rel
            || aspect > thresholds.aspect_abs
        {
            return Some(InvalidationReason::ProjectionChanged);
        }
        if translation > thresholds.translation_m || rot > thresholds.rotation_rad {
            return Some(InvalidationReason::CameraMoved);
        }
        None
    }
}

/// Owns scene versions, last invalidation, and camera change detection.
#[derive(Debug, Clone)]
pub struct SceneVersionRegistry {
    pub versions: SceneVersions,
    pub last_reason: InvalidationReason,
    pub camera_thresholds: CameraChangeThresholds,
    last_camera: Option<CameraSnapshot>,
    /// Set when a meaningful scene change occurred this frame.
    meaningful_this_frame: bool,
}

impl Default for SceneVersionRegistry {
    fn default() -> Self {
        Self {
            versions: SceneVersions::default(),
            last_reason: InvalidationReason::None,
            camera_thresholds: CameraChangeThresholds::default(),
            last_camera: None,
            meaningful_this_frame: false,
        }
    }
}

impl SceneVersionRegistry {
    pub fn begin_frame(&mut self) {
        self.meaningful_this_frame = false;
    }

    pub fn meaningful_this_frame(&self) -> bool {
        self.meaningful_this_frame
    }

    pub fn last_reason(&self) -> InvalidationReason {
        self.last_reason
    }

    pub fn notify(&mut self, reason: InvalidationReason) {
        if reason == InvalidationReason::None {
            return;
        }
        self.last_reason = reason;
        self.meaningful_this_frame = true;
        match reason {
            InvalidationReason::CameraCut
            | InvalidationReason::CameraMoved
            | InvalidationReason::ProjectionChanged => self.versions.bump_camera(),
            InvalidationReason::ViewportResized => self.versions.bump_viewport(),
            InvalidationReason::TerrainChanged => self.versions.bump_terrain(),
            InvalidationReason::GeometryChanged => self.versions.bump_geometry(),
            InvalidationReason::MaterialChanged => self.versions.bump_material(),
            InvalidationReason::LightingChanged => self.versions.bump_lighting(),
            InvalidationReason::AtmosphereChanged => self.versions.bump_atmosphere(),
            InvalidationReason::RenderModeChanged => {
                self.versions.bump_lighting();
                self.versions.bump_viewport();
            }
            InvalidationReason::None => {}
        }
    }

    /// Compare camera against last snapshot; notify if changed beyond thresholds.
    pub fn update_camera(
        &mut self,
        camera: &OrbitCamera,
        aspect: f32,
    ) -> Option<InvalidationReason> {
        let snap = CameraSnapshot::from_camera(camera, aspect);
        let reason = snap.detect_change(self.last_camera, &self.camera_thresholds);
        self.last_camera = Some(snap);
        if let Some(reason) = reason {
            self.notify(reason);
        }
        reason
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_below_threshold_does_not_invalidate() {
        let cam = OrbitCamera::default();
        let a = CameraSnapshot::from_camera(&cam, 16.0 / 9.0);
        let mut b = a;
        b.eye.x += 0.001;
        b.target.x += 0.001;
        assert!(b
            .detect_change(Some(a), &CameraChangeThresholds::default())
            .is_none());
    }

    #[test]
    fn large_move_is_camera_cut() {
        let cam = OrbitCamera::default();
        let a = CameraSnapshot::from_camera(&cam, 1.0);
        let mut b = a;
        b.eye.x += 200.0;
        b.target.x += 200.0;
        assert_eq!(
            b.detect_change(Some(a), &CameraChangeThresholds::default()),
            Some(InvalidationReason::CameraCut)
        );
    }
}
