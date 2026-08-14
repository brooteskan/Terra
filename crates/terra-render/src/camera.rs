use glam::{Mat4, Vec3};

#[derive(Debug, Clone)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 1500.0,
            yaw: 0.7,
            pitch: 0.6,
            fov_y: 50f32.to_radians(),
            near: 1.0,
            far: 100_000.0,
        }
    }
}

impl OrbitCamera {
    /// Unit vector from target toward the eye (orbit offset direction).
    fn offset_dir(&self) -> Vec3 {
        let cp = self.pitch.cos();
        Vec3::new(self.yaw.cos() * cp, self.pitch.sin(), self.yaw.sin() * cp)
    }

    pub fn eye(&self) -> Vec3 {
        self.target + self.offset_dir() * self.distance
    }

    /// Look direction from eye toward target.
    pub fn forward(&self) -> Vec3 {
        -self.offset_dir()
    }

    pub fn right(&self) -> Vec3 {
        let f = self.forward();
        let r = f.cross(Vec3::Y);
        if r.length_squared() < 1e-8 {
            Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos())
        } else {
            r.normalize()
        }
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fov_y, aspect.max(0.01), self.near, self.far);
        proj * view
    }

    /// Orbit around the look-at target (classic turntable).
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.005;
        self.pitch = (self.pitch + dy * 0.005).clamp(-1.45, 1.45);
    }

    /// Rotate around the eye (game-engine / FPS look). Eye stays fixed; target retargets.
    pub fn look(&mut self, dx: f32, dy: f32) {
        let eye = self.eye();
        self.yaw += dx * 0.005;
        self.pitch = (self.pitch + dy * 0.005).clamp(-1.45, 1.45);
        self.target = eye - self.offset_dir() * self.distance;
    }

    pub fn zoom(&mut self, delta: f32) {
        // Exponential zoom so wheel steps feel even near and far.
        let factor = (1.0 - delta * 0.0015).clamp(0.5, 1.5);
        self.distance = (self.distance * factor).clamp(10.0, 50_000.0);
    }

    /// Pan in the camera view plane (right + camera-up), moving eye and target together.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let right = self.right();
        let up = right.cross(self.forward()).normalize_or_zero();
        let scale = self.distance * 0.00125;
        // Drag right → world moves right under the camera (invert screen X).
        // Drag up → pan along camera up.
        self.target += right * (-dx * scale) + up * (dy * scale);
    }

    /// Fly the camera rig: `forward`/`right`/`up` are typically -1..1 from WASD/QE.
    ///
    /// Game-engine free-fly: W/S move along the look direction (pitch included), A/D
    /// strafe along camera right, Q/E along world up. Speed scales with distance.
    pub fn fly(&mut self, forward: f32, right: f32, up: f32, dt: f32, boost: bool) {
        let f = self.forward();
        let r = self.right();
        let mut dir = f * forward + r * right + Vec3::Y * up;
        let len = dir.length();
        if len < 1e-6 || dt <= 0.0 {
            return;
        }
        dir /= len;
        let mut speed = self.distance.max(50.0) * 0.9;
        if boost {
            speed *= 3.0;
        }
        self.target += dir * speed * dt;
    }

    /// Keep the camera rig over the heightfield with margin for FPS look/fly.
    ///
    /// Clamping the look-at target into the world footprint fights `look()`: rotating around a
    /// distant eye puts the target outside the world, then yanking it back teleports the eye.
    /// Instead, clamp the eye with padding and translate target by the same delta (rigid rig).
    pub fn clamp_to_world(&mut self, world_size: (f32, f32)) {
        let max_x = world_size.0.max(1.0);
        let max_z = world_size.1.max(1.0);
        let pad = (max_x.max(max_z) * 0.75).max(self.distance);
        let eye = self.eye();
        let clamped_x = eye.x.clamp(-pad, max_x + pad);
        let clamped_z = eye.z.clamp(-pad, max_z + pad);
        let dx = clamped_x - eye.x;
        let dz = clamped_z - eye.z;
        if dx.abs() > 1e-6 || dz.abs() > 1e-6 {
            self.target.x += dx;
            self.target.z += dz;
        }
    }

    pub fn reset(&mut self, target: Vec3, distance: f32) {
        self.target = target;
        self.distance = distance;
        self.yaw = 0.7;
        self.pitch = 0.6;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn look_keeps_eye_fixed() {
        let mut cam = OrbitCamera {
            target: Vec3::new(100.0, 0.0, 200.0),
            distance: 500.0,
            yaw: 0.4,
            pitch: 0.5,
            ..OrbitCamera::default()
        };
        let eye_before = cam.eye();
        cam.look(40.0, -25.0);
        let eye_after = cam.eye();
        assert!(
            (eye_before - eye_after).length() < 1e-3,
            "look must pivot around the eye: before={eye_before:?} after={eye_after:?}"
        );
    }

    #[test]
    fn fly_forward_moves_toward_look() {
        let mut cam = OrbitCamera {
            target: Vec3::new(0.0, 0.0, 0.0),
            distance: 100.0,
            yaw: 0.0,
            pitch: 0.0,
            ..OrbitCamera::default()
        };
        // yaw=0, pitch=0 → offset=(1,0,0), forward=(-1,0,0)
        let before = cam.target;
        cam.fly(1.0, 0.0, 0.0, 1.0, false);
        assert!(
            cam.target.x < before.x,
            "W should move along look (toward -X)"
        );
    }

    #[test]
    fn fly_forward_dives_when_looking_down() {
        let mut cam = OrbitCamera {
            target: Vec3::new(500.0, 100.0, 500.0),
            distance: 800.0,
            yaw: 0.4,
            pitch: 0.9, // steep look-down
            ..OrbitCamera::default()
        };
        let y_before = cam.target.y;
        let eye_y_before = cam.eye().y;
        cam.fly(1.0, 0.0, 0.0, 1.0, false);
        assert!(
            cam.target.y < y_before,
            "W along look-down must lower the look-at: y {y_before} → {}",
            cam.target.y
        );
        assert!(
            cam.eye().y < eye_y_before,
            "eye should dive with the rig: {} → {}",
            eye_y_before,
            cam.eye().y
        );
    }

    #[test]
    fn look_then_clamp_keeps_eye_stable_near_world_edge() {
        let world = (4096.0, 4096.0);
        let mut cam = OrbitCamera {
            target: Vec3::new(2048.0, 100.0, 2048.0),
            distance: 4500.0,
            yaw: 0.7,
            pitch: 0.6,
            ..OrbitCamera::default()
        };
        // Overview framing puts the eye outside the footprint.
        assert!(cam.eye().x > world.0 || cam.eye().z > world.1);
        let eye_before = cam.eye();
        cam.look(80.0, -40.0);
        cam.clamp_to_world(world);
        let eye_after = cam.eye();
        assert!(
            (eye_before - eye_after).length() < 1e-2,
            "look+clamp must not teleport the eye: before={eye_before:?} after={eye_after:?}"
        );
    }

    #[test]
    fn fly_qe_changes_altitude() {
        let mut cam = OrbitCamera {
            target: Vec3::new(100.0, 50.0, 200.0),
            distance: 400.0,
            yaw: 1.0,
            pitch: 0.7,
            ..OrbitCamera::default()
        };
        let y0 = cam.target.y;
        cam.fly(0.0, 0.0, 1.0, 1.0, false); // E
        assert!(cam.target.y > y0, "E should raise altitude");
        let y1 = cam.target.y;
        cam.fly(0.0, 0.0, -1.0, 1.0, false); // Q
        assert!(cam.target.y < y1, "Q should lower altitude");
    }
}
