//! Direct-manipulation shape objects that compile into authoring constraints.

use crate::authoring::{
    SculptPoint, TerrainConstraint, TerrainConstraintKind, TerrainConstraintParams,
};
use crate::ids::LayerId;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable identity for a shape object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShapeObjectId(pub Uuid);

impl ShapeObjectId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ShapeObjectId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeKind {
    CoastlinePolygon,
    LandmassPolygon,
    MountainSpine,
    RidgeSpline,
    ValleySpline,
    RiverPath,
    CanyonPath,
    PlateauPolygon,
    LakeBasin,
    Volcano,
    UpliftCentre,
    HeightStamp,
}

impl ShapeKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::CoastlinePolygon => "Coastline",
            Self::LandmassPolygon => "Landmass",
            Self::MountainSpine => "Mountain Spine",
            Self::RidgeSpline => "Ridge",
            Self::ValleySpline => "Valley",
            Self::RiverPath => "River",
            Self::CanyonPath => "Canyon",
            Self::PlateauPolygon => "Plateau",
            Self::LakeBasin => "Lake Basin",
            Self::Volcano => "Volcano",
            Self::UpliftCentre => "Uplift Centre",
            Self::HeightStamp => "Height Stamp",
        }
    }

    pub fn constraint_kind(self) -> TerrainConstraintKind {
        match self {
            Self::CoastlinePolygon | Self::LandmassPolygon => TerrainConstraintKind::Coastline,
            Self::MountainSpine | Self::RidgeSpline => TerrainConstraintKind::Ridge,
            Self::ValleySpline | Self::CanyonPath => TerrainConstraintKind::Valley,
            Self::RiverPath => TerrainConstraintKind::River,
            Self::PlateauPolygon => TerrainConstraintKind::Plateau,
            Self::LakeBasin => TerrainConstraintKind::MinElevation,
            Self::Volcano | Self::UpliftCentre | Self::HeightStamp => TerrainConstraintKind::Ridge,
        }
    }
}

/// World-space axis-aligned bounds in metres.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct WorldBounds {
    pub min_x: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_z: f32,
}

impl WorldBounds {
    pub fn from_points(points: &[(f32, f32)], pad: f32) -> Self {
        if points.is_empty() {
            return Self::default();
        }
        let mut min_x = f32::MAX;
        let mut min_z = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_z = f32::MIN;
        for &(x, z) in points {
            min_x = min_x.min(x);
            min_z = min_z.min(z);
            max_x = max_x.max(x);
            max_z = max_z.max(z);
        }
        Self {
            min_x: min_x - pad,
            min_z: min_z - pad,
            max_x: max_x + pad,
            max_z: max_z + pad,
        }
    }

    pub fn intersects_page(&self, origin_x: f32, origin_z: f32, extent: f32) -> bool {
        let max_x = origin_x + extent;
        let max_z = origin_z + extent;
        self.min_x <= max_x
            && self.max_x >= origin_x
            && self.min_z <= max_z
            && self.max_z >= origin_z
    }

    pub fn expand(&self, other: &WorldBounds) -> WorldBounds {
        if self.max_x < self.min_x {
            return *other;
        }
        if other.max_x < other.min_x {
            return *self;
        }
        WorldBounds {
            min_x: self.min_x.min(other.min_x),
            min_z: self.min_z.min(other.min_z),
            max_x: self.max_x.max(other.max_x),
            max_z: self.max_z.max(other.max_z),
        }
    }
}

/// Editable shape object in the World Design model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeObject {
    pub id: ShapeObjectId,
    pub name: String,
    pub kind: ShapeKind,
    /// Control points in normalized UV (0–1) for compile compatibility with authoring.
    pub points: Vec<SculptPoint>,
    /// Width / influence in metres.
    pub width_m: f32,
    /// Inner falloff (0–1 of width).
    #[serde(default)]
    pub inner_falloff: f32,
    /// Outer falloff (0–1 of width).
    #[serde(default = "default_outer")]
    pub outer_falloff: f32,
    /// Constraint value (elevation delta, protect strength, etc.).
    pub value: f32,
    pub strength: f32,
    /// Translation in UV.
    #[serde(default)]
    pub offset_u: f32,
    #[serde(default)]
    pub offset_v: f32,
    #[serde(default)]
    pub rotation_deg: f32,
    #[serde(default = "default_scale")]
    pub scale: f32,
    #[serde(default)]
    pub enabled: bool,
    /// Managed constraint layer this compiles into (if any).
    #[serde(default)]
    pub compiled_layer: Option<LayerId>,
}

fn default_outer() -> f32 {
    1.0
}

fn default_scale() -> f32 {
    1.0
}

impl ShapeObject {
    pub fn new(name: impl Into<String>, kind: ShapeKind) -> Self {
        Self {
            id: ShapeObjectId::new(),
            name: name.into(),
            kind,
            points: Vec::new(),
            width_m: 120.0,
            inner_falloff: 0.25,
            outer_falloff: 1.0,
            value: 30.0,
            strength: 0.85,
            offset_u: 0.0,
            offset_v: 0.0,
            rotation_deg: 0.0,
            scale: 1.0,
            enabled: true,
            compiled_layer: None,
        }
    }

    pub fn with_points(mut self, points: Vec<SculptPoint>) -> Self {
        self.points = points;
        self
    }

    pub fn transformed_points(&self) -> Vec<SculptPoint> {
        let cx = 0.5 + self.offset_u;
        let cy = 0.5 + self.offset_v;
        let theta = self.rotation_deg.to_radians();
        let (s, c) = theta.sin_cos();
        let scale = self.scale.max(1e-4);
        self.points
            .iter()
            .map(|p| {
                let dx = (p.u - 0.5) * scale;
                let dy = (p.v - 0.5) * scale;
                let rx = dx * c - dy * s;
                let ry = dx * s + dy * c;
                SculptPoint {
                    u: (cx + rx).clamp(0.0, 1.0),
                    v: (cy + ry).clamp(0.0, 1.0),
                    pressure: p.pressure,
                }
            })
            .collect()
    }

    pub fn world_bounds(&self, world_size_x: f32, world_size_z: f32) -> WorldBounds {
        let pts: Vec<(f32, f32)> = self
            .transformed_points()
            .iter()
            .map(|p| (p.u * world_size_x, p.v * world_size_z))
            .collect();
        WorldBounds::from_points(&pts, self.width_m)
    }

    pub fn to_constraint(&self) -> TerrainConstraint {
        TerrainConstraint {
            kind: self.kind.constraint_kind(),
            points: self.transformed_points(),
            width_m: self.width_m,
            value: self.value,
            strength: self.strength,
        }
    }

    pub fn translate_uv(&mut self, du: f32, dv: f32) {
        self.offset_u += du;
        self.offset_v += dv;
    }

    /// Move control point `index` so its transformed UV matches `(tu, tv)`.
    pub fn set_point_world_uv(&mut self, index: usize, tu: f32, tv: f32) -> bool {
        let Some(p) = self.points.get_mut(index) else {
            return false;
        };
        let cx = 0.5 + self.offset_u;
        let cy = 0.5 + self.offset_v;
        let theta = self.rotation_deg.to_radians();
        let (s, c) = theta.sin_cos();
        let scale = self.scale.max(1e-4);
        let rx = tu - cx;
        let ry = tv - cy;
        let dx = (rx * c + ry * s) / scale;
        let dy = (-rx * s + ry * c) / scale;
        p.u = (dx + 0.5).clamp(0.0, 1.0);
        p.v = (dy + 0.5).clamp(0.0, 1.0);
        true
    }

    pub fn duplicate(&self) -> Self {
        let mut s = self.clone();
        s.id = ShapeObjectId::new();
        s.name = format!("{} Copy", self.name);
        s.offset_u += 0.02;
        s.offset_v += 0.02;
        s.compiled_layer = None;
        s
    }
}

/// Document-level shape collection.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ShapeObjectStore {
    pub shapes: Vec<ShapeObject>,
    #[serde(default)]
    pub selected: Option<ShapeObjectId>,
    /// Layer id of the managed TerrainConstraints layer shapes compile into.
    #[serde(default)]
    pub managed_constraints_layer: Option<LayerId>,
}

impl ShapeObjectStore {
    pub fn get(&self, id: ShapeObjectId) -> Option<&ShapeObject> {
        self.shapes.iter().find(|s| s.id == id)
    }

    pub fn get_mut(&mut self, id: ShapeObjectId) -> Option<&mut ShapeObject> {
        self.shapes.iter_mut().find(|s| s.id == id)
    }

    pub fn push(&mut self, shape: ShapeObject) -> ShapeObjectId {
        let id = shape.id;
        self.shapes.push(shape);
        self.selected = Some(id);
        id
    }

    pub fn remove(&mut self, id: ShapeObjectId) -> Option<ShapeObject> {
        let idx = self.shapes.iter().position(|s| s.id == id)?;
        let removed = self.shapes.remove(idx);
        if self.selected == Some(id) {
            self.selected = self.shapes.first().map(|s| s.id);
        }
        Some(removed)
    }

    /// Compile all enabled shapes into constraint params.
    pub fn compile_constraints(&self) -> TerrainConstraintParams {
        TerrainConstraintParams {
            preview_strength: 0.6,
            constraints: self
                .shapes
                .iter()
                .filter(|s| s.enabled && !s.points.is_empty())
                .map(|s| s.to_constraint())
                .collect(),
        }
    }

    pub fn dirty_bounds(&self, world_size_x: f32, world_size_z: f32) -> WorldBounds {
        let mut b = WorldBounds::default();
        for s in &self.shapes {
            if s.enabled {
                b = b.expand(&s.world_bounds(world_size_x, world_size_z));
            }
        }
        b
    }

    /// Tropical Island starter shapes (UV space).
    pub fn tropical_island_shapes() -> Self {
        let coastline = ShapeObject::new("Island Coastline", ShapeKind::CoastlinePolygon)
            .with_points(vec![
                SculptPoint {
                    u: 0.28,
                    v: 0.42,
                    pressure: 1.0,
                },
                SculptPoint {
                    u: 0.38,
                    v: 0.28,
                    pressure: 1.0,
                },
                SculptPoint {
                    u: 0.58,
                    v: 0.26,
                    pressure: 1.0,
                },
                SculptPoint {
                    u: 0.74,
                    v: 0.40,
                    pressure: 1.0,
                },
                SculptPoint {
                    u: 0.72,
                    v: 0.62,
                    pressure: 1.0,
                },
                SculptPoint {
                    u: 0.52,
                    v: 0.76,
                    pressure: 1.0,
                },
                SculptPoint {
                    u: 0.32,
                    v: 0.68,
                    pressure: 1.0,
                },
            ]);
        let mut coastline = coastline;
        coastline.width_m = 90.0;
        coastline.value = 0.0;
        coastline.strength = 0.9;

        let mut uplift =
            ShapeObject::new("Main Uplift", ShapeKind::UpliftCentre).with_points(vec![
                SculptPoint {
                    u: 0.48,
                    v: 0.50,
                    pressure: 1.0,
                },
            ]);
        uplift.width_m = 220.0;
        uplift.value = 55.0;
        uplift.strength = 0.85;

        let mut spine =
            ShapeObject::new("Mountain Spine", ShapeKind::MountainSpine).with_points(vec![
                SculptPoint {
                    u: 0.34,
                    v: 0.58,
                    pressure: 0.7,
                },
                SculptPoint {
                    u: 0.47,
                    v: 0.48,
                    pressure: 1.0,
                },
                SculptPoint {
                    u: 0.62,
                    v: 0.43,
                    pressure: 0.65,
                },
            ]);
        spine.width_m = 145.0;
        spine.value = 42.0;
        spine.strength = 0.9;

        let mut store = Self::default();
        store.push(coastline);
        store.push(uplift);
        store.push(spine);
        store
    }

    /// Alpine range starter shapes.
    pub fn alpine_shapes() -> Self {
        let mut spine =
            ShapeObject::new("Main Range Spine", ShapeKind::MountainSpine).with_points(vec![
                SculptPoint {
                    u: 0.18,
                    v: 0.62,
                    pressure: 0.8,
                },
                SculptPoint {
                    u: 0.42,
                    v: 0.48,
                    pressure: 1.0,
                },
                SculptPoint {
                    u: 0.68,
                    v: 0.40,
                    pressure: 0.9,
                },
                SculptPoint {
                    u: 0.88,
                    v: 0.52,
                    pressure: 0.7,
                },
            ]);
        spine.width_m = 120.0;
        spine.value = 55.0;
        spine.strength = 0.95;

        let mut uplift =
            ShapeObject::new("Range Uplift", ShapeKind::UpliftCentre).with_points(vec![
                SculptPoint {
                    u: 0.5,
                    v: 0.48,
                    pressure: 1.0,
                },
            ]);
        uplift.width_m = 280.0;
        uplift.value = 70.0;
        uplift.strength = 0.85;

        let mut ridge = ShapeObject::new("Side Ridge", ShapeKind::RidgeSpline).with_points(vec![
            SculptPoint {
                u: 0.30,
                v: 0.30,
                pressure: 0.7,
            },
            SculptPoint {
                u: 0.55,
                v: 0.35,
                pressure: 1.0,
            },
            SculptPoint {
                u: 0.78,
                v: 0.28,
                pressure: 0.6,
            },
        ]);
        ridge.width_m = 90.0;
        ridge.value = 35.0;

        let mut store = Self::default();
        store.push(spine);
        store.push(uplift);
        store.push(ridge);
        store
    }

    /// Desert mesa / canyon starter shapes.
    pub fn desert_shapes() -> Self {
        let mut mesa =
            ShapeObject::new("Mesa Outline", ShapeKind::LandmassPolygon).with_points(vec![
                SculptPoint {
                    u: 0.28,
                    v: 0.32,
                    pressure: 1.0,
                },
                SculptPoint {
                    u: 0.62,
                    v: 0.28,
                    pressure: 1.0,
                },
                SculptPoint {
                    u: 0.72,
                    v: 0.55,
                    pressure: 1.0,
                },
                SculptPoint {
                    u: 0.40,
                    v: 0.68,
                    pressure: 1.0,
                },
            ]);
        mesa.width_m = 60.0;
        mesa.value = 40.0;

        let mut uplift =
            ShapeObject::new("Mesa Uplift", ShapeKind::UpliftCentre).with_points(vec![
                SculptPoint {
                    u: 0.48,
                    v: 0.45,
                    pressure: 1.0,
                },
            ]);
        uplift.width_m = 200.0;
        uplift.value = 50.0;

        let mut store = Self::default();
        store.push(mesa);
        store.push(uplift);
        store
    }

    /// River valley starter shapes.
    pub fn river_valley_shapes() -> Self {
        let mut corridor =
            ShapeObject::new("Valley Corridor", ShapeKind::RidgeSpline).with_points(vec![
                SculptPoint {
                    u: 0.15,
                    v: 0.20,
                    pressure: 0.6,
                },
                SculptPoint {
                    u: 0.40,
                    v: 0.45,
                    pressure: 1.0,
                },
                SculptPoint {
                    u: 0.70,
                    v: 0.65,
                    pressure: 0.9,
                },
                SculptPoint {
                    u: 0.88,
                    v: 0.82,
                    pressure: 0.7,
                },
            ]);
        corridor.width_m = 160.0;
        corridor.value = -20.0;
        corridor.strength = 0.8;

        let mut uplift =
            ShapeObject::new("Valley Flanks", ShapeKind::UpliftCentre).with_points(vec![
                SculptPoint {
                    u: 0.35,
                    v: 0.55,
                    pressure: 1.0,
                },
            ]);
        uplift.width_m = 320.0;
        uplift.value = 45.0;

        let mut store = Self::default();
        store.push(corridor);
        store.push(uplift);
        store
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_produces_constraints() {
        let store = ShapeObjectStore::tropical_island_shapes();
        let params = store.compile_constraints();
        assert_eq!(params.constraints.len(), 3);
        let b = store.dirty_bounds(8192.0, 8192.0);
        assert!(b.max_x > b.min_x);
    }

    #[test]
    fn translate_and_duplicate() {
        let mut s = ShapeObject::new("Ridge", ShapeKind::RidgeSpline).with_points(vec![
            SculptPoint {
                u: 0.4,
                v: 0.5,
                pressure: 1.0,
            },
            SculptPoint {
                u: 0.6,
                v: 0.5,
                pressure: 1.0,
            },
        ]);
        s.translate_uv(0.1, 0.0);
        let pts = s.transformed_points();
        assert!((pts[0].u - 0.5).abs() < 0.05);
        let d = s.duplicate();
        assert_ne!(d.id, s.id);
    }

    #[test]
    fn set_point_world_uv_moves_control() {
        let mut s = ShapeObject::new("Ridge", ShapeKind::RidgeSpline).with_points(vec![
            SculptPoint {
                u: 0.4,
                v: 0.5,
                pressure: 1.0,
            },
            SculptPoint {
                u: 0.6,
                v: 0.5,
                pressure: 1.0,
            },
        ]);
        assert!(s.set_point_world_uv(0, 0.3, 0.4));
        let pts = s.transformed_points();
        assert!((pts[0].u - 0.3).abs() < 1e-3);
        assert!((pts[0].v - 0.4).abs() < 1e-3);
    }
}
