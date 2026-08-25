use super::{WorldSculptPoint, WorldSculptStroke, DEFAULT_RECONCILE};
use crate::{AuthoredFeatureId, FeatureChange};
use std::collections::BTreeSet;
use std::fmt;
use terra_world::{WorldBounds, WorldError, WorldPosition};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SculptStoreError {
    World(WorldError),
    DuplicateFeatureId(AuthoredFeatureId),
    CoordinateSpaceMismatch,
}

impl fmt::Display for SculptStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::World(error) => error.fmt(f),
            Self::DuplicateFeatureId(id) => {
                write!(f, "authored feature id {:?} occurs more than once", id.0)
            }
            Self::CoordinateSpaceMismatch => {
                write!(
                    f,
                    "bounded-UV and world-metre sculpt histories cannot be mixed"
                )
            }
        }
    }
}

impl std::error::Error for SculptStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::World(error) => Some(error),
            Self::DuplicateFeatureId(_) | Self::CoordinateSpaceMismatch => None,
        }
    }
}

impl From<WorldError> for SculptStoreError {
    fn from(value: WorldError) -> Self {
        Self::World(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeletedWorldSculpt {
    pub index: usize,
    pub record: WorldSculptStroke,
    pub change: FeatureChange,
}

/// Ordered, validated world-metre sculpt records.
#[derive(Debug, Clone, PartialEq)]
pub struct WorldSculptStore {
    records: Vec<WorldSculptStroke>,
    reconcile: f32,
}

impl Default for WorldSculptStore {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            reconcile: DEFAULT_RECONCILE,
        }
    }
}

impl WorldSculptStore {
    pub fn try_from_records(
        records: Vec<WorldSculptStroke>,
        reconcile: f32,
    ) -> Result<Self, SculptStoreError> {
        validate_records(&records)?;
        Ok(Self { records, reconcile })
    }

    pub fn reconcile(&self) -> f32 {
        self.reconcile
    }

    pub fn set_reconcile(&mut self, reconcile: f32) {
        self.reconcile = reconcile;
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &WorldSculptStroke> {
        self.records.iter()
    }

    pub fn get(&self, id: AuthoredFeatureId) -> Option<&WorldSculptStroke> {
        self.records.iter().find(|record| record.id == id)
    }

    pub fn append(&mut self, record: WorldSculptStroke) -> Result<FeatureChange, SculptStoreError> {
        if self.get(record.id).is_some() {
            return Err(SculptStoreError::DuplicateFeatureId(record.id));
        }
        let replacement_bounds = active_bounds(&record)?;
        let id = record.id;
        self.records.push(record);
        Ok(FeatureChange {
            id,
            previous_bounds: None,
            replacement_bounds,
        })
    }

    /// Append a record or extend the compatible active tail in history order.
    pub fn append_or_extend(
        &mut self,
        record: WorldSculptStroke,
        continuing: bool,
    ) -> Result<FeatureChange, SculptStoreError> {
        let extend_tail = continuing
            && self.records.last().is_some_and(|tail| {
                tail.enabled
                    && tail.kind == record.kind
                    && compatible_radius(tail.radius_m, record.radius_m)
            });
        if !extend_tail {
            return self.append(record);
        }

        let index = self.records.len() - 1;
        let previous_bounds = active_bounds(&self.records[index])?;
        let mut candidate = self.records[index].clone();
        candidate.points.extend(record.points);
        candidate.strength = record.strength;
        candidate.target_height = record.target_height;
        candidate.falloff = record.falloff;
        let replacement_bounds = active_bounds(&candidate)?;
        let id = candidate.id;
        self.records[index] = candidate;
        Ok(FeatureChange {
            id,
            previous_bounds,
            replacement_bounds,
        })
    }

    pub fn replace(
        &mut self,
        id: AuthoredFeatureId,
        mut replacement: WorldSculptStroke,
    ) -> Result<Option<FeatureChange>, SculptStoreError> {
        let Some(index) = self.position(id) else {
            return Ok(None);
        };
        let previous_bounds = active_bounds(&self.records[index])?;
        replacement.id = id;
        let replacement_bounds = active_bounds(&replacement)?;
        self.records[index] = replacement;
        Ok(Some(FeatureChange {
            id,
            previous_bounds,
            replacement_bounds,
        }))
    }

    pub fn translate(
        &mut self,
        id: AuthoredFeatureId,
        dx_m: f64,
        dz_m: f64,
    ) -> Result<Option<FeatureChange>, SculptStoreError> {
        WorldPosition::try_new(dx_m, dz_m)?;
        let Some(index) = self.position(id) else {
            return Ok(None);
        };
        let previous_bounds = active_bounds(&self.records[index])?;
        let mut candidate = self.records[index].clone();
        candidate.points = candidate
            .points
            .iter()
            .map(|point| {
                WorldPosition::try_new(point.position.x_m() + dx_m, point.position.z_m() + dz_m)
                    .map(|position| WorldSculptPoint {
                        position,
                        pressure: point.pressure,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let replacement_bounds = active_bounds(&candidate)?;
        self.records[index] = candidate;
        Ok(Some(FeatureChange {
            id,
            previous_bounds,
            replacement_bounds,
        }))
    }

    pub fn set_enabled(
        &mut self,
        id: AuthoredFeatureId,
        enabled: bool,
    ) -> Result<Option<FeatureChange>, SculptStoreError> {
        let Some(index) = self.position(id) else {
            return Ok(None);
        };
        let previous_bounds = active_bounds(&self.records[index])?;
        let mut candidate = self.records[index].clone();
        candidate.enabled = enabled;
        let replacement_bounds = active_bounds(&candidate)?;
        self.records[index] = candidate;
        Ok(Some(FeatureChange {
            id,
            previous_bounds,
            replacement_bounds,
        }))
    }

    pub fn delete(
        &mut self,
        id: AuthoredFeatureId,
    ) -> Result<Option<DeletedWorldSculpt>, SculptStoreError> {
        let Some(index) = self.position(id) else {
            return Ok(None);
        };
        let previous_bounds = active_bounds(&self.records[index])?;
        let record = self.records.remove(index);
        Ok(Some(DeletedWorldSculpt {
            index,
            record,
            change: FeatureChange {
                id,
                previous_bounds,
                replacement_bounds: None,
            },
        }))
    }

    /// Append a source history atomically, assigning fresh IDs to incoming records.
    pub fn merge(&mut self, source: &Self) -> Result<Vec<FeatureChange>, SculptStoreError> {
        let mut used = self
            .records
            .iter()
            .map(|record| record.id)
            .collect::<BTreeSet<_>>();
        let mut incoming = source.records.clone();
        let mut changes = Vec::with_capacity(incoming.len());
        for record in &mut incoming {
            record.id = next_unused_id(&mut used);
            let replacement_bounds = active_bounds(record)?;
            changes.push(FeatureChange {
                id: record.id,
                previous_bounds: None,
                replacement_bounds,
            });
        }
        self.records.extend(incoming);
        Ok(changes)
    }

    /// Assign fresh identities while preserving record order and semantics.
    pub fn reseed_feature_ids(&mut self) {
        let mut used = BTreeSet::new();
        for record in &mut self.records {
            record.id = next_unused_id(&mut used);
        }
    }

    pub fn reseeded_clone(&self) -> Self {
        let mut clone = self.clone();
        clone.reseed_feature_ids();
        clone
    }

    pub fn active_bounds(
        &self,
        id: AuthoredFeatureId,
    ) -> Result<Option<WorldBounds>, SculptStoreError> {
        self.get(id)
            .map(active_bounds)
            .transpose()
            .map(Option::flatten)
    }

    pub fn index_records(&self) -> Result<Vec<(AuthoredFeatureId, WorldBounds)>, SculptStoreError> {
        let mut records = Vec::new();
        for record in &self.records {
            if let Some(bounds) = active_bounds(record)? {
                records.push((record.id, bounds));
            }
        }
        Ok(records)
    }

    pub(crate) fn records(&self) -> &[WorldSculptStroke] {
        &self.records
    }

    pub(crate) fn indexed_ordinals(&self) -> Result<Vec<(usize, WorldBounds)>, SculptStoreError> {
        let mut records = Vec::new();
        for (index, record) in self.records.iter().enumerate() {
            if let Some(bounds) = active_bounds(record)? {
                records.push((index, bounds));
            }
        }
        Ok(records)
    }

    fn position(&self, id: AuthoredFeatureId) -> Option<usize> {
        self.records.iter().position(|record| record.id == id)
    }
}

fn validate_records(records: &[WorldSculptStroke]) -> Result<(), SculptStoreError> {
    let mut ids = BTreeSet::new();
    for record in records {
        if !ids.insert(record.id) {
            return Err(SculptStoreError::DuplicateFeatureId(record.id));
        }
        record_bounds(record)?;
    }
    Ok(())
}

fn record_bounds(record: &WorldSculptStroke) -> Result<Option<WorldBounds>, SculptStoreError> {
    if !record.radius_m.is_finite() || record.radius_m <= 0.0 {
        return Err(WorldError::InvalidWorldExpansion.into());
    }
    let Some(bounds) = WorldBounds::try_from_points(record.points.iter().map(|p| p.position))?
    else {
        return Ok(None);
    };
    Ok(Some(bounds.checked_expand(
        f64::from(record.radius_m),
        f64::from(record.radius_m),
    )?))
}

fn active_bounds(record: &WorldSculptStroke) -> Result<Option<WorldBounds>, SculptStoreError> {
    if record.enabled {
        record_bounds(record)
    } else {
        // Disabled records still validate their persisted geometry so every
        // store state can be safely enabled later without discovering damage.
        record_bounds(record).map(|_| None)
    }
}

fn compatible_radius(previous: f32, replacement: f32) -> bool {
    (previous - replacement).abs() <= replacement.max(1.0) * 0.05
}

fn next_unused_id(used: &mut BTreeSet<AuthoredFeatureId>) -> AuthoredFeatureId {
    loop {
        let id = AuthoredFeatureId::new();
        if used.insert(id) {
            return id;
        }
    }
}
