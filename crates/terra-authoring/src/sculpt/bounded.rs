use super::{SculptPoint, SculptStroke, DEFAULT_RECONCILE};

/// Ordered legacy bounded-UV sculpt records.
///
/// The record vector is deliberately private so coordinate-space selection and
/// future evaluator-facing invariants cannot be bypassed by callers.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedSculptStore {
    records: Vec<SculptStroke>,
    reconcile: f32,
}

impl Default for BoundedSculptStore {
    fn default() -> Self {
        Self::from_records(Vec::new(), DEFAULT_RECONCILE)
    }
}

impl BoundedSculptStore {
    pub fn from_records(records: Vec<SculptStroke>, reconcile: f32) -> Self {
        Self { records, reconcile }
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

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &SculptStroke> {
        self.records.iter()
    }

    pub fn get(&self, index: usize) -> Option<&SculptStroke> {
        self.records.get(index)
    }

    pub fn append(&mut self, record: SculptStroke) -> usize {
        let index = self.records.len();
        self.records.push(record);
        index
    }

    /// Append a record or extend the compatible active tail in history order.
    pub fn append_or_extend(&mut self, record: SculptStroke, continuing: bool) -> usize {
        let append_to_tail = continuing
            && self.records.last().is_some_and(|tail| {
                tail.enabled
                    && tail.kind == record.kind
                    && compatible_radius(tail.radius_m, record.radius_m)
            });
        if append_to_tail {
            let index = self.records.len() - 1;
            self.records[index].points.extend(record.points);
            index
        } else {
            self.append(record)
        }
    }

    pub fn extend_last(&mut self, point: SculptPoint) -> Option<usize> {
        let index = self.records.len().checked_sub(1)?;
        self.records[index].points.push(point);
        Some(index)
    }

    pub fn replace(&mut self, index: usize, replacement: SculptStroke) -> Option<SculptStroke> {
        let record = self.records.get_mut(index)?;
        Some(std::mem::replace(record, replacement))
    }

    pub fn set_enabled(&mut self, index: usize, enabled: bool) -> Option<bool> {
        let record = self.records.get_mut(index)?;
        let previous = record.enabled;
        record.enabled = enabled;
        Some(previous)
    }

    pub fn delete(&mut self, index: usize) -> Option<SculptStroke> {
        (index < self.records.len()).then(|| self.records.remove(index))
    }

    pub fn merge(&mut self, source: &Self) {
        self.records.extend(source.records.iter().cloned());
    }

    pub(crate) fn records(&self) -> &[SculptStroke] {
        &self.records
    }
}

fn compatible_radius(previous: f32, replacement: f32) -> bool {
    (previous - replacement).abs() <= replacement.max(1.0) * 0.05
}
