use super::{TerrainDemandClass, TerrainTileKey};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Complete semantic identity of tile content. This is deliberately independent
/// of residency: the atlas/cache remains the sole authority for published pages.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct TerrainContentStamp {
    pub document_revision: u64,
    pub plan_revision: u64,
    pub output_revision: u64,
    pub content_revision: u64,
}

/// Deduplication identity required by the terrain work contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TerrainTileWorkKey {
    pub tile: TerrainTileKey,
    pub plan_revision: u64,
    pub output_revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainTileWorkSource {
    CpuHeight,
    GpuPyramid,
    GpuCompiledPlan,
}

/// One concrete request to publish a height tile from an already existing CPU
/// heightfield or immutable GPU pyramid. Both sources have production executors.
#[derive(Debug, Clone)]
pub struct TerrainTileWorkRequest {
    pub key: TerrainTileWorkKey,
    pub content: TerrainContentStamp,
    pub source: TerrainTileWorkSource,
    pub class: TerrainDemandClass,
    pub visible: bool,
    pub projected_error_px: f32,
    pub distance_m: f32,
    pub estimated_us: u64,
}

impl TerrainTileWorkRequest {
    fn is_required_coverage(&self) -> bool {
        matches!(
            self.class,
            TerrainDemandClass::CoarseCoverage | TerrainDemandClass::FallbackAncestor
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerrainTileWorkBudget {
    pub max_estimated_us: u64,
    pub max_items: usize,
    pub max_in_flight: usize,
}

impl TerrainTileWorkBudget {
    pub const fn new(max_estimated_us: u64, max_items: usize, max_in_flight: usize) -> Self {
        Self {
            max_estimated_us,
            max_items,
            max_in_flight,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TerrainTileWorkLease {
    pub id: u64,
    pub request: TerrainTileWorkRequest,
    started_at: Instant,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TerrainTileWorkStats {
    pub queued: usize,
    pub in_flight: usize,
    pub enqueued: u64,
    pub deduplicated: u64,
    pub reprioritized: u64,
    pub cancelled: u64,
    pub stale_dropped: u64,
    pub resident_skipped: u64,
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub budget_overruns: u64,
    pub estimated_us_dispatched: u64,
    pub queue_latency_us_total: u64,
    pub queue_latency_us_max: u64,
    pub completion_latency_us_total: u64,
    pub completion_latency_us_max: u64,
}

#[derive(Debug, Clone)]
struct QueuedEntry {
    request: TerrainTileWorkRequest,
    first_seen_epoch: u64,
    enqueued_at: Instant,
}

/// Bounded, revision-aware policy queue. It stores demand and execution state,
/// never atlas residency, page handles, or page-table data.
#[derive(Debug, Clone)]
pub struct TerrainTileWorkScheduler {
    queued: HashMap<TerrainTileWorkKey, QueuedEntry>,
    in_flight: HashMap<u64, QueuedEntry>,
    live_content: Option<TerrainContentStamp>,
    capacity: usize,
    epoch: u64,
    next_id: u64,
    stats: TerrainTileWorkStats,
}

impl Default for TerrainTileWorkScheduler {
    fn default() -> Self {
        Self::new(256)
    }
}

impl TerrainTileWorkScheduler {
    const FORCE_RUN_AFTER_EPOCHS: u64 = 60;

    pub fn new(capacity: usize) -> Self {
        Self {
            queued: HashMap::new(),
            in_flight: HashMap::new(),
            live_content: None,
            capacity: capacity.max(1),
            epoch: 0,
            next_id: 0,
            stats: TerrainTileWorkStats::default(),
        }
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
        self.trim_to_capacity();
        self.refresh_counts();
    }

    /// Replace the current demand set while preserving age for retained work.
    pub fn reconcile(
        &mut self,
        live_content: TerrainContentStamp,
        requests: impl IntoIterator<Item = TerrainTileWorkRequest>,
    ) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.live_content != Some(live_content) {
            self.stats.stale_dropped = self
                .stats
                .stale_dropped
                .saturating_add(self.queued.len() as u64);
            self.stats.cancelled = self
                .stats
                .cancelled
                .saturating_add(self.in_flight.len() as u64);
            self.queued.clear();
            self.in_flight.clear();
            self.live_content = Some(live_content);
        }

        let mut demanded = HashSet::new();
        for mut request in requests {
            if request.content != live_content {
                self.stats.stale_dropped = self.stats.stale_dropped.saturating_add(1);
                continue;
            }
            request.estimated_us = request.estimated_us.max(1);
            if !demanded.insert(request.key.clone()) {
                self.stats.deduplicated = self.stats.deduplicated.saturating_add(1);
            }
            if self.in_flight.values().any(|entry| {
                entry.request.key == request.key && entry.request.content == live_content
            }) {
                self.stats.deduplicated = self.stats.deduplicated.saturating_add(1);
                continue;
            }
            match self.queued.get_mut(&request.key) {
                Some(entry) => {
                    entry.request = request;
                    self.stats.deduplicated = self.stats.deduplicated.saturating_add(1);
                    self.stats.reprioritized = self.stats.reprioritized.saturating_add(1);
                }
                None => {
                    self.queued.insert(
                        request.key.clone(),
                        QueuedEntry {
                            request,
                            first_seen_epoch: self.epoch,
                            enqueued_at: Instant::now(),
                        },
                    );
                    self.stats.enqueued = self.stats.enqueued.saturating_add(1);
                }
            }
        }

        let before = self.queued.len();
        self.queued.retain(|key, _| demanded.contains(key));
        self.stats.cancelled = self
            .stats
            .cancelled
            .saturating_add(before.saturating_sub(self.queued.len()) as u64);
        self.trim_to_capacity();
        self.refresh_counts();
    }

    pub fn mark_resident_skip(&mut self, key: &TerrainTileWorkKey) {
        if self.queued.remove(key).is_some() {
            self.stats.resident_skipped = self.stats.resident_skipped.saturating_add(1);
        }
        self.refresh_counts();
    }

    pub fn skip_in_flight_as_resident(&mut self, lease: TerrainTileWorkLease) {
        if self.in_flight.remove(&lease.id).is_some() {
            self.stats.resident_skipped = self.stats.resident_skipped.saturating_add(1);
        }
        self.refresh_counts();
    }

    pub fn dequeue_budgeted(&mut self, budget: TerrainTileWorkBudget) -> Vec<TerrainTileWorkLease> {
        self.epoch = self.epoch.wrapping_add(1);
        if budget.max_items == 0 || self.in_flight.len() >= budget.max_in_flight {
            return Vec::new();
        }
        let required_pending = self
            .queued
            .values()
            .any(|entry| entry.request.is_required_coverage());
        let mut candidates: Vec<_> = self
            .queued
            .iter()
            .filter(|(_, entry)| !required_pending || entry.request.is_required_coverage())
            .map(|(key, entry)| (key.clone(), entry.clone()))
            .collect();
        let epoch = self.epoch;
        candidates.sort_by(|(_, a), (_, b)| compare_entries(a, b, epoch));

        let available_slots = budget
            .max_in_flight
            .saturating_sub(self.in_flight.len())
            .min(budget.max_items);
        let mut used = 0u64;
        let mut selected = Vec::new();
        for (key, entry) in candidates {
            if selected.len() >= available_slots {
                break;
            }
            let estimate = entry.request.estimated_us;
            let fits = used.saturating_add(estimate) <= budget.max_estimated_us;
            if !fits && !selected.is_empty() {
                continue;
            }
            if !fits {
                self.stats.budget_overruns = self.stats.budget_overruns.saturating_add(1);
            }
            let Some(entry) = self.queued.remove(&key) else {
                continue;
            };
            self.next_id = self.next_id.wrapping_add(1).max(1);
            let id = self.next_id;
            let now = Instant::now();
            let queue_us = micros_u64(now.saturating_duration_since(entry.enqueued_at));
            self.stats.queue_latency_us_total =
                self.stats.queue_latency_us_total.saturating_add(queue_us);
            self.stats.queue_latency_us_max = self.stats.queue_latency_us_max.max(queue_us);
            self.stats.submitted = self.stats.submitted.saturating_add(1);
            self.stats.estimated_us_dispatched =
                self.stats.estimated_us_dispatched.saturating_add(estimate);
            used = used.saturating_add(estimate);
            self.in_flight.insert(id, entry.clone());
            selected.push(TerrainTileWorkLease {
                id,
                request: entry.request,
                started_at: now,
            });
        }
        self.refresh_counts();
        selected
    }

    pub fn complete(&mut self, lease: TerrainTileWorkLease, live: TerrainContentStamp) -> bool {
        let Some(entry) = self.in_flight.remove(&lease.id) else {
            return false;
        };
        let elapsed = micros_u64(Instant::now().saturating_duration_since(lease.started_at));
        if self.live_content != Some(live) || entry.request.content != live {
            self.stats.stale_dropped = self.stats.stale_dropped.saturating_add(1);
            self.refresh_counts();
            return false;
        }
        self.stats.completed = self.stats.completed.saturating_add(1);
        self.stats.completion_latency_us_total = self
            .stats
            .completion_latency_us_total
            .saturating_add(elapsed);
        self.stats.completion_latency_us_max = self.stats.completion_latency_us_max.max(elapsed);
        self.refresh_counts();
        true
    }

    pub fn fail(&mut self, lease: TerrainTileWorkLease) {
        if self.in_flight.remove(&lease.id).is_some() {
            self.stats.failed = self.stats.failed.saturating_add(1);
        }
        self.refresh_counts();
    }

    pub fn clear(&mut self) {
        self.stats.cancelled = self
            .stats
            .cancelled
            .saturating_add((self.queued.len() + self.in_flight.len()) as u64);
        self.queued.clear();
        self.in_flight.clear();
        self.live_content = None;
        self.refresh_counts();
    }

    pub fn is_empty(&self) -> bool {
        self.queued.is_empty() && self.in_flight.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queued.len()
    }

    pub fn queued_requests(&self) -> impl Iterator<Item = &TerrainTileWorkRequest> {
        self.queued.values().map(|entry| &entry.request)
    }

    pub fn stats(&self) -> TerrainTileWorkStats {
        self.stats
    }

    pub fn live_content(&self) -> Option<TerrainContentStamp> {
        self.live_content
    }

    pub fn lease_is_live(&self, id: u64, content: TerrainContentStamp) -> bool {
        self.live_content == Some(content)
            && self
                .in_flight
                .get(&id)
                .is_some_and(|entry| entry.request.content == content)
    }

    fn trim_to_capacity(&mut self) {
        while self.queued.len().saturating_add(self.in_flight.len()) > self.capacity {
            let epoch = self.epoch;
            let victim = self
                .queued
                .iter()
                .filter(|(_, entry)| !entry.request.is_required_coverage())
                .max_by(|(_, a), (_, b)| compare_entries(a, b, epoch))
                .map(|(key, _)| key.clone())
                .or_else(|| self.queued.keys().next().cloned());
            let Some(victim) = victim else {
                break;
            };
            self.queued.remove(&victim);
            self.stats.cancelled = self.stats.cancelled.saturating_add(1);
        }
    }

    fn refresh_counts(&mut self) {
        self.stats.queued = self.queued.len();
        self.stats.in_flight = self.in_flight.len();
    }
}

fn compare_entries(a: &QueuedEntry, b: &QueuedEntry, epoch: u64) -> Ordering {
    let a_age = epoch.saturating_sub(a.first_seen_epoch);
    let b_age = epoch.saturating_sub(b.first_seen_epoch);
    let a_forced = a_age >= TerrainTileWorkScheduler::FORCE_RUN_AFTER_EPOCHS;
    let b_forced = b_age >= TerrainTileWorkScheduler::FORCE_RUN_AFTER_EPOCHS;
    b_forced
        .cmp(&a_forced)
        .then_with(|| {
            if a_forced && b_forced {
                b_age.cmp(&a_age)
            } else {
                Ordering::Equal
            }
        })
        .then_with(|| b.request.visible.cmp(&a.request.visible))
        .then_with(|| a.request.class.cmp(&b.request.class))
        .then_with(|| {
            b.request
                .key
                .tile
                .address
                .lod
                .cmp(&a.request.key.tile.address.lod)
        })
        .then_with(|| {
            b.request
                .projected_error_px
                .total_cmp(&a.request.projected_error_px)
        })
        .then_with(|| a.request.distance_m.total_cmp(&b.request.distance_m))
        .then_with(|| b_age.cmp(&a_age))
        .then_with(|| {
            a.request
                .key
                .tile
                .address
                .coord
                .z
                .cmp(&b.request.key.tile.address.coord.z)
        })
        .then_with(|| {
            a.request
                .key
                .tile
                .address
                .coord
                .x
                .cmp(&b.request.key.tile.address.coord.x)
        })
}

fn micros_u64(duration: std::time::Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FieldId;

    fn stamp(content_revision: u64) -> TerrainContentStamp {
        TerrainContentStamp {
            document_revision: 3,
            plan_revision: 5,
            output_revision: 7,
            content_revision,
        }
    }

    fn request(
        level: u8,
        tx: u32,
        class: TerrainDemandClass,
        visible: bool,
        error: f32,
        estimate: u64,
    ) -> TerrainTileWorkRequest {
        TerrainTileWorkRequest {
            key: TerrainTileWorkKey {
                tile: TerrainTileKey::new(
                    None,
                    FieldId::Height,
                    crate::TileAddress::new(
                        crate::Lod::try_new(10 - level).unwrap(),
                        crate::TileCoord {
                            x: i64::from(tx),
                            z: 0,
                        },
                    ),
                ),
                plan_revision: 5,
                output_revision: 7,
            },
            content: stamp(11),
            source: TerrainTileWorkSource::GpuPyramid,
            class,
            visible,
            projected_error_px: error,
            distance_m: tx as f32,
            estimated_us: estimate,
        }
    }

    #[test]
    fn coarse_visible_work_outranks_fine_and_offscreen_work() {
        let mut scheduler = TerrainTileWorkScheduler::new(8);
        let fine = request(4, 0, TerrainDemandClass::Refinement, true, 100.0, 10);
        let offscreen = request(0, 1, TerrainDemandClass::CoarseCoverage, false, 100.0, 10);
        let coarse = request(0, 2, TerrainDemandClass::CoarseCoverage, true, 1.0, 10);
        scheduler.reconcile(stamp(11), [fine, offscreen, coarse]);
        let selected = scheduler.dequeue_budgeted(TerrainTileWorkBudget::new(10, 1, 8));
        assert_eq!(selected[0].request.key.tile.address.coord.x, 2);
    }

    #[test]
    fn duplicate_demand_is_one_work_item() {
        let mut scheduler = TerrainTileWorkScheduler::new(8);
        let work = request(2, 4, TerrainDemandClass::Refinement, true, 3.0, 10);
        scheduler.reconcile(stamp(11), [work.clone(), work]);
        assert_eq!(scheduler.len(), 1);
        assert!(scheduler.stats().deduplicated >= 1);
    }

    #[test]
    fn newer_content_supersedes_queued_and_in_flight_work() {
        let mut scheduler = TerrainTileWorkScheduler::new(8);
        scheduler.reconcile(
            stamp(11),
            [request(2, 1, TerrainDemandClass::Refinement, true, 3.0, 10)],
        );
        let lease = scheduler
            .dequeue_budgeted(TerrainTileWorkBudget::new(10, 1, 8))
            .pop()
            .unwrap();
        let mut newer = request(2, 1, TerrainDemandClass::Refinement, true, 3.0, 10);
        newer.content = stamp(12);
        scheduler.reconcile(stamp(12), [newer]);
        assert!(!scheduler.complete(lease, stamp(12)));
        assert!(scheduler.stats().cancelled >= 1);
    }

    #[test]
    fn oversized_live_request_runs_and_records_overrun() {
        let mut scheduler = TerrainTileWorkScheduler::new(8);
        scheduler.reconcile(
            stamp(11),
            [request(
                2,
                1,
                TerrainDemandClass::Refinement,
                true,
                3.0,
                100,
            )],
        );
        let selected = scheduler.dequeue_budgeted(TerrainTileWorkBudget::new(10, 1, 8));
        assert_eq!(selected.len(), 1);
        assert_eq!(scheduler.stats().budget_overruns, 1);
    }

    #[test]
    fn camera_reprioritizes_retained_work_without_resetting_queue() {
        let mut scheduler = TerrainTileWorkScheduler::new(8);
        let a = request(2, 1, TerrainDemandClass::Refinement, true, 10.0, 10);
        let b = request(2, 2, TerrainDemandClass::Refinement, true, 1.0, 10);
        scheduler.reconcile(stamp(11), [a.clone(), b.clone()]);
        let mut a2 = a;
        let mut b2 = b;
        a2.projected_error_px = 1.0;
        b2.projected_error_px = 20.0;
        scheduler.reconcile(stamp(11), [a2, b2]);
        let selected = scheduler.dequeue_budgeted(TerrainTileWorkBudget::new(10, 1, 8));
        assert_eq!(selected[0].request.key.tile.address.coord.x, 2);
        assert_eq!(scheduler.stats().reprioritized, 2);
    }

    #[test]
    fn aging_live_request_cannot_starve_under_repeated_new_demand() {
        let mut scheduler = TerrainTileWorkScheduler::new(128);
        let old = request(2, 1, TerrainDemandClass::Refinement, true, 1.0, 10);
        let mut selected_old = false;
        for epoch in 0..=TerrainTileWorkScheduler::FORCE_RUN_AFTER_EPOCHS {
            let hot = request(
                2,
                100 + epoch as u32,
                TerrainDemandClass::Refinement,
                true,
                100.0,
                10,
            );
            scheduler.reconcile(stamp(11), [old.clone(), hot]);
            let lease = scheduler
                .dequeue_budgeted(TerrainTileWorkBudget::new(10, 1, 128))
                .pop()
                .unwrap();
            selected_old = lease.request.key == old.key;
            scheduler.complete(lease, stamp(11));
            if selected_old {
                break;
            }
        }
        assert!(
            selected_old,
            "the retained request must eventually be forced"
        );
    }
}
