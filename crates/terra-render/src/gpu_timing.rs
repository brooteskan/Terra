//! Optional GPU timestamp queries for pass-level profiling.
//!
//! Readback is asynchronous with frame latency — never `Maintain::Wait` on the
//! interactive path. Maps complete via `Maintain::Poll` when the callback fires.
//!
//! Query indices are allocated densely per frame so skipped passes (e.g. shadow
//! while path-tracing) never leave holes that DX12 refuses to resolve.

use std::sync::mpsc::{self, TryRecvError};

/// Enough room for terrain / shadow / path-trace / temporal / denoise pairs.
const QUERY_CAPACITY: u32 = 12;
const READBACK_LATENCY: u64 = 2;

#[derive(Debug, Clone, Copy, Default)]
pub struct GpuTimings {
    pub terrain_us: u64,
    pub shadow_us: u64,
    pub path_trace_us: u64,
    pub temporal_us: u64,
    pub denoise_us: u64,
    pub supported: bool,
    pub source_frame: u64,
    pub context: GpuPresentationTraceContext,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuPresentationTraceContext {
    pub frame_id: u64,
    pub generation: u64,
    pub evaluation_id: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct PassPair {
    begin: u32,
    end: u32,
    valid: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct FrameStampLayout {
    terrain: PassPair,
    shadow: PassPair,
    path_trace: PassPair,
    temporal: PassPair,
    denoise: PassPair,
    count: u32,
}

#[derive(Debug, Clone, Copy)]
struct SlotState {
    submitted_frame: u64,
    layout: FrameStampLayout,
    context: GpuPresentationTraceContext,
}

enum MapPending {
    Idle,
    Waiting {
        idx: usize,
        count: u32,
        rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    },
}

/// Owns a timestamp query set + resolve buffer ring when the device supports it.
pub struct GpuTimestampTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback: [wgpu::Buffer; 2],
    slots: [SlotState; 2],
    write_idx: usize,
    frame: u64,
    period_ns: f32,
    /// Dense allocation cursor for the current frame's query indices.
    next_query: u32,
    /// Pass pairs stamped this frame (resolved at end-of-frame).
    current: FrameStampLayout,
    current_context: GpuPresentationTraceContext,
    last: GpuTimings,
    map_pending: MapPending,
}

impl GpuTimestampTimer {
    pub fn try_new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("terra-gpu-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_CAPACITY,
        });
        let resolve_size = u64::from(QUERY_CAPACITY) * 8;
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terra-gpu-ts-resolve"),
            size: resolve_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let make_readback = |i: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("terra-gpu-ts-readback-{i}")),
                size: resolve_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        Some(Self {
            query_set,
            resolve_buffer,
            readback: [make_readback(0), make_readback(1)],
            slots: [SlotState {
                submitted_frame: 0,
                layout: FrameStampLayout::default(),
                context: GpuPresentationTraceContext::default(),
            }; 2],
            write_idx: 0,
            frame: 0,
            period_ns: queue.get_timestamp_period(),
            next_query: 0,
            current: FrameStampLayout::default(),
            current_context: GpuPresentationTraceContext::default(),
            last: GpuTimings {
                supported: true,
                ..Default::default()
            },
            map_pending: MapPending::Idle,
        })
    }

    pub fn last(&self) -> GpuTimings {
        self.last
    }

    pub fn begin_frame(&mut self, context: GpuPresentationTraceContext) {
        self.frame = self.frame.wrapping_add(1);
        self.next_query = 0;
        self.current = FrameStampLayout::default();
        self.current_context = context;
    }

    fn alloc_pair(&mut self) -> Option<(u32, u32)> {
        if self.next_query + 2 > QUERY_CAPACITY {
            return None;
        }
        let begin = self.next_query;
        let end = begin + 1;
        self.next_query += 2;
        self.current.count = self.next_query;
        Some((begin, end))
    }

    fn stamp_render(
        &mut self,
        pair_slot: fn(&mut FrameStampLayout) -> &mut PassPair,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let (begin, end) = self.alloc_pair()?;
        let pair = pair_slot(&mut self.current);
        *pair = PassPair {
            begin,
            end,
            valid: true,
        };
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    pub fn terrain_timestamp_writes(&mut self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.stamp_render(|l| &mut l.terrain)
    }

    pub fn shadow_timestamp_writes(&mut self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.stamp_render(|l| &mut l.shadow)
    }

    pub fn temporal_timestamp_writes(&mut self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.stamp_render(|l| &mut l.temporal)
    }

    pub fn denoise_timestamp_writes(&mut self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.stamp_render(|l| &mut l.denoise)
    }

    /// Compute-pass timestamps (path tracer).
    pub fn path_trace_timestamp_writes(&mut self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let (begin, end) = self.alloc_pair()?;
        self.current.path_trace = PassPair {
            begin,
            end,
            valid: true,
        };
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    pub fn progressive_timestamp_writes(
        &mut self,
    ) -> (
        Option<wgpu::RenderPassTimestampWrites<'_>>,
        Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let Some((tb, te)) = self.alloc_pair() else {
            return (None, None);
        };
        let Some((db, de)) = self.alloc_pair() else {
            // Keep temporal allocation consistent if denoise cannot fit.
            self.current.temporal = PassPair {
                begin: tb,
                end: te,
                valid: true,
            };
            return (
                Some(wgpu::RenderPassTimestampWrites {
                    query_set: &self.query_set,
                    beginning_of_pass_write_index: Some(tb),
                    end_of_pass_write_index: Some(te),
                }),
                None,
            );
        };
        self.current.temporal = PassPair {
            begin: tb,
            end: te,
            valid: true,
        };
        self.current.denoise = PassPair {
            begin: db,
            end: de,
            valid: true,
        };
        (
            Some(wgpu::RenderPassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: Some(tb),
                end_of_pass_write_index: Some(te),
            }),
            Some(wgpu::RenderPassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: Some(db),
                end_of_pass_write_index: Some(de),
            }),
        )
    }

    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let count = self.current.count;
        if count == 0 {
            return;
        }
        // Cannot resolve into a buffer that is still mapped.
        if !matches!(self.map_pending, MapPending::Idle) {
            self.current = FrameStampLayout::default();
            self.next_query = 0;
            return;
        }
        let mut idx = self.write_idx;
        if !self.slot_is_writable(idx) {
            idx = 1 - idx;
            if !self.slot_is_writable(idx) {
                self.current = FrameStampLayout::default();
                self.next_query = 0;
                return;
            }
        }

        // Dense 0..count — every index was written by a pass this frame.
        encoder.resolve_query_set(&self.query_set, 0..count, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback[idx],
            0,
            u64::from(count) * 8,
        );
        self.slots[idx].submitted_frame = self.frame;
        self.slots[idx].layout = self.current;
        self.slots[idx].context = self.current_context;
        self.write_idx = 1 - idx;
        self.current = FrameStampLayout::default();
        self.next_query = 0;
    }

    /// Non-blocking: poll device, finish any pending map, or start mapping a ripe slot.
    pub fn poll_readback(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::Maintain::Poll);

        if let MapPending::Waiting { idx, count, rx } = &self.map_pending {
            match rx.try_recv() {
                Ok(Ok(())) => {
                    let idx = *idx;
                    let count = *count;
                    self.map_pending = MapPending::Idle;
                    self.finish_mapped(idx, count);
                }
                Ok(Err(_)) => {
                    let idx = *idx;
                    self.map_pending = MapPending::Idle;
                    self.slots[idx].submitted_frame = 0;
                }
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    let idx = *idx;
                    self.map_pending = MapPending::Idle;
                    self.slots[idx].submitted_frame = 0;
                }
            }
            return;
        }

        let Some(idx) = self.pick_readable_slot() else {
            return;
        };
        let count = self.slots[idx].layout.count.max(1);
        let buffer = &self.readback[idx];
        let slice = buffer.slice(..u64::from(count) * 8);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.map_pending = MapPending::Waiting { idx, count, rx };
        let _ = device.poll(wgpu::Maintain::Poll);
    }

    fn finish_mapped(&mut self, idx: usize, count: u32) {
        let buffer = &self.readback[idx];
        let slice = buffer.slice(..u64::from(count) * 8);
        let data = slice.get_mapped_range();
        let stamps: Vec<u64> = data
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
            .collect();
        drop(data);
        buffer.unmap();

        let layout = self.slots[idx].layout;
        let source_frame = self.slots[idx].submitted_frame;
        let context = self.slots[idx].context;
        self.slots[idx].submitted_frame = 0;

        let to_us = |pair: PassPair| -> u64 {
            if !pair.valid {
                return 0;
            }
            let begin = pair.begin as usize;
            let end = pair.end as usize;
            if begin >= stamps.len() || end >= stamps.len() {
                return 0;
            }
            let delta = stamps[end].saturating_sub(stamps[begin]);
            ((delta as f64) * f64::from(self.period_ns) / 1000.0) as u64
        };
        self.last.terrain_us = to_us(layout.terrain);
        self.last.shadow_us = to_us(layout.shadow);
        self.last.path_trace_us = to_us(layout.path_trace);
        self.last.temporal_us = to_us(layout.temporal);
        self.last.denoise_us = to_us(layout.denoise);
        self.last.supported = true;
        self.last.source_frame = source_frame;
        self.last.context = context;
    }

    fn slot_is_writable(&self, idx: usize) -> bool {
        if matches!(self.map_pending, MapPending::Waiting { idx: pending, .. } if pending == idx) {
            return false;
        }
        let slot = &self.slots[idx];
        slot.submitted_frame == 0
            || self.frame.saturating_sub(slot.submitted_frame) >= READBACK_LATENCY
    }

    fn pick_readable_slot(&self) -> Option<usize> {
        let mut best: Option<(u64, usize)> = None;
        for (idx, slot) in self.slots.iter().enumerate() {
            if slot.submitted_frame == 0 {
                continue;
            }
            if self.frame.saturating_sub(slot.submitted_frame) < READBACK_LATENCY {
                continue;
            }
            match best {
                Some((f, _)) if slot.submitted_frame >= f => {}
                _ => best = Some((slot.submitted_frame, idx)),
            }
        }
        best.map(|(_, idx)| idx)
    }
}

pub fn requested_timestamp_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let available = adapter.features();
    let mut features = wgpu::Features::empty();
    if available.contains(wgpu::Features::TIMESTAMP_QUERY) {
        features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    features
}
