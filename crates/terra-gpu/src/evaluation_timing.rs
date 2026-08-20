//! Delayed, non-blocking timestamp queries for compiled terrain evaluations.

use std::sync::mpsc::{self, TryRecvError};

const SLOT_COUNT: usize = 3;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuEvaluationTraceContext {
    pub frame_id: u64,
    pub generation: u64,
    pub evaluation_id: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuEvaluationTiming {
    pub context: GpuEvaluationTraceContext,
    pub gpu_us: u64,
}

enum SlotState {
    Idle,
    Submitted(GpuEvaluationTraceContext),
    Mapping {
        context: GpuEvaluationTraceContext,
        receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    },
}

struct TimingSlot {
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    state: SlotState,
}

pub(crate) struct GpuEvaluationTimer {
    query_set: wgpu::QuerySet,
    slots: Vec<TimingSlot>,
    next_slot: usize,
    completed: Vec<GpuEvaluationTiming>,
}

impl GpuEvaluationTimer {
    pub(crate) fn try_new(device: &wgpu::Device) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("terra-evaluation-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: (SLOT_COUNT * 2) as u32,
        });
        let slots = (0..SLOT_COUNT)
            .map(|index| TimingSlot {
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("terra-evaluation-timestamp-resolve-{index}")),
                    size: 16,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("terra-evaluation-timestamp-readback-{index}")),
                    size: 16,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: SlotState::Idle,
            })
            .collect();
        Some(Self {
            query_set,
            slots,
            next_slot: 0,
            completed: Vec::new(),
        })
    }

    pub(crate) fn begin(&mut self, encoder: &mut wgpu::CommandEncoder) -> Option<usize> {
        let slot = (0..SLOT_COUNT)
            .map(|offset| (self.next_slot + offset) % SLOT_COUNT)
            .find(|index| matches!(self.slots[*index].state, SlotState::Idle))?;
        encoder.write_timestamp(&self.query_set, (slot * 2) as u32);
        Some(slot)
    }

    pub(crate) fn finish(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        slot: usize,
        context: GpuEvaluationTraceContext,
    ) {
        let first_query = (slot * 2) as u32;
        encoder.write_timestamp(&self.query_set, first_query + 1);
        encoder.resolve_query_set(
            &self.query_set,
            first_query..first_query + 2,
            &self.slots[slot].resolve,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.slots[slot].resolve,
            0,
            &self.slots[slot].readback,
            0,
            16,
        );
        self.slots[slot].state = SlotState::Submitted(context);
        self.next_slot = (slot + 1) % SLOT_COUNT;
    }

    pub(crate) fn poll(&mut self, device: &wgpu::Device, timestamp_period_ns: f32) {
        let _ = device.poll(wgpu::Maintain::Poll);
        for slot in &mut self.slots {
            let state = std::mem::replace(&mut slot.state, SlotState::Idle);
            slot.state = match state {
                SlotState::Idle => SlotState::Idle,
                SlotState::Submitted(context) => {
                    let (sender, receiver) = mpsc::channel();
                    slot.readback
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |result| {
                            let _ = sender.send(result);
                        });
                    SlotState::Mapping { context, receiver }
                }
                SlotState::Mapping { context, receiver } => match receiver.try_recv() {
                    Ok(Ok(())) => {
                        let data = slot.readback.slice(..).get_mapped_range();
                        let begin = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0; 8]));
                        let end = u64::from_le_bytes(data[8..16].try_into().unwrap_or([0; 8]));
                        drop(data);
                        slot.readback.unmap();
                        let gpu_us = ((end.saturating_sub(begin) as f64)
                            * f64::from(timestamp_period_ns)
                            / 1_000.0) as u64;
                        self.completed.push(GpuEvaluationTiming { context, gpu_us });
                        SlotState::Idle
                    }
                    Ok(Err(_)) | Err(TryRecvError::Disconnected) => SlotState::Idle,
                    Err(TryRecvError::Empty) => SlotState::Mapping { context, receiver },
                },
            };
        }
    }

    pub(crate) fn take_completed(&mut self) -> Vec<GpuEvaluationTiming> {
        std::mem::take(&mut self.completed)
    }
}
