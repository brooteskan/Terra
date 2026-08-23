//! Asynchronous sampled comparison outside regional terrain edits.

use std::sync::mpsc::{self, TryRecvError};

use bytemuck::{Pod, Zeroable};
use terra_core::tiling::SampleRect;
use terra_gpu::output_identity::{GpuOutputCoverage, GpuOutputId, GpuTerrainOutputIdentity};

use crate::TerrainPresentationMode;

const READBACK_SLOTS: usize = 4;
const DEFAULT_DEBUG_PROBES: u32 = 64;
const MAX_PROBES: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ProbeParams {
    width: u32,
    height: u32,
    rect_x: u32,
    rect_y: u32,
    rect_w: u32,
    rect_h: u32,
    probe_count: u32,
    compare: u32,
    epsilon: f32,
    _pad0: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RawProbeResult {
    failed: u32,
    max_delta_bits: u32,
    first_probe_encoded: u32,
    compared: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct TerrainIntegrityProbeResult {
    pub candidate: GpuTerrainOutputIdentity,
    pub expected_base: Option<GpuOutputId>,
    pub rect: SampleRect,
    pub passed: bool,
    pub max_delta: f32,
    pub first_failing_probe: Option<u32>,
    pub probes_compared: u32,
}

#[derive(Clone, Copy)]
struct ProbeContext {
    candidate: GpuTerrainOutputIdentity,
    expected_base: Option<GpuOutputId>,
    rect: SampleRect,
}

enum SlotState {
    Idle,
    Submitted(ProbeContext),
    Mapping {
        context: ProbeContext,
        receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    },
}

struct ReadbackSlot {
    buffer: wgpu::Buffer,
    state: SlotState,
}

pub(crate) struct TerrainIntegrityProbe {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    baseline: wgpu::Buffer,
    result: wgpu::Buffer,
    params: wgpu::Buffer,
    readback: Vec<ReadbackSlot>,
    next_slot: usize,
    probe_count: u32,
    epsilon: f32,
    skipped_readbacks: u64,
}

impl TerrainIntegrityProbe {
    pub(crate) fn try_new(
        device: &wgpu::Device,
        pipelines: &terra_gpu::PipelineCacheRegistry,
    ) -> Option<Self> {
        let configured = std::env::var("TERRA_GPU_INTEGRITY_PROBE").ok();
        let probe_count = match configured.as_deref() {
            Some(value) if value.eq_ignore_ascii_case("off") || value == "0" => 0,
            Some(value) => value.parse::<u32>().unwrap_or(DEFAULT_DEBUG_PROBES),
            None if cfg!(debug_assertions) => DEFAULT_DEBUG_PROBES,
            None => 0,
        }
        .clamp(0, MAX_PROBES);
        if probe_count == 0 {
            return None;
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain-integrity-probe-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-integrity-probe-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain-integrity-probe-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/outside_region_probe.wgsl").into(),
            ),
        });
        let pipeline = pipelines.compute_pipeline("terrain-integrity-probe-pipeline", || {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("terrain-integrity-probe-pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: pipelines.driver_cache(),
            })
        });
        let baseline = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain-integrity-probe-baseline"),
            size: u64::from(probe_count) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let result = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain-integrity-probe-result"),
            size: std::mem::size_of::<RawProbeResult>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain-integrity-probe-params"),
            size: std::mem::size_of::<ProbeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = (0..READBACK_SLOTS)
            .map(|index| ReadbackSlot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("terrain-integrity-probe-readback-{index}")),
                    size: std::mem::size_of::<RawProbeResult>() as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: SlotState::Idle,
            })
            .collect();
        Some(Self {
            pipeline,
            bgl,
            baseline,
            result,
            params,
            readback,
            next_slot: 0,
            probe_count,
            epsilon: 1.0e-5,
            skipped_readbacks: 0,
        })
    }

    pub(crate) fn submit(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::TextureView,
        candidate: GpuTerrainOutputIdentity,
        actual_mode: TerrainPresentationMode,
        actual_rect: Option<SampleRect>,
    ) {
        let full = SampleRect {
            x: 0,
            y: 0,
            w: candidate.extent.0,
            h: candidate.extent.1,
        };
        let rect = actual_rect.unwrap_or(full);
        let compare = actual_mode == TerrainPresentationMode::RegionalCopy;
        let expected_base = match candidate.coverage {
            GpuOutputCoverage::Patch { expected_base, .. } => expected_base,
            GpuOutputCoverage::WholeField => None,
        };
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&ProbeParams {
                width: candidate.extent.0,
                height: candidate.extent.1,
                rect_x: rect.x,
                rect_y: rect.y,
                rect_w: rect.w,
                rect_h: rect.h,
                probe_count: self.probe_count,
                compare: u32::from(compare),
                epsilon: self.epsilon,
                _pad0: [0; 3],
            }),
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-integrity-probe-bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.baseline.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain-integrity-probe-encoder"),
        });
        encoder.clear_buffer(&self.result, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain-integrity-probe"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(self.probe_count.div_ceil(64), 1, 1);
        }
        if let Some(index) = (0..self.readback.len())
            .map(|offset| (self.next_slot + offset) % self.readback.len())
            .find(|index| matches!(self.readback[*index].state, SlotState::Idle))
        {
            encoder.copy_buffer_to_buffer(
                &self.result,
                0,
                &self.readback[index].buffer,
                0,
                std::mem::size_of::<RawProbeResult>() as u64,
            );
            self.readback[index].state = SlotState::Submitted(ProbeContext {
                candidate,
                expected_base,
                rect,
            });
            self.next_slot = (index + 1) % self.readback.len();
        } else {
            self.skipped_readbacks = self.skipped_readbacks.saturating_add(1);
        }
        queue.submit(Some(encoder.finish()));
    }

    pub(crate) fn poll(&mut self, device: &wgpu::Device) -> Vec<TerrainIntegrityProbeResult> {
        let _ = device.poll(wgpu::Maintain::Poll);
        let mut completed = Vec::new();
        for slot in &mut self.readback {
            let state = std::mem::replace(&mut slot.state, SlotState::Idle);
            slot.state = match state {
                SlotState::Idle => SlotState::Idle,
                SlotState::Submitted(context) => {
                    let (sender, receiver) = mpsc::channel();
                    slot.buffer
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |result| {
                            let _ = sender.send(result);
                        });
                    SlotState::Mapping { context, receiver }
                }
                SlotState::Mapping { context, receiver } => match receiver.try_recv() {
                    Ok(Ok(())) => {
                        let bytes = slot.buffer.slice(..).get_mapped_range();
                        let raw = *bytemuck::from_bytes::<RawProbeResult>(&bytes);
                        drop(bytes);
                        slot.buffer.unmap();
                        completed.push(TerrainIntegrityProbeResult {
                            candidate: context.candidate,
                            expected_base: context.expected_base,
                            rect: context.rect,
                            passed: raw.failed == 0,
                            max_delta: f32::from_bits(raw.max_delta_bits),
                            first_failing_probe: (raw.first_probe_encoded != 0)
                                .then_some(self.probe_count - raw.first_probe_encoded),
                            probes_compared: raw.compared,
                        });
                        SlotState::Idle
                    }
                    Ok(Err(_)) | Err(TryRecvError::Disconnected) => SlotState::Idle,
                    Err(TryRecvError::Empty) => SlotState::Mapping { context, receiver },
                },
            };
        }
        completed
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn production_probe_never_uses_a_synchronous_wait() {
        let source = include_str!("integrity_probe.rs");
        let forbidden = ["Maintain::", "Wait"].concat();
        assert!(!source.contains(&forbidden));
    }
}
