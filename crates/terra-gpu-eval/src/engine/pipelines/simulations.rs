//! GPU evaluator pipelines simulations implementation.

use super::*;

impl GpuTerrainEngine {
    pub(in super::super) fn river_accumulation_iters(&self, quality: PreviewQuality) -> u32 {
        match quality {
            PreviewQuality::Draft => 12,
            PreviewQuality::Medium => 32,
            PreviewQuality::Full | PreviewQuality::Export => {
                (self.metrics.width.min(self.metrics.height) / 4).clamp(48, 160)
            }
        }
    }

    /// Run the shared iterative D8 accumulation preview against `height_slot` and
    /// return the scratch texture containing the final accumulation field.
    pub(in super::super) fn run_river_accumulation(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        height_slot: TexSlot,
        quality: PreviewQuality,
    ) -> TexSlot {
        let iters = self.river_accumulation_iters(quality);

        // Seed accumulation with unit rainfall.
        self.fill_slot(device, queue, encoder, TexSlot::WaterA, 1.0);
        self.fill_slot(device, queue, encoder, TexSlot::WaterB, 0.0);

        let accum_u = RiverAccumU {
            width: self.metrics.width,
            height: self.metrics.height,
            _p0: 0.0,
            _p1: 0.0,
        };

        let mut src_a = true;
        for _ in 0..iters {
            let u_buf = self.write_uniform(device, queue, &accum_u);
            let (acc_in, acc_out) = if src_a {
                (&self.water_a.view, &self.water_b.view)
            } else {
                (&self.water_b.view, &self.water_a.view)
            };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("river-accum-bg"),
                layout: &self.river_accum.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(self.view_of(height_slot)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(acc_in),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(acc_out),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("river-accum"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.river_accum.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(
                    self.metrics.width.div_ceil(8),
                    self.metrics.height.div_ceil(8),
                    1,
                );
            }
            src_a = !src_a;
        }

        if src_a {
            TexSlot::WaterA
        } else {
            TexSlot::WaterB
        }
    }

    pub(in super::super) fn run_river_carve(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &terra_core::layer::RiverCarveParams,
        quality: PreviewQuality,
    ) {
        let height_slot = if self.current == 0 {
            TexSlot::Ping
        } else {
            TexSlot::Pong
        };
        let accum_slot = self.run_river_accumulation(device, queue, encoder, height_slot, quality);

        let carve_u = RiverCarveU {
            width: self.metrics.width,
            height: self.metrics.height,
            threshold: p.accumulation_threshold.max(1.0),
            depth: p.depth,
            channel_width: p.width.max(1.0),
            bank_smooth: p.bank_smooth.max(0.0),
            max_radius: match quality {
                PreviewQuality::Draft => 12,
                PreviewQuality::Medium => 20,
                PreviewQuality::Full | PreviewQuality::Export => RIVER_CARVE_MAX_RADIUS,
            },
            _pad: 0,
        };
        let u_buf = self.write_uniform(device, queue, &carve_u);
        let acc_view = self.view_of(accum_slot);
        let (src, dst) = if self.current == 0 {
            (&self.ping.view, &self.pong.view)
        } else {
            (&self.pong.view, &self.ping.view)
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("river-carve-bg"),
            layout: &self.river_carve.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(acc_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(dst),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("river-carve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.river_carve.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
        }
        self.swap_current();
        self.expand_range(self.approx_range.0 - p.depth * 2.0, self.approx_range.1);
    }

    pub(in super::super) fn run_stream_power(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &terra_core::layer::StreamPowerParams,
        quality: PreviewQuality,
    ) {
        let authored_iters = match quality {
            PreviewQuality::Draft => p.iterations.clamp(1, 8),
            PreviewQuality::Medium => p.iterations.clamp(1, 16),
            PreviewQuality::Full | PreviewQuality::Export => p.iterations.max(1),
        };
        // Mirror the CPU processor's default level-step averaging. Non-default
        // authored level controls are rejected by the planner, and document-level
        // schedule variants remain part of the configuration-fallback work in #138.
        let base = match quality {
            PreviewQuality::Draft => draft_sim_levels(self.metrics.width),
            PreviewQuality::Medium | PreviewQuality::Full | PreviewQuality::Export => {
                default_sim_levels(self.metrics.width)
            }
        };
        let levels = LevelStepSettings::default().schedule_for_filter(
            base,
            p.level_count,
            p.start_level,
            p.level_step_strength,
            &p.level_step_curve,
            quality,
        );
        let (iter_scale, effect_scale) = if levels.is_empty() {
            (1.0, 1.0)
        } else {
            let count = levels.len() as f32;
            (
                levels.iter().map(|level| level.iter_scale).sum::<f32>() / count,
                levels.iter().map(|level| level.effect_scale).sum::<f32>() / count,
            )
        };
        let iters = ((authored_iters as f32 * iter_scale).round() as u32)
            .max(1)
            .min(match quality {
                PreviewQuality::Draft => self.max_sim_iters_per_tick.max(1),
                PreviewQuality::Medium | PreviewQuality::Full | PreviewQuality::Export => u32::MAX,
            });
        let drainage_stride = match quality {
            PreviewQuality::Draft => p.drainage_reuse_stride.max(2),
            PreviewQuality::Medium | PreviewQuality::Full | PreviewQuality::Export => {
                p.drainage_reuse_stride.max(1)
            }
        };
        let cell_area = (self.metrics.dx() * self.metrics.dz()).max(1.0e-6);
        let uniform = StreamPowerU {
            width: self.metrics.width,
            height: self.metrics.height,
            k: (p.k * effect_scale).max(0.0),
            m: p.m.max(0.0),
            n: p.n.max(0.0),
            dt: p.dt.max(0.0),
            uplift: p.uplift_rate,
            base_level: p.base_level,
            cell_area,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        self.fill_slot(
            device,
            queue,
            encoder,
            TexSlot::Hardness,
            p.hardness.clamp(0.0, 1.0),
        );

        let mut accum_slot = TexSlot::WaterA;
        for iter in 0..iters {
            let height_slot = if self.current == 0 {
                TexSlot::Ping
            } else {
                TexSlot::Pong
            };
            if iter == 0 || iter % drainage_stride == 0 {
                accum_slot =
                    self.run_river_accumulation(device, queue, encoder, height_slot, quality);
            }

            let u_buf = self.write_uniform(device, queue, &uniform);
            let (src, dst) = if self.current == 0 {
                (&self.ping.view, &self.pong.view)
            } else {
                (&self.pong.view, &self.ping.view)
            };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stream-power-bg"),
                layout: &self.stream_power.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(self.view_of(accum_slot)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.hardness.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(dst),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("stream-power-incision"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.stream_power.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(
                    self.metrics.width.div_ceil(8),
                    self.metrics.height.div_ceil(8),
                    1,
                );
            }
            self.swap_current();
        }

        let iter_scale = iters as f32;
        self.expand_range(
            (self.approx_range.0 - 50.0 * iter_scale + p.uplift_rate * iter_scale)
                .max(p.base_level),
            (self.approx_range.1 + p.uplift_rate.max(0.0) * iter_scale).max(p.base_level),
        );
    }
}
