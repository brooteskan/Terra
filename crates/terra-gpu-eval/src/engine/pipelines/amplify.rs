//! GPU evaluator pipelines amplify implementation.

use super::*;

impl GpuTerrainEngine {
    pub(in super::super) fn run_amplify_accumulation(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        level_res: u32,
        height_a: bool,
        quality: PreviewQuality,
    ) -> bool {
        let water_a = self.water_a.view.clone();
        let water_b = self.water_b.view.clone();
        self.fill_view_extent(device, queue, encoder, &water_a, [level_res; 2], 1.0);
        self.fill_view_extent(device, queue, encoder, &water_b, [level_res; 2], 0.0);
        let iterations = match quality {
            PreviewQuality::Draft => 12,
            PreviewQuality::Medium => 32,
            PreviewQuality::Full | PreviewQuality::Export => (level_res / 4).clamp(48, 160),
        };
        let uniform = RiverAccumU {
            width: level_res,
            height: level_res,
            _p0: 0.0,
            _p1: 0.0,
        };
        let mut src_a = true;
        for _ in 0..iterations {
            let buffer = self.write_uniform(device, queue, &uniform);
            let height = if height_a {
                &self.amplify_a.view
            } else {
                &self.amplify_b.view
            };
            let (acc_in, acc_out) = if src_a {
                (&self.water_a.view, &self.water_b.view)
            } else {
                (&self.water_b.view, &self.water_a.view)
            };
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("amplify-river-accum-bg"),
                layout: &self.river_accum.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(height),
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
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amplify-river-accum"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.river_accum.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
            drop(pass);
            src_a = !src_a;
        }
        src_a
    }

    pub(in super::super) fn run_multi_scale_amplify(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        authored: &MultiScaleAmplifyParams,
        quality: PreviewQuality,
    ) {
        queue.write_buffer(
            &self.simulation_invalid_state_buffer,
            0,
            bytemuck::bytes_of(&0u32),
        );
        let mut params = authored.clone();
        match quality {
            PreviewQuality::Draft => {
                params.thermal_iters = params.thermal_iters.clamp(1, 6);
                params.spe_iters = params.spe_iters.min(2);
                params.level_count = if params.level_count == 0 {
                    2
                } else {
                    params.level_count.min(2)
                };
            }
            PreviewQuality::Medium => {
                params.thermal_iters = params.thermal_iters.clamp(1, 10);
                params.spe_iters = params.spe_iters.min(4);
            }
            PreviewQuality::Full | PreviewQuality::Export => {}
        }
        let levels = match quality {
            PreviewQuality::Draft => {
                amplify_sim_levels(self.metrics.width, params.level_count.clamp(1, 2))
            }
            PreviewQuality::Medium | PreviewQuality::Full | PreviewQuality::Export => {
                amplify_sim_levels(self.metrics.width, params.level_count)
            }
        };
        let required_side = levels
            .iter()
            .map(|level| level.resolution)
            .max()
            .unwrap_or(self.metrics.width)
            .max(1);
        if self.amplify_a.width < required_side || self.amplify_a.height < required_side {
            self.hardness = HeightTex::new(device, "hardness", required_side, required_side);
            self.water_a = HeightTex::new(device, "water-a", required_side, required_side);
            self.water_b = HeightTex::new(device, "water-b", required_side, required_side);
            self.delta = HeightTex::new(device, "thermal-delta", required_side, required_side);
            self.sed_a = HeightTex::new(device, "sed-a", required_side, required_side);
            self.sed_b = HeightTex::new(device, "sed-b", required_side, required_side);
            self.rainfall = HeightTex::new(device, "rainfall", required_side, required_side);
            self.loose_sediment =
                HeightTex::new(device, "loose-sediment", required_side, required_side);
            self.outflow = RgbaTex::new(device, "hydraulic-outflow", required_side, required_side);
            self.amplify_a = HeightTex::new(device, "amplify-a", required_side, required_side);
            self.amplify_b = HeightTex::new(device, "amplify-b", required_side, required_side);
        }
        let hardness = match params.hardness_source {
            MaskSource::Constant(value) => value,
            MaskSource::None => params.hardness,
            _ => unreachable!("planner rejected non-uniform amplify hardness"),
        }
        .clamp(0.0, 1.0);
        let ridge_lock = match params.ridge_lock {
            MaskSource::Constant(value) => value,
            MaskSource::None => 0.0,
            _ => unreachable!("planner rejected non-uniform amplify ridge lock"),
        }
        .clamp(0.0, 1.0);

        let hardness_view = self.hardness.view.clone();
        let rainfall_view = self.rainfall.view.clone();
        let loose_view = self.loose_sediment.view.clone();
        for (index, level) in levels.iter().enumerate() {
            let level_res = level.resolution;
            let source_slot = if self.current == 0 {
                TexSlot::Ping
            } else {
                TexSlot::Pong
            };
            self.copy_slots(device, queue, encoder, source_slot, TexSlot::Layer);

            let downsample = AmplifyDownsampleU {
                src_width: self.metrics.width,
                src_height: self.metrics.height,
                dst_width: level_res,
                dst_height: level_res,
                area_mode: u32::from(
                    level_res < self.metrics.width || level_res < self.metrics.height,
                ),
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let buffer = self.write_uniform(device, queue, &downsample);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("amplify-downsample-bg"),
                layout: &self.amplify_downsample.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.amplify_a.view),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("amplify-downsample"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.amplify_downsample.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
            }

            self.fill_view_extent(
                device,
                queue,
                encoder,
                &hardness_view,
                [level_res; 2],
                hardness,
            );
            let fine_t = if levels.len() <= 1 {
                1.0
            } else {
                index as f32 / (levels.len() - 1) as f32
            };
            let thermal_w = (1.0 - 0.65 * fine_t) * level.iter_scale;
            let spe_w = (0.25 + 0.75 * fine_t) * level.effect_scale;
            let dep_w = fine_t * level.effect_scale;
            let level_dx = self.metrics.world_size_x / level_res.max(1) as f32;
            let level_dz = self.metrics.world_size_z / level_res.max(1) as f32;

            let thermal_iters = ((params.thermal_iters as f32 * thermal_w).round() as u32).max(1);
            let thermal = ThermalU {
                width: level_res,
                height: level_res,
                dx: level_dx,
                talus: params.talus_angle_deg.to_radians().tan() * level_dx,
                strength: (params.thermal_strength * thermal_w).clamp(0.0, 1.0),
                _p2: 0.0,
                _p3: 0.0,
                _pad: 0.0,
            };
            let mut height_a = true;
            for _ in 0..thermal_iters {
                let uniform = self.write_uniform(device, queue, &thermal);
                let (src, dst) = if height_a {
                    (&self.amplify_a.view, &self.amplify_b.view)
                } else {
                    (&self.amplify_b.view, &self.amplify_a.view)
                };
                let delta_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("amplify-thermal-delta-bg"),
                    layout: &self.thermal.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.delta.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.hardness.view),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("amplify-thermal-delta"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.thermal.pipeline);
                    pass.set_bind_group(0, &delta_group, &[]);
                    pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
                }
                let apply_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("amplify-thermal-apply-bg"),
                    layout: &self.thermal_apply.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.delta.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(dst),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("amplify-thermal-apply"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.thermal_apply.pipeline);
                    pass.set_bind_group(0, &apply_group, &[]);
                    pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
                }
                height_a = !height_a;
            }

            if params.spe_strength > 1.0e-6 && params.spe_iters > 0 {
                let spe_iters = ((params.spe_iters as f32 * spe_w).round() as u32)
                    .max(if spe_w > 0.15 { 1 } else { 0 });
                let stream_power = StreamPowerU {
                    width: level_res,
                    height: level_res,
                    k: 0.05 * params.spe_strength * spe_w,
                    m: 0.5,
                    n: 1.0,
                    dt: 0.85,
                    uplift: 0.0,
                    base_level: 0.0,
                    cell_area: (level_dx * level_dz).max(1.0e-6),
                    _pad0: 0.0,
                    _pad1: 0.0,
                    _pad2: 0.0,
                };
                for _ in 0..spe_iters {
                    let accum_a = self.run_amplify_accumulation(
                        device, queue, encoder, level_res, height_a, quality,
                    );
                    let uniform = self.write_uniform(device, queue, &stream_power);
                    let (src, dst) = if height_a {
                        (&self.amplify_a.view, &self.amplify_b.view)
                    } else {
                        (&self.amplify_b.view, &self.amplify_a.view)
                    };
                    let accumulation = if accum_a {
                        &self.water_a.view
                    } else {
                        &self.water_b.view
                    };
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("amplify-stream-power-bg"),
                        layout: &self.stream_power.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(accumulation),
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
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("amplify-stream-power"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.stream_power.pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
                    drop(pass);
                    height_a = !height_a;
                }
            }

            if params.deposition_strength > 1.0e-6 && dep_w > 0.2 {
                let hydraulic = terra_core::layer::HydraulicErosionParams {
                    iterations: ((8.0 * dep_w).round() as u32).max(2),
                    rainfall: 0.02 * dep_w,
                    evaporation: 0.015,
                    capacity: 0.12,
                    erosion: 0.12 * dep_w,
                    deposition: (0.55 * params.deposition_strength * dep_w).clamp(0.0, 1.0),
                    timestep: 0.2,
                    hardness: 0.0,
                    hardness_source: MaskSource::None,
                    fan_boost: 0.8 * params.deposition_strength,
                    floodplain_bias: 0.5 * params.deposition_strength,
                    bank_slip: 0.0,
                    sediment_softness: 0.0,
                    ..Default::default()
                };
                let water_a = self.water_a.view.clone();
                let water_b = self.water_b.view.clone();
                let sed_a = self.sed_a.view.clone();
                let sed_b = self.sed_b.view.clone();
                self.fill_view_extent(device, queue, encoder, &water_a, [level_res; 2], 0.0);
                self.fill_view_extent(device, queue, encoder, &water_b, [level_res; 2], 0.0);
                self.fill_view_extent(device, queue, encoder, &sed_a, [level_res; 2], 0.0);
                self.fill_view_extent(device, queue, encoder, &sed_b, [level_res; 2], 0.0);
                self.fill_view_extent(device, queue, encoder, &rainfall_view, [level_res; 2], 1.0);
                self.fill_view_extent(device, queue, encoder, &loose_view, [level_res; 2], 0.0);
                let uniform = HydraulicU {
                    width: level_res,
                    height: level_res,
                    timestep: clamp_timestep_cfl(hydraulic.timestep, level_dx, 4.0),
                    rainfall: hydraulic.rainfall,
                    evaporation: hydraulic.evaporation,
                    erosion: hydraulic.erosion,
                    deposition: hydraulic.deposition,
                    capacity: hydraulic.capacity,
                    fan_boost: hydraulic.fan_boost,
                    floodplain_bias: hydraulic.floodplain_bias,
                    dx: level_dx,
                    incision_bias: hydraulic.incision_bias.max(0.05),
                    bedrock_k: hydraulic.bedrock_hardness.clamp(0.0, 1.0),
                    sediment_k: hydraulic.sediment_hardness.clamp(0.0, 1.0),
                    layered: 0.0,
                    _pad1: 0.0,
                };
                let mut water_flip = false;
                for _ in 0..hydraulic.iterations {
                    let buffer = self.write_uniform(device, queue, &uniform);
                    let (height_src, height_dst) = if height_a {
                        (&self.amplify_a.view, &self.amplify_b.view)
                    } else {
                        (&self.amplify_b.view, &self.amplify_a.view)
                    };
                    let (water_src, water_dst) = if water_flip {
                        (&self.water_b.view, &self.water_a.view)
                    } else {
                        (&self.water_a.view, &self.water_b.view)
                    };
                    let (sed_src, sed_dst) = if water_flip {
                        (&self.sed_b.view, &self.sed_a.view)
                    } else {
                        (&self.sed_a.view, &self.sed_b.view)
                    };
                    let outflow_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("amplify-hydraulic-outflow-bg"),
                        layout: &self.hydraulic_outflow.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(height_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(water_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&self.outflow.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&self.rainfall.view),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("amplify-hydraulic-outflow"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.hydraulic_outflow.pipeline);
                        pass.set_bind_group(0, &outflow_group, &[]);
                        pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
                    }
                    let hydraulic_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("amplify-hydraulic-bg"),
                        layout: &self.hydraulic.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(height_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(water_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(sed_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&self.outflow.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::TextureView(height_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(water_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: wgpu::BindingResource::TextureView(sed_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 8,
                                resource: wgpu::BindingResource::TextureView(&self.hardness.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 9,
                                resource: wgpu::BindingResource::TextureView(&self.rainfall.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 10,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.loose_sediment.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 11,
                                resource: self.simulation_invalid_state_buffer.as_entire_binding(),
                            },
                        ],
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("amplify-hydraulic"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.hydraulic.pipeline);
                    pass.set_bind_group(0, &hydraulic_group, &[]);
                    pass.dispatch_workgroups(level_res.div_ceil(8), level_res.div_ceil(8), 1);
                    drop(pass);
                    height_a = !height_a;
                    water_flip = !water_flip;
                }
            }

            let blend = AmplifyBlendU {
                src_width: level_res,
                src_height: level_res,
                dst_width: self.metrics.width,
                dst_height: self.metrics.height,
                hardness,
                ridge_lock,
                lock_strength: params.lock_strength,
                detail_boost: params.detail_boost,
            };
            let buffer = self.write_uniform(device, queue, &blend);
            let processed = if height_a {
                &self.amplify_a.view
            } else {
                &self.amplify_b.view
            };
            let destination = if self.current == 0 {
                &self.pong.view
            } else {
                &self.ping.view
            };
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("amplify-upsample-blend-bg"),
                layout: &self.amplify_upsample_blend.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(processed),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(destination),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amplify-upsample-blend"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.amplify_upsample_blend.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
            drop(pass);
            self.swap_current();
        }
    }
}
