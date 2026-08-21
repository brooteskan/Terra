//! GPU evaluator pipelines dispatch implementation.

use super::*;

impl GpuTerrainEngine {
    pub(in super::super) fn eval_layer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        layer: &Layer,
        kernel: GpuKernel,
        quality: PreviewQuality,
    ) -> Result<(), GpuError> {
        #[cfg(test)]
        self.executed_kernels.push(kernel);
        match (kernel, &layer.kind) {
            (GpuKernel::HeightmapSample, LayerKind::ImportHeightmap(p)) => {
                self.run_heightmap_sample(device, queue, encoder, layer)?;
                let endpoint = p.height_offset + p.height_scale;
                self.expand_range(p.height_offset.min(endpoint), p.height_offset.max(endpoint));
                self.blend_into_current_with_mask(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                    [TexSlot::MaskOnes, TexSlot::StampMask],
                )?;
            }
            (GpuKernel::HeightmapSample, LayerKind::Stamp2d(p)) => {
                self.run_heightmap_sample(device, queue, encoder, layer)?;
                let endpoint = p.heightmap.height_offset + p.heightmap.height_scale;
                self.expand_range(
                    p.heightmap.height_offset.min(endpoint),
                    p.heightmap.height_offset.max(endpoint),
                );
                self.blend_into_current_with_mask(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                    [TexSlot::MaskOnes, TexSlot::StampMask],
                )?;
            }
            (GpuKernel::Sculpt, LayerKind::SculptBase(p)) => {
                // `layer_tex` was filled by `record_sculpt_to_layer` before this call.
                if self.last_dirty_rect.is_none() {
                    let (lo, hi) = p.sample_range();
                    self.expand_range(lo, hi);
                }
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::SculptStrokes, LayerKind::SculptStrokes(p)) => {
                // Stamp every stroke, then reconcile into `layer_tex`; the standard
                // blend below reproduces the CPU composite for the supported blends.
                let source = if self.current == 0 {
                    TexSlot::Ping
                } else {
                    TexSlot::Pong
                };
                self.run_sculpt_strokes(
                    device,
                    queue,
                    encoder,
                    layer.id(),
                    p,
                    source,
                    self.last_dirty_rect
                        .unwrap_or((0, 0, self.metrics.width, self.metrics.height)),
                );
                // Presentation range: fold in only the *absolute* stamp targets, like
                // every other kernel expands with stable values. The additive kinds
                // are relative to the (already-ranged) input, so widening by their
                // magnitude here would be relative to the accumulating range and
                // compound across incremental dabs — sinking the render's slab base
                // (`min_h - f(span)`) a little further on every drag step. Their exact
                // extent is left to the async CPU refine; height itself is unaffected.
                // Flatten is deliberately absent: its target is a mean of heights
                // already in range and it settles `h` toward that mean, so it cannot
                // exceed the current extent — and its `target_height` is ignored (#117).
                for stroke in &p.strokes {
                    if stroke.enabled
                        && matches!(
                            stroke.kind,
                            SculptStrokeKind::HeightStamp | SculptStrokeKind::PlateauStamp
                        )
                    {
                        self.expand_range(stroke.target_height, stroke.target_height);
                    }
                }
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Path, LayerKind::Path(p)) => {
                self.run_path_height(device, queue, encoder, p);
                let node_height = p
                    .nodes
                    .iter()
                    .map(|node| node.height.abs())
                    .fold(0.0, f32::max);
                let reach = p.height_offset.abs() + node_height + p.noise_strength.abs();
                self.expand_range(self.approx_range.0 - reach, self.approx_range.1 + reach);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::PolygonHeight, LayerKind::PolygonHeight(p)) => {
                self.run_polygon_height(device, queue, encoder, p);
                match p.mode {
                    PolygonHeightMode::RaiseBy => {
                        let reach = p.height.abs();
                        self.expand_range(self.approx_range.0 - reach, self.approx_range.1 + reach);
                    }
                    PolygonHeightMode::SetElevation if p.carve => {
                        self.expand_range(
                            self.approx_range.0 - p.height.abs(),
                            self.approx_range.1,
                        );
                    }
                    PolygonHeightMode::SetElevation => {
                        self.expand_range(p.height, p.height);
                    }
                }
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::ProceduralShape, LayerKind::ProceduralShape(p)) => {
                self.run_procedural_shape(device, queue, encoder, p)?;
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Fill, LayerKind::Flat(p)) => {
                self.fill_slot(device, queue, encoder, TexSlot::Layer, p.height);
                self.expand_range(p.height, p.height);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Ramp, LayerKind::Ramp(p)) => {
                let u = RampU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    height_min: p.height_min,
                    height_max: p.height_max,
                    direction: p.direction,
                    _pad: 0.0,
                };
                let u_buf = self.write_uniform(device, queue, &u);
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ramp-bg"),
                    layout: &self.ramp.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: u_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ramp"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.ramp.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(
                        self.metrics.width.div_ceil(8),
                        self.metrics.height.div_ceil(8),
                        1,
                    );
                }
                self.expand_range(p.height_min, p.height_max);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::NoiseValue(p)) => {
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    p,
                    NoiseDispatch::new(0, NoiseKernelMode::LegacyValue),
                );
                self.expand_range(0.0, p.amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::NoisePerlin(p)) => {
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    p,
                    NoiseDispatch::new(1, NoiseKernelMode::Perlin),
                );
                let amplitude = p.amplitude.abs();
                self.expand_range(-amplitude, amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::Fbm(p)) => {
                let nt = Self::noise_type_u(p.noise).ok_or_else(|| {
                    cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "noise",
                        "fBm noise type is outside the compiled plan",
                    )
                })?;
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    &p.base,
                    NoiseDispatch::new(nt, NoiseKernelMode::Fbm),
                );
                let amplitude = p.base.amplitude.abs();
                self.expand_range(-amplitude, amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::Ridged(p)) => {
                let nt = Self::noise_type_u(p.noise).ok_or_else(|| {
                    cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "noise",
                        "ridged noise type is outside the compiled plan",
                    )
                })?;
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    &p.base,
                    NoiseDispatch::new(nt, NoiseKernelMode::Ridged),
                );
                self.expand_range(p.base.amplitude.min(0.0), p.base.amplitude.max(0.0));
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            // Dedicated range-mask / dune asymmetry / canyon meander kernels.
            (GpuKernel::Shape, LayerKind::Mountains(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.base.seed & 0xFFFF_FFFF) as u32,
                    octaves: p.base.octaves.max(1),
                    frequency: p.base.frequency,
                    amplitude: p.base.amplitude,
                    lacunarity: p.base.lacunarity,
                    persistence: p.base.persistence,
                    offset_x: p.base.offset_x,
                    offset_z: p.base.offset_z,
                    ridge_sharpness: p.ridge_sharpness,
                    range_angle: p.range_angle,
                    range_width: p.range_width,
                    wave_frequency: 0.0,
                    asymmetry: 0.0,
                    depth: 0.0,
                    canyon_width: 0.0,
                    meander: p.crest_detail,
                    shape_mode: 0,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.base.amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Dunes(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.base.seed & 0xFFFF_FFFF) as u32,
                    octaves: p.base.octaves.max(1),
                    frequency: p.base.frequency,
                    amplitude: p.effective_height(),
                    lacunarity: p.base.lacunarity,
                    persistence: p.base.persistence,
                    offset_x: p.base.offset_x,
                    offset_z: p.base.offset_z,
                    ridge_sharpness: p.effective_crest_sharpness(),
                    range_angle: p.direction_deg,
                    range_width: p.linearity,
                    wave_frequency: p.effective_scale(),
                    asymmetry: p.effective_crest_sharpness(),
                    depth: p.trough_depth,
                    canyon_width: p.basin_floor,
                    meander: p.wind_strength,
                    shape_mode: 1,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.effective_height());
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Canyons(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.seed & 0xFFFF_FFFF) as u32,
                    octaves: 1,
                    frequency: 1.0,
                    amplitude: 1.0,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: 0.0,
                    offset_z: 0.0,
                    ridge_sharpness: 0.0,
                    range_angle: 0.0,
                    range_width: 0.0,
                    wave_frequency: 0.0,
                    asymmetry: 0.0,
                    depth: p.depth,
                    canyon_width: p.width,
                    meander: p.meander,
                    shape_mode: 2,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(-p.depth, 0.0);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::DomainWarp(p)) => {
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    &p.base,
                    NoiseDispatch::domain_warp(p.warp_strength, p.warp_frequency),
                );
                let amplitude = p.base.amplitude.abs();
                self.expand_range(-amplitude, amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::VoronoiRegions(p)) => {
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    &p.base,
                    NoiseDispatch::voronoi_regions(p.cell_jitter, p.height_per_cell),
                );
                let worley_lo = 0.25 * p.base.amplitude * p.base.remap_min;
                let worley_hi = 0.25 * p.base.amplitude * p.base.remap_max;
                let cell_span = (p.height_per_cell * p.cell_jitter).abs();
                self.expand_range(
                    worley_lo.min(worley_hi) - cell_span,
                    worley_lo.max(worley_hi) + cell_span,
                );
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Thermal, LayerKind::ThermalErosion(p)) => {
                let talus = p.talus_angle_deg.to_radians().tan() * self.metrics.dx();
                let iters = Self::scale_iters(quality, p.iterations).min(match quality {
                    PreviewQuality::Draft => self.max_sim_iters_per_tick.max(1),
                    PreviewQuality::Medium => 24,
                    PreviewQuality::Full | PreviewQuality::Export => u32::MAX,
                });
                self.fill_slot(
                    device,
                    queue,
                    encoder,
                    TexSlot::Hardness,
                    p.hardness.clamp(0.0, 1.0),
                );
                for _ in 0..iters {
                    let strength = p.strength;
                    let talus_v = talus;
                    // Inline thermal step (avoid borrowing self.thermal while mutably borrowing self)
                    let u = ThermalU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        dx: self.metrics.dx(),
                        talus: talus_v,
                        strength,
                        _p2: 0.0,
                        _p3: 0.0,
                        _pad: 0.0,
                    };
                    let u_buf = self.write_uniform(device, queue, &u);
                    let src_ping = self.current == 0;
                    let (src, dst) = if src_ping {
                        (&self.ping.view, &self.pong.view)
                    } else {
                        (&self.pong.view, &self.ping.view)
                    };
                    let delta_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("thermal-delta-bg"),
                        layout: &self.thermal.bgl,
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
                            label: Some("thermal-delta"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.thermal.pipeline);
                        pass.set_bind_group(0, &delta_bg, &[]);
                        pass.dispatch_workgroups(
                            self.metrics.width.div_ceil(8),
                            self.metrics.height.div_ceil(8),
                            1,
                        );
                    }
                    let apply_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("thermal-apply-bg"),
                        layout: &self.thermal_apply.bgl,
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
                            label: Some("thermal-apply"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.thermal_apply.pipeline);
                        pass.set_bind_group(0, &apply_bg, &[]);
                        pass.dispatch_workgroups(
                            self.metrics.width.div_ceil(8),
                            self.metrics.height.div_ceil(8),
                            1,
                        );
                    }
                    self.swap_current();
                }
            }
            (GpuKernel::Hydraulic, LayerKind::HydraulicErosion(p)) => {
                let p = apply_transport_model(p, p.transport_model);
                self.fill_slot(device, queue, encoder, TexSlot::WaterA, 0.0);
                self.fill_slot(device, queue, encoder, TexSlot::WaterB, 0.0);
                self.fill_slot(device, queue, encoder, TexSlot::SedA, 0.0);
                self.fill_slot(device, queue, encoder, TexSlot::SedB, 0.0);
                self.fill_slot(device, queue, encoder, TexSlot::Rainfall, 1.0);
                self.fill_slot(
                    device,
                    queue,
                    encoder,
                    TexSlot::LooseSediment,
                    if p.layered_materials {
                        p.initial_sediment_thickness.max(0.0)
                    } else {
                        0.0
                    },
                );
                let eff_k = if p.layered_materials {
                    p.sediment_hardness.clamp(0.0, 1.0)
                } else {
                    p.hardness.clamp(0.0, 1.0)
                };
                self.fill_slot(device, queue, encoder, TexSlot::Hardness, eff_k);
                let iters = Self::scale_iters(quality, p.iterations).min(match quality {
                    PreviewQuality::Draft => self.max_sim_iters_per_tick.max(1),
                    PreviewQuality::Medium => 24,
                    PreviewQuality::Full | PreviewQuality::Export => u32::MAX,
                });
                let timestep = clamp_timestep_cfl(p.timestep, self.metrics.dx(), 4.0);
                let mut water_flip = false;
                for _ in 0..iters {
                    let u = HydraulicU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        timestep,
                        rainfall: p.rainfall,
                        evaporation: p.evaporation,
                        erosion: p.erosion,
                        deposition: p.deposition,
                        capacity: p.capacity,
                        fan_boost: p.fan_boost,
                        floodplain_bias: p.floodplain_bias,
                        dx: self.metrics.dx(),
                        incision_bias: p.incision_bias.max(0.05),
                        bedrock_k: p.bedrock_hardness.clamp(0.0, 1.0),
                        sediment_k: p.sediment_hardness.clamp(0.0, 1.0),
                        layered: if p.layered_materials { 1.0 } else { 0.0 },
                        _pad1: 0.0,
                    };
                    let u_buf = self.write_uniform(device, queue, &u);
                    let src_ping = self.current == 0;
                    let (h_src, h_dst) = if src_ping {
                        (&self.ping.view, &self.pong.view)
                    } else {
                        (&self.pong.view, &self.ping.view)
                    };
                    let (w_src, w_dst) = if water_flip {
                        (&self.water_b.view, &self.water_a.view)
                    } else {
                        (&self.water_a.view, &self.water_b.view)
                    };
                    let (s_src, s_dst) = if water_flip {
                        (&self.sed_b.view, &self.sed_a.view)
                    } else {
                        (&self.sed_a.view, &self.sed_b.view)
                    };
                    let outflow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hydraulic-outflow-bg"),
                        layout: &self.hydraulic_outflow.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(h_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(w_src),
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
                            label: Some("hydraulic-outflow"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.hydraulic_outflow.pipeline);
                        pass.set_bind_group(0, &outflow_bg, &[]);
                        pass.dispatch_workgroups(
                            self.metrics.width.div_ceil(8),
                            self.metrics.height.div_ceil(8),
                            1,
                        );
                    }
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hydraulic-bg"),
                        layout: &self.hydraulic.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(h_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(w_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(s_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&self.outflow.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::TextureView(h_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(w_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: wgpu::BindingResource::TextureView(s_dst),
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
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("hydraulic"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.hydraulic.pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(
                            self.metrics.width.div_ceil(8),
                            self.metrics.height.div_ceil(8),
                            1,
                        );
                    }
                    self.swap_current();
                    water_flip = !water_flip;
                }
            }
            (GpuKernel::RiverCarve, LayerKind::RiverCarve(p)) => {
                self.run_river_carve(device, queue, encoder, p, quality);
            }
            (GpuKernel::StreamPower, LayerKind::StreamPowerErosion(p)) => {
                self.run_stream_power(device, queue, encoder, p, quality);
            }
            (GpuKernel::MultiScaleAmplify, LayerKind::MultiScaleAmplify(p)) => {
                self.run_multi_scale_amplify(device, queue, encoder, p, quality);
            }
            (GpuKernel::Blur, LayerKind::Blur(p)) => {
                let iters = Self::blur_iters(p);
                for _ in 0..iters {
                    let u = BlurU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        radius: p.radius.clamp(1, BLUR_MAX_RADIUS),
                        _pad: 0,
                    };
                    let u_buf = self.write_uniform(device, queue, &u);
                    let src_ping = self.current == 0;
                    let (src, dst) = if src_ping {
                        (&self.ping.view, &self.pong.view)
                    } else {
                        (&self.pong.view, &self.ping.view)
                    };
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("blur-bg"),
                        layout: &self.blur.bgl,
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
                                resource: wgpu::BindingResource::TextureView(dst),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("blur"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.blur.pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(
                            self.metrics.width.div_ceil(8),
                            self.metrics.height.div_ceil(8),
                            1,
                        );
                    }
                    self.swap_current();
                }
            }
            (GpuKernel::EffectFilter, LayerKind::EffectFilter(p)) => {
                self.run_effect_filter(device, queue, encoder, p, quality);
                let amp = p.amount.abs().max(1.0);
                self.expand_range(self.approx_range.0 - amp, self.approx_range.1 + amp);
            }
            (GpuKernel::Shape, LayerKind::Mesa(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.seed & 0xFFFF_FFFF) as u32,
                    octaves: 3,
                    frequency: 0.001,
                    amplitude: p.height,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: p.center_u,
                    offset_z: p.center_v,
                    ridge_sharpness: p.edge_steepness,
                    range_angle: 0.0,
                    range_width: p.radius,
                    wave_frequency: 0.0,
                    asymmetry: 0.0,
                    depth: p.cap_noise,
                    canyon_width: 0.0,
                    meander: p.soft,
                    shape_mode: 5,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.height);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Volcano(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.seed & 0xFFFF_FFFF) as u32,
                    octaves: 3,
                    frequency: 0.001,
                    amplitude: p.height,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: p.center_u,
                    offset_z: p.center_v,
                    ridge_sharpness: p.flank_power,
                    range_angle: 0.0,
                    range_width: p.radius,
                    wave_frequency: 0.0,
                    asymmetry: 0.0,
                    depth: p.crater_depth,
                    canyon_width: p.crater_radius,
                    meander: p.roughness,
                    shape_mode: 4,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.height);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Uplift(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.seed & 0xFFFF_FFFF) as u32,
                    octaves: p.detail_octaves.max(1),
                    frequency: p.frequency,
                    amplitude: p.amplitude,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: 0.0,
                    offset_z: 0.0,
                    ridge_sharpness: p.ridge_power,
                    range_angle: p.range_angle,
                    range_width: p.corridor_width,
                    wave_frequency: p.detail_frequency,
                    asymmetry: p.altitude_fade,
                    depth: p.detail_amplitude,
                    canyon_width: 0.0,
                    meander: p.warp_strength,
                    shape_mode: 3,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Island(p)) => {
                if p.archetype == IslandArchetype::VolcanicHighIsland {
                    // Preserve the already-admitted volcanic compatibility preview.
                    let u = ShapeU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        world_x: self.metrics.world_size_x,
                        world_z: self.metrics.world_size_z,
                        seed: p.seed as u32,
                        octaves: 4,
                        frequency: p.ridge_frequency.max(0.0001),
                        amplitude: p.mountain_height,
                        lacunarity: 2.0,
                        persistence: 0.5,
                        offset_x: p.center_u,
                        offset_z: p.center_v,
                        ridge_sharpness: p.mountain_power,
                        range_angle: p.rotation_deg,
                        range_width: p.radius,
                        wave_frequency: p.coastline_frequency,
                        asymmetry: p.aspect,
                        depth: p.beach_height,
                        canyon_width: p.lagoon_radius,
                        meander: p.coastline_warp,
                        shape_mode: 6,
                        _pad: 0,
                    };
                    self.gen_shape(device, queue, encoder, u);
                } else {
                    self.gen_island(device, queue, encoder, p);
                }
                self.expand_range(p.ocean_floor, p.mountain_height);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Plateau(p)) => {
                self.gen_plateau(device, queue, encoder, p);
                self.expand_range(p.low, p.high);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Terrace, LayerKind::Terrace(p)) => {
                // Terrace quantization is defined by the range of the field entering
                // this layer. `approx_range` is deliberately conservative and may
                // contain unrelated history, so it is not authoritative here.
                self.reduce_current_height_range(device, queue, encoder);
                let u = TerraceU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    levels: p.levels,
                    sharpness: p.sharpness,
                    _p0: 0.0,
                    _p1: 0.0,
                    _p2: 0.0,
                    _p3: 0.0,
                };
                let u_buf = self.write_uniform(device, queue, &u);
                let src_ping = self.current == 0;
                let (src, dst) = if src_ping {
                    (&self.ping.view, &self.pong.view)
                } else {
                    (&self.pong.view, &self.ping.view)
                };
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrace-bg"),
                    layout: &self.terrace.bgl,
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
                            resource: wgpu::BindingResource::TextureView(dst),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.effect_filter_range_buffer.as_entire_binding(),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("terrace"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.terrace.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(
                        self.metrics.width.div_ceil(8),
                        self.metrics.height.div_ceil(8),
                        1,
                    );
                }
                self.swap_current();
            }
            (planned, actual) => {
                return Err(GpuError::Wgpu(format!(
                    "GPU support plan {planned:?} does not match layer {actual:?}"
                )));
            }
        }
        Ok(())
    }
}
