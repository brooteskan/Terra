//! GPU evaluator pipelines filters implementation.

use super::*;

impl GpuTerrainEngine {
    pub(in super::super) fn scale_iters(quality: PreviewQuality, iters: u32) -> u32 {
        match quality {
            // Draft must still read as a real filter change (WC interactive), not a no-op.
            PreviewQuality::Draft => iters.clamp(2, 8),
            PreviewQuality::Medium => iters.clamp(4, 12),
            PreviewQuality::Full | PreviewQuality::Export => iters.max(1),
        }
    }

    pub(in super::super) fn effect_filter_iters(
        quality: PreviewQuality,
        p: &EffectFilterParams,
    ) -> u32 {
        match effect_filter_gpu_spec(p).map(|spec| spec.passes) {
            Some(EffectFilterGpuPasses::LegacyQualityScaled) => {
                Self::scale_iters(quality, p.iterations.max(1)).min(8)
            }
            Some(EffectFilterGpuPasses::Once) | None => 1,
        }
    }

    pub(in super::super) fn blur_iters(p: &terra_core::layer::BlurParams) -> u32 {
        p.iterations.clamp(1, 8)
    }

    /// Executed iteration count for a layer's kernel — the single source of truth
    /// shared by the kernel dispatch and the dirty-region halo sizing so the two
    /// never disagree about how far a filter reaches.
    pub(in super::super) fn dirty_dispatch_extent(&self) -> (u32, u32, u32, u32, u32, u32) {
        // Returns (region_x, region_y, region_w, region_h, groups_x, groups_y).
        // `last_dirty_rect` is already expanded by the plan halo in `evaluate`, so
        // no further padding here — this is exactly the region the kernels rewrite.
        if let Some((x, y, w, h)) = self.last_dirty_rect {
            let gx = w.div_ceil(8);
            let gy = h.div_ceil(8);
            (x, y, w, h, gx.max(1), gy.max(1))
        } else {
            let w = self.metrics.width;
            let h = self.metrics.height;
            (0, 0, 0, 0, w.div_ceil(8), h.div_ceil(8))
        }
    }

    pub(in super::super) fn reduce_effect_filter_range(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // Ordered-f32 encodings of +infinity (min initializer) and -infinity
        // (max initializer). The WGSL transform preserves total numeric ordering
        // across negative and positive finite terrain heights.
        let initial = [0xff80_0000u32, 0x007f_ffffu32];
        queue.write_buffer(
            &self.effect_filter_range_buffer,
            0,
            bytemuck::cast_slice(&initial),
        );
        let uniform = EffectRangeU {
            width: self.metrics.width,
            height: self.metrics.height,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform = self.write_uniform(device, queue, &uniform);
        let src = if self.current == 0 {
            &self.ping.view
        } else {
            &self.pong.view
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("effect-filter-range-bg"),
            layout: &self.effect_filter_range.bgl,
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
                    resource: self.effect_filter_range_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("effect-filter-range"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.effect_filter_range.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
    }

    pub(super) fn effect_filter_uniform(
        &self,
        p: &EffectFilterParams,
        mode: u32,
        iterations: u32,
        region: (u32, u32, u32, u32),
    ) -> EffectFilterU {
        let (region_x, region_y, region_w, region_h) = region;
        EffectFilterU {
            width: self.metrics.width,
            height: self.metrics.height,
            world_x: self.metrics.world_size_x,
            world_z: self.metrics.world_size_z,
            mode,
            radius: p.radius.clamp(1, EFFECT_FILTER_MAX_RADIUS),
            iterations,
            seed: (p.seed & 0xFFFF_FFFF) as u32,
            strength: p.strength.clamp(0.0, 1.0),
            amount: p.amount,
            frequency: p.effective_frequency(),
            sea_level: p.sea_level,
            beach_width: p
                .beach_width
                .max(p.crater_radius * self.metrics.world_size_x * 0.5),
            slope_min: p.slope_min,
            slope_max: p.slope_max,
            rock_hardness: p.rock_hardness,
            terrace_height: p.terrace_height,
            terrace_offset: p.terrace_offset,
            rotation_deg: p.rotation_deg,
            anisotropy: p.anisotropy,
            warp_strength: p.warp_strength,
            warp_frequency: p.warp_frequency,
            dx: self.metrics.dx(),
            invert: if p.invert { 1.0 } else { 0.0 },
            flow_threshold: p.flow_threshold,
            wall_steepness: p.wall_steepness,
            valley_floor: p.valley_floor,
            talus_mix: p.talus_mix,
            top_smoothness: p.top_smoothness,
            riser_sharpness: p.riser_sharpness,
            lacunarity: p.lacunarity,
            persistence: p.persistence,
            octaves: p.octaves,
            voronoi_feature: match p.voronoi_feature {
                terra_core::noise::WorleyFeature::F1 => 0,
                terra_core::noise::WorleyFeature::F2 => 1,
                terra_core::noise::WorleyFeature::F2MinusF1 => 2,
            },
            tileable: u32::from(p.tileable),
            _pad_params: 0,
            crater_radius: p.crater_radius,
            dz: self.metrics.dz(),
            _pad_metric0: 0.0,
            _pad_metric1: 0.0,
            region_x,
            region_y,
            region_w,
            region_h,
        }
    }

    pub(in super::super) fn run_effect_filter(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &EffectFilterParams,
        quality: PreviewQuality,
    ) {
        let spec = effect_filter_gpu_spec(p)
            .expect("compiled EffectFilter plan must retain an executable spec");
        let mode = spec.mode;
        let iters = Self::effect_filter_iters(quality, p);
        if spec.needs_height_range {
            self.reduce_effect_filter_range(device, queue, encoder);
        }
        let (rx, ry, rw, rh, gx, gy) = self.dirty_dispatch_extent();
        for _ in 0..iters {
            let u = self.effect_filter_uniform(p, mode, iters, (rx, ry, rw, rh));
            let u_buf = self.write_uniform(device, queue, &u);
            let src_ping = self.current == 0;
            let (src, dst) = if src_ping {
                (&self.ping.view, &self.pong.view)
            } else {
                (&self.pong.view, &self.ping.view)
            };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("effect-filter-bg"),
                layout: &self.effect_filter.bgl,
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
                    label: Some("effect-filter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.effect_filter.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            self.swap_current();
        }
    }
}
