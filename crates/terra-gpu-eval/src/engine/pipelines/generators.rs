//! GPU evaluator pipelines generators implementation.

use super::*;

impl GpuTerrainEngine {
    pub(in super::super) fn noise_type_u(t: FractalNoiseType) -> Option<u32> {
        match t {
            FractalNoiseType::Value => Some(0),
            FractalNoiseType::Perlin => Some(1),
            FractalNoiseType::OpenSimplex => None,
        }
    }

    pub(super) fn gen_noise(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &NoiseParams,
        dispatch: NoiseDispatch,
    ) {
        self.gen_noise_to(device, queue, encoder, p, dispatch, TexSlot::Layer);
    }

    pub(super) fn gen_noise_to(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &NoiseParams,
        dispatch: NoiseDispatch,
        destination: TexSlot,
    ) {
        let u = NoiseU {
            width: self.metrics.width,
            height: self.metrics.height,
            world_x: self.metrics.world_size_x,
            world_z: self.metrics.world_size_z,
            seed: (p.seed & 0xFFFF_FFFF) as u32,
            octaves: p.octaves.max(1),
            frequency: p.frequency,
            amplitude: p.amplitude,
            lacunarity: p.lacunarity,
            persistence: p.persistence,
            offset_x: p.offset_x,
            offset_z: p.offset_z,
            remap_min: p.remap_min,
            remap_max: p.remap_max,
            noise_type: dispatch.noise_type,
            mode: dispatch.mode as u32,
            warp_strength: dispatch.warp_strength,
            warp_frequency: dispatch.warp_frequency,
            cell_jitter: dispatch.cell_jitter,
            height_per_cell: dispatch.height_per_cell,
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let destination = self.view_of(destination);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("noise-bg"),
            layout: &self.noise.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(destination),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("noise"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.noise.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
        }
    }

    pub(super) fn gen_shape(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        u: ShapeU,
    ) {
        let u_buf = self.write_uniform(device, queue, &u);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shapes-bg"),
            layout: &self.shapes.bgl,
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
                label: Some("shapes"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shapes.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
        }
    }

    pub(in super::super) fn gen_island(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &IslandParams,
    ) {
        let archetype = match p.archetype {
            IslandArchetype::VolcanicHighIsland => 0,
            IslandArchetype::Archipelago => 1,
            IslandArchetype::Atoll => 2,
        };
        let u = IslandU {
            width: self.metrics.width,
            height: self.metrics.height,
            world_x: self.metrics.world_size_x,
            world_z: self.metrics.world_size_z,
            seed: p.seed as u32,
            archetype,
            _pad_u: [0; 2],
            center_u: p.center_u,
            center_v: p.center_v,
            rotation_deg: p.rotation_deg,
            radius: p.radius,
            aspect: p.aspect,
            sea_level: p.sea_level,
            ocean_floor: p.ocean_floor,
            mountain_height: p.mountain_height,
            shelf_width: p.shelf_width,
            shelf_depth: p.shelf_depth,
            beach_width: p.beach_width,
            beach_height: p.beach_height,
            reef_width: p.reef_width,
            reef_depth: p.reef_depth,
            coastline_warp: p.coastline_warp,
            coastline_frequency: p.coastline_frequency,
            mountain_power: p.mountain_power,
            ridge_strength: p.ridge_strength,
            ridge_frequency: p.ridge_frequency,
            lagoon_radius: p.lagoon_radius,
            _pad_f: [0.0; 4],
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("island-bg"),
            layout: &self.island.bgl,
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
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("island"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.island.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
    }

    pub(in super::super) fn gen_plateau(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &PlateauParams,
    ) {
        let source = if self.current == 0 {
            TexSlot::Ping
        } else {
            TexSlot::Pong
        };
        self.gen_plateau_between(device, queue, encoder, p, source, TexSlot::Layer);
    }

    pub(in super::super) fn gen_plateau_between(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &PlateauParams,
        source: TexSlot,
        destination: TexSlot,
    ) {
        let u = PlateauU {
            width: self.metrics.width,
            height: self.metrics.height,
            low: p.low,
            high: p.high,
            soft: p.soft,
            _pad: [0.0; 3],
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let src_view = self.view_of(source);
        let dst_view = self.view_of(destination);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("plateau-bg"),
            layout: &self.plateau.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(dst_view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("plateau"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.plateau.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
    }

    pub(in super::super) fn run_path_height(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &PathParams,
    ) {
        let samples = terra_core::generators::path_samples(
            p,
            self.metrics.world_size_x,
            self.metrics.world_size_z,
        );
        let mut points: Vec<[f32; 4]> = samples
            .iter()
            .map(|sample| [sample.x, sample.z, sample.height, sample.width])
            .collect();
        let point_count = points.len() as u32;
        if points.is_empty() {
            points.push([0.0; 4]);
        }
        let point_buffer = make_storage_buffer(
            device,
            queue,
            "path-height-points",
            bytemuck::cast_slice(&points),
        );
        let uniform = PathU {
            width: self.metrics.width,
            height: self.metrics.height,
            world_x: self.metrics.world_size_x,
            world_z: self.metrics.world_size_z,
            point_count,
            carve: u32::from(p.carve),
            seed: p.seed as u32,
            _pad0: 0,
            base_width: p.width,
            falloff: p.falloff,
            noise_strength: p.noise_strength,
            noise_scale: p.noise_scale,
            height_offset: p.height_offset,
            profile: p.profile,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let uniform_buffer = self.write_uniform(device, queue, &uniform);
        let src = if self.current == 0 {
            &self.ping.view
        } else {
            &self.pong.view
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path-height-bg"),
            layout: &self.path_height.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: point_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("path-height"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.path_height.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
    }

    pub(in super::super) fn run_polygon_height(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &PolygonHeightParams,
    ) {
        let mut points: Vec<[f32; 4]> = p
            .points
            .iter()
            .map(|point| [point[0], point[1], 0.0, 0.0])
            .collect();
        let point_count = points.len() as u32;
        if points.is_empty() {
            points.push([0.0; 4]);
        }
        let point_buffer = make_storage_buffer(
            device,
            queue,
            "polygon-height-points",
            bytemuck::cast_slice(&points),
        );
        let short_axis = self
            .metrics
            .world_size_x
            .min(self.metrics.world_size_z)
            .max(1.0);
        let uniform = PolygonHeightU {
            width: self.metrics.width,
            height: self.metrics.height,
            world_x: self.metrics.world_size_x,
            world_z: self.metrics.world_size_z,
            point_count,
            mode: match p.mode {
                PolygonHeightMode::RaiseBy => 0,
                PolygonHeightMode::SetElevation => 1,
            },
            carve: u32::from(p.carve),
            _pad0: 0,
            target_height: p.height,
            falloff: (p.falloff.clamp(0.0, 0.5) * short_axis).max(1.0e-3),
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let uniform_buffer = self.write_uniform(device, queue, &uniform);
        let src = if self.current == 0 {
            &self.ping.view
        } else {
            &self.pong.view
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("polygon-height-bg"),
            layout: &self.polygon_height.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: point_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("polygon-height"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.polygon_height.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
    }

    pub(in super::super) fn run_procedural_crater(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &EffectFilterParams,
    ) {
        self.fill_slot(device, queue, encoder, TexSlot::SculptStamp, 80.0);
        let spec = effect_filter_gpu_spec(p)
            .expect("planner admitted only an executable procedural Crater");
        let uniform = self.effect_filter_uniform(p, spec.mode, 1, (0, 0, 0, 0));
        let uniform_buffer = self.write_uniform(device, queue, &uniform);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("procedural-crater-bg"),
            layout: &self.effect_filter.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.sculpt_stamp.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.effect_filter_range_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("procedural-crater"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.effect_filter.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
    }

    pub(in super::super) fn run_procedural_shape(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &ProceduralShapeParams,
    ) -> Result<(), GpuError> {
        match p.generator {
            ProceduralGenerator::Mountain => {
                let q = &p.mountain;
                self.gen_shape(
                    device,
                    queue,
                    encoder,
                    ShapeU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        world_x: self.metrics.world_size_x,
                        world_z: self.metrics.world_size_z,
                        seed: q.base.seed as u32,
                        octaves: q.base.octaves.max(1),
                        frequency: q.base.frequency,
                        amplitude: q.base.amplitude,
                        lacunarity: q.base.lacunarity,
                        persistence: q.base.persistence,
                        offset_x: q.base.offset_x,
                        offset_z: q.base.offset_z,
                        ridge_sharpness: q.ridge_sharpness,
                        range_angle: q.range_angle,
                        range_width: q.range_width,
                        wave_frequency: 0.0,
                        asymmetry: 0.0,
                        depth: 0.0,
                        canyon_width: 0.0,
                        meander: q.crest_detail,
                        shape_mode: 0,
                        _pad: 0,
                    },
                );
                self.expand_range(0.0, q.base.amplitude);
            }
            ProceduralGenerator::Hills => {
                let noise_type = Self::noise_type_u(p.hills.noise).ok_or_else(|| {
                    cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "procedural shape",
                        "Hills noise type is outside the compiled GPU plan",
                    )
                })?;
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    &p.hills.base,
                    NoiseDispatch::new(noise_type, NoiseKernelMode::Fbm),
                );
                let amplitude = p.hills.base.amplitude.abs();
                self.expand_range(-amplitude, amplitude);
            }
            ProceduralGenerator::Plateau => {
                let noise_type = Self::noise_type_u(p.hills.noise).ok_or_else(|| {
                    cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "procedural shape",
                        "Plateau noise type is outside the compiled GPU plan",
                    )
                })?;
                self.gen_noise_to(
                    device,
                    queue,
                    encoder,
                    &p.hills.base,
                    NoiseDispatch::new(noise_type, NoiseKernelMode::Fbm),
                    TexSlot::SculptStamp,
                );
                self.gen_plateau_between(
                    device,
                    queue,
                    encoder,
                    &p.plateau,
                    TexSlot::SculptStamp,
                    TexSlot::Layer,
                );
                self.expand_range(p.plateau.low, p.plateau.high);
            }
            ProceduralGenerator::Mesa => {
                let q = &p.mesa;
                self.gen_shape(
                    device,
                    queue,
                    encoder,
                    ShapeU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        world_x: self.metrics.world_size_x,
                        world_z: self.metrics.world_size_z,
                        seed: q.seed as u32,
                        octaves: 3,
                        frequency: 0.001,
                        amplitude: q.height,
                        lacunarity: 2.0,
                        persistence: 0.5,
                        offset_x: q.center_u,
                        offset_z: q.center_v,
                        ridge_sharpness: q.edge_steepness,
                        range_angle: 0.0,
                        range_width: q.radius,
                        wave_frequency: 0.0,
                        asymmetry: 0.0,
                        depth: q.cap_noise,
                        canyon_width: 0.0,
                        meander: q.soft,
                        shape_mode: 5,
                        _pad: 0,
                    },
                );
                self.expand_range(0.0, q.height);
            }
            ProceduralGenerator::Volcano => {
                let q = &p.volcano;
                self.gen_shape(
                    device,
                    queue,
                    encoder,
                    ShapeU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        world_x: self.metrics.world_size_x,
                        world_z: self.metrics.world_size_z,
                        seed: q.seed as u32,
                        octaves: 3,
                        frequency: 0.001,
                        amplitude: q.height,
                        lacunarity: 2.0,
                        persistence: 0.5,
                        offset_x: q.center_u,
                        offset_z: q.center_v,
                        ridge_sharpness: q.flank_power,
                        range_angle: 0.0,
                        range_width: q.radius,
                        wave_frequency: 0.0,
                        asymmetry: 0.0,
                        depth: q.crater_depth,
                        canyon_width: q.crater_radius,
                        meander: q.roughness,
                        shape_mode: 4,
                        _pad: 0,
                    },
                );
                self.expand_range(0.0, q.height);
            }
            ProceduralGenerator::Canyon => {
                let q = &p.canyon;
                self.gen_shape(
                    device,
                    queue,
                    encoder,
                    ShapeU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        world_x: self.metrics.world_size_x,
                        world_z: self.metrics.world_size_z,
                        seed: q.seed as u32,
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
                        depth: q.depth,
                        canyon_width: q.width,
                        meander: q.meander,
                        shape_mode: 2,
                        _pad: 0,
                    },
                );
                self.expand_range(-q.depth, 0.0);
            }
            ProceduralGenerator::Noise => {
                self.gen_noise(
                    device,
                    queue,
                    encoder,
                    &p.noise,
                    NoiseDispatch::new(1, NoiseKernelMode::Perlin),
                );
                let amplitude = p.noise.amplitude.abs();
                self.expand_range(-amplitude, amplitude);
            }
            ProceduralGenerator::Crater => {
                self.run_procedural_crater(device, queue, encoder, &p.crater);
                self.expand_range(80.0 - p.crater.amount.abs(), 80.0 + p.crater.amount.abs());
            }
            ProceduralGenerator::Dunes => {
                return Err(cpu_required(
                    GpuFallbackCode::UnsupportedOptions,
                    "procedural shape",
                    "Dunes generator is not parity-covered",
                ));
            }
        }
        Ok(())
    }

    /// Resample and upload only a warm edit's destination footprint. Destination
    /// coordinates remain absolute so differing authoring/preview resolutions use
    /// the same bilinear mapping as the full upload.
    pub(in super::super) fn ensure_source_raster(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
    ) -> Result<PathBuf, GpuError> {
        let key = if path.is_empty() {
            PathBuf::from("<empty-heightmap>")
        } else {
            std::fs::canonicalize(path).unwrap_or_else(|_| Path::new(path).to_path_buf())
        };
        let fingerprint = if path.is_empty() {
            None
        } else {
            let metadata = std::fs::metadata(path)
                .map_err(|error| GpuError::SourceAsset(format!("{path}: {error}")))?;
            Some(SourceFingerprint {
                len: metadata.len(),
                modified: metadata.modified().ok(),
            })
        };
        if self
            .source_rasters
            .get(&key)
            .is_some_and(|entry| entry.fingerprint == fingerprint)
        {
            return Ok(key);
        }
        let decoded = if path.is_empty() {
            terra_core::generators::DecodedHeightmap {
                width: 1,
                height: 1,
                samples: vec![0.0],
            }
        } else {
            terra_core::generators::load_heightmap(path)
                .map_err(|error| GpuError::SourceAsset(error.to_string()))?
        };
        let limit = device.limits().max_texture_dimension_2d;
        if decoded.width > limit || decoded.height > limit {
            return Err(cpu_required(
                GpuFallbackCode::RuntimeResourceLimit,
                "heightmap",
                format!(
                    "source {}x{} exceeds device texture limit {}",
                    decoded.width, decoded.height, limit
                ),
            ));
        }
        let tex = HeightTex::new(
            device,
            "source-heightmap",
            decoded.width.max(1),
            decoded.height.max(1),
        );
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&decoded.samples),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(decoded.width * 4),
                rows_per_image: Some(decoded.height),
            },
            wgpu::Extent3d {
                width: decoded.width,
                height: decoded.height,
                depth_or_array_layers: 1,
            },
        );
        self.source_rasters
            .insert(key.clone(), SourceRasterTex { tex, fingerprint });
        #[cfg(test)]
        {
            self.source_upload_count += 1;
        }
        Ok(key)
    }

    pub(in super::super) fn run_heightmap_sample(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        layer: &Layer,
    ) -> Result<(), GpuError> {
        let (params, transform) = match &layer.kind {
            LayerKind::ImportHeightmap(params) => (params, None),
            LayerKind::Stamp2d(params) => {
                (&params.heightmap, layer.common.shape_transform.as_ref())
            }
            _ => {
                return Err(cpu_required(
                    GpuFallbackCode::UnsupportedLayerKind,
                    "heightmap",
                    "heightmap sampler received a non-heightmap layer",
                ));
            }
        };
        let key = self.ensure_source_raster(device, queue, &params.path)?;
        let source = self.source_rasters.get(&key).expect("source just loaded");
        let (mode, offset_x, offset_z, inv_scale, sin_t, cos_t, blend_size, roundness) =
            if let Some(transform) = transform {
                let theta = -transform.rotation_deg.to_radians();
                let (sin_t, cos_t) = theta.sin_cos();
                (
                    1,
                    transform.offset_x,
                    transform.offset_z,
                    1.0 / transform.scale.max(1e-6),
                    sin_t,
                    cos_t,
                    transform.blend_size,
                    transform.blend_roundness,
                )
            } else {
                (0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
            };
        let uniform = HeightmapSampleU {
            width: self.metrics.width,
            height: self.metrics.height,
            source_width: source.tex.width,
            source_height: source.tex.height,
            mode,
            _pad0: 0,
            height_scale: if params.path.is_empty() {
                0.0
            } else {
                params.height_scale
            },
            height_offset: if params.path.is_empty() {
                0.0
            } else {
                params.height_offset
            },
            world_x: self.metrics.world_size_x,
            world_z: self.metrics.world_size_z,
            offset_x,
            offset_z,
            inv_scale,
            sin_t,
            cos_t,
            blend_size,
            blend_roundness: roundness,
            _pad1: [0.0; 3],
        };
        let uniform_buf = self.write_uniform(device, queue, &uniform);
        let source = self.source_rasters.get(&key).expect("source retained");
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("heightmap-sample-bg"),
            layout: &self.heightmap_sample.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source.tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.stamp_mask.view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("heightmap-sample"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.heightmap_sample.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            self.metrics.width.div_ceil(8),
            self.metrics.height.div_ceil(8),
            1,
        );
        Ok(())
    }
}
