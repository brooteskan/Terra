//! GPU evaluator pipelines sculpt implementation.

use super::*;

impl GpuTerrainEngine {
    pub(in super::super) fn record_sculpt_to_layer(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &SculptParams,
        region: Option<(u32, u32, u32, u32)>,
    ) {
        let full_width = self.metrics.width;
        let full_height = self.metrics.height;
        let (origin_x, origin_y, width, height) = region.unwrap_or((0, 0, full_width, full_height));
        let row_bytes = width.saturating_mul(4);
        let padded_row_bytes = row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let mut upload = vec![0u8; padded_row_bytes as usize * height as usize];
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for local_y in 0..height {
            let row_start = local_y as usize * padded_row_bytes as usize;
            let row = &mut upload[row_start..row_start + row_bytes as usize];
            for local_x in 0..width {
                let x = origin_x + local_x;
                let y = origin_y + local_y;
                let u = (x as f32 + 0.5) / full_width.max(1) as f32;
                let v = (y as f32 + 0.5) / full_height.max(1) as f32;
                let sample = params.sample_bilinear(u, v);
                lo = lo.min(sample);
                hi = hi.max(sample);
                let offset = local_x as usize * 4;
                row[offset..offset + 4].copy_from_slice(&sample.to_ne_bytes());
            }
        }
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("compiled-plan-sculpt-upload"),
            contents: &upload,
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(height),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.layer_tex.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: origin_x,
                    y: origin_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        if lo <= hi {
            self.expand_range(lo, hi);
        }
        let texels = u64::from(width) * u64::from(height);
        self.last_eval_stats.sculpt_resampled_texels = self
            .last_eval_stats
            .sculpt_resampled_texels
            .saturating_add(texels);
        self.last_eval_stats.upload_bytes = self
            .last_eval_stats
            .upload_bytes
            .saturating_add(texels.saturating_mul(4));
    }

    /// Stamp the stroke set into the running height, measure each Flatten target,
    /// then relax into `layer_tex` (the layer contribution the standard blend
    /// consumes). Full-field like every other layer's contribution — `layer_tex` is
    /// shared scratch, so it must be valid everywhere the full-field blend reads it
    /// (#113).
    ///
    /// Most kinds stamp in a single pass over the whole set. Flatten (#117) splits
    /// the set: each Flatten's target is the brush-weighted mean of the *running*
    /// field over its footprint, so the stroke run is cut before every Flatten, the
    /// prior segment is stamped into a ping-pong height buffer, and a reduce/resolve
    /// pair measures that buffer into `targets[f]` before the Flatten (in the next
    /// segment) reads it. With no Flatten present this degenerates to one stamp of
    /// `[0, n)` reading the layer input — the pre-#117 path. `edited` is order- and
    /// target-independent, so a single pass computes it over the whole set.
    fn stroke_runtime_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer: LayerId,
        headers: Vec<StrokeHeaderGpu>,
        points: Vec<[f32; 4]>,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let header_size = std::mem::size_of::<StrokeHeaderGpu>();
        let point_size = std::mem::size_of::<[f32; 4]>();
        let existing = self.stroke_runtime.remove(&layer);
        let runtime = if let Some(mut runtime) = existing {
            let points_append = runtime.uploaded_points.len() <= points.len()
                && runtime.uploaded_points == points[..runtime.uploaded_points.len()];
            let header_start = if runtime.uploaded_headers.len() == headers.len()
                && !headers.is_empty()
                && runtime.uploaded_headers[..headers.len() - 1] == headers[..headers.len() - 1]
            {
                Some(headers.len() - 1)
            } else if runtime.uploaded_headers.len() < headers.len()
                && runtime.uploaded_headers == headers[..runtime.uploaded_headers.len()]
            {
                Some(runtime.uploaded_headers.len())
            } else if runtime.uploaded_headers == headers {
                Some(headers.len())
            } else {
                None
            };
            let has_capacity =
                headers.len() <= runtime.header_capacity && points.len() <= runtime.point_capacity;

            if let Some(header_start) = header_start.filter(|_| points_append && has_capacity) {
                if header_start < headers.len() {
                    let bytes = bytemuck::cast_slice(&headers[header_start..]);
                    queue.write_buffer(
                        &runtime.headers,
                        (header_start * header_size) as u64,
                        bytes,
                    );
                    self.last_eval_stats.stroke_header_upload_bytes = self
                        .last_eval_stats
                        .stroke_header_upload_bytes
                        .saturating_add(bytes.len() as u64);
                }
                let point_start = runtime.uploaded_points.len();
                if point_start < points.len() {
                    let bytes = bytemuck::cast_slice(&points[point_start..]);
                    queue.write_buffer(&runtime.points, (point_start * point_size) as u64, bytes);
                    self.last_eval_stats.stroke_point_upload_bytes = self
                        .last_eval_stats
                        .stroke_point_upload_bytes
                        .saturating_add(bytes.len() as u64);
                }
                runtime.uploaded_headers = headers;
                runtime.uploaded_points = points;
                runtime
            } else {
                let header_capacity = headers
                    .len()
                    .max(INITIAL_STROKE_HEADER_CAPACITY)
                    .next_power_of_two();
                let point_capacity = points
                    .len()
                    .max(INITIAL_STROKE_POINT_CAPACITY)
                    .next_power_of_two();
                let header_bytes = bytemuck::cast_slice(&headers);
                let point_bytes = bytemuck::cast_slice(&points);
                self.last_eval_stats.stroke_header_upload_bytes = self
                    .last_eval_stats
                    .stroke_header_upload_bytes
                    .saturating_add(header_bytes.len() as u64);
                self.last_eval_stats.stroke_point_upload_bytes = self
                    .last_eval_stats
                    .stroke_point_upload_bytes
                    .saturating_add(point_bytes.len() as u64);
                self.last_eval_stats.stroke_payload_rebuilds = self
                    .last_eval_stats
                    .stroke_payload_rebuilds
                    .saturating_add(1);
                StrokeRuntimeBuffers {
                    headers: make_runtime_storage_buffer(
                        device,
                        queue,
                        "sculpt-stroke-headers",
                        header_size,
                        header_capacity,
                        header_bytes,
                    ),
                    points: make_runtime_storage_buffer(
                        device,
                        queue,
                        "sculpt-stroke-points",
                        point_size,
                        point_capacity,
                        point_bytes,
                    ),
                    header_capacity,
                    point_capacity,
                    uploaded_headers: headers,
                    uploaded_points: points,
                }
            }
        } else {
            let header_capacity = headers
                .len()
                .max(INITIAL_STROKE_HEADER_CAPACITY)
                .next_power_of_two();
            let point_capacity = points
                .len()
                .max(INITIAL_STROKE_POINT_CAPACITY)
                .next_power_of_two();
            let header_bytes = bytemuck::cast_slice(&headers);
            let point_bytes = bytemuck::cast_slice(&points);
            self.last_eval_stats.stroke_header_upload_bytes = self
                .last_eval_stats
                .stroke_header_upload_bytes
                .saturating_add(header_bytes.len() as u64);
            self.last_eval_stats.stroke_point_upload_bytes = self
                .last_eval_stats
                .stroke_point_upload_bytes
                .saturating_add(point_bytes.len() as u64);
            self.last_eval_stats.stroke_payload_rebuilds = self
                .last_eval_stats
                .stroke_payload_rebuilds
                .saturating_add(1);
            StrokeRuntimeBuffers {
                headers: make_runtime_storage_buffer(
                    device,
                    queue,
                    "sculpt-stroke-headers",
                    header_size,
                    header_capacity,
                    header_bytes,
                ),
                points: make_runtime_storage_buffer(
                    device,
                    queue,
                    "sculpt-stroke-points",
                    point_size,
                    point_capacity,
                    point_bytes,
                ),
                header_capacity,
                point_capacity,
                uploaded_headers: headers,
                uploaded_points: points,
            }
        };
        let result = (runtime.headers.clone(), runtime.points.clone());
        self.stroke_runtime.insert(layer, runtime);
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn run_sculpt_strokes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        layer: LayerId,
        p: &SculptStrokeParams,
        source: TexSlot,
        region: (u32, u32, u32, u32),
    ) {
        let strokes: Vec<&SculptStroke> = p.strokes.iter().filter(|s| s.enabled).collect();
        let (headers, points) = build_stroke_buffers(&strokes, &self.metrics);
        let (header_buf, point_buf) =
            self.stroke_runtime_buffers(device, queue, layer, headers, points);

        let width = self.metrics.width;
        let height = self.metrics.height;
        let world_x = self.metrics.world_size_x;
        let world_z = self.metrics.world_size_z;
        let n = strokes.len() as u32;
        // `region` is the rectangle the plan will composite and publish. Reconcile
        // samples the stamped field one texel beyond it, so stamp into a private
        // guard domain while keeping the final pass bounded to `region`.
        let stamp_region = expand_sample_region(region, sculpt_stamp_guard(p), width, height);
        let (region_x, region_y, region_w, region_h) = stamp_region;
        let gx = region_w.div_ceil(8).max(1);
        let gy = region_h.div_ceil(8).max(1);
        let num_partials = gx * gy;

        // Flatten footprint means, indexed by global stroke id; `partials` is the
        // reduce pass's per-workgroup scratch. Both are written by the GPU, so they
        // only need a valid (zeroed) backing until then.
        let targets_buf = make_storage_buffer(
            device,
            queue,
            "sculpt-stroke-flatten-targets",
            bytemuck::cast_slice(&vec![0f32; n.max(1) as usize]),
        );
        let partials_buf = make_storage_buffer(
            device,
            queue,
            "sculpt-stroke-flatten-partials",
            bytemuck::cast_slice(&vec![[0f32; 2]; num_partials.max(1) as usize]),
        );

        // The running field lives in one of three textures: the layer input (`Src`,
        // ping/pong) or the two ping-pong scratch buffers. `Src` is never written.
        #[derive(Clone, Copy)]
        enum RunSlot {
            Src,
            A,
            B,
        }
        fn flip(s: RunSlot) -> RunSlot {
            match s {
                RunSlot::Src | RunSlot::B => RunSlot::A,
                RunSlot::A => RunSlot::B,
            }
        }
        #[derive(Clone, Copy)]
        struct StampOp {
            lo: u32,
            hi: u32,
            in_slot: RunSlot,
            out_slot: RunSlot,
        }
        #[derive(Clone, Copy)]
        struct ReduceOp {
            stroke_index: u32,
            field: RunSlot,
            target_index: u32,
            fallback: f32,
        }
        #[derive(Clone, Copy)]
        enum Op {
            Stamp(StampOp),
            Reduce(ReduceOp),
        }

        // Cut the run before each Flatten. `cur` is the field entering the next
        // segment; a Flatten's reduce measures it, and the segment that finally
        // applies the Flatten reads `targets[f]` the resolve just wrote.
        let mut ops: Vec<Op> = Vec::new();
        let mut cur = RunSlot::Src;
        let mut prev = 0u32;
        for (idx, stroke) in strokes.iter().enumerate() {
            if !matches!(stroke.kind, SculptStrokeKind::Flatten) {
                continue;
            }
            let f = idx as u32;
            if f > prev {
                let out = flip(cur);
                ops.push(Op::Stamp(StampOp {
                    lo: prev,
                    hi: f,
                    in_slot: cur,
                    out_slot: out,
                }));
                cur = out;
            }
            ops.push(Op::Reduce(ReduceOp {
                stroke_index: f,
                field: cur,
                target_index: f,
                fallback: stroke.target_height,
            }));
            prev = f;
        }
        if n > prev {
            let out = flip(cur);
            ops.push(Op::Stamp(StampOp {
                lo: prev,
                hi: n,
                in_slot: cur,
                out_slot: out,
            }));
            cur = out;
        }
        let final_slot = cur;

        // All `&mut self` (uniform-pool) writes happen before any texture-view
        // borrow, matching `blend_into_current`; the ops carry their slots so the
        // dispatch phase needs no further planning.
        enum PassU {
            Stamp(wgpu::Buffer),
            Reduce {
                reduce: wgpu::Buffer,
                resolve: wgpu::Buffer,
            },
        }
        let mut pass_us: Vec<PassU> = Vec::with_capacity(ops.len());
        for op in &ops {
            match *op {
                Op::Stamp(s) => {
                    let u = SculptStrokesU {
                        width,
                        height,
                        world_x,
                        world_z,
                        stroke_lo: s.lo,
                        stroke_hi: s.hi,
                        region_x,
                        region_y,
                        region_w,
                        region_h,
                    };
                    pass_us.push(PassU::Stamp(self.write_uniform(device, queue, &u)));
                }
                Op::Reduce(r) => {
                    let ru = SculptReduceU {
                        width,
                        height,
                        world_x,
                        world_z,
                        stroke_index: r.stroke_index,
                        region_x,
                        region_y,
                        region_w,
                        region_h,
                    };
                    let sv = SculptResolveU {
                        num_partials,
                        target_index: r.target_index,
                        fallback: r.fallback,
                        _p0: 0.0,
                    };
                    let reduce = self.write_uniform(device, queue, &ru);
                    let resolve = self.write_uniform(device, queue, &sv);
                    pass_us.push(PassU::Reduce { reduce, resolve });
                }
            }
        }
        let edited_u = SculptStrokesU {
            width,
            height,
            world_x,
            world_z,
            stroke_lo: 0,
            stroke_hi: n,
            region_x,
            region_y,
            region_w,
            region_h,
        };
        let edited_u_buf = self.write_uniform(device, queue, &edited_u);
        let (output_x, output_y, output_w, output_h) = region;
        let recon_u = SculptReconcileU {
            width,
            height,
            reconcile: p.reconcile,
            _p0: 0.0,
            region_x: output_x,
            region_y: output_y,
            region_w: output_w,
            region_h: output_h,
        };
        let recon_u_buf = self.write_uniform(device, queue, &recon_u);

        // Immutable view borrows only, from here down.
        let src_view = self.view_of(source);
        let stamp_a = &self.sculpt_stamp.view;
        let stamp_b = &self.sculpt_stamp_b.view;
        let slot_view = |slot: RunSlot| match slot {
            RunSlot::Src => src_view,
            RunSlot::A => stamp_a,
            RunSlot::B => stamp_b,
        };

        // Edited coverage: one order-independent pass over the whole set.
        let edited_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sculpt-strokes-edited-bg"),
            layout: &self.sculpt_strokes_edited.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: edited_u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: header_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: point_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.sculpt_edited.view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sculpt-strokes-edited"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sculpt_strokes_edited.pipeline);
            pass.set_bind_group(0, &edited_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Segmented stamp + Flatten reductions, in execution order.
        for (op, pu) in ops.iter().zip(pass_us.iter()) {
            match (op, pu) {
                (Op::Stamp(s), PassU::Stamp(u_buf)) => {
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("sculpt-strokes-bg"),
                        layout: &self.sculpt_strokes.bgl,
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
                                resource: wgpu::BindingResource::TextureView(slot_view(s.in_slot)),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: header_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: point_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: targets_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(slot_view(s.out_slot)),
                            },
                        ],
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("sculpt-strokes-stamp"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.sculpt_strokes.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                (Op::Reduce(r), PassU::Reduce { reduce, resolve }) => {
                    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("sculpt-strokes-flatten-reduce-bg"),
                        layout: &self.sculpt_strokes_flatten_reduce.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: reduce.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(slot_view(r.field)),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: header_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: point_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: partials_buf.as_entire_binding(),
                            },
                        ],
                    });
                    let resolve_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("sculpt-strokes-flatten-resolve-bg"),
                        layout: &self.sculpt_strokes_flatten_resolve.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: resolve.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: partials_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: targets_buf.as_entire_binding(),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("sculpt-strokes-flatten-reduce"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.sculpt_strokes_flatten_reduce.pipeline);
                        pass.set_bind_group(0, &reduce_bg, &[]);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("sculpt-strokes-flatten-resolve"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.sculpt_strokes_flatten_resolve.pipeline);
                        pass.set_bind_group(0, &resolve_bg, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                }
                _ => unreachable!("ops and pass uniforms are built in lockstep"),
            }
        }

        // Reconcile the final running field into the layer contribution.
        let recon_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sculpt-strokes-reconcile-bg"),
            layout: &self.sculpt_strokes_reconcile.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: recon_u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(slot_view(final_slot)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.sculpt_edited.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sculpt-strokes-reconcile"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sculpt_strokes_reconcile.pipeline);
            pass.set_bind_group(0, &recon_bg, &[]);
            pass.dispatch_workgroups(output_w.div_ceil(8).max(1), output_h.div_ceil(8).max(1), 1);
        }
    }
}
