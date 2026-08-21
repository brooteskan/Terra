//! GPU evaluator pipelines common implementation.

use super::*;

impl GpuTerrainEngine {
    pub(in super::super) fn fill_slot(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        slot: TexSlot,
        value: f32,
    ) {
        let u = FillU {
            width: self.metrics.width,
            height: self.metrics.height,
            value,
            _pad: 0.0,
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let view = self.view_of(slot);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fill-bg"),
            layout: &self.fill.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fill"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fill.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
        }
    }

    pub(in super::super) fn fill_view_extent(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        extent: [u32; 2],
        value: f32,
    ) {
        let [width, height] = extent;
        let uniform = FillU {
            width,
            height,
            value,
            _pad: 0.0,
        };
        let buffer = self.write_uniform(device, queue, &uniform);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fill-extent-bg"),
            layout: &self.fill.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fill-extent"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.fill.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    pub(in super::super) fn copy_slots(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: TexSlot,
        dst: TexSlot,
    ) {
        let u = CopyU {
            width: self.metrics.width,
            height: self.metrics.height,
            region_x: 0,
            region_y: 0,
            region_w: self.metrics.width,
            region_h: self.metrics.height,
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copy-bg"),
            layout: &self.copy.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(self.view_of(src)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(self.view_of(dst)),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
        }
    }

    pub(in super::super) fn blend_into_current(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        opacity: f32,
        mode: BlendMode,
    ) -> Result<(), GpuError> {
        self.blend_into_current_with_mask(
            device,
            queue,
            encoder,
            opacity,
            mode,
            [TexSlot::MaskOnes, TexSlot::UnitMask],
        )
    }

    pub(in super::super) fn blend_into_current_with_mask(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        opacity: f32,
        mode: BlendMode,
        masks: [TexSlot; 2],
    ) -> Result<(), GpuError> {
        let u = BlendU {
            width: self.metrics.width,
            height: self.metrics.height,
            opacity,
            mode: gpu_blend_mode(mode).ok_or_else(|| {
                cpu_required(
                    GpuFallbackCode::BlendMode,
                    "blend",
                    format!("{mode:?} is not implemented"),
                )
            })?,
            region_x: 0,
            region_y: 0,
            region_w: self.metrics.width,
            region_h: self.metrics.height,
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let src_ping = self.current == 0;
        let (base, dst) = if src_ping {
            (&self.ping.view, &self.pong.view)
        } else {
            (&self.pong.view, &self.ping.view)
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blend-bg"),
            layout: &self.blend.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(base),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(self.view_of(masks[0])),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(self.view_of(masks[1])),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(dst),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("blend"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blend.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.metrics.width.div_ceil(8),
                self.metrics.height.div_ceil(8),
                1,
            );
        }
        self.swap_current();
        Ok(())
    }
}
