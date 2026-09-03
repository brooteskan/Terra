//! GPU evaluator readback implementation.

use super::*;

impl GpuTerrainEngine {
    pub(super) fn readback_height_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        label: &str,
    ) -> Result<Heightfield, GpuError> {
        let w = self.metrics.width;
        let h = self.metrics.height;
        let unpadded = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let padded_f32 = readback_f32(device, queue, &buf, (padded * h / 4) as usize)?;
        let mut dense = Vec::with_capacity((w * h) as usize);
        let row_floats = (padded / 4) as usize;
        for y in 0..h as usize {
            let start = y * row_floats;
            dense.extend_from_slice(&padded_f32[start..start + w as usize]);
        }
        Ok(Heightfield::from_dense(self.metrics, &dense))
    }

    pub fn readback_current(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Heightfield, GpuError> {
        self.readback_height_texture(device, queue, self.output_texture(), "gpu-height-readback")
    }

    #[cfg(feature = "gpu-parity")]
    pub fn readback_simulation_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<crate::GpuSimulationStateReadback, GpuError> {
        let invalid_bits =
            readback_f32(device, queue, &self.simulation_invalid_state_buffer, 1)?[0].to_bits();
        let w = self.metrics.width;
        let h = self.metrics.height;
        let unpadded = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let sources = [
            (&self.hardness.texture, "gpu-hardness-readback"),
            (&self.water_a.texture, "gpu-water-a-readback"),
            (&self.water_b.texture, "gpu-water-b-readback"),
            (&self.sed_a.texture, "gpu-sediment-a-readback"),
            (&self.sed_b.texture, "gpu-sediment-b-readback"),
            (&self.delta.texture, "gpu-redistribution-readback"),
            (&self.rainfall.texture, "gpu-rainfall-readback"),
            (&self.loose_sediment.texture, "gpu-loose-sediment-readback"),
        ];
        let buffers: Vec<_> = sources
            .iter()
            .map(|(_, label)| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (padded * h) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-simulation-state-readback"),
        });
        for ((texture, _), buffer) in sources.iter().zip(&buffers) {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }
        queue.submit(Some(encoder.finish()));
        let mut fields = Vec::with_capacity(buffers.len());
        let row_floats = (padded / 4) as usize;
        for buffer in &buffers {
            let padded_f32 = readback_f32(device, queue, buffer, (padded * h / 4) as usize)?;
            let mut dense = Vec::with_capacity((w * h) as usize);
            for y in 0..h as usize {
                let start = y * row_floats;
                dense.extend_from_slice(&padded_f32[start..start + w as usize]);
            }
            fields.push(Heightfield::from_dense(self.metrics, &dense));
        }
        let mut fields = fields.into_iter();
        Ok(crate::GpuSimulationStateReadback {
            invalid_state_bits: invalid_bits,
            hardness: fields.next().expect("hardness readback"),
            water_a: fields.next().expect("water-a readback"),
            water_b: fields.next().expect("water-b readback"),
            sediment_a: fields.next().expect("sediment-a readback"),
            sediment_b: fields.next().expect("sediment-b readback"),
            redistribution: fields.next().expect("redistribution readback"),
            rainfall: fields.next().expect("rainfall readback"),
            loose_sediment: fields.next().expect("loose-sediment readback"),
        })
    }
}
