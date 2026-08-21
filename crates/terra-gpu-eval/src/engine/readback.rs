//! GPU evaluator readback implementation.

use super::*;

impl GpuTerrainEngine {
    pub fn readback_current(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Heightfield, GpuError> {
        let w = self.metrics.width;
        let h = self.metrics.height;
        let unpadded = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-readback-buf"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-readback-enc"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: self.output_texture(),
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
}
