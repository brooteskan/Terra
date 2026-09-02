//! Staging ring for CPU→GPU texture uploads.
//!
//! Rotates through reusable `COPY_DST | COPY_SRC` buffers so large height/aux
//! uploads use `copy_buffer_to_texture` with 256-byte row alignment instead of
//! one-shot unaligned `queue.write_texture` for every region.

/// One slot in the staging ring.
struct StagingSlot {
    buffer: wgpu::Buffer,
    capacity: u64,
    /// Frame index when this slot was last submitted (for recycling).
    submitted_frame: u64,
}

/// Multi-buffer staging ring aligned for wgpu texture copies (256-byte rows).
pub struct StagingRing {
    slots: Vec<StagingSlot>,
    cursor: usize,
    frame: u64,
    /// Frames before a slot may be rewritten (matches surface latency).
    latency: u64,
}

impl StagingRing {
    pub fn new(device: &wgpu::Device, slot_count: usize, initial_capacity: u64) -> Self {
        let slots = (0..slot_count.max(2))
            .map(|i| {
                let capacity = initial_capacity.max(256);
                StagingSlot {
                    buffer: create_staging_buffer(device, capacity, i),
                    capacity,
                    submitted_frame: 0,
                }
            })
            .collect();
        Self {
            slots,
            cursor: 0,
            frame: 0,
            latency: 2,
        }
    }

    pub fn begin_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
    }

    /// Upload tightly packed `f32` texels into `dst` at `origin`.
    ///
    /// Tiny payloads stay on `queue.write_texture` (lower overhead). Larger
    /// regions go through a ring slot with padded rows.
    // Staging upload: device/queue + optional encoder + destination texture and
    // its region geometry (origin/width/height) + the source slice, each used
    // once. Kept flat.
    #[allow(clippy::too_many_arguments)]
    pub fn write_r32_region(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: Option<&mut wgpu::CommandEncoder>,
        dst: &wgpu::Texture,
        origin: wgpu::Origin3d,
        width: u32,
        height: u32,
        data: &[f32],
    ) {
        let bytes = bytemuck::cast_slice::<f32, u8>(data);
        let row_bytes = (width * 4) as u64;
        let aligned_row = align256(row_bytes);
        let needed = aligned_row * u64::from(height);

        // Prefer write_texture when we have no encoder or the payload is small.
        if encoder.is_none() || needed <= 64 * 1024 {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: dst,
                    mip_level: 0,
                    origin,
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            return;
        }

        let encoder = encoder.expect("checked above");
        let slot_index = self.acquire_slot(device, needed);
        let slot = &mut self.slots[slot_index];

        if aligned_row == row_bytes {
            queue.write_buffer(&slot.buffer, 0, bytes);
        } else {
            let mut padded = vec![0u8; needed as usize];
            for row in 0..height as usize {
                let src = row * width as usize * 4;
                let dst_off = row * aligned_row as usize;
                padded[dst_off..dst_off + row_bytes as usize]
                    .copy_from_slice(&bytes[src..src + row_bytes as usize]);
            }
            queue.write_buffer(&slot.buffer, 0, &padded);
        }

        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_row as u32),
                    rows_per_image: Some(height),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst,
                mip_level: 0,
                origin,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        slot.submitted_frame = self.frame;
        self.cursor = (slot_index + 1) % self.slots.len();
    }

    fn acquire_slot(&mut self, device: &wgpu::Device, needed: u64) -> usize {
        let n = self.slots.len();
        for offset in 0..n {
            let idx = (self.cursor + offset) % n;
            let free = self.frame.saturating_sub(self.slots[idx].submitted_frame) >= self.latency
                || self.slots[idx].submitted_frame == 0;
            if !free {
                continue;
            }
            if self.slots[idx].capacity < needed {
                self.slots[idx].capacity = needed.next_power_of_two().max(needed);
                self.slots[idx].buffer =
                    create_staging_buffer(device, self.slots[idx].capacity, idx);
            }
            return idx;
        }
        let idx = self.cursor % n;
        if self.slots[idx].capacity < needed {
            self.slots[idx].capacity = needed.next_power_of_two().max(needed);
            self.slots[idx].buffer = create_staging_buffer(device, self.slots[idx].capacity, idx);
        }
        idx
    }
}

fn align256(bytes: u64) -> u64 {
    (bytes + 255) & !255
}

fn create_staging_buffer(device: &wgpu::Device, capacity: u64, index: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("terrain-staging-{index}")),
        size: capacity,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
