//! Narrow shared compute-binding layouts used by height/auxiliary texture kernels.

/// Uniform buffer + read-only float texture + caller-selected third binding.
pub fn uniform_texture_compute_layout_entries(
    third_binding: wgpu::BindingType,
) -> [wgpu::BindGroupLayoutEntry; 3] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: third_binding,
            count: None,
        },
    ]
}

/// Create the common three-binding compute layout without owning pipeline policy.
pub fn uniform_texture_compute_layout(
    device: &wgpu::Device,
    label: &str,
    third_binding: wgpu::BindingType,
) -> wgpu::BindGroupLayout {
    let entries = uniform_texture_compute_layout_entries(third_binding);
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

pub fn write_storage_texture_binding(format: wgpu::TextureFormat) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access: wgpu::StorageTextureAccess::WriteOnly,
        format,
        view_dimension: wgpu::TextureViewDimension::D2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_layout_keeps_exact_binding_contract() {
        let entries = uniform_texture_compute_layout_entries(write_storage_texture_binding(
            wgpu::TextureFormat::Rgba16Float,
        ));
        assert_eq!(entries.map(|entry| entry.binding), [0, 1, 2]);
        assert!(matches!(
            entries[0].ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                ..
            }
        ));
        assert!(matches!(
            entries[1].ty,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            }
        ));
        assert!(matches!(
            entries[2].ty,
            wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            }
        ));
    }
}
