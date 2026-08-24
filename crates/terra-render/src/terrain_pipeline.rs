use std::sync::Arc;

use crate::{terrain_shader, TerrainGrid, TerrainShaderVariant};

/// Complete set of pipelines that must switch together for one terrain variant.
pub struct TerrainPipelineBundle {
    pub(crate) terrain: wgpu::RenderPipeline,
    pub(crate) variant: TerrainShaderVariant,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) family: Arc<()>,
}

pub struct OceanPipelineBundle {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) variant: TerrainShaderVariant,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) family: Arc<()>,
}

pub struct WireframePipelineBundle {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) variant: TerrainShaderVariant,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) family: Arc<()>,
}

impl TerrainPipelineBundle {
    pub fn variant(&self) -> TerrainShaderVariant {
        self.variant
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TerrainPipelineCompileError {
    #[error("{variant} terrain WGSL parse failed: {message}")]
    ShaderParse {
        variant: &'static str,
        message: String,
    },
    #[error("{variant} terrain WGSL validation failed: {message}")]
    ShaderValidation {
        variant: &'static str,
        message: String,
    },
}

/// Renderer-issued, thread-safe capability for constructing compatible bundles.
#[derive(Clone)]
pub struct TerrainPipelineCompiler {
    device: wgpu::Device,
    pipelines: Arc<terra_gpu::PipelineCacheRegistry>,
    bind_group_layout: wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    family: Arc<()>,
}

impl TerrainPipelineCompiler {
    pub(crate) fn new(
        device: wgpu::Device,
        pipelines: Arc<terra_gpu::PipelineCacheRegistry>,
        bind_group_layout: wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
        family: Arc<()>,
    ) -> Self {
        Self {
            device,
            pipelines,
            bind_group_layout,
            format,
            family,
        }
    }

    pub fn compile(
        &self,
        variant: TerrainShaderVariant,
    ) -> Result<TerrainPipelineBundle, TerrainPipelineCompileError> {
        let source = terrain_shader::compose(variant);
        let module = naga::front::wgsl::parse_str(&source).map_err(|error| {
            TerrainPipelineCompileError::ShaderParse {
                variant: variant.name(),
                message: error.to_string(),
            }
        })?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|error| TerrainPipelineCompileError::ShaderValidation {
            variant: variant.name(),
            message: error.to_string(),
        })?;

        log::info!(
            "terra-render: compiling {} terrain shader/pipelines…",
            variant.name()
        );
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(variant.shader_label()),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(match variant {
                    TerrainShaderVariant::Bounded => "terrain-bounded-pl",
                    TerrainShaderVariant::Infinite => "terrain-infinite-pl",
                }),
                bind_group_layouts: &[&self.bind_group_layout],
                push_constant_ranges: &[],
            });

        let terrain =
            self.pipelines
                .render_pipeline(variant.terrain_pipeline_label(), self.format, || {
                    self.device
                        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: Some(variant.terrain_pipeline_label()),
                            layout: Some(&layout),
                            vertex: wgpu::VertexState {
                                module: &shader,
                                entry_point: Some("vs_main"),
                                buffers: &[TerrainGrid::vertex_layout()],
                                compilation_options: Default::default(),
                            },
                            fragment: Some(wgpu::FragmentState {
                                module: &shader,
                                entry_point: Some("fs_main"),
                                targets: &[Some(wgpu::ColorTargetState {
                                    format: self.format,
                                    blend: Some(wgpu::BlendState::REPLACE),
                                    write_mask: wgpu::ColorWrites::ALL,
                                })],
                                compilation_options: Default::default(),
                            }),
                            primitive: wgpu::PrimitiveState {
                                topology: wgpu::PrimitiveTopology::TriangleList,
                                cull_mode: Some(wgpu::Face::Back),
                                ..Default::default()
                            },
                            depth_stencil: Some(wgpu::DepthStencilState {
                                format: wgpu::TextureFormat::Depth32Float,
                                depth_write_enabled: true,
                                depth_compare: wgpu::CompareFunction::Less,
                                stencil: Default::default(),
                                bias: Default::default(),
                            }),
                            multisample: Default::default(),
                            multiview: None,
                            cache: self.pipelines.driver_cache(),
                        })
                });
        Ok(TerrainPipelineBundle {
            terrain,
            variant,
            format: self.format,
            family: Arc::clone(&self.family),
        })
    }

    pub fn compile_ocean(
        &self,
        variant: TerrainShaderVariant,
    ) -> Result<OceanPipelineBundle, TerrainPipelineCompileError> {
        let (shader, layout) = self.shader_and_layout(variant)?;
        let pipeline =
            self.pipelines
                .render_pipeline(variant.ocean_pipeline_label(), self.format, || {
                    self.device
                        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: Some(variant.ocean_pipeline_label()),
                            layout: Some(&layout),
                            vertex: wgpu::VertexState {
                                module: &shader,
                                entry_point: Some("vs_ocean"),
                                buffers: &[TerrainGrid::vertex_layout()],
                                compilation_options: Default::default(),
                            },
                            fragment: Some(wgpu::FragmentState {
                                module: &shader,
                                entry_point: Some("fs_ocean"),
                                targets: &[Some(wgpu::ColorTargetState {
                                    format: self.format,
                                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                                    write_mask: wgpu::ColorWrites::ALL,
                                })],
                                compilation_options: Default::default(),
                            }),
                            primitive: wgpu::PrimitiveState {
                                topology: wgpu::PrimitiveTopology::TriangleList,
                                cull_mode: None,
                                ..Default::default()
                            },
                            depth_stencil: Some(wgpu::DepthStencilState {
                                format: wgpu::TextureFormat::Depth32Float,
                                depth_write_enabled: true,
                                depth_compare: wgpu::CompareFunction::Less,
                                stencil: Default::default(),
                                bias: Default::default(),
                            }),
                            multisample: Default::default(),
                            multiview: None,
                            cache: self.pipelines.driver_cache(),
                        })
                });
        Ok(OceanPipelineBundle {
            pipeline,
            variant,
            format: self.format,
            family: Arc::clone(&self.family),
        })
    }

    pub fn compile_wireframe(
        &self,
        variant: TerrainShaderVariant,
    ) -> Result<WireframePipelineBundle, TerrainPipelineCompileError> {
        let (shader, layout) = self.shader_and_layout(variant)?;
        let pipeline =
            self.pipelines
                .render_pipeline(variant.wireframe_pipeline_label(), self.format, || {
                    self.device
                        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: Some(variant.wireframe_pipeline_label()),
                            layout: Some(&layout),
                            vertex: wgpu::VertexState {
                                module: &shader,
                                entry_point: Some("vs_main"),
                                buffers: &[TerrainGrid::vertex_layout()],
                                compilation_options: Default::default(),
                            },
                            fragment: Some(wgpu::FragmentState {
                                module: &shader,
                                entry_point: Some("fs_wireframe"),
                                targets: &[Some(wgpu::ColorTargetState {
                                    format: self.format,
                                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                                    write_mask: wgpu::ColorWrites::ALL,
                                })],
                                compilation_options: Default::default(),
                            }),
                            primitive: wgpu::PrimitiveState {
                                topology: wgpu::PrimitiveTopology::LineList,
                                cull_mode: None,
                                ..Default::default()
                            },
                            depth_stencil: Some(wgpu::DepthStencilState {
                                format: wgpu::TextureFormat::Depth32Float,
                                depth_write_enabled: false,
                                depth_compare: wgpu::CompareFunction::LessEqual,
                                stencil: Default::default(),
                                bias: wgpu::DepthBiasState {
                                    constant: -4,
                                    slope_scale: -2.0,
                                    clamp: 0.0,
                                },
                            }),
                            multisample: Default::default(),
                            multiview: None,
                            cache: self.pipelines.driver_cache(),
                        })
                });
        Ok(WireframePipelineBundle {
            pipeline,
            variant,
            format: self.format,
            family: Arc::clone(&self.family),
        })
    }

    fn shader_and_layout(
        &self,
        variant: TerrainShaderVariant,
    ) -> Result<(wgpu::ShaderModule, wgpu::PipelineLayout), TerrainPipelineCompileError> {
        let source = terrain_shader::compose(variant);
        let module = naga::front::wgsl::parse_str(&source).map_err(|error| {
            TerrainPipelineCompileError::ShaderParse {
                variant: variant.name(),
                message: error.to_string(),
            }
        })?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|error| TerrainPipelineCompileError::ShaderValidation {
            variant: variant.name(),
            message: error.to_string(),
        })?;
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(variant.shader_label()),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(match variant {
                    TerrainShaderVariant::Bounded => "terrain-bounded-optional-pl",
                    TerrainShaderVariant::Infinite => "terrain-infinite-optional-pl",
                }),
                bind_group_layouts: &[&self.bind_group_layout],
                push_constant_ranges: &[],
            });
        Ok((shader, layout))
    }
}
