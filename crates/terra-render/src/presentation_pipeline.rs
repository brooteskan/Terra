use std::sync::Arc;

use crate::{
    BrushOverlay, GuideOverlay, OceanPipelineBundle, OverhangOverlay, PathTracer,
    TerrainPipelineBundle, TerrainPipelineCompiler, TerrainShaderVariant, VegetationOverlay,
    WireframePipelineBundle,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PresentationPipelineFeature {
    InfiniteTerrain,
    Ocean(TerrainShaderVariant),
    Wireframe(TerrainShaderVariant),
    Progressive,
    Overhang,
    Vegetation,
    Guides,
    Brush,
}

impl PresentationPipelineFeature {
    pub const fn label(self) -> &'static str {
        match self {
            Self::InfiniteTerrain => "Infinite terrain",
            Self::Ocean(_) => "ocean",
            Self::Wireframe(_) => "wireframe",
            Self::Progressive => "progressive renderer",
            Self::Overhang => "overhang",
            Self::Vegetation => "vegetation",
            Self::Guides => "guides",
            Self::Brush => "brush",
        }
    }
}

pub struct ProgressivePresentationBundle {
    pub(crate) progressive: crate::progressive::ProgressiveRenderer,
    pub(crate) path_tracer: PathTracer,
}

pub enum PresentationPipelineBundle {
    Terrain(TerrainPipelineBundle),
    Ocean(OceanPipelineBundle),
    Wireframe(WireframePipelineBundle),
    Progressive(ProgressivePresentationBundle),
    Overhang(OverhangOverlay),
    Vegetation(VegetationOverlay),
    Guides(GuideOverlay),
    Brush(BrushOverlay),
}

impl PresentationPipelineBundle {
    pub fn feature(&self) -> PresentationPipelineFeature {
        match self {
            Self::Terrain(_) => PresentationPipelineFeature::InfiniteTerrain,
            Self::Ocean(bundle) => PresentationPipelineFeature::Ocean(bundle.variant),
            Self::Wireframe(bundle) => PresentationPipelineFeature::Wireframe(bundle.variant),
            Self::Progressive(_) => PresentationPipelineFeature::Progressive,
            Self::Overhang(_) => PresentationPipelineFeature::Overhang,
            Self::Vegetation(_) => PresentationPipelineFeature::Vegetation,
            Self::Guides(_) => PresentationPipelineFeature::Guides,
            Self::Brush(_) => PresentationPipelineFeature::Brush,
        }
    }
}

#[derive(Clone)]
pub struct PresentationPipelineCompiler {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Arc<terra_gpu::PipelineCacheRegistry>,
    terrain: TerrainPipelineCompiler,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    internal_scale: f32,
}

impl PresentationPipelineCompiler {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipelines: Arc<terra_gpu::PipelineCacheRegistry>,
        terrain: TerrainPipelineCompiler,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        internal_scale: f32,
    ) -> Self {
        Self {
            device,
            queue,
            pipelines,
            terrain,
            format,
            width,
            height,
            internal_scale,
        }
    }

    pub fn compile(
        &self,
        feature: PresentationPipelineFeature,
    ) -> Result<PresentationPipelineBundle, String> {
        let bundle = match feature {
            PresentationPipelineFeature::InfiniteTerrain => PresentationPipelineBundle::Terrain(
                self.terrain
                    .compile(TerrainShaderVariant::Infinite)
                    .map_err(|error| error.to_string())?,
            ),
            PresentationPipelineFeature::Ocean(variant) => PresentationPipelineBundle::Ocean(
                self.terrain
                    .compile_ocean(variant)
                    .map_err(|error| error.to_string())?,
            ),
            PresentationPipelineFeature::Wireframe(variant) => {
                PresentationPipelineBundle::Wireframe(
                    self.terrain
                        .compile_wireframe(variant)
                        .map_err(|error| error.to_string())?,
                )
            }
            PresentationPipelineFeature::Progressive => {
                let progressive = crate::progressive::ProgressiveRenderer::new(
                    &self.device,
                    &self.pipelines,
                    self.width,
                    self.height,
                    self.format,
                );
                let path_tracer = PathTracer::new(
                    &self.device,
                    &self.queue,
                    &self.pipelines,
                    self.width,
                    self.height,
                    self.internal_scale,
                );
                PresentationPipelineBundle::Progressive(ProgressivePresentationBundle {
                    progressive,
                    path_tracer,
                })
            }
            PresentationPipelineFeature::Overhang => PresentationPipelineBundle::Overhang(
                OverhangOverlay::new(&self.device, &self.pipelines, self.format),
            ),
            PresentationPipelineFeature::Vegetation => PresentationPipelineBundle::Vegetation(
                VegetationOverlay::new(&self.device, &self.pipelines, self.format),
            ),
            PresentationPipelineFeature::Guides => PresentationPipelineBundle::Guides(
                GuideOverlay::new(&self.device, &self.pipelines, self.format),
            ),
            PresentationPipelineFeature::Brush => PresentationPipelineBundle::Brush(
                BrushOverlay::new(&self.device, &self.pipelines, self.format),
            ),
        };
        Ok(bundle)
    }
}
