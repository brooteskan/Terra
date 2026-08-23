use terra_render::{
    BrushOverlay, GpuContext, GuideOverlay, GuideState, PresentationBackendId, SurfacePick,
    TerrainRenderer,
};

/// Editor-only viewport visuals owned and scheduled by the application shell.
pub(crate) struct EditorOverlays {
    pub(crate) brush: BrushOverlay,
    guides: GuideOverlay,
    bound_height_revision: u64,
}

impl EditorOverlays {
    pub(crate) fn new(gpu: &GpuContext, renderer: &TerrainRenderer) -> Self {
        let mut brush = BrushOverlay::new(&gpu.device, gpu.pipeline_registry(), gpu.surface_format);
        brush.rebind_height(&gpu.device, renderer.heights.display_height_view());
        Self {
            brush,
            guides: GuideOverlay::new(&gpu.device, gpu.pipeline_registry(), gpu.surface_format),
            bound_height_revision: renderer.height_binding_revision(),
        }
    }

    pub(crate) fn set_guides(&mut self, grid: bool, bounds: bool) {
        self.guides.set_state(GuideState { grid, bounds });
    }

    pub(crate) fn sync_height(&mut self, gpu: &GpuContext, renderer: &TerrainRenderer) {
        let revision = renderer.height_binding_revision();
        if revision != self.bound_height_revision {
            self.brush
                .rebind_height(&gpu.device, renderer.heights.display_height_view());
            self.bound_height_revision = revision;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn request_surface_pick(
        &mut self,
        gpu: &GpuContext,
        renderer: &TerrainRenderer,
        cursor: (f32, f32),
        screen: (f32, f32),
        radius_uv: f32,
        color: [f32; 4],
    ) {
        self.sync_height(gpu, renderer);
        if renderer.traversal_mode() == terra_render::TerrainTraversalMode::Infinite {
            // Sparse authored editing is a later slice. Do not run the bounded
            // monolithic picker in an incorrect coordinate frame.
            self.brush.hide(&gpu.queue);
            return;
        }
        let (width, height) = renderer.size();
        let aspect = width as f32 / height.max(1) as f32;
        self.brush.request_surface_pick(
            &gpu.device,
            &gpu.queue,
            &renderer.camera,
            aspect,
            cursor,
            screen,
            renderer.heights.world_size,
            renderer.heights.height_range,
            radius_uv,
            color,
        );
    }

    pub(crate) fn latest_surface_pick(
        &self,
        renderer: &TerrainRenderer,
        cursor: (f32, f32),
        screen: (f32, f32),
    ) -> Option<SurfacePick> {
        if renderer.traversal_mode() == terra_render::TerrainTraversalMode::Infinite {
            return None;
        }
        let (width, height) = renderer.size();
        let aspect = width as f32 / height.max(1) as f32;
        self.brush
            .latest_pick_for(&renderer.camera, aspect, cursor, screen)
    }

    pub(crate) fn render(
        &mut self,
        gpu: &GpuContext,
        renderer: &TerrainRenderer,
        view: &wgpu::TextureView,
    ) {
        self.sync_height(gpu, renderer);
        let (width, height) = renderer.size();
        let view_proj = renderer.camera_view_proj(width as f32 / height.max(1) as f32);
        self.brush.upload_view_proj(&gpu.queue, view_proj);
        self.guides.upload_view_proj(&gpu.queue, view_proj);
        let render_origin = renderer.render_origin_xz();
        self.guides.sync_geometry(
            &gpu.queue,
            renderer.heights.world_size,
            renderer.heights.height_range,
            (
                (renderer.camera.target.x - render_origin.x) as f32,
                (renderer.camera.target.z - render_origin.y) as f32,
            ),
        );

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("editor-overlay-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("editor-guide-overlay-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &renderer.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.guides.draw(&mut pass);
        }
        {
            let depth_tested = renderer.presentation_backend() == PresentationBackendId::RasterLit;
            let depth_attachment = depth_tested.then_some(wgpu::RenderPassDepthStencilAttachment {
                view: &renderer.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("editor-brush-overlay-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.brush.draw(&mut pass, depth_tested);
        }
        gpu.queue.submit(Some(encoder.finish()));
    }
}
