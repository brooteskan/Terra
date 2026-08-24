use terra_render::{
    BrushOverlay, GpuContext, GuideOverlay, GuideState, OptionalResource, OptionalResourceState,
    PresentationBackendId, PresentationPipelineBundle, PresentationPipelineFeature, SurfacePick,
    TerrainRenderer,
};

/// Editor-only viewport visuals owned and scheduled by the application shell.
pub(crate) struct EditorOverlays {
    brush: OptionalResource<BrushOverlay>,
    guides: OptionalResource<GuideOverlay>,
    guide_state: GuideState,
    bound_height_revision: u64,
}

impl EditorOverlays {
    pub(crate) fn new(gpu: &GpuContext, renderer: &TerrainRenderer) -> Self {
        let _ = gpu;
        Self {
            brush: OptionalResource::default(),
            guides: OptionalResource::default(),
            guide_state: GuideState::default(),
            bound_height_revision: renderer.height_binding_revision(),
        }
    }

    pub(crate) fn state(&self, feature: PresentationPipelineFeature) -> &OptionalResourceState {
        match feature {
            PresentationPipelineFeature::Guides => self.guides.state(),
            PresentationPipelineFeature::Brush => self.brush.state(),
            _ => panic!("renderer-owned feature queried through EditorOverlays"),
        }
    }

    pub(crate) fn begin_compile(
        &mut self,
        feature: PresentationPipelineFeature,
        request_id: u64,
    ) -> bool {
        match feature {
            PresentationPipelineFeature::Guides => self.guides.begin(request_id),
            PresentationPipelineFeature::Brush => self.brush.begin(request_id),
            _ => false,
        }
    }

    pub(crate) fn fail_compile(
        &mut self,
        feature: PresentationPipelineFeature,
        request_id: u64,
        message: String,
    ) -> bool {
        match feature {
            PresentationPipelineFeature::Guides => self.guides.fail(request_id, message),
            PresentationPipelineFeature::Brush => self.brush.fail(request_id, message),
            _ => false,
        }
    }

    pub(crate) fn install(
        &mut self,
        gpu: &GpuContext,
        renderer: &TerrainRenderer,
        request_id: u64,
        bundle: PresentationPipelineBundle,
    ) -> Result<(), String> {
        match bundle {
            PresentationPipelineBundle::Guides(mut guides) => {
                guides.set_state(self.guide_state);
                self.guides
                    .install(request_id, guides)
                    .map_err(|_| "stale guide pipeline bundle".to_string())
            }
            PresentationPipelineBundle::Brush(mut brush) => {
                brush.rebind_height(&gpu.device, renderer.heights.display_height_view());
                brush.rebind_depth(&gpu.device, &renderer.depth);
                self.bound_height_revision = renderer.height_binding_revision();
                self.brush
                    .install(request_id, brush)
                    .map_err(|_| "stale brush pipeline bundle".to_string())
            }
            _ => Err("renderer bundle cannot be installed in EditorOverlays".into()),
        }
    }

    pub(crate) fn cancel_compiles(&mut self) {
        self.guides.reset_unready();
        self.brush.reset_unready();
    }

    pub(crate) fn invalidate(&mut self) {
        self.guides.reset();
        self.brush.reset();
    }

    pub(crate) fn set_guides(&mut self, grid: bool, bounds: bool) {
        self.guide_state = GuideState { grid, bounds };
        if let Some(guides) = self.guides.ready_mut() {
            guides.set_state(self.guide_state);
        }
    }

    pub(crate) fn sync_height(&mut self, gpu: &GpuContext, renderer: &TerrainRenderer) {
        let revision = renderer.height_binding_revision();
        if revision != self.bound_height_revision {
            if let Some(brush) = self.brush.ready_mut() {
                brush.rebind_height(&gpu.device, renderer.heights.display_height_view());
                brush.rebind_depth(&gpu.device, &renderer.depth);
            }
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
        radius: f32,
        color: [f32; 4],
        visible: bool,
    ) {
        self.sync_height(gpu, renderer);
        let (width, height) = renderer.size();
        let aspect = width as f32 / height.max(1) as f32;
        let Some(brush) = self.brush.ready_mut() else {
            return;
        };
        brush.request_surface_pick(
            &gpu.device,
            &gpu.queue,
            &renderer.camera,
            aspect,
            cursor,
            screen,
            renderer.heights.world_size,
            renderer.heights.height_range,
            radius,
            color,
            visible,
            renderer.traversal_mode(),
            renderer.render_origin_xz(),
            renderer.height_binding_revision(),
        );
    }

    pub(crate) fn latest_surface_pick(
        &self,
        renderer: &TerrainRenderer,
        cursor: (f32, f32),
        screen: (f32, f32),
    ) -> Option<SurfacePick> {
        let (width, height) = renderer.size();
        let aspect = width as f32 / height.max(1) as f32;
        self.brush.ready().and_then(|brush| {
            brush.latest_pick_for(
                &renderer.camera,
                aspect,
                cursor,
                screen,
                renderer.traversal_mode(),
                renderer.render_origin_xz(),
                renderer.height_binding_revision(),
            )
        })
    }

    pub(crate) fn poll_brush(&mut self, device: &wgpu::Device) {
        if let Some(brush) = self.brush.ready_mut() {
            brush.poll(device);
        }
    }

    pub(crate) fn hide_brush(&mut self, queue: &wgpu::Queue) {
        if let Some(brush) = self.brush.ready_mut() {
            brush.hide(queue);
        }
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
        if let Some(brush) = self.brush.ready_mut() {
            brush.upload_view_proj(&gpu.queue, view_proj);
        }
        if let Some(guides) = self.guides.ready_mut() {
            guides.upload_view_proj(&gpu.queue, view_proj);
        }
        let render_origin = renderer.render_origin_xz();
        if let Some(guides) = self.guides.ready_mut() {
            guides.sync_geometry(
                &gpu.queue,
                renderer.heights.world_size,
                renderer.heights.height_range,
                (
                    (renderer.camera.target.x - render_origin.x) as f32,
                    (renderer.camera.target.z - render_origin.y) as f32,
                ),
            );
        }

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
            if let Some(guides) = self.guides.ready() {
                guides.draw(&mut pass);
            }
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
            if let Some(brush) = self.brush.ready() {
                brush.draw(&mut pass, depth_tested);
            }
        }
        gpu.queue.submit(Some(encoder.finish()));
    }
}
