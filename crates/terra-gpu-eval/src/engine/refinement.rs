//! GPU evaluator refinement implementation.

use super::compiled_plan_support::{plan_fallback_diagnostic, plan_operation_fallback};
use super::*;

impl GpuTerrainEngine {
    /// Start an isolated, resumable Complete-quality evaluation. This API is
    /// intentionally separate from the required interactive evaluator: it
    /// always recomputes into staged resources and never exposes a partial
    /// candidate through `plan_resources.current()`.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_compiled_refinement(
        &mut self,
        device: &wgpu::Device,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        plan: &CompiledTerrainPlan,
        expected_revision: PlanStructureRevision,
        invalidation: &PlanInvalidation,
        metrics: HeightfieldMetrics,
        quality: PreviewQuality,
    ) -> Result<GpuRefinementJob, GpuError> {
        if !plan.matches_structure_revision(expected_revision) {
            return Err(GpuError::StalePlan {
                plan_revision: plan.stamp().structure_revision.get(),
                expected_revision: expected_revision.get(),
            });
        }
        if !matches!(quality, PreviewQuality::Medium | PreviewQuality::Full) {
            return Err(GpuError::Wgpu(
                "resumable refinement only accepts Medium or Full quality".into(),
            ));
        }

        self.ensure_size(device, metrics);
        self.uniform_pool.reset();
        self.plan_operations.begin_evaluation();
        self.last_plan_operation_trace.clear();
        self.last_eval_stats = GpuEvalStats {
            resolution: metrics.width.max(metrics.height),
            cold_execution: true,
            ..GpuEvalStats::default()
        };

        let selected: Vec<_> = plan
            .operations()
            .iter()
            .enumerate()
            .filter_map(|(index, _)| {
                let id = PlanOpId::from_index(index);
                plan.analysis().operation_is_live(id).then_some(id)
            })
            .collect();
        let graph = compile_gpu_graph(stack, mask_assets);
        let flat_layers = stack.flatten_layers();
        let decisions: HashMap<LayerId, (Option<GpuLayerPlan>, Option<GpuFallbackReason>)> =
            flat_layers
                .iter()
                .enumerate()
                .map(|(index, layer)| {
                    (
                        layer.id(),
                        (
                            graph.plans.get(index).copied().flatten(),
                            graph.fallback_reasons.get(index).cloned().flatten(),
                        ),
                    )
                })
                .collect();
        let mut kernels = HashMap::new();
        for operation_id in &selected {
            let operation = plan.operation(*operation_id).expect("live plan operation");
            if let TerrainOpKind::RunLayerKernel {
                layer,
                output_fields,
                ..
            } = &operation.kind
            {
                if output_fields.iter().any(|field| {
                    plan.analysis().field_is_live(*field)
                        && plan.analysis().consumers(*field).iter().any(|consumer| {
                            !plan.operation(*consumer).is_some_and(|operation| {
                                matches!(operation.kind, TerrainOpKind::PublishOutput { .. })
                            })
                        })
                }) {
                    return Err(cpu_required(
                        GpuFallbackCode::AuxiliaryDependency,
                        "auxiliary field",
                        "the GPU kernel does not yet publish a live auxiliary output",
                    ));
                }
                match decisions.get(layer) {
                    Some((Some(kernel), _)) => {
                        kernels.insert(*operation_id, *kernel);
                    }
                    decision => {
                        let reason = decision
                            .and_then(|(_, reason)| reason.clone())
                            .unwrap_or_else(|| {
                                GpuFallbackReason::new(
                                    GpuFallbackCode::UnsupportedOptions,
                                    "GPU capability graph",
                                    "layer has no executable GPU kernel choice",
                                )
                            });
                        return Err(GpuError::RequiresCpu(reason));
                    }
                }
            }
        }
        let key = GpuPlanResourceKey::new(metrics.width, metrics.height, self.device_generation);
        let builder = self
            .plan_resources
            .begin_candidate(device, plan, key)
            .map_err(|error| GpuError::Wgpu(error.to_string()))?;
        let published_output_slots = plan
            .operations()
            .iter()
            .filter_map(|operation| match operation.kind {
                TerrainOpKind::PublishOutput { output, source } => Some((output, source)),
                _ => None,
            })
            .collect();

        Ok(GpuRefinementJob {
            stack: stack.clone(),
            masks: mask_assets.to_vec(),
            graph,
            plan: plan.clone(),
            expected_revision,
            invalidation: invalidation.clone(),
            metrics,
            quality,
            builder: Some(builder),
            candidate: None,
            selected,
            kernels,
            published_output_slots,
            cursor: 0,
            last_height: None,
            completion: None,
            final_copy_submitted: false,
            final_copy_complete: false,
            submissions_issued: 0,
            submissions_completed: 0,
            resource_prepare_us: 0,
            encode_us: 0,
            range_before: self.approx_range,
            trace_context: self.pending_evaluation_trace.unwrap_or_default(),
            final_submission_serial: GpuSubmissionSerial(0),
        })
    }

    /// Advance by one allocation or one GPU submission. A job never has more
    /// than one refinement submission outstanding; required interactive work is
    /// free to submit behind that single unavoidable command buffer.
    pub fn advance_compiled_refinement(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        job: &mut GpuRefinementJob,
    ) -> Result<GpuRefinementStep, GpuError> {
        if job.final_copy_complete {
            return Ok(GpuRefinementStep::ReadyToPublish);
        }
        if let Some(completion) = job.completion.as_ref() {
            let _ = device.poll(wgpu::Maintain::Poll);
            match completion.try_recv() {
                Ok(()) | Err(TryRecvError::Disconnected) => {
                    job.completion = None;
                    job.submissions_completed = job.submissions_completed.saturating_add(1);
                    if job.final_copy_submitted && job.cursor == job.selected.len() {
                        job.final_copy_complete = true;
                        return Ok(GpuRefinementStep::ReadyToPublish);
                    }
                    return Ok(GpuRefinementStep::Progressed);
                }
                Err(TryRecvError::Empty) => return Ok(GpuRefinementStep::AwaitingGpu),
            }
        }

        if let Some(builder) = job.builder.as_mut() {
            let started = Instant::now();
            let complete = builder.advance(device);
            job.resource_prepare_us = job
                .resource_prepare_us
                .saturating_add(started.elapsed().as_micros() as u64);
            if complete {
                let builder = job.builder.take().expect("builder exists");
                job.candidate = Some(
                    builder
                        .finish()
                        .map_err(|error| GpuError::Wgpu(error.to_string()))?,
                );
            }
            return Ok(GpuRefinementStep::Progressed);
        }

        if !self.retired_refinement_completions.is_empty() {
            let _ = device.poll(wgpu::Maintain::Poll);
            self.retired_refinement_completions
                .retain_mut(|completion| matches!(completion.try_recv(), Err(TryRecvError::Empty)));
            if !self.retired_refinement_completions.is_empty() {
                return Ok(GpuRefinementStep::AwaitingGpu);
            }
        }

        let candidate = job
            .candidate
            .as_ref()
            .expect("completed candidate resources");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compiled-terrain-refinement-unit"),
        });
        let encode_started = Instant::now();
        if let Some(operation_id) = job.selected.get(job.cursor).copied() {
            let scope = PropagatedDirtyScope::new(PlanDirtyScope::FullField);
            if let Err(error) = self.record_compiled_operation(
                device,
                queue,
                &mut encoder,
                &job.stack,
                &job.masks,
                &job.plan,
                candidate,
                operation_id,
                job.kernels.get(&operation_id).copied(),
                job.quality,
                scope,
                true,
                &job.invalidation,
                &job.published_output_slots,
            ) {
                let diagnostic = plan_fallback_diagnostic(
                    &job.plan,
                    &job.stack,
                    operation_id,
                    plan_operation_fallback(error),
                );
                return Err(GpuError::RequiresCpu(diagnostic.reason));
            }
            let operation = job
                .plan
                .operation(operation_id)
                .expect("selected operation");
            if matches!(operation.kind, TerrainOpKind::PublishOutput { .. }) {
                self.last_eval_stats.operations_published =
                    self.last_eval_stats.operations_published.saturating_add(1);
            } else {
                self.last_eval_stats.operations_dispatched =
                    self.last_eval_stats.operations_dispatched.saturating_add(1);
                self.last_eval_stats.plan_workgroups =
                    self.last_eval_stats.plan_workgroups.saturating_add(
                        u64::from(job.metrics.width.div_ceil(8))
                            * u64::from(job.metrics.height.div_ceil(8)),
                    );
            }
            for field in job.plan.analysis().outputs(operation_id) {
                if job.plan.field(*field).is_some_and(|field| {
                    matches!(
                        field.kind,
                        terra_core::terrain_plan::LogicalFieldKind::Height
                    )
                }) {
                    job.last_height = Some(*field);
                }
            }
            job.cursor += 1;
        } else {
            let presentation = candidate
                .view(job.plan.final_height())
                .map_err(|error| GpuError::Wgpu(error.to_string()))?;
            record_copy_views_region(
                device,
                &mut encoder,
                &self.copy,
                presentation,
                &self.ping.view,
                job.metrics.width,
                job.metrics.height,
                (0, 0, job.metrics.width, job.metrics.height),
            );
            job.final_copy_submitted = true;
        }
        job.encode_us = job
            .encode_us
            .saturating_add(encode_started.elapsed().as_micros() as u64);
        let submit_started = Instant::now();
        queue.submit(Some(encoder.finish()));
        let submission_serial = self.allocate_submission_serial();
        if job.final_copy_submitted && job.cursor == job.selected.len() {
            job.final_submission_serial = submission_serial;
        }
        self.last_eval_stats.queue_submit_us = self
            .last_eval_stats
            .queue_submit_us
            .saturating_add(submit_started.elapsed().as_micros() as u64);
        job.submissions_issued = job.submissions_issued.saturating_add(1);
        let ordinal = job.submissions_issued;
        let (sender, receiver) = mpsc::channel();
        queue.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        job.completion = Some(receiver);
        Ok(GpuRefinementStep::Submitted { ordinal })
    }

    pub fn publish_compiled_refinement(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mut job: GpuRefinementJob,
    ) -> Result<GpuEvalResult, GpuError> {
        if !job.final_copy_complete {
            return Err(GpuError::Wgpu(
                "refinement candidate published before GPU completion".into(),
            ));
        }
        let candidate = job.candidate.take().expect("completed candidate");
        let final_field = job.plan.final_height();
        let final_binding = candidate
            .layout()
            .binding(final_field)
            .expect("refinement final field has a physical binding");
        let source_resource_incarnation = candidate.incarnation();
        // The completed candidate was fenced into private ping scratch. Only the
        // freshness-checked publication call is allowed to mutate the renderer's
        // stable output allocation.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-refinement-publish"),
        });
        record_copy_views_region(
            device,
            &mut encoder,
            &self.copy,
            &self.ping.view,
            &self.published_height.view,
            job.metrics.width,
            job.metrics.height,
            (0, 0, job.metrics.width, job.metrics.height),
        );
        queue.submit(Some(encoder.finish()));
        let publication_serial = self.allocate_submission_serial();
        self.plan_resources.commit_candidate(candidate);
        self.last_graph = job.graph;
        self.active_plan_revision = Some(job.expected_revision);
        self.deferred_plan_resume = None;
        self.last_quality = Some(job.quality);
        self.current = 0;
        self.last_dirty_rect = None;
        self.mark_all_tiles_dirty();
        self.last_eval_stats.selected_operations = job.selected.len() as u32;
        self.last_eval_stats.resource_prepare_us = job.resource_prepare_us;
        self.last_eval_stats.command_encode_us = job.encode_us;
        self.last_eval_stats.dirty_texels =
            u64::from(job.metrics.width).saturating_mul(u64::from(job.metrics.height));
        self.pending_evaluation_trace = None;
        let output_identity = GpuTerrainOutputIdentity {
            output: self.allocate_output_id(),
            frame_id: job.trace_context.frame_id,
            generation: job.trace_context.generation,
            evaluation_id: job.trace_context.evaluation_id,
            plan_revision: job.expected_revision.get(),
            requested_quality: job.quality,
            actual_quality: job.quality,
            intent: GpuEvaluationIntent::Complete,
            selected_field: terra_gpu::output_identity::GpuSelectedFieldIdentity {
                selected: final_field,
                expected_final: final_field,
                resource_incarnation: source_resource_incarnation,
                physical_allocation: final_binding.physical.index(),
            },
            output_resource: terra_gpu::output_identity::GpuOutputResourceIdentity {
                device_generation: self.device_generation,
                incarnation: self.output_resource_incarnation,
                slot: self.output_slot(),
            },
            extent: (job.metrics.width, job.metrics.height),
            coverage: terra_gpu::output_identity::GpuOutputCoverage::WholeField,
            completeness: terra_gpu::output_identity::GpuOutputCompleteness::Complete,
            invalidation: terra_gpu::output_identity::GpuInvalidationKind::Cold,
            last_write: terra_gpu::output_identity::GpuLastWriteIdentity {
                serial: publication_serial,
                completion: terra_gpu::output_identity::GpuSubmissionCompletion::Submitted,
            },
        };
        self.last_output_identity = Some(output_identity);
        Ok(GpuEvalResult {
            width: job.metrics.width,
            height: job.metrics.height,
            world_size: (job.metrics.world_size_x, job.metrics.world_size_z),
            height_range: self.approx_range,
            fully_gpu: true,
            freshness: GpuPreviewFreshness::Current,
            cpu: None,
            resume_cpu_from: None,
            cpu_fallback: None,
            did_eval: !job.selected.is_empty(),
            output_identity: Some(output_identity),
        })
    }

    pub fn abandon_compiled_refinement(&mut self, job: GpuRefinementJob) {
        if let Some(completion) = job.completion {
            self.retired_refinement_completions.push(completion);
        }
        self.approx_range = job.range_before;
        self.last_dirty_rect = None;
        self.pending_evaluation_trace = None;
    }

    /// Global refinement queue depth, including a fence retained from a
    /// superseded generation. Required interactive submissions are not counted.
    pub fn refinement_submissions_in_flight(&self, job: &GpuRefinementJob) -> usize {
        self.retired_refinement_completions.len() + usize::from(job.completion.is_some())
    }
}
