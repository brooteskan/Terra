//! Single-channel interchange exports with one file per selected field and shared metadata.

use std::collections::HashSet;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use terra_core::document::TerrainDocument;
use terra_core::fields::FieldId;
use terra_core::heightfield::Heightfield;
use terra_core::terrain_plan::{
    compile_terrain_plan, PlanStructureRevision, TerrainOpKind, TerrainPlanStamp,
};
use terra_cpu_eval::EvalContext;
use terra_jobs::CancelToken;

use crate::{BackgroundExporter, ExportError, IoError};

const METADATA_FILENAME: &str = "export_metadata.json";

#[derive(serde::Serialize)]
struct ExportMetadata {
    version: u32,
    width: u32,
    height: u32,
    world_size_x: f32,
    world_size_z: f32,
    dx: f32,
    dz: f32,
    world_units: &'static str,
    up_axis: &'static str,
    sample_location: &'static str,
    sample_order: &'static str,
    format: &'static str,
    sample_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    byte_order: Option<&'static str>,
    fields: Vec<ExportFieldMetadata>,
}

#[derive(serde::Serialize)]
struct ExportFieldMetadata {
    field: String,
    file: String,
    min: f32,
    max: f32,
    /// Recover original values (subject to quantization) as offset + sample * scale.
    value_offset: f64,
    value_scale: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FieldExportFormat {
    #[default]
    Png,
    Tiff16,
    R16,
    R8,
}

impl FieldExportFormat {
    pub const ALL: [Self; 4] = [Self::Png, Self::Tiff16, Self::R16, Self::R8];

    pub fn label(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Tiff16 => "Unsigned 16bit grayscale TIFF",
            Self::R16 => "R16",
            Self::R8 => "R8",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Tiff16 => "tiff",
            Self::R16 => "r16",
            Self::R8 => "r8",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FieldExportOptions {
    pub format: FieldExportFormat,
    pub fields: Vec<FieldId>,
}

impl Default for FieldExportOptions {
    fn default() -> Self {
        Self {
            format: FieldExportFormat::Png,
            fields: vec![FieldId::Height],
        }
    }
}

impl FieldExportOptions {
    /// Never retain a hidden selection from a previous project or a disabled layer.
    pub fn retain_available(&mut self, available: &[FieldId]) {
        self.fields.retain(|field| available.contains(field));
    }
}

#[derive(Debug)]
pub struct FieldExportResult {
    /// Selected field files followed by the mandatory metadata file.
    pub paths: Vec<PathBuf>,
}

pub type BackgroundFieldExporter = BackgroundExporter<FieldExportResult>;

/// Use the compiled stack's selection rules, including disabled groups and solo.
/// Slope and curvature can always be derived from the final evaluated height.
pub fn exportable_fields(document: &TerrainDocument) -> Result<Vec<FieldId>, String> {
    document
        .metrics
        .at_resolution(document.export_resolution)
        .map_err(|e| e.to_string())?;
    let plan = compile_terrain_plan(
        &document.stack,
        &document.masks,
        TerrainPlanStamp::new(PlanStructureRevision::new(1)),
    )
    .map_err(|errors| {
        errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    let mut fields = vec![FieldId::Height, FieldId::Slope, FieldId::Curvature];
    for operation in plan.operations() {
        if let TerrainOpKind::RunLayerKernel { layer, .. } = &operation.kind {
            let kind = &document
                .stack
                .find(*layer)
                .expect("compiled layer exists")
                .kind;
            for id in terra_cpu_eval::ProcessorRegistry::scalar_export_fields(
                kind,
                fields.contains(&FieldId::StrataReference),
                document.export_resolution,
                &document.level_steps,
            ) {
                if !fields.contains(&id) {
                    fields.push(id);
                }
            }
        }
    }
    fields[3..].sort_by_key(FieldId::display_name);
    Ok(fields)
}

pub fn field_export_filename(field: &FieldId, format: FieldExportFormat) -> String {
    let key = field.cache_key();
    let stem: String = key
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!(
        "{}.{}",
        if stem.is_empty() { "field" } else { &stem },
        format.extension()
    )
}

impl BackgroundFieldExporter {
    pub fn start_fields(
        &mut self,
        mut document: TerrainDocument,
        out_dir: PathBuf,
        options: FieldExportOptions,
    ) {
        self.start_job(move |job| {
            if options.fields.is_empty() {
                return Err(IoError::Msg("Select at least one field".into()).into());
            }
            document.sync_all_biome_paint_masks();
            let available = exportable_fields(&document).map_err(IoError::Msg)?;
            if options
                .fields
                .iter()
                .any(|field| !available.contains(field))
            {
                return Err(IoError::Msg("A selected field is no longer available".into()).into());
            }
            job.set_progress(0.1);
            let (height, mut context) =
                crate::evaluate_document_for_export(&document, job.token().clone())?;
            if options.fields.contains(&FieldId::Slope)
                || options.fields.contains(&FieldId::Curvature)
            {
                context.aux_maps.refresh_derived(&height);
                context.sync_aux_hashmap();
            }
            job.set_progress(0.7);
            write_field_exports(&height, &context, &out_dir, &options, job.token())
        });
    }
}

fn check_cancelled(cancel: &CancelToken) -> Result<(), ExportError> {
    if cancel.is_cancelled() {
        Err(terra_cpu_eval::EvalError::Cancelled.into())
    } else {
        Ok(())
    }
}

/// Height is normalized to its evaluated range. Unit-interval masks preserve
/// their weights; other scalar fields use their evaluated range. All formats
/// share the same conversion, orientation, and sample grid.
fn write_field_exports(
    height: &Heightfield,
    context: &EvalContext,
    out_dir: &Path,
    options: &FieldExportOptions,
    cancel: &CancelToken,
) -> Result<FieldExportResult, ExportError> {
    height.metrics.validate().map_err(IoError::from)?;
    if options.fields.is_empty() {
        return Err(IoError::Msg("Select at least one field".into()).into());
    }
    let mut names = HashSet::new();
    // Validate every requested source before creating any output files. Missing
    // fields must not silently turn into blank maps or unrelated height data.
    for field in &options.fields {
        let name = field_export_filename(field, options.format);
        if !names.insert(name.to_ascii_lowercase()) {
            return Err(IoError::Msg(format!("Duplicate output filename: {name}")).into());
        }
        if *field != FieldId::Height {
            let source = context.aux.get(&field.cache_key()).ok_or_else(|| {
                IoError::Msg(format!(
                    "{} was not produced by this project",
                    field.display_name()
                ))
            })?;
            if source.metrics.width != height.metrics.width
                || source.metrics.height != height.metrics.height
            {
                return Err(IoError::Msg(format!(
                    "{} has an unexpected resolution",
                    field.display_name()
                ))
                .into());
            }
        }
    }
    check_cancelled(cancel)?;
    std::fs::create_dir_all(out_dir).map_err(IoError::from)?;
    let metadata_path = out_dir.join(METADATA_FILENAME);
    // A failed overwrite must not leave an old manifest describing changed files.
    if metadata_path.exists() {
        std::fs::remove_file(&metadata_path).map_err(IoError::from)?;
    }
    let width = height.metrics.width;
    let height_px = height.metrics.height;
    let mut paths = Vec::new();
    let mut fields = Vec::new();
    for field in &options.fields {
        check_cancelled(cancel)?;
        let aux = context.aux.get(&field.cache_key());
        let sample = |x, z| {
            if *field == FieldId::Height {
                height.get(x, z)
            } else {
                aux.expect("validated field").get(x, z)
            }
        };
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for z in 0..height_px {
            check_cancelled(cancel)?;
            for x in 0..width {
                let v = sample(x, z);
                if !v.is_finite() {
                    return Err(IoError::Msg(format!(
                        "{} contains non-finite samples",
                        field.display_name()
                    ))
                    .into());
                }
                min = min.min(v);
                max = max.max(v);
            }
        }
        let (base, span) = if *field != FieldId::Height && min >= 0.0 && max <= 1.0 {
            (0.0, 1.0)
        } else {
            (
                f64::from(min),
                (f64::from(max) - f64::from(min)).max(1.0e-6),
            )
        };
        let limit = if options.format == FieldExportFormat::R8 {
            255.0
        } else {
            65535.0
        };
        let mut pixels = Vec::with_capacity(width as usize * height_px as usize);
        for z in 0..height_px {
            check_cancelled(cancel)?;
            for x in 0..width {
                pixels.push(
                    (((f64::from(sample(x, z)) - base) / span).clamp(0.0, 1.0) * limit).round()
                        as u16,
                );
            }
        }
        let path = out_dir.join(field_export_filename(field, options.format));
        let temp = path.with_extension(format!("{}.partial", options.format.extension()));
        let result = (|| -> Result<(), ExportError> {
            match options.format {
                FieldExportFormat::Png | FieldExportFormat::Tiff16 => {
                    let image = image::ImageBuffer::<image::Luma<u16>, _>::from_raw(
                        width, height_px, pixels,
                    )
                    .expect("complete sample grid");
                    let format = if options.format == FieldExportFormat::Png {
                        image::ImageFormat::Png
                    } else {
                        image::ImageFormat::Tiff
                    };
                    image
                        .save_with_format(&temp, format)
                        .map_err(IoError::from)?;
                }
                FieldExportFormat::R16 | FieldExportFormat::R8 => {
                    let mut writer =
                        BufWriter::new(std::fs::File::create(&temp).map_err(IoError::from)?);
                    for row in pixels.chunks(width as usize) {
                        check_cancelled(cancel)?;
                        for value in row {
                            if options.format == FieldExportFormat::R8 {
                                writer.write_all(&[*value as u8])
                            } else {
                                writer.write_all(&value.to_le_bytes())
                            }
                            .map_err(IoError::from)?;
                        }
                    }
                    writer.flush().map_err(IoError::from)?;
                }
            }
            check_cancelled(cancel)?;
            if path.exists() {
                std::fs::remove_file(&path).map_err(IoError::from)?;
            }
            std::fs::rename(&temp, &path).map_err(IoError::from)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result?;
        fields.push(ExportFieldMetadata {
            field: field.cache_key(),
            file: field_export_filename(field, options.format),
            min,
            max,
            value_offset: base,
            value_scale: span / limit,
        });
        paths.push(path);
    }
    let metadata = ExportMetadata {
        version: 1,
        width,
        height: height_px,
        world_size_x: height.metrics.world_size_x,
        world_size_z: height.metrics.world_size_z,
        dx: height.metrics.dx(),
        dz: height.metrics.dz(),
        world_units: "metres",
        up_axis: "y",
        sample_location: "cell_center",
        sample_order: "rows_increasing_z_columns_increasing_x",
        format: options.format.extension(),
        sample_type: if options.format == FieldExportFormat::R8 {
            "uint8"
        } else {
            "uint16"
        },
        byte_order: match options.format {
            FieldExportFormat::R16 => Some("little_endian"),
            _ => None,
        },
        fields,
    };
    let temp = metadata_path.with_extension("json.partial");
    let result = (|| -> Result<(), ExportError> {
        check_cancelled(cancel)?;
        let bytes = serde_json::to_vec_pretty(&metadata).map_err(IoError::from)?;
        std::fs::write(&temp, bytes).map_err(IoError::from)?;
        check_cancelled(cancel)?;
        std::fs::rename(&temp, &metadata_path).map_err(IoError::from)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result?;
    paths.push(metadata_path);
    Ok(FieldExportResult { paths })
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_core::heightfield::HeightfieldMetrics;
    use terra_core::layer::{FlatParams, Layer, LayerGroup, LayerKind, LayerStack, StackNode};
    use terra_core::mask::MaskField;

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("terra-field-export-{name}-{}", std::process::id()))
    }

    fn fixture() -> (Heightfield, EvalContext) {
        let metrics = HeightfieldMetrics::new(3, 2, 30.0, 20.0);
        let mut height = Heightfield::zeros(metrics);
        for z in 0..2 {
            for x in 0..3 {
                height.set(x, z, (z * 3 + x) as f32 * 10.0 - 10.0);
            }
        }
        let mut context = EvalContext::new(metrics);
        context.aux_insert("wetness", MaskField::filled(metrics, 0.25));
        (height, context)
    }

    fn flat_document() -> TerrainDocument {
        let mut doc = TerrainDocument::new_default();
        doc.export_resolution = 16;
        doc.stack = LayerStack::new();
        doc.stack.push(Layer::new(
            "Height",
            LayerKind::Flat(FlatParams { height: 37.0 }),
        ));
        doc
    }

    #[test]
    fn all_formats_write_only_selected_fields_with_matching_orientation_and_precision() {
        let root = scratch("formats");
        let _ = std::fs::remove_dir_all(&root);
        let (height, context) = fixture();
        for format in FieldExportFormat::ALL {
            let dir = root.join(format.extension());
            let result = write_field_exports(
                &height,
                &context,
                &dir,
                &FieldExportOptions {
                    format,
                    fields: vec![FieldId::Height, FieldId::Wetness],
                },
                &CancelToken::never(),
            )
            .unwrap();
            assert_eq!(result.paths.len(), 3);
            assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 3);
            assert_eq!(result.paths[2], dir.join(METADATA_FILENAME));
            let metadata: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&result.paths[2]).unwrap()).unwrap();
            assert_eq!(metadata["width"], 3);
            assert_eq!(metadata["height"], 2);
            assert_eq!(metadata["world_size_x"], 30.0);
            assert_eq!(metadata["world_size_z"], 20.0);
            assert_eq!(metadata["dx"], 10.0);
            assert_eq!(metadata["dz"], 10.0);
            assert_eq!(metadata["world_units"], "metres");
            assert_eq!(metadata["format"], format.extension());
            assert_eq!(metadata["fields"].as_array().unwrap().len(), 2);
            assert_eq!(metadata["fields"][0]["min"], -10.0);
            assert_eq!(metadata["fields"][0]["max"], 40.0);
            let limit = if format == FieldExportFormat::R8 {
                255.0
            } else {
                65535.0
            };
            assert_eq!(
                metadata["sample_type"],
                if limit == 255.0 { "uint8" } else { "uint16" }
            );
            if format == FieldExportFormat::R16 {
                assert_eq!(metadata["byte_order"], "little_endian");
            } else {
                assert!(metadata.get("byte_order").is_none());
            }
            for (index, (field, min, max)) in [("height", -10.0, 40.0), ("wetness", 0.25, 0.25)]
                .into_iter()
                .enumerate()
            {
                let entry = &metadata["fields"][index];
                assert_eq!(entry["field"], field);
                assert_eq!(entry["file"], format!("{field}.{}", format.extension()));
                assert_eq!(entry["min"], min);
                assert_eq!(entry["max"], max);
                let offset = entry["value_offset"].as_f64().unwrap();
                let scale = entry["value_scale"].as_f64().unwrap();
                let file = dir.join(entry["file"].as_str().unwrap());
                let samples: Vec<f64> = match format {
                    FieldExportFormat::Png | FieldExportFormat::Tiff16 => image::open(file)
                        .unwrap()
                        .to_luma16()
                        .as_raw()
                        .iter()
                        .map(|v| f64::from(*v))
                        .collect(),
                    FieldExportFormat::R16 => std::fs::read(file)
                        .unwrap()
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|bytes| f64::from(u16::from_le_bytes(*bytes)))
                        .collect(),
                    FieldExportFormat::R8 => std::fs::read(file)
                        .unwrap()
                        .into_iter()
                        .map(f64::from)
                        .collect(),
                };
                for (i, encoded) in samples.into_iter().enumerate() {
                    let value = if field == "height" {
                        i as f64 * 10.0 - 10.0
                    } else {
                        0.25
                    };
                    assert!((offset + encoded * scale - value).abs() <= scale / 2.0 + 1.0e-10);
                }
            }
            let path = &result.paths[0];
            assert_eq!(
                path.file_name().unwrap(),
                format!("height.{}", format.extension()).as_str()
            );
            match format {
                FieldExportFormat::Png | FieldExportFormat::Tiff16 => {
                    let decoded = image::open(path).unwrap();
                    assert_eq!(decoded.color(), image::ColorType::L16);
                    let decoded = decoded.to_luma16();
                    assert_eq!(decoded.dimensions(), (3, 2));
                    assert_eq!(decoded.as_raw(), &[0, 13107, 26214, 39321, 52428, 65535]);
                    assert!(image::open(&result.paths[1])
                        .unwrap()
                        .to_luma16()
                        .as_raw()
                        .iter()
                        .all(|v| *v == 16384));
                }
                FieldExportFormat::R16 => {
                    let bytes = std::fs::read(path).unwrap();
                    assert_eq!(
                        bytes,
                        [0_u16, 13107, 26214, 39321, 52428, 65535]
                            .into_iter()
                            .flat_map(u16::to_le_bytes)
                            .collect::<Vec<_>>()
                    );
                }
                FieldExportFormat::R8 => {
                    assert_eq!(std::fs::read(path).unwrap(), [0, 51, 102, 153, 204, 255])
                }
            }
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn height_is_optional_and_empty_or_missing_selections_create_no_files() {
        let root = scratch("selection");
        let _ = std::fs::remove_dir_all(&root);
        let (height, context) = fixture();
        let options = FieldExportOptions {
            format: FieldExportFormat::R8,
            fields: vec![FieldId::Wetness],
        };
        let result =
            write_field_exports(&height, &context, &root, &options, &CancelToken::never()).unwrap();
        assert_eq!(
            result.paths,
            [root.join("wetness.r8"), root.join(METADATA_FILENAME)]
        );
        assert_eq!(std::fs::read(&result.paths[0]).unwrap(), vec![64; 6]);
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&result.paths[1]).unwrap()).unwrap();
        assert_eq!(metadata["fields"].as_array().unwrap().len(), 1);
        assert_eq!(metadata["fields"][0]["field"], "wetness");
        for (name, fields) in [
            ("empty", vec![]),
            ("missing", vec![FieldId::Height, FieldId::Snow]),
        ] {
            let dir = root.join(name);
            assert!(write_field_exports(
                &height,
                &context,
                &dir,
                &FieldExportOptions {
                    fields,
                    ..options.clone()
                },
                &CancelToken::never()
            )
            .is_err());
            assert!(!dir.exists());
        }
        let (cancel, flag) = CancelToken::flag();
        flag.cancel();
        assert!(write_field_exports(
            &height,
            &context,
            &root.join("cancelled"),
            &options,
            &cancel
        )
        .is_err());
        assert!(!root.join("cancelled").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn metadata_is_replaced_for_each_export_and_not_left_stale_on_failure() {
        let root = scratch("metadata-overwrite");
        let _ = std::fs::remove_dir_all(&root);
        let (mut height, context) = fixture();
        for (format, field) in [
            (FieldExportFormat::Png, FieldId::Wetness),
            (FieldExportFormat::R8, FieldId::Height),
        ] {
            for z in 0..height.metrics.height {
                for x in 0..height.metrics.width {
                    height.set(x, z, 37.0);
                }
            }
            write_field_exports(
                &height,
                &context,
                &root,
                &FieldExportOptions {
                    format,
                    fields: vec![field.clone()],
                },
                &CancelToken::never(),
            )
            .unwrap();
            let metadata: serde_json::Value =
                serde_json::from_slice(&std::fs::read(root.join(METADATA_FILENAME)).unwrap())
                    .unwrap();
            assert_eq!(metadata["format"], format.extension());
            assert_eq!(metadata["fields"].as_array().unwrap().len(), 1);
            assert_eq!(
                metadata["fields"][0]["file"],
                field_export_filename(&field, format)
            );
            if field == FieldId::Height {
                assert_eq!(metadata["fields"][0]["min"], 37.0);
                assert_eq!(metadata["fields"][0]["max"], 37.0);
                assert_eq!(metadata["fields"][0]["value_offset"], 37.0);
                assert_eq!(std::fs::read(root.join("height.r8")).unwrap(), vec![0; 6]);
            }
        }
        height.set(0, 0, f32::NAN);
        assert!(write_field_exports(
            &height,
            &context,
            &root,
            &FieldExportOptions::default(),
            &CancelToken::never()
        )
        .is_err());
        assert!(!root.join(METADATA_FILENAME).exists());
        assert!(!root.join("export_metadata.json.partial").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn metadata_write_failure_fails_the_export() {
        let root = scratch("metadata-failure");
        let _ = std::fs::remove_dir_all(&root);
        // A directory at the staging filename forces a deterministic write error.
        std::fs::create_dir_all(root.join("export_metadata.json.partial")).unwrap();
        let (height, context) = fixture();
        assert!(write_field_exports(
            &height,
            &context,
            &root,
            &FieldExportOptions::default(),
            &CancelToken::never()
        )
        .is_err());
        assert!(!root.join(METADATA_FILENAME).exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn available_fields_follow_enabled_groups_and_solo_and_clear_stale_selections() {
        let mut doc = flat_document();
        let baseline = exportable_fields(&doc).unwrap();
        assert_eq!(
            baseline,
            [FieldId::Height, FieldId::Slope, FieldId::Curvature]
        );
        let mut group = LayerGroup::new("Surface");
        group.children.push(StackNode::Layer(Layer::new(
            "Materials",
            LayerKind::Materials(Default::default()),
        )));
        let group_id = group.id;
        doc.stack.push_group(group);
        assert!(exportable_fields(&doc)
            .unwrap()
            .contains(&FieldId::Materials));
        doc.stack.find_group_mut(group_id).unwrap().enabled = false;
        assert_eq!(exportable_fields(&doc).unwrap(), baseline);
        doc.stack.find_group_mut(group_id).unwrap().enabled = true;
        let height_id = doc.stack.flatten_layers()[0].id();
        doc.stack.find_mut(height_id).unwrap().common.solo = true;
        assert_eq!(exportable_fields(&doc).unwrap(), baseline);
        let mut options = FieldExportOptions {
            fields: vec![FieldId::Materials],
            ..Default::default()
        };
        options.retain_available(&baseline);
        assert!(options.fields.is_empty());
    }

    #[test]
    fn background_export_evaluates_current_document_and_exports_without_height() {
        let root = scratch("worker");
        let _ = std::fs::remove_dir_all(&root);
        let mut exporter = BackgroundFieldExporter::new();
        exporter.start_fields(
            flat_document(),
            root.clone(),
            FieldExportOptions {
                format: FieldExportFormat::Tiff16,
                fields: vec![FieldId::Slope],
            },
        );
        let started = std::time::Instant::now();
        while !exporter.job.done {
            exporter.poll();
            assert!(started.elapsed() < std::time::Duration::from_secs(15));
            std::thread::yield_now();
        }
        let result = exporter.job.result.take().unwrap().unwrap();
        assert_eq!(
            result.paths,
            [root.join("slope.tiff"), root.join(METADATA_FILENAME)]
        );
        let image = image::open(&result.paths[0]).unwrap().to_luma16();
        assert_eq!(image.dimensions(), (16, 16));
        assert!(image.as_raw().iter().all(|v| *v == 0));
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 2);
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&result.paths[1]).unwrap()).unwrap();
        assert_eq!(metadata["width"], 16);
        assert_eq!(metadata["height"], 16);
        assert_eq!(metadata["fields"][0]["file"], "slope.tiff");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn available_fields_follow_hydraulic_output_options() {
        let mut doc = flat_document();
        let mut params = terra_core::layer::HydraulicErosionParams {
            output_water: false,
            output_sediment: false,
            output_flow: false,
            output_channels: false,
            ..Default::default()
        };
        let layer = Layer::new("Hydraulic", LayerKind::HydraulicErosion(params.clone()));
        let id = layer.id();
        doc.stack.push(layer);
        let optional = [
            FieldId::WaterDepth,
            FieldId::WaterVelocity,
            FieldId::Sediment,
            FieldId::FlowAccumulation,
            FieldId::ChannelMask,
        ];
        let fields = exportable_fields(&doc).unwrap();
        assert!(optional.iter().all(|field| !fields.contains(field)));

        params.output_water = true;
        params.output_sediment = true;
        params.output_flow = true;
        params.output_channels = true;
        doc.stack.find_mut(id).unwrap().kind = LayerKind::HydraulicErosion(params);
        let fields = exportable_fields(&doc).unwrap();
        assert!(optional.iter().all(|field| fields.contains(field)));
    }

    #[test]
    fn advertised_surface_and_simulation_fields_are_produced_by_evaluation() {
        let mut failures = Vec::new();
        for mut kind in [
            LayerKind::SculptStrokes(Default::default()),
            LayerKind::TerrainConstraints(Default::default()),
            LayerKind::GradientReconstruct(Default::default()),
            LayerKind::LandscapeEvolution(Default::default()),
            LayerKind::HydrologyRepair(Default::default()),
            LayerKind::GeomorphicDetail(Default::default()),
            LayerKind::EcosystemFeedback(Default::default()),
            LayerKind::Island(Default::default()),
            LayerKind::Dunes(Default::default()),
            LayerKind::Materials(Default::default()),
            LayerKind::ThermalErosion(Default::default()),
            LayerKind::HydraulicErosion(Default::default()),
            LayerKind::DebrisFlow(Default::default()),
            LayerKind::RiverCarve(Default::default()),
            LayerKind::StreamPowerErosion(Default::default()),
            LayerKind::MultiScaleAmplify(Default::default()),
            LayerKind::Path(Default::default()),
            LayerKind::RiverNetwork(Default::default()),
            LayerKind::SandSimulation(Default::default()),
            LayerKind::FluidSimulation(Default::default()),
            LayerKind::Biomes(Default::default()),
            LayerKind::Vegetation(Default::default()),
            LayerKind::OverhangStamp(Default::default()),
            LayerKind::LocalSdf(Default::default()),
        ] {
            // Verify output availability, not the cost of full default simulations.
            match &mut kind {
                LayerKind::GradientReconstruct(p) => p.iterations = 1,
                LayerKind::LandscapeEvolution(p) => {
                    p.iterations = 1;
                    p.fixed_point_iters = 1;
                }
                LayerKind::HydrologyRepair(p) => p.iterations = 1,
                LayerKind::ThermalErosion(p) => p.iterations = 1,
                LayerKind::HydraulicErosion(p) => p.iterations = 1,
                LayerKind::DebrisFlow(p) => p.iterations = 1,
                LayerKind::StreamPowerErosion(p) => p.iterations = 1,
                LayerKind::MultiScaleAmplify(p) => {
                    p.thermal_iters = 1;
                    p.spe_iters = 1;
                    p.level_count = 1;
                }
                LayerKind::SandSimulation(p) => {
                    p.iterations = 1;
                    p.avalanche_iters = 1;
                }
                _ => {}
            }
            let name = kind.type_id();
            let mut doc = flat_document();
            doc.export_resolution = 64;
            doc.metrics.world_size_x = 128.0;
            doc.metrics.world_size_z = 128.0;
            doc.stack.push(Layer::new(name, kind));
            let fields = exportable_fields(&doc).unwrap();
            let (height, mut context) =
                crate::evaluate_document_for_export(&doc, CancelToken::never()).unwrap();
            context.ensure_derived_fields(&height);
            let missing: Vec<_> = fields
                .iter()
                .filter(|field| {
                    **field != FieldId::Height && !context.aux.contains_key(&field.cache_key())
                })
                .collect();
            if !missing.is_empty() {
                failures.push(format!("{name}: {missing:?}"));
            }
        }
        assert!(
            failures.is_empty(),
            "advertised fields missing after evaluation: {failures:?}"
        );
    }
}
