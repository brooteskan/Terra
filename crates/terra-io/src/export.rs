use crate::IoError;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use terra_core::fields::keys;
use terra_core::heightfield::Heightfield;
use terra_core::mask::MaskField;
use terra_cpu_eval::EvalContext;

pub struct ExportRequest {
    pub out_dir: PathBuf,
    pub include_masks: bool,
    /// Pack material IDs into splatmap RGBA (up to 4 IDs).
    pub include_splat: bool,
    /// Bake albedo color map from material IDs / height tint.
    pub include_color: bool,
    /// Height-derived tangent-space normal map.
    pub include_normal: bool,
    /// When true, color map uses only procedural tints (no texture lookups).
    pub tint_only: bool,
    /// Write `splat_ids.json` listing stable material ID → RGBA channel mapping.
    pub include_splat_ids: bool,
    /// Write `vegetation_instances.json` from Poisson sampling of Vegetation layers.
    pub include_vegetation_instances: bool,
    /// Write a coarse collision/preview mesh (`terrain_collision.obj`).
    pub include_collision_mesh: bool,
    /// Grid stride for collision mesh (higher = coarser).
    pub collision_mesh_stride: u32,
}

impl Default for ExportRequest {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::new(),
            include_masks: true,
            include_splat: true,
            include_color: true,
            include_normal: true,
            tint_only: true,
            include_splat_ids: true,
            include_vegetation_instances: true,
            include_collision_mesh: true,
            collision_mesh_stride: 8,
        }
    }
}

#[derive(Debug)]
pub struct ExportResult {
    pub height_path: PathBuf,
    pub height_metadata_path: PathBuf,
    pub raw_path: PathBuf,
    pub raw_metadata_path: PathBuf,
    pub mask_paths: Vec<PathBuf>,
    pub splat_path: Option<PathBuf>,
    pub color_path: Option<PathBuf>,
    pub normal_path: Option<PathBuf>,
    pub splat_ids_path: Option<PathBuf>,
    pub vegetation_instances_path: Option<PathBuf>,
    pub collision_mesh_path: Option<PathBuf>,
    pub hash: u64,
    pub manifest_path: Option<PathBuf>,
    pub tiles_changed: usize,
    pub tile_paths: Vec<PathBuf>,
}

#[derive(serde::Serialize)]
struct HeightMetadata {
    min: f32,
    max: f32,
    width: u32,
    height: u32,
    world_size_x: f32,
    world_size_z: f32,
    dx: f32,
    dz: f32,
}

pub fn export_package(
    hf: &Heightfield,
    ctx: &EvalContext,
    req: &ExportRequest,
) -> Result<ExportResult, IoError> {
    std::fs::create_dir_all(&req.out_dir)?;
    let height_path = req.out_dir.join("height.png");
    write_height_png(hf, &height_path)?;
    let height_metadata_path = req.out_dir.join("height_meta.json");
    write_height_metadata(hf, &height_metadata_path)?;
    let mut mask_paths = Vec::new();
    if req.include_masks {
        for (name, mask) in &ctx.aux {
            let path = req.out_dir.join(format!("mask_{name}.png"));
            write_mask_png(mask, &path)?;
            mask_paths.push(path);
        }
    }

    let materials = ctx.aux.get(keys::MATERIALS);
    let (splat_path, splat_ids) = if req.include_splat {
        let path = req.out_dir.join("splat.png");
        let ids = write_splat_png(hf, materials, &path)?;
        (Some(path), ids)
    } else {
        (None, collect_material_ids(materials))
    };

    let splat_ids_path = if req.include_splat_ids {
        let path = req.out_dir.join("splat_ids.json");
        write_splat_ids_json(&splat_ids, &path)?;
        Some(path)
    } else {
        None
    };

    let color_path = if req.include_color {
        let path = req.out_dir.join("color.png");
        write_color_png(hf, materials, req.tint_only, &path)?;
        Some(path)
    } else {
        None
    };

    let normal_path = if req.include_normal {
        let path = req.out_dir.join("normal.png");
        write_normal_png(hf, &path)?;
        Some(path)
    } else {
        None
    };

    let vegetation_instances_path = if req.include_vegetation_instances {
        let path = req.out_dir.join("vegetation_instances.json");
        write_vegetation_instances(hf, ctx, &path)?;
        Some(path)
    } else {
        None
    };

    let collision_mesh_path = if req.include_collision_mesh {
        let path = req.out_dir.join("terrain_collision.obj");
        write_collision_obj(hf, req.collision_mesh_stride.max(1), &path)?;
        Some(path)
    } else {
        None
    };

    let hash = hash_heights(hf);
    let raw_path = req.out_dir.join("height.r32");
    write_height_raw_streamed(hf, &raw_path)?;
    let raw_metadata_path = req.out_dir.join("height.r32.meta.json");
    write_height_metadata(hf, &raw_metadata_path)?;

    // Tile manifest for incremental / engine export.
    let tile_size = hf.metrics.tile_size.max(64);
    let manifest = build_tile_manifest(hf, tile_size, hash);
    let manifest_path = req.out_dir.join("tile_manifest.json");
    let previous = std::fs::read(&manifest_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<TileManifest>(&bytes).ok());
    let changed = previous.as_ref().map_or_else(
        || {
            manifest
                .tiles
                .iter()
                .map(|tile| (tile.tx, tile.ty))
                .collect()
        },
        |old| changed_tiles(old, &manifest),
    );
    let tile_paths = write_changed_height_tiles(hf, &req.out_dir, &manifest, &changed)?;
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

    Ok(ExportResult {
        height_path,
        height_metadata_path,
        raw_path,
        raw_metadata_path,
        mask_paths,
        splat_path,
        color_path,
        normal_path,
        splat_ids_path,
        vegetation_instances_path,
        collision_mesh_path,
        hash,
        manifest_path: Some(manifest_path),
        tiles_changed: changed.len(),
        tile_paths,
    })
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct TileManifestEntry {
    pub tx: u32,
    pub ty: u32,
    pub content_hash: u64,
    pub origin_x: f32,
    pub origin_z: f32,
    pub size_m: f32,
    #[serde(default)]
    pub width_samples: u32,
    #[serde(default)]
    pub height_samples: u32,
    #[serde(default)]
    pub file: String,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct TileManifest {
    pub world_size_x: f32,
    pub world_size_z: f32,
    pub tile_size: u32,
    pub height_hash: u64,
    pub tiles: Vec<TileManifestEntry>,
}

/// Build a deterministic tile manifest for incremental export / change detection.
pub fn build_tile_manifest(hf: &Heightfield, tile_size: u32, height_hash: u64) -> TileManifest {
    let tw = tile_size.max(1);
    let tiles_x = hf.metrics.width.div_ceil(tw);
    let tiles_z = hf.metrics.height.div_ceil(tw);
    let dx = hf.metrics.dx();
    let dz = hf.metrics.dz();
    let mut tiles = Vec::new();
    for ty in 0..tiles_z {
        for tx in 0..tiles_x {
            let mut h: u64 = 0xcbf29ce484222325;
            let i0 = tx * tw;
            let j0 = ty * tw;
            let i1 = (i0 + tw).min(hf.metrics.width);
            let j1 = (j0 + tw).min(hf.metrics.height);
            for j in j0..j1 {
                for i in i0..i1 {
                    let bits = hf.get(i, j).to_bits() as u64;
                    h ^= bits;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
            tiles.push(TileManifestEntry {
                tx,
                ty,
                content_hash: h,
                origin_x: i0 as f32 * dx,
                origin_z: j0 as f32 * dz,
                size_m: tw as f32 * dx.max(dz),
                width_samples: i1 - i0,
                height_samples: j1 - j0,
                file: format!("height_tiles/height_{tx}_{ty}.r32"),
            });
        }
    }
    TileManifest {
        world_size_x: hf.metrics.world_size_x,
        world_size_z: hf.metrics.world_size_z,
        tile_size: tw,
        height_hash,
        tiles,
    }
}

/// Return tile indices whose content hash differs from `previous`.
pub fn changed_tiles(previous: &TileManifest, current: &TileManifest) -> Vec<(u32, u32)> {
    let mut prev = std::collections::HashMap::new();
    for t in &previous.tiles {
        prev.insert((t.tx, t.ty), t.content_hash);
    }
    current
        .tiles
        .iter()
        .filter(|t| prev.get(&(t.tx, t.ty)).copied() != Some(t.content_hash))
        .map(|t| (t.tx, t.ty))
        .collect()
}

/// Write the canonical row-major R32 heightmap without allocating a second full field.
fn write_height_raw_streamed(hf: &Heightfield, path: &std::path::Path) -> Result<(), IoError> {
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);
    for j in 0..hf.metrics.height {
        for i in 0..hf.metrics.width {
            writer.write_all(&hf.get(i, j).to_le_bytes())?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Materialize only changed virtual tiles. Ordering and bytes are deterministic, while the
/// manifest remains the authority when an older export directory contains stale extra files.
fn write_changed_height_tiles(
    hf: &Heightfield,
    out_dir: &std::path::Path,
    manifest: &TileManifest,
    changed: &[(u32, u32)],
) -> Result<Vec<PathBuf>, IoError> {
    let tile_dir = out_dir.join("height_tiles");
    std::fs::create_dir_all(&tile_dir)?;
    let changed: std::collections::HashSet<_> = changed.iter().copied().collect();
    let mut paths = Vec::with_capacity(manifest.tiles.len());
    for tile in &manifest.tiles {
        let path = out_dir.join(&tile.file);
        let needs_write = changed.contains(&(tile.tx, tile.ty)) || !path.exists();
        if needs_write {
            let file = std::fs::File::create(&path)?;
            let mut writer = BufWriter::with_capacity(
                (tile.width_samples as usize)
                    .saturating_mul(tile.height_samples as usize)
                    .saturating_mul(4)
                    .min(1024 * 1024),
                file,
            );
            let i0 = tile.tx * manifest.tile_size;
            let j0 = tile.ty * manifest.tile_size;
            for j in j0..j0 + tile.height_samples {
                for i in i0..i0 + tile.width_samples {
                    writer.write_all(&hf.get(i, j).to_le_bytes())?;
                }
            }
            writer.flush()?;
        }
        paths.push(path);
    }
    Ok(paths)
}

fn write_height_metadata(hf: &Heightfield, path: &std::path::Path) -> Result<(), IoError> {
    let (min, max) = hf.min_max();
    let metrics = hf.metrics;
    let metadata = HeightMetadata {
        min,
        max,
        width: metrics.width,
        height: metrics.height,
        world_size_x: metrics.world_size_x,
        world_size_z: metrics.world_size_z,
        dx: metrics.dx(),
        dz: metrics.dz(),
    };
    std::fs::write(path, serde_json::to_vec_pretty(&metadata)?)?;
    Ok(())
}

pub fn write_height_png(hf: &Heightfield, path: &std::path::Path) -> Result<(), IoError> {
    let (min_h, max_h) = hf.min_max();
    let span = (max_h - min_h).max(1e-6);
    let w = hf.metrics.width;
    let h = hf.metrics.height;
    let mut img16 = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(w, h);
    for j in 0..h {
        for i in 0..w {
            let t = ((hf.get(i, j) - min_h) / span).clamp(0.0, 1.0);
            img16.put_pixel(i, j, image::Luma([(t * 65535.0) as u16]));
        }
    }
    img16.save(path)?;
    Ok(())
}

pub fn write_mask_png(mask: &MaskField, path: &std::path::Path) -> Result<(), IoError> {
    let w = mask.metrics.width;
    let h = mask.metrics.height;
    let mut img = image::GrayImage::new(w, h);
    for j in 0..h {
        for i in 0..w {
            img.put_pixel(i, j, image::Luma([(mask.get(i, j) * 255.0) as u8]));
        }
    }
    img.save(path)?;
    Ok(())
}

/// Pack up to four distinct material IDs into RGBA channels (weight 1 where ID matches).
/// Returns the ordered ID list written into R,G,B,A (0 = unused).
pub fn write_splat_png(
    hf: &Heightfield,
    materials: Option<&MaskField>,
    path: &std::path::Path,
) -> Result<Vec<u32>, IoError> {
    let w = hf.metrics.width;
    let h = hf.metrics.height;
    let mut ids = collect_material_ids(materials);
    while ids.len() < 4 {
        ids.push(0);
    }
    ids.truncate(4);
    let mut img = image::RgbaImage::new(w, h);
    for j in 0..h {
        for i in 0..w {
            let id = materials
                .map(|m| (m.get(i, j) * 16.0).round() as u32)
                .unwrap_or(0);
            let mut px = [0u8; 4];
            for (c, &slot) in ids.iter().enumerate() {
                if slot != 0 && slot == id {
                    px[c] = 255;
                }
            }
            img.put_pixel(i, j, image::Rgba(px));
        }
    }
    img.save(path)?;
    Ok(ids)
}

fn collect_material_ids(materials: Option<&MaskField>) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    let Some(mats) = materials else {
        return ids;
    };
    let w = mats.metrics.width;
    let h = mats.metrics.height;
    for j in 0..h {
        for i in 0..w {
            let id = (mats.get(i, j) * 16.0).round() as u32;
            if id > 0 && !ids.contains(&id) {
                ids.push(id);
                if ids.len() >= 4 {
                    return ids;
                }
            }
        }
    }
    ids
}

fn write_splat_ids_json(ids: &[u32], path: &std::path::Path) -> Result<(), IoError> {
    #[derive(serde::Serialize)]
    struct SplatIdEntry {
        channel: &'static str,
        material_id: u32,
    }
    #[derive(serde::Serialize)]
    struct SplatIdDoc {
        description: &'static str,
        channels: Vec<SplatIdEntry>,
    }
    let channel_names = ["R", "G", "B", "A"];
    let channels = ids
        .iter()
        .enumerate()
        .take(4)
        .map(|(i, &id)| SplatIdEntry {
            channel: channel_names[i],
            material_id: id,
        })
        .collect();
    let doc = SplatIdDoc {
        description: "Stable material IDs packed into splat.png RGBA channels (255 = weight).",
        channels,
    };
    std::fs::write(path, serde_json::to_vec_pretty(&doc)?)?;
    Ok(())
}

fn write_vegetation_instances(
    hf: &Heightfield,
    ctx: &EvalContext,
    path: &std::path::Path,
) -> Result<(), IoError> {
    #[derive(serde::Serialize)]
    struct VegInstance {
        x: f32,
        y: f32,
        z: f32,
        scale: f32,
    }
    #[derive(serde::Serialize)]
    struct VegDoc {
        count: usize,
        instances: Vec<VegInstance>,
    }
    let density = ctx.aux.get(keys::VEGETATION);
    let mut params = terra_core::layer::VegetationParams::default();
    // Prefer denser export sampling when a density mask exists.
    if density.is_some() {
        params.density = 0.45;
        params.min_distance = 4.0;
    }
    let points = terra_core::scatter::poisson_disk(hf, &params);
    let mut instances = Vec::with_capacity(points.len());
    for (x, z) in points {
        let (i, j) = hf.metrics.sample_index(x, z);
        let y = hf.get(i, j);
        let dens = density.map(|d| d.get(i, j)).unwrap_or(1.0);
        if dens < 0.08 {
            continue;
        }
        instances.push(VegInstance {
            x,
            y,
            z,
            scale: 0.7 + dens * 0.8,
        });
    }
    let doc = VegDoc {
        count: instances.len(),
        instances,
    };
    std::fs::write(path, serde_json::to_vec_pretty(&doc)?)?;
    Ok(())
}

/// Coarse grid mesh for collision / DCC preview (Y-up, metres).
pub fn write_collision_obj(
    hf: &Heightfield,
    stride: u32,
    path: &std::path::Path,
) -> Result<(), IoError> {
    let stride = stride.max(1);
    let w = hf.metrics.width;
    let h = hf.metrics.height;
    let dx = hf.metrics.dx();
    let dz = hf.metrics.dz();
    let mut verts = String::new();
    let mut faces = String::new();
    let cols = w.div_ceil(stride);
    let rows = h.div_ceil(stride);
    for jj in 0..rows {
        for ii in 0..cols {
            let i = (ii * stride).min(w - 1);
            let j = (jj * stride).min(h - 1);
            let x = i as f32 * dx;
            let z = j as f32 * dz;
            let y = hf.get(i, j);
            verts.push_str(&format!("v {x:.4} {y:.4} {z:.4}\n"));
        }
    }
    for jj in 0..rows.saturating_sub(1) {
        for ii in 0..cols.saturating_sub(1) {
            let a = jj * cols + ii + 1;
            let b = a + 1;
            let c = a + cols;
            let d = c + 1;
            faces.push_str(&format!("f {a} {c} {d} {b}\n"));
        }
    }
    let mut out = String::from("# Terra collision mesh\n");
    out.push_str(&verts);
    out.push_str(&faces);
    std::fs::write(path, out)?;
    Ok(())
}

fn material_tint(id: u32) -> [f32; 3] {
    // Matches terra-render terrain.wgsl material_color palette intent.
    match id % 5 {
        1 => [0.45, 0.42, 0.38],
        2 => [0.28, 0.48, 0.22],
        3 => [0.62, 0.52, 0.32],
        4 => [0.55, 0.55, 0.58],
        _ => [0.35, 0.40, 0.28],
    }
}

pub fn write_color_png(
    hf: &Heightfield,
    materials: Option<&MaskField>,
    tint_only: bool,
    path: &std::path::Path,
) -> Result<(), IoError> {
    write_color_png_ex(hf, materials, None, tint_only, path)
}

/// Bake color map; when `material_params` has albedo_path entries and `tint_only` is false,
/// multiply procedural tint by a tiled sample of each PNG.
pub fn write_color_png_ex(
    hf: &Heightfield,
    materials: Option<&MaskField>,
    material_params: Option<&terra_core::layer::MaterialsParams>,
    tint_only: bool,
    path: &std::path::Path,
) -> Result<(), IoError> {
    let (min_h, max_h) = hf.min_max();
    let span = (max_h - min_h).max(1e-6);
    let w = hf.metrics.width;
    let h = hf.metrics.height;
    let mut albedo_cache: std::collections::HashMap<u32, Option<image::RgbaImage>> =
        std::collections::HashMap::new();
    if !tint_only {
        if let Some(params) = material_params {
            for rule in &params.rules {
                if let Some(p) = rule.albedo_path.as_ref().filter(|s| !s.is_empty()) {
                    let img = image::open(p).ok().map(|i| i.to_rgba8());
                    albedo_cache.insert(rule.id, img);
                }
            }
        }
    }
    let mut img = image::RgbImage::new(w, h);
    let wx = hf.metrics.world_size_x.max(1e-3);
    let wz = hf.metrics.world_size_z.max(1e-3);
    for j in 0..h {
        for i in 0..w {
            let hn = ((hf.get(i, j) - min_h) / span).clamp(0.0, 1.0);
            let height_col = [0.15 + hn * 0.35, 0.22 + hn * 0.40, 0.12 + hn * 0.25];
            let id = materials
                .map(|m| (m.get(i, j) * 16.0).round() as u32)
                .unwrap_or(0);
            let mut tint = if id > 0 {
                material_tint(id)
            } else {
                height_col
            };
            if let Some(params) = material_params {
                if let Some(rule) = params.rules.iter().find(|r| r.id == id) {
                    tint = rule.tint;
                }
            }
            if !tint_only {
                if let Some(Some(tex)) = albedo_cache.get(&id) {
                    let u = ((i as f32 + 0.5) / w as f32 * wx * 0.085).fract();
                    let v = ((j as f32 + 0.5) / h as f32 * wz * 0.085).fract();
                    let tx = ((u * tex.width() as f32) as u32).min(tex.width() - 1);
                    let ty = ((v * tex.height() as f32) as u32).min(tex.height() - 1);
                    let px = tex.get_pixel(tx, ty).0;
                    tint[0] *= px[0] as f32 / 255.0;
                    tint[1] *= px[1] as f32 / 255.0;
                    tint[2] *= px[2] as f32 / 255.0;
                }
            }
            let mix = if id > 0 { 0.75 } else { 1.0 };
            let r = ((height_col[0] * (1.0 - mix) + tint[0] * mix) * 255.0) as u8;
            let g = ((height_col[1] * (1.0 - mix) + tint[1] * mix) * 255.0) as u8;
            let b = ((height_col[2] * (1.0 - mix) + tint[2] * mix) * 255.0) as u8;
            img.put_pixel(i, j, image::Rgb([r, g, b]));
        }
    }
    img.save(path)?;
    Ok(())
}

pub fn write_normal_png(hf: &Heightfield, path: &std::path::Path) -> Result<(), IoError> {
    let w = hf.metrics.width;
    let h = hf.metrics.height;
    let dx = hf.metrics.dx().max(1e-5);
    let dz = hf.metrics.dz().max(1e-5);
    let mut img = image::RgbImage::new(w, h);
    for j in 0..h {
        for i in 0..w {
            let z = hf.get(i, j);
            let zx = if i + 1 < w { hf.get(i + 1, j) } else { z };
            let zy = if j + 1 < h { hf.get(i, j + 1) } else { z };
            let gx = (zx - z) / dx;
            let gy = (zy - z) / dz;
            // Object-space normal → tangent-ish encoding for DCC.
            let nx = -gx;
            let ny = 1.0;
            let nz = -gy;
            let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-6);
            let r = (((nx / len) * 0.5 + 0.5) * 255.0) as u8;
            let g = (((nz / len) * 0.5 + 0.5) * 255.0) as u8;
            let b = (((ny / len) * 0.5 + 0.5) * 255.0) as u8;
            img.put_pixel(i, j, image::Rgb([r, g, b]));
        }
    }
    img.save(path)?;
    Ok(())
}

pub fn hash_heights(hf: &Heightfield) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for j in 0..hf.metrics.height {
        for i in 0..hf.metrics.width {
            h ^= hf.get(i, j).to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_core::heightfield::HeightfieldMetrics;

    #[test]
    fn hash_deterministic() {
        let m = HeightfieldMetrics::new(8, 8, 8.0, 8.0);
        let hf = Heightfield::filled(m, 3.5);
        assert_eq!(hash_heights(&hf), hash_heights(&hf));
    }

    #[test]
    fn export_maps_write() {
        let dir = std::env::temp_dir().join(format!("terra_export_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let m = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..16u32 {
            for i in 0..16u32 {
                hf.set(i, j, i as f32 + j as f32);
            }
        }
        let mut mats = MaskField::zeros(m);
        for j in 0..16u32 {
            for i in 0..16u32 {
                let id = if i > 8 { 2.0 / 16.0 } else { 1.0 / 16.0 };
                mats.set(i, j, id);
            }
        }
        let mut ctx = EvalContext::new(m);
        ctx.aux.insert(keys::MATERIALS.to_string(), mats);
        let req = ExportRequest {
            out_dir: dir.clone(),
            ..Default::default()
        };
        let result = export_package(&hf, &ctx, &req).expect("export");
        assert!(result.height_path.exists());
        assert!(result.splat_path.unwrap().exists());
        assert!(result.color_path.unwrap().exists());
        assert!(result.normal_path.unwrap().exists());
        assert!(result.splat_ids_path.unwrap().exists());
        assert!(result.vegetation_instances_path.unwrap().exists());
        assert!(result.collision_mesh_path.unwrap().exists());
        assert!(result.manifest_path.unwrap().exists());
        assert!(result.tiles_changed > 0);
        assert!(!result.tile_paths.is_empty());
        assert!(result.tile_paths.iter().all(|path| path.exists()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn raw_export_is_row_major_without_dense_staging() {
        let dir = std::env::temp_dir().join(format!("terra_raw_stream_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let m = HeightfieldMetrics::new(3, 2, 3.0, 2.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..2 {
            for i in 0..3 {
                hf.set(i, j, (j * 3 + i) as f32 + 0.25);
            }
        }
        let path = dir.join("height.r32");
        write_height_raw_streamed(&hf, &path).unwrap();
        let bytes = std::fs::read(path).unwrap();
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![0.25, 1.25, 2.25, 3.25, 4.25, 5.25]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn repeated_export_only_rewrites_changed_height_tiles() {
        let dir = std::env::temp_dir().join(format!("terra_tile_stream_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut m = HeightfieldMetrics::new(128, 64, 128.0, 64.0);
        m.tile_size = 64;
        let mut hf = Heightfield::zeros(m);
        let ctx = EvalContext::new(m);
        let req = ExportRequest {
            out_dir: dir.clone(),
            include_masks: false,
            include_splat: false,
            include_color: false,
            include_normal: false,
            include_splat_ids: false,
            include_vegetation_instances: false,
            include_collision_mesh: false,
            ..Default::default()
        };
        assert_eq!(export_package(&hf, &ctx, &req).unwrap().tiles_changed, 2);
        assert_eq!(export_package(&hf, &ctx, &req).unwrap().tiles_changed, 0);
        hf.set(96, 12, 9.0);
        assert_eq!(export_package(&hf, &ctx, &req).unwrap().tiles_changed, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn changed_tiles_detects_edits() {
        let m = HeightfieldMetrics::new(64, 64, 640.0, 640.0);
        let mut hf = Heightfield::zeros(m);
        let a = build_tile_manifest(&hf, 32, 1);
        hf.set(2, 2, 10.0);
        let b = build_tile_manifest(&hf, 32, 2);
        let changed = changed_tiles(&a, &b);
        assert!(!changed.is_empty());
    }
}
