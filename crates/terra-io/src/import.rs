use crate::IoError;
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};

pub fn import_heightmap_png(
    path: &std::path::Path,
    metrics: HeightfieldMetrics,
    height_scale: f32,
    height_offset: f32,
) -> Result<Heightfield, IoError> {
    metrics.validate()?;
    let img = image::open(path)?.to_luma16();
    let (iw, ih) = img.dimensions();
    let mut hf = Heightfield::zeros(metrics);
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            let u = i as f32 / metrics.width as f32;
            let v = j as f32 / metrics.height as f32;
            let x = ((u * iw as f32) as u32).min(iw - 1);
            let y = ((v * ih as f32) as u32).min(ih - 1);
            let pix = img.get_pixel(x, y).0[0] as f32 / 65535.0;
            hf.set(i, j, pix * height_scale + height_offset);
        }
    }
    hf.refresh_halos();
    Ok(hf)
}

pub fn import_heightmap_raw(
    path: &std::path::Path,
    metrics: HeightfieldMetrics,
) -> Result<Heightfield, IoError> {
    metrics.validate()?;
    let bytes = std::fs::read(path)?;
    let expected = (metrics.width * metrics.height) as usize * 4;
    if bytes.len() < expected {
        return Err(IoError::Msg(format!(
            "RAW too small: {} < {}",
            bytes.len(),
            expected
        )));
    }
    let mut data = Vec::with_capacity(expected / 4);
    for chunk in bytes.chunks_exact(4).take(expected / 4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Heightfield::from_dense(metrics, &data))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invalid_metrics() -> HeightfieldMetrics {
        HeightfieldMetrics {
            width: 0,
            height: 16,
            world_size_x: 16.0,
            world_size_z: 16.0,
            tile_size: 16,
            halo: 2,
        }
    }

    #[test]
    fn png_import_rejects_invalid_metrics_before_touching_the_file() {
        // Validation runs before any file access, so an otherwise-missing path
        // still surfaces the metrics error rather than an IO or decode error.
        let err = import_heightmap_png(
            std::path::Path::new("terra-nonexistent-fixture.png"),
            invalid_metrics(),
            1.0,
            0.0,
        )
        .expect_err("invalid metrics must be rejected");
        assert!(matches!(err, IoError::Metrics(_)), "got: {err}");
    }

    #[test]
    fn raw_import_rejects_invalid_metrics_before_touching_the_file() {
        let err = import_heightmap_raw(
            std::path::Path::new("terra-nonexistent-fixture.raw"),
            invalid_metrics(),
        )
        .expect_err("invalid metrics must be rejected");
        assert!(matches!(err, IoError::Metrics(_)), "got: {err}");
    }
}
