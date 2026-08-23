use super::{TerrainPyramid, TerrainTileKey};

/// Measure one exported child tile against its immediate parent using the same
/// cell-center mapping and clamped bilinear reconstruction as the GPU pyramid
/// error pass. `sample` addresses complete level-local sample coordinates.
pub fn measure_tile_geometric_error(
    pyramid: &TerrainPyramid,
    key: &TerrainTileKey,
    mut sample: impl FnMut(u8, u32, u32) -> f32,
) -> Option<f32> {
    let (level, tile) = pyramid.level_and_tile(key.address)?;
    if level == 0 {
        return Some(0.0);
    }
    let child = pyramid.level_metrics(level)?;
    let parent = pyramid.level_metrics(level - 1)?;
    let extent = pyramid.tile_extent(level, tile)?;
    let mut maximum = 0.0f32;
    for y in extent.origin_z..extent.origin_z + extent.height {
        for x in extent.origin_x..extent.origin_x + extent.width {
            let uv_x = (x as f32 + 0.5) / child.width as f32;
            let uv_y = (y as f32 + 0.5) / child.height as f32;
            let px = uv_x * parent.width as f32 - 0.5;
            let py = uv_y * parent.height as f32 - 0.5;
            let x0 = px.floor() as i32;
            let y0 = py.floor() as i32;
            let tx = px.fract();
            let ty = py.fract();
            let clamp_x = |value: i32| value.clamp(0, parent.width as i32 - 1) as u32;
            let clamp_y = |value: i32| value.clamp(0, parent.height as i32 - 1) as u32;
            let ax = clamp_x(x0);
            let ay = clamp_y(y0);
            let bx = clamp_x(x0 + 1);
            let by = clamp_y(y0 + 1);
            let h00 = sample(level - 1, ax, ay);
            let h10 = sample(level - 1, bx, ay);
            let h01 = sample(level - 1, ax, by);
            let h11 = sample(level - 1, bx, by);
            let top = h00 + (h10 - h00) * tx;
            let bottom = h01 + (h11 - h01) * tx;
            let reconstructed = top + (bottom - top) * ty;
            let mut error = (sample(level, x, y) - reconstructed).abs();
            if !error.is_finite() || error < 0.0 {
                error = f32::MAX;
            }
            maximum = maximum.max(error);
        }
    }
    Some(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PyramidConfig, TileId};

    #[test]
    fn constant_levels_have_zero_error() {
        let mut config = PyramidConfig::new(17, 170.0, 90.0);
        config.tile_size = 8;
        let pyramid = TerrainPyramid::new(config);
        let key = TerrainTileKey::height(
            pyramid
                .address(pyramid.max_level(), TileId { tx: 1, tz: 1 })
                .unwrap(),
        );
        assert_eq!(
            measure_tile_geometric_error(&pyramid, &key, |_, _, _| 42.0),
            Some(0.0)
        );
    }

    #[test]
    fn root_error_is_zero_without_sampling() {
        let pyramid = TerrainPyramid::new(PyramidConfig::new(8, 8.0, 8.0));
        let key = pyramid.height_tiles_at_level(0).unwrap().next().unwrap();
        assert_eq!(
            measure_tile_geometric_error(&pyramid, &key, |_, _, _| panic!("root sampled")),
            Some(0.0)
        );
    }
}
