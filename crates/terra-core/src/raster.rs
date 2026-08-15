//! Typed, allocation-checked resampling for project-owned dense rasters.

use thiserror::Error;

/// Dimensions of a stored, imported, or evaluated raster grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridDimensions {
    pub width: u32,
    pub height: u32,
}

impl GridDimensions {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub const fn square(resolution: u32) -> Self {
        Self::new(resolution, resolution)
    }

    pub const fn component_min(self, other: Self) -> Self {
        Self::new(
            if self.width < other.width {
                self.width
            } else {
                other.width
            },
            if self.height < other.height {
                self.height
            } else {
                other.height
            },
        )
    }

    pub const fn is_smaller_than(self, other: Self) -> bool {
        self.width < other.width || self.height < other.height
    }
}

pub const MIB: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasterSemantic {
    Height,
    Mask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RasterResizeLimits {
    pub min_dimension: u32,
    pub max_dimension: u32,
    pub max_stored_bytes: usize,
}

impl Default for RasterResizeLimits {
    fn default() -> Self {
        Self {
            min_dimension: 128,
            max_dimension: 8192,
            max_stored_bytes: 256 * MIB,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RasterResizeError {
    #[error("source raster dimensions must be non-zero (found {width} x {height})")]
    InvalidSourceDimensions { width: u32, height: u32 },
    #[error("raster dimensions must be at least {minimum} per axis (requested {requested_width} x {requested_height})")]
    BelowMinimum {
        minimum: u32,
        requested_width: u32,
        requested_height: u32,
    },
    #[error("raster dimensions may not exceed {maximum} per axis (requested {requested_width} x {requested_height})")]
    AboveMaximum {
        maximum: u32,
        requested_width: u32,
        requested_height: u32,
    },
    #[error("raster allocation size overflow for {width} x {height}")]
    AllocationOverflow { width: u32, height: u32 },
    #[error(
        "raster requires {required_bytes} bytes, exceeding the {budget_bytes}-byte source budget"
    )]
    MemoryBudgetExceeded {
        required_bytes: usize,
        budget_bytes: usize,
    },
    #[error("source buffer has {actual} samples; {width} x {height} requires {expected}")]
    MalformedSource {
        width: u32,
        height: u32,
        expected: usize,
        actual: usize,
    },
    #[error("could not allocate {samples} raster samples")]
    AllocationFailed { samples: usize },
}

pub fn checked_sample_count(dimensions: GridDimensions) -> Result<usize, RasterResizeError> {
    let width =
        usize::try_from(dimensions.width).map_err(|_| RasterResizeError::AllocationOverflow {
            width: dimensions.width,
            height: dimensions.height,
        })?;
    let height =
        usize::try_from(dimensions.height).map_err(|_| RasterResizeError::AllocationOverflow {
            width: dimensions.width,
            height: dimensions.height,
        })?;
    width
        .checked_mul(height)
        .ok_or(RasterResizeError::AllocationOverflow {
            width: dimensions.width,
            height: dimensions.height,
        })
}

pub fn validate_resize(
    source_dimensions: GridDimensions,
    source_samples: &[f32],
    target_dimensions: GridDimensions,
    limits: RasterResizeLimits,
) -> Result<(), RasterResizeError> {
    if source_dimensions.width == 0 || source_dimensions.height == 0 {
        return Err(RasterResizeError::InvalidSourceDimensions {
            width: source_dimensions.width,
            height: source_dimensions.height,
        });
    }
    if target_dimensions.width < limits.min_dimension
        || target_dimensions.height < limits.min_dimension
    {
        return Err(RasterResizeError::BelowMinimum {
            minimum: limits.min_dimension,
            requested_width: target_dimensions.width,
            requested_height: target_dimensions.height,
        });
    }
    if target_dimensions.width > limits.max_dimension
        || target_dimensions.height > limits.max_dimension
    {
        return Err(RasterResizeError::AboveMaximum {
            maximum: limits.max_dimension,
            requested_width: target_dimensions.width,
            requested_height: target_dimensions.height,
        });
    }
    let expected = checked_sample_count(source_dimensions)?;
    if source_samples.len() != expected {
        return Err(RasterResizeError::MalformedSource {
            width: source_dimensions.width,
            height: source_dimensions.height,
            expected,
            actual: source_samples.len(),
        });
    }
    let target_samples = checked_sample_count(target_dimensions)?;
    let required_bytes = target_samples
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(RasterResizeError::AllocationOverflow {
            width: target_dimensions.width,
            height: target_dimensions.height,
        })?;
    if required_bytes > limits.max_stored_bytes {
        return Err(RasterResizeError::MemoryBudgetExceeded {
            required_bytes,
            budget_bytes: limits.max_stored_bytes,
        });
    }
    Ok(())
}

pub fn resample_f32_grid(
    source: &[f32],
    source_dimensions: GridDimensions,
    target_dimensions: GridDimensions,
    semantic: RasterSemantic,
    limits: RasterResizeLimits,
) -> Result<Vec<f32>, RasterResizeError> {
    validate_resize(source_dimensions, source, target_dimensions, limits)?;
    if source_dimensions == target_dimensions {
        return Ok(source.to_vec());
    }

    let sw = source_dimensions.width as usize;
    let sh = source_dimensions.height as usize;
    let tw = target_dimensions.width as usize;
    let th = target_dimensions.height as usize;

    let mut horizontal = allocate_zeroed(tw.checked_mul(sh).ok_or(
        RasterResizeError::AllocationOverflow {
            width: target_dimensions.width,
            height: source_dimensions.height,
        },
    )?)?;
    for y in 0..sh {
        let row = &source[y * sw..(y + 1) * sw];
        for x in 0..tw {
            horizontal[y * tw + x] = sample_axis(row, sw, tw, x);
        }
    }

    let mut output = allocate_zeroed(tw.checked_mul(th).ok_or(
        RasterResizeError::AllocationOverflow {
            width: target_dimensions.width,
            height: target_dimensions.height,
        },
    )?)?;
    let mut column = allocate_zeroed(sh)?;
    for x in 0..tw {
        for y in 0..sh {
            column[y] = horizontal[y * tw + x];
        }
        for y in 0..th {
            let value = sample_axis(&column, sh, th, y);
            output[y * tw + x] = match semantic {
                RasterSemantic::Height => value,
                RasterSemantic::Mask => value.clamp(0.0, 1.0),
            };
        }
    }
    Ok(output)
}

fn allocate_zeroed(samples: usize) -> Result<Vec<f32>, RasterResizeError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(samples)
        .map_err(|_| RasterResizeError::AllocationFailed { samples })?;
    values.resize(samples, 0.0);
    Ok(values)
}

fn sample_axis(source: &[f32], source_len: usize, target_len: usize, target_index: usize) -> f32 {
    if source_len == target_len {
        return source[target_index];
    }
    if target_len < source_len {
        return area_sample_axis(source, source_len, target_len, target_index);
    }

    let position = ((target_index as f32 + 0.5) * source_len as f32 / target_len as f32 - 0.5)
        .clamp(0.0, (source_len - 1) as f32);
    let left = position.floor() as usize;
    let right = (left + 1).min(source_len - 1);
    let blend = position - left as f32;
    source[left] + (source[right] - source[left]) * blend
}

fn area_sample_axis(
    source: &[f32],
    source_len: usize,
    target_len: usize,
    target_index: usize,
) -> f32 {
    let start = target_index as f32 * source_len as f32 / target_len as f32;
    let end = (target_index + 1) as f32 * source_len as f32 / target_len as f32;
    let first = start.floor() as usize;
    let last = end.ceil().min(source_len as f32) as usize;
    let mut sum = 0.0;
    let mut weight = 0.0;
    for source_index in first..last {
        let overlap =
            (end.min((source_index + 1) as f32) - start.max(source_index as f32)).max(0.0);
        sum += source[source_index] * overlap;
        weight += overlap;
    }
    if weight <= f32::EPSILON {
        0.0
    } else {
        sum / weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn permissive_limits() -> RasterResizeLimits {
        RasterResizeLimits {
            min_dimension: 1,
            max_dimension: 8192,
            max_stored_bytes: 256 * MIB,
        }
    }

    #[test]
    fn rectangular_mixed_axis_resize_preserves_constant() {
        let source = vec![7.5; 5 * 3];
        let resized = resample_f32_grid(
            &source,
            GridDimensions::new(5, 3),
            GridDimensions::new(3, 7),
            RasterSemantic::Height,
            permissive_limits(),
        )
        .unwrap();
        assert_eq!(resized.len(), 21);
        assert!(resized.iter().all(|value| (*value - 7.5).abs() < 1.0e-6));
    }

    #[test]
    fn area_downsample_rejects_checkerboard_aliasing() {
        let source: Vec<f32> = (0..64)
            .flat_map(|y| (0..64).map(move |x| if (x + y) % 2 == 0 { 0.0 } else { 1.0 }))
            .collect();
        let resized = resample_f32_grid(
            &source,
            GridDimensions::square(64),
            GridDimensions::square(8),
            RasterSemantic::Mask,
            permissive_limits(),
        )
        .unwrap();
        assert!(resized.iter().all(|value| (*value - 0.5).abs() < 1.0e-6));
    }

    #[test]
    fn invalid_target_is_rejected_before_allocation() {
        let error = resample_f32_grid(
            &[0.0; 4],
            GridDimensions::square(2),
            GridDimensions::new(u32::MAX, 128),
            RasterSemantic::Height,
            RasterResizeLimits::default(),
        )
        .unwrap_err();
        assert!(matches!(error, RasterResizeError::AboveMaximum { .. }));
    }

    #[test]
    fn zero_sized_source_fails_without_sampling() {
        let error = resample_f32_grid(
            &[],
            GridDimensions::new(0, 0),
            GridDimensions::square(128),
            RasterSemantic::Mask,
            RasterResizeLimits::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            RasterResizeError::InvalidSourceDimensions { .. }
        ));
    }
}
