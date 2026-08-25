//! Application-owned pipeline-cache location policy.

use std::path::{Path, PathBuf};

pub const DIRECTORY_ENV: &str = "TERRA_PIPELINE_CACHE_DIR";

pub fn config() -> Option<terra_gpu::PipelineCacheConfig> {
    directory().map(terra_gpu::PipelineCacheConfig::new)
}

pub fn directory() -> Option<PathBuf> {
    std::env::var_os(DIRECTORY_ENV)
        .map(PathBuf::from)
        .or_else(|| {
            directories::BaseDirs::new()
                .map(|directories| directory_from_local_data(directories.data_local_dir()))
        })
}

fn directory_from_local_data(local_data: &Path) -> PathBuf {
    local_data.join("Terra").join("cache").join("pipelines")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_data_path_maps_to_terra_pipeline_cache() {
        let root = Path::new("local-data");
        assert_eq!(
            directory_from_local_data(root),
            root.join("Terra").join("cache").join("pipelines")
        );
    }
}
