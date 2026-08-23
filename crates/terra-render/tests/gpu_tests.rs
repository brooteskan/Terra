//! Consolidated GPU integration suite.
//!
//! One test process means all cases share the headless device and immutable
//! pipeline handles while retaining independent renderer and texture state.

#[path = "gpu_region_presentation.rs"]
mod gpu_region_presentation;
#[path = "offscreen_render.rs"]
mod offscreen_render;
#[path = "partial_region_upload.rs"]
mod partial_region_upload;
#[path = "tile_stream_residency.rs"]
mod tile_stream_residency;
