use terra_core::{
    InfiniteTopology, InfiniteTopologyConfig, Lod, TerrainContentStamp, TerrainTileKey,
    TileAddress, TileCoord, WorldPosition,
};
use terra_gpu::{GpuTileAtlas, GpuVirtualPageEntry};

fn topology() -> InfiniteTopology {
    InfiniteTopology::try_new(InfiniteTopologyConfig {
        origin: WorldPosition::ORIGIN,
        tile_size: 8,
        finest_spacing_m: 1.0,
        max_lod: Lod::try_new(8).unwrap(),
    })
    .unwrap()
}

fn key(x: i64, z: i64) -> TerrainTileKey {
    TerrainTileKey::height(TileAddress::new(Lod::FINEST, TileCoord { x, z }))
}

fn stamp(revision: u64) -> TerrainContentStamp {
    TerrainContentStamp {
        document_revision: 3,
        plan_revision: 5,
        output_revision: 7,
        content_revision: revision,
    }
}

fn entry_matches(entry: &GpuVirtualPageEntry, key: &TerrainTileKey) -> bool {
    entry.valid != 0
        && entry.lod == u32::from(key.address.lod.get())
        && entry.tile_x == key.address.coord.x as u64 as u32
        && entry.tile_x_hi == ((key.address.coord.x as u64) >> 32) as u32
        && entry.tile_z == key.address.coord.z as u64 as u32
        && entry.tile_z_hi == ((key.address.coord.z as u64) >> 32) as u32
}

fn publish(
    atlas: &mut GpuTileAtlas,
    gpu: &terra_test_gpu::TestGpu,
    key: TerrainTileKey,
    value: f32,
    content: TerrainContentStamp,
) -> terra_gpu::GpuTileUpload {
    let samples = vec![value; 100];
    atlas
        .upload_packed_height_tile_current_at_frame(
            &gpu.queue, key, &samples, 10, 10, 1, content, content, 0,
        )
        .unwrap()
}

#[test]
fn signed_sparse_addresses_do_not_alias_and_counts_stay_authoritative() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let mut atlas = GpuTileAtlas::new(&gpu.device, 8, 1, 2).unwrap();
    atlas.configure_infinite(&gpu.device, &gpu.queue, topology());
    let negative = key(-1, -2);
    let same_low_word = key(u32::MAX as i64, -2);
    publish(&mut atlas, gpu, negative.clone(), 11.0, stamp(9));
    publish(&mut atlas, gpu, same_low_word.clone(), 22.0, stamp(9));

    let directory = atlas.read_virtual_page_table_blocking(&gpu.device, &gpu.queue);
    assert_eq!(directory.iter().filter(|entry| entry.valid != 0).count(), 2);
    assert!(directory
        .iter()
        .any(|entry| entry_matches(entry, &negative)));
    assert!(directory
        .iter()
        .any(|entry| entry_matches(entry, &same_low_word)));
    let physical = atlas.read_page_table_blocking(&gpu.device, &gpu.queue);
    assert_eq!(physical.iter().filter(|entry| entry.valid != 0).count(), 2);
    assert_eq!(atlas.residency().stats().resident_tiles, 2);
}

#[test]
fn sparse_eviction_protects_current_coarse_page_and_regenerates() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let mut atlas = GpuTileAtlas::new(&gpu.device, 8, 1, 2).unwrap();
    atlas.configure_infinite(&gpu.device, &gpu.queue, topology());
    let coarse = key(-1, 0);
    let old = key(0, 0);
    let incoming = key(1, 0);
    let content = stamp(11);
    publish(&mut atlas, gpu, coarse.clone(), 1.0, content);
    publish(&mut atlas, gpu, old.clone(), 2.0, content);
    atlas.apply_demand(std::slice::from_ref(&coarse), std::slice::from_ref(&coarse));

    let replacement = publish(&mut atlas, gpu, incoming.clone(), 3.0, content);
    assert_eq!(replacement.evicted, vec![old.clone()]);
    assert!(atlas.is_current(&coarse, content));
    assert!(atlas.is_current(&incoming, content));
    assert!(!atlas.is_current(&old, content));

    atlas.apply_demand(std::slice::from_ref(&old), &[]);
    let regenerated = publish(&mut atlas, gpu, old.clone(), 2.0, content);
    let page = atlas.read_page_blocking(&gpu.device, &gpu.queue, regenerated.handle.slot);
    assert!(page
        .iter()
        .all(|sample| sample.to_bits() == 2.0f32.to_bits()));
    assert_eq!(atlas.residency().stats().resident_tiles, 2);
    assert_eq!(
        atlas
            .read_virtual_page_table_blocking(&gpu.device, &gpu.queue)
            .iter()
            .filter(|entry| entry.valid != 0)
            .count(),
        2
    );
}

#[test]
fn stale_packed_result_cannot_allocate_residency() {
    let Some(gpu) = terra_test_gpu::headless() else {
        return;
    };
    let mut atlas = GpuTileAtlas::new(&gpu.device, 8, 1, 1).unwrap();
    atlas.configure_infinite(&gpu.device, &gpu.queue, topology());
    let samples = vec![4.0; 100];
    assert!(atlas
        .upload_packed_height_tile_current_at_frame(
            &gpu.queue,
            key(-1, 0),
            &samples,
            10,
            10,
            1,
            stamp(1),
            stamp(2),
            0,
        )
        .is_err());
    assert_eq!(atlas.residency().stats().resident_tiles, 0);
    assert!(atlas
        .read_page_table_blocking(&gpu.device, &gpu.queue)
        .iter()
        .all(|entry| entry.valid == 0));
}
