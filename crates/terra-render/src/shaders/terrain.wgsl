struct FrameUniforms {
    view_proj: mat4x4<f32>,
    // xyz = light direction (from light toward scene), w = sun intensity
    light_dir: vec4<f32>,
    // x=world_size_x, y=world_size_z, z=height_min, w=height_max
    world: vec4<f32>,
    // x=tex_w, y=tex_h, z=ocean_level, w=slab base height
    grid: vec4<f32>,
    // Legacy slot: x=origin_x, y=origin_z, z=spacing_x, w=grid_size (mesh is world-fixed UV*world)
    clipmap: vec4<f32>,
    // xyz = camera eye, w = exposure
    eye: vec4<f32>,
    // x = stochastic frame seed, y = progressive enabled, z = accumulated samples, w = biome tint
    render: vec4<f32>,
    // x = shading_mode (0 lit, 1 height, 2 slope, 3 flow), y = contours on, z = contour interval m
    // w = clipmap hole half-extent (Chebyshev metres); 0 = no hole
    viz: vec4<f32>,
    // Directional light view-projection for shadow map sampling.
    light_view_proj: mat4x4<f32>,
    // x=use_tile_stream, y=tile_size, z=halo, w=max_pages
    stream: vec4<f32>,
    // x=fog_density, y=height_falloff, z=max_amount, w=sun_scatter
    fog: vec4<f32>,
    // x=shadow_enabled, y=depth_bias, z=stream_level, w=soft_scale
    shadow: vec4<f32>,
    // Raster shading: x=ambient_strength, y=shadow_strength, z=fog_strength, w=unused
    raster: vec4<f32>,
    // Complete streamed-content identity: document/plan then output/content halves.
    stream2: vec4<u32>,
    stream3: vec4<u32>,
    // x=level_count, y=target_level, z=current_frame_lo bits, w=transition_frames
    stream4: vec4<u32>,
    // x=terminal monolithic allowed, y=debug mode
    stream5: vec4<f32>,
    // x=Infinite sparse addressing, y=directory capacity, z=max LOD
    stream6: vec4<u32>,
    // Signed finest-tile render anchor: x low/high, z low/high
    stream7: vec4<u32>,
    // x=finest spacing metres, y=finest tile span metres,
    // z/w=camera-relative X/Z centre of this draw's exclusion hole
    stream8: vec4<f32>,
};

/// Physical page-table row (must match `GpuPageTableEntry`).
struct PageTableEntry {
    key_hash_lo: u32,
    key_hash_hi: u32,
    generation: u32,
    valid: u32,
    level: u32,
    tile_x: u32,
    tile_x_hi: u32,
    tile_z: u32,
    tile_z_hi: u32,
    width: u32,
    height: u32,
    halo: u32,
    document_revision_lo: u32,
    document_revision_hi: u32,
    plan_revision_lo: u32,
    plan_revision_hi: u32,
    output_revision_lo: u32,
    output_revision_hi: u32,
    content_revision_lo: u32,
    content_revision_hi: u32,
    published_frame_lo: u32,
    published_frame_hi: u32,
};

struct VirtualPageEntry {
    physical_slot: u32,
    generation: u32,
    valid: u32,
    lod: u32,
    tile_x: u32,
    tile_x_hi: u32,
    tile_z: u32,
    tile_z_hi: u32,
};

struct TerrainLevelEntry {
    resolution: u32,
    tiles_x: u32,
    tiles_z: u32,
    metadata_offset: u32,
};


struct MaterialGpu {
    albedo_roughness: vec4<f32>,
    // x = metalness, y = authored valid, z = albedo array layer + 1 (0 = tint only)
    metalness_valid: vec4<f32>,
};

struct MaterialPalette {
    entries: array<MaterialGpu, 17>,
};
@group(0) @binding(0) var<uniform> u: FrameUniforms;
@group(0) @binding(1) var height_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var normal_samp: sampler;
@group(0) @binding(4) var materials_tex: texture_2d<f32>;
@group(0) @binding(5) var wetness_tex: texture_2d<f32>;
@group(0) @binding(6) var vegetation_tex: texture_2d<f32>;
@group(0) @binding(7) var flow_tex: texture_2d<f32>;
@group(0) @binding(8) var<uniform> material_palette: MaterialPalette;
@group(0) @binding(9) var snow_tex: texture_2d<f32>;
@group(0) @binding(10) var temperature_tex: texture_2d<f32>;
@group(0) @binding(11) var rainfall_tex: texture_2d<f32>;
@group(0) @binding(12) var albedo_array: texture_2d_array<f32>;
@group(0) @binding(13) var albedo_samp: sampler;
@group(0) @binding(14) var placement_tint_tex: texture_2d<f32>;
@group(0) @binding(15) var tile_atlas: texture_2d_array<f32>;
@group(0) @binding(16) var<storage, read> page_table: array<PageTableEntry>;
@group(0) @binding(17) var shadow_map: texture_depth_2d;
@group(0) @binding(18) var shadow_samp: sampler_comparison;
@group(0) @binding(19) var<storage, read> virtual_page_table: array<VirtualPageEntry>;
@group(0) @binding(20) var<storage, read> terrain_levels: array<TerrainLevelEntry>;

struct VsIn {
    @location(0) uv: vec2<f32>,
    // 0 = top surface, 1 = skirt wall, 2 = underside
    @location(1) face: f32,
    // 0 = sample heightfield, 1 = slab base (grid.w)
    @location(2) use_base: f32,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) terrain_uv: vec2<f32>,
    @location(3) face: f32,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var o: VsOut;
    let cells = max(u.clipmap.w - 1.0, 1.0);
    let wx = u.clipmap.x + v.uv.x * u.clipmap.z * cells;
    let wz = u.clipmap.y + v.uv.y * u.clipmap.z * cells;
    let huv_raw = vec2<f32>(
        wx / max(u.world.x, 1e-3),
        wz / max(u.world.y, 1e-3),
    );
    let huv = clamp(huv_raw, vec2<f32>(0.0), vec2<f32>(1.0));

    // terra-variant:infinite
    let surface_h = resolve_height_infinite(vec2<f32>(wx, wz)).height;
    // terra-variant:bounded
    let surface_h = sample_height_bilinear(huv);
    // terra-variant:end
    // grid.w = slab base height (below terrain min).
    let base_y = u.grid.w;
    let h = select(surface_h, base_y, v.use_base > 0.5);

    let world_pos = vec3<f32>(wx, h, wz);
    o.position = u.view_proj * vec4<f32>(world_pos, 1.0);
    o.world_pos = world_pos;
    o.face = v.face;
    // terra-variant:infinite
    o.terrain_uv = vec2<f32>(0.5);
    // terra-variant:bounded
    o.terrain_uv = huv_raw;
    // terra-variant:end

    if (v.face > 1.5) {
        o.normal = vec3<f32>(0.0, -1.0, 0.0);
    } else if (v.face > 0.5) {
        // Outward wall normal from which border we're on.
        let eps = 1e-3;
        if (v.uv.x <= eps) {
            o.normal = vec3<f32>(-1.0, 0.0, 0.0);
        } else if (v.uv.x >= 1.0 - eps) {
            o.normal = vec3<f32>(1.0, 0.0, 0.0);
        } else if (v.uv.y <= eps) {
            o.normal = vec3<f32>(0.0, 0.0, -1.0);
        } else {
            o.normal = vec3<f32>(0.0, 0.0, 1.0);
        }
    } else {
        // terra-variant:infinite
        let d = max(u.stream8.x, 0.01);
        let hx0 = resolve_height_infinite(vec2<f32>(wx - d, wz)).height;
        let hx1 = resolve_height_infinite(vec2<f32>(wx + d, wz)).height;
        let hz0 = resolve_height_infinite(vec2<f32>(wx, wz - d)).height;
        let hz1 = resolve_height_infinite(vec2<f32>(wx, wz + d)).height;
        o.normal = normalize(vec3<f32>(hx0 - hx1, 2.0 * d, hz0 - hz1));
        // terra-variant:bounded
        let ntex = textureDimensions(normal_tex);
        let nx = i32(clamp(huv.x * f32(ntex.x - 1u), 0.0, f32(ntex.x - 1u)));
        let ny = i32(clamp(huv.y * f32(ntex.y - 1u), 0.0, f32(ntex.y - 1u)));
        o.normal = textureLoad(normal_tex, vec2<i32>(nx, ny), 0).xyz;
        // terra-variant:end
    }
    return o;
}


@vertex
fn vs_ocean(v: VsIn) -> VsOut {
    var o: VsOut;
    let wx = v.uv.x * u.world.x;
    let wz = v.uv.y * u.world.y;
    let huv_raw = vec2<f32>(
        wx / max(u.world.x, 1e-3),
        wz / max(u.world.y, 1e-3),
    );
    let wave = sin(wx * 0.017 + wz * 0.011) * 0.18
        + sin(wx * -0.009 + wz * 0.021) * 0.11;
    let world_pos = vec3<f32>(wx, u.grid.z + wave, wz);
    o.position = u.view_proj * vec4<f32>(world_pos, 1.0);
    o.world_pos = world_pos;
    let dx = 0.18 * 0.017 * cos(wx * 0.017 + wz * 0.011)
        - 0.11 * 0.009 * cos(wx * -0.009 + wz * 0.021);
    let dz = 0.18 * 0.011 * cos(wx * 0.017 + wz * 0.011)
        + 0.11 * 0.021 * cos(wx * -0.009 + wz * 0.021);
    o.normal = normalize(vec3<f32>(-dx, 1.0, -dz));
    o.terrain_uv = huv_raw;
    o.face = 0.0;
    return o;
}
fn sample_map(map: texture_2d<f32>, uv: vec2<f32>) -> f32 {
    let dim = textureDimensions(map);
    let x = i32(clamp(uv.x * f32(dim.x - 1u), 0.0, f32(dim.x - 1u)));
    let y = i32(clamp(uv.y * f32(dim.y - 1u), 0.0, f32(dim.y - 1u)));
    return textureLoad(map, vec2<i32>(x, y), 0).r;
}

fn sample_height_monolithic(uv: vec2<f32>) -> f32 {
    let dim = textureDimensions(height_tex);
    let x = i32(clamp(uv.x * f32(dim.x - 1u), 0.0, f32(dim.x - 1u)));
    let y = i32(clamp(uv.y * f32(dim.y - 1u), 0.0, f32(dim.y - 1u)));
    return textureLoad(height_tex, vec2<i32>(x, y), 0).r;
}

const STREAM_EXACT: u32 = 0u;
const STREAM_ANCESTOR: u32 = 1u;
const STREAM_TERMINAL: u32 = 2u;
const STREAM_BLEND: u32 = 3u;

struct ResolvedHeightSample {
    height: f32,
    sample_class: u32,
    level: u32,
};

fn page_identity_current(e: PageTableEntry) -> bool {
    return e.document_revision_lo == u.stream2.x
        && e.document_revision_hi == u.stream2.y
        && e.plan_revision_lo == u.stream2.z
        && e.plan_revision_hi == u.stream2.w
        && e.output_revision_lo == u.stream3.x
        && e.output_revision_hi == u.stream3.y
        && e.content_revision_lo == u.stream3.z
        && e.content_revision_hi == u.stream3.w;
}

// terra-variant:infinite
struct Signed64Words {
    lo: u32,
    hi: u32,
};

fn signed64_add_i32(value: Signed64Words, delta: i32) -> Signed64Words {
    let delta_lo = bitcast<u32>(delta);
    let delta_hi = select(0u, 0xffffffffu, delta < 0);
    let lo = value.lo + delta_lo;
    let carry = select(0u, 1u, lo < value.lo);
    return Signed64Words(lo, value.hi + delta_hi + carry);
}

fn signed64_shift_right(value: Signed64Words, shift: u32) -> Signed64Words {
    if (shift == 0u) { return value; }
    let sign_word = select(0u, 0xffffffffu, bitcast<i32>(value.hi) < 0);
    if (shift < 32u) {
        return Signed64Words(
            (value.lo >> shift) | (value.hi << (32u - shift)),
            bitcast<u32>(bitcast<i32>(value.hi) >> shift),
        );
    }
    if (shift < 64u) {
        return Signed64Words(
            bitcast<u32>(bitcast<i32>(value.hi) >> (shift - 32u)),
            sign_word,
        );
    }
    return Signed64Words(sign_word, sign_word);
}

fn sparse_hash(lod: u32, x: Signed64Words, z: Signed64Words) -> u32 {
    var hash = 2166136261u;
    hash = (hash ^ lod) * 16777619u;
    hash = (hash ^ x.lo) * 16777619u;
    hash = (hash ^ x.hi) * 16777619u;
    hash = (hash ^ z.lo) * 16777619u;
    hash = (hash ^ z.hi) * 16777619u;
    return hash;
}

fn infinite_address(local_xz: vec2<f32>, lod: u32) -> vec4<u32> {
    let tile_span = max(u.stream8.y, 1.0e-6);
    let delta_x = i32(floor(local_xz.x / tile_span));
    let delta_z = i32(floor(local_xz.y / tile_span));
    let fine_x = signed64_add_i32(Signed64Words(u.stream7.x, u.stream7.y), delta_x);
    let fine_z = signed64_add_i32(Signed64Words(u.stream7.z, u.stream7.w), delta_z);
    let tile_x = signed64_shift_right(fine_x, lod);
    let tile_z = signed64_shift_right(fine_z, lod);
    return vec4<u32>(tile_x.lo, tile_x.hi, tile_z.lo, tile_z.hi);
}

fn lookup_tile_page_sparse(lod: u32, address: vec4<u32>) -> i32 {
    let capacity = u.stream6.y;
    if (capacity == 0u) { return -1; }
    let mask = capacity - 1u;
    var index = sparse_hash(
        lod,
        Signed64Words(address.x, address.y),
        Signed64Words(address.z, address.w),
    ) & mask;
    for (var probe = 0u; probe < capacity; probe = probe + 1u) {
        let mapping = virtual_page_table[index];
        if (mapping.valid == 0u) { return -1; }
        if (mapping.lod == lod
            && mapping.tile_x == address.x && mapping.tile_x_hi == address.y
            && mapping.tile_z == address.z && mapping.tile_z_hi == address.w) {
            let max_pages = u32(max(u.stream.w, 1.0));
            if (mapping.physical_slot >= max_pages) { return -1; }
            let e = page_table[mapping.physical_slot];
            if (e.valid != 0u && e.generation == mapping.generation
                && e.level == lod
                && e.tile_x == address.x && e.tile_x_hi == address.y
                && e.tile_z == address.z && e.tile_z_hi == address.w
                && page_identity_current(e)) {
                return i32(mapping.physical_slot);
            }
            return -1;
        }
        index = (index + 1u) & mask;
    }
    return -1;
}
// terra-variant:end

fn lookup_tile_page(level: u32, tile_x: u32, tile_z: u32) -> i32 {
    let level_count = u.stream4.x;
    if (level >= level_count) { return -1; }
    let level_info = terrain_levels[level];
    if (tile_x >= level_info.tiles_x || tile_z >= level_info.tiles_z) { return -1; }
    let virtual_index = level_info.metadata_offset + tile_z * level_info.tiles_x + tile_x;
    let mapping = virtual_page_table[virtual_index];
    let max_pages = u32(max(u.stream.w, 1.0));
    if (mapping.valid == 0u || mapping.physical_slot >= max_pages) { return -1; }
    let e = page_table[mapping.physical_slot];
    if (e.valid == 0u || e.generation != mapping.generation
        || e.level != level || e.tile_x != tile_x || e.tile_z != tile_z
        || !page_identity_current(e)) {
        return -1;
    }
    return i32(mapping.physical_slot);
}

fn sample_page_bilinear(page: u32, uv: vec2<f32>) -> f32 {
    let e = page_table[page];
    let level_info = terrain_levels[e.level];
    let last = f32(max(level_info.resolution, 1u) - 1u);
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * last;
    let tile_size = max(u.stream.y, 1.0);
    let local = p - vec2<f32>(f32(e.tile_x), f32(e.tile_z)) * tile_size
        + vec2<f32>(f32(e.halo));
    let limit = vec2<i32>(
        i32(e.width + e.halo * 2u - 1u),
        i32(e.height + e.halo * 2u - 1u),
    );
    let p0 = clamp(vec2<i32>(floor(local)), vec2<i32>(0), limit);
    let p1 = min(p0 + vec2<i32>(1), limit);
    let f = fract(local);
    let h00 = textureLoad(tile_atlas, p0, i32(page), 0).r;
    let h10 = textureLoad(tile_atlas, vec2<i32>(p1.x, p0.y), i32(page), 0).r;
    let h01 = textureLoad(tile_atlas, vec2<i32>(p0.x, p1.y), i32(page), 0).r;
    let h11 = textureLoad(tile_atlas, p1, i32(page), 0).r;
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// terra-variant:infinite
fn infinite_tile_origin_local(local_xz: vec2<f32>, lod: u32) -> vec2<f32> {
    let fine_span = max(u.stream8.y, 1.0e-6);
    let level_scale = exp2(f32(lod));
    var mask = 0u;
    if (lod > 0u && lod < 32u) {
        mask = (1u << lod) - 1u;
    }
    let anchor_phase = vec2<f32>(
        f32(u.stream7.x & mask),
        f32(u.stream7.z & mask),
    );
    let fine_position = local_xz / fine_span;
    let group = floor((anchor_phase + fine_position) / level_scale) * level_scale
        - anchor_phase;
    return group * fine_span;
}

fn sample_page_infinite(page: u32, local_xz: vec2<f32>, lod: u32) -> f32 {
    let e = page_table[page];
    let spacing = max(u.stream8.x, 1.0e-6) * exp2(f32(lod));
    let tile_origin = infinite_tile_origin_local(local_xz, lod);
    let p = (local_xz - tile_origin) / spacing + vec2<f32>(f32(e.halo));
    let limit = vec2<i32>(
        i32(e.width + e.halo * 2u - 1u),
        i32(e.height + e.halo * 2u - 1u),
    );
    let p0 = clamp(vec2<i32>(floor(p)), vec2<i32>(0), limit);
    let p1 = min(p0 + vec2<i32>(1), limit);
    let f = fract(p);
    let h00 = textureLoad(tile_atlas, p0, i32(page), 0).r;
    let h10 = textureLoad(tile_atlas, vec2<i32>(p1.x, p0.y), i32(page), 0).r;
    let h01 = textureLoad(tile_atlas, vec2<i32>(p0.x, p1.y), i32(page), 0).r;
    let h11 = textureLoad(tile_atlas, p1, i32(page), 0).r;
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

fn resolve_height_infinite_from(local_xz: vec2<f32>, start_lod: u32) -> ResolvedHeightSample {
    let max_lod = u.stream6.z;
    // Infinite presentation has no bounded monolithic fallback. Until the
    // current coarse-coverage set is complete, resolve to terminal so the
    // fragment stage leaves the viewport clear instead of drawing a local square
    // or exposing a partially resident page set.
    if (u.stream.x <= 0.5 || u.stream6.y == 0u) {
        return ResolvedHeightSample(u.world.z, STREAM_TERMINAL, max_lod);
    }
    for (var lod = start_lod; lod <= max_lod; lod = lod + 1u) {
        let address = infinite_address(local_xz, lod);
        let page = lookup_tile_page_sparse(lod, address);
        if (page >= 0) {
            return ResolvedHeightSample(
                sample_page_infinite(u32(page), local_xz, lod),
                select(STREAM_ANCESTOR, STREAM_EXACT, lod == 0u),
                lod,
            );
        }
    }
    return ResolvedHeightSample(u.world.z, STREAM_TERMINAL, max_lod);
}

fn infinite_page_edge_weight(page: u32, local_xz: vec2<f32>, lod: u32) -> f32 {
    let e = page_table[page];
    let spacing = max(u.stream8.x, 1.0e-6) * exp2(f32(lod));
    let level_span = max(u.stream8.y, 1.0e-6) * exp2(f32(lod));
    let tile_origin = infinite_tile_origin_local(local_xz, lod);
    let local = (local_xz - tile_origin) / spacing;
    let band = 4.0;
    var weight = 1.0;
    if (local.x < band
        && lookup_tile_page_sparse(lod, infinite_address(local_xz - vec2<f32>(level_span, 0.0), lod)) < 0) {
        weight = min(weight, smoothstep(0.0, band, local.x));
    }
    if (f32(e.width - 1u) - local.x < band
        && lookup_tile_page_sparse(lod, infinite_address(local_xz + vec2<f32>(level_span, 0.0), lod)) < 0) {
        weight = min(weight, smoothstep(0.0, band, f32(e.width - 1u) - local.x));
    }
    if (local.y < band
        && lookup_tile_page_sparse(lod, infinite_address(local_xz - vec2<f32>(0.0, level_span), lod)) < 0) {
        weight = min(weight, smoothstep(0.0, band, local.y));
    }
    if (f32(e.height - 1u) - local.y < band
        && lookup_tile_page_sparse(lod, infinite_address(local_xz + vec2<f32>(0.0, level_span), lod)) < 0) {
        weight = min(weight, smoothstep(0.0, band, f32(e.height - 1u) - local.y));
    }
    return weight;
}

fn resolve_height_infinite(local_xz: vec2<f32>) -> ResolvedHeightSample {
    let selected = resolve_height_infinite_from(local_xz, 0u);
    if (selected.sample_class == STREAM_TERMINAL || selected.level >= u.stream6.z) {
        return selected;
    }
    let address = infinite_address(local_xz, selected.level);
    let page = lookup_tile_page_sparse(selected.level, address);
    if (page < 0) { return selected; }
    let e = page_table[u32(page)];
    let transition_frames = f32(max(u.stream4.w, 1u));
    let time_weight = select(
        smoothstep(0.0, 1.0, f32(u.stream4.z - e.published_frame_lo) / transition_frames),
        1.0,
        u.stream4.w == 0u,
    );
    let weight = min(time_weight, infinite_page_edge_weight(u32(page), local_xz, selected.level));
    if (weight >= 0.999) { return selected; }
    let ancestor = resolve_height_infinite_from(local_xz, selected.level + 1u);
    if (ancestor.sample_class == STREAM_TERMINAL) { return selected; }
    return ResolvedHeightSample(
        mix(ancestor.height, selected.height, weight),
        STREAM_BLEND,
        selected.level,
    );
}
// terra-variant:end

fn resolve_from_level(uv: vec2<f32>, start_level: i32) -> ResolvedHeightSample {
    let c = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let target_level = u.stream4.y;
    var level = start_level;
    loop {
        if (level < 0) { break; }
        let li = u32(level);
        let level_info = terrain_levels[li];
        let last = f32(max(level_info.resolution, 1u) - 1u);
        let p = c * last;
        let tile_size = max(u.stream.y, 1.0);
        let tx = min(u32(floor(p.x / tile_size)), level_info.tiles_x - 1u);
        let tz = min(u32(floor(p.y / tile_size)), level_info.tiles_z - 1u);
        let page = lookup_tile_page(li, tx, tz);
        if (page >= 0) {
            return ResolvedHeightSample(
                sample_page_bilinear(u32(page), c),
                select(STREAM_ANCESTOR, STREAM_EXACT, li == target_level),
                li,
            );
        }
        level = level - 1;
    }
    return ResolvedHeightSample(sample_height_monolithic(c), STREAM_TERMINAL, 0u);
}

fn page_edge_weight(page: u32, uv: vec2<f32>) -> f32 {
    let e = page_table[page];
    let level_info = terrain_levels[e.level];
    let last = f32(max(level_info.resolution, 1u) - 1u);
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * last;
    let tile_size = max(u.stream.y, 1.0);
    let local = p - vec2<f32>(f32(e.tile_x), f32(e.tile_z)) * tile_size;
    let band = 4.0;
    var weight = 1.0;
    if (e.tile_x > 0u && lookup_tile_page(e.level, e.tile_x - 1u, e.tile_z) < 0) {
        weight = min(weight, smoothstep(0.0, band, local.x));
    }
    if (e.tile_x + 1u < level_info.tiles_x
        && lookup_tile_page(e.level, e.tile_x + 1u, e.tile_z) < 0) {
        weight = min(weight, smoothstep(0.0, band, f32(e.width - 1u) - local.x));
    }
    if (e.tile_z > 0u && lookup_tile_page(e.level, e.tile_x, e.tile_z - 1u) < 0) {
        weight = min(weight, smoothstep(0.0, band, local.y));
    }
    if (e.tile_z + 1u < level_info.tiles_z
        && lookup_tile_page(e.level, e.tile_x, e.tile_z + 1u) < 0) {
        weight = min(weight, smoothstep(0.0, band, f32(e.height - 1u) - local.y));
    }
    return weight;
}

fn resolve_height_streamed(uv: vec2<f32>) -> ResolvedHeightSample {
    let target_level = i32(u.stream4.y);
    let selected = resolve_from_level(uv, target_level);
    if (selected.sample_class == STREAM_TERMINAL || selected.level == 0u) {
        return selected;
    }
    let level_info = terrain_levels[selected.level];
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0))
        * f32(max(level_info.resolution, 1u) - 1u);
    let tile_size = max(u.stream.y, 1.0);
    let tx = min(u32(floor(p.x / tile_size)), level_info.tiles_x - 1u);
    let tz = min(u32(floor(p.y / tile_size)), level_info.tiles_z - 1u);
    let page = lookup_tile_page(selected.level, tx, tz);
    if (page < 0) { return selected; }
    let e = page_table[u32(page)];
    let current_frame = u.stream4.z;
    let transition_frames = f32(max(u.stream4.w, 1u));
    let time_weight = select(
        smoothstep(
            0.0,
            1.0,
            f32(current_frame - e.published_frame_lo) / transition_frames,
        ),
        1.0,
        u.stream4.w == 0u,
    );
    let weight = min(time_weight, page_edge_weight(u32(page), uv));
    if (weight >= 0.999) { return selected; }
    let ancestor = resolve_from_level(uv, i32(selected.level) - 1);
    return ResolvedHeightSample(
        mix(ancestor.height, selected.height, weight),
        select(STREAM_BLEND, STREAM_TERMINAL, ancestor.sample_class == STREAM_TERMINAL),
        selected.level,
    );
}

fn sample_height_uv(uv: vec2<f32>) -> f32 {
    if (u.stream.x > 0.5) {
        return resolve_height_streamed(uv).height;
    }
    return sample_height_monolithic(uv);
}

fn material_color(id: u32) -> vec3<f32> {
    // Neutral editor palette — rock/dirt/grass/sand/snow without blowing out.
    switch (id % 5u) {
        case 0u: { return vec3<f32>(0.42, 0.40, 0.36); } // rock
        case 1u: { return vec3<f32>(0.48, 0.36, 0.24); } // dirt
        case 2u: { return vec3<f32>(0.30, 0.42, 0.22); } // grass
        case 3u: { return vec3<f32>(0.62, 0.54, 0.36); } // sand
        default: { return vec3<f32>(0.78, 0.80, 0.82); } // snow
    }
}

fn material_surface(sample: f32) -> vec4<f32> {
    let id = u32(clamp(round(sample * 16.0), 0.0, 16.0));
    let authored = clamp(material_palette.entries[id].metalness_valid.y, 0.0, 1.0);
    let color = mix(material_color(id), material_palette.entries[id].albedo_roughness.rgb, authored);
    let roughness = mix(0.82, material_palette.entries[id].albedo_roughness.a, authored);
    return vec4<f32>(color, roughness);
}

fn material_tex_layer(sample: f32) -> i32 {
    let id = u32(clamp(round(sample * 16.0), 0.0, 16.0));
    let packed = material_palette.entries[id].metalness_valid.z;
    return i32(packed) - 1;
}

fn value_noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/// World-space triplanar albedo sample from the material texture array.
/// Always samples (uniform control flow) — callers blend with `weight`.
fn triplanar_albedo(world: vec3<f32>, n: vec3<f32>, layer: i32, scale: f32) -> vec3<f32> {
    let safe_layer = max(layer, 0);
    let an = abs(n);
    var w = an / max(an.x + an.y + an.z, 1e-4);
    // Bias toward top-down for gentle slopes so cliffs take side projections.
    w = pow(w, vec3<f32>(1.35));
    w = w / max(w.x + w.y + w.z, 1e-4);
    let ux = world.zy * scale;
    let uy = world.xz * scale;
    let uz = world.xy * scale;
    let cx = textureSample(albedo_array, albedo_samp, ux, safe_layer).rgb;
    let cy = textureSample(albedo_array, albedo_samp, uy, safe_layer).rgb;
    let cz = textureSample(albedo_array, albedo_samp, uz, safe_layer).rgb;
    return cx * w.x + cy * w.y + cz * w.z;
}

fn material_metalness(sample: f32) -> f32 {
    let id = u32(clamp(round(sample * 16.0), 0.0, 16.0));
    return material_palette.entries[id].metalness_valid.x
        * clamp(material_palette.entries[id].metalness_valid.y, 0.0, 1.0);
}

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

/// Local height-field AO — samples a few texels as a cheap GI occlusion term.
/// Offsets are unrolled (no WGSL array ctors — some NVIDIA drivers crash on those).
fn height_ao(uv: vec2<f32>, h: f32) -> f32 {
    let dim = textureDimensions(height_tex);
    let texel = vec2<f32>(1.0 / f32(dim.x), 1.0 / f32(dim.y));
    var occl = 0.0;
    occl += max(sample_height_uv(clamp(uv + vec2<f32>( 2.0,  0.0) * texel, vec2<f32>(0.0), vec2<f32>(1.0))) - h, 0.0);
    occl += max(sample_height_uv(clamp(uv + vec2<f32>(-2.0,  0.0) * texel, vec2<f32>(0.0), vec2<f32>(1.0))) - h, 0.0);
    occl += max(sample_height_uv(clamp(uv + vec2<f32>( 0.0,  2.0) * texel, vec2<f32>(0.0), vec2<f32>(1.0))) - h, 0.0);
    occl += max(sample_height_uv(clamp(uv + vec2<f32>( 0.0, -2.0) * texel, vec2<f32>(0.0), vec2<f32>(1.0))) - h, 0.0);
    occl += max(sample_height_uv(clamp(uv + vec2<f32>( 4.0,  4.0) * texel, vec2<f32>(0.0), vec2<f32>(1.0))) - h, 0.0);
    occl += max(sample_height_uv(clamp(uv + vec2<f32>(-4.0,  4.0) * texel, vec2<f32>(0.0), vec2<f32>(1.0))) - h, 0.0);
    occl += max(sample_height_uv(clamp(uv + vec2<f32>( 4.0, -4.0) * texel, vec2<f32>(0.0), vec2<f32>(1.0))) - h, 0.0);
    occl += max(sample_height_uv(clamp(uv + vec2<f32>(-4.0, -4.0) * texel, vec2<f32>(0.0), vec2<f32>(1.0))) - h, 0.0);
    let span = max(u.world.w - u.world.z, 1.0);
    return clamp(1.0 - (occl / 8.0) / (span * 0.08), 0.35, 1.0);
}

fn sample_height_bilinear(uv: vec2<f32>) -> f32 {
    let uvc = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    if (u.stream.x > 0.5) {
        return resolve_height_streamed(uvc).height;
    }
    let dim = textureDimensions(height_tex);
    let p = uvc * vec2<f32>(dim - vec2<u32>(1u));
    let p0 = vec2<i32>(floor(p));
    let p1 = min(p0 + vec2<i32>(1), vec2<i32>(dim) - vec2<i32>(1));
    let f = fract(p);
    let h00 = textureLoad(height_tex, p0, 0).r;
    let h10 = textureLoad(height_tex, vec2<i32>(p1.x, p0.y), 0).r;
    let h01 = textureLoad(height_tex, vec2<i32>(p0.x, p1.y), 0).r;
    let h11 = textureLoad(height_tex, p1, 0).r;
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

fn sample_shadow_map(world_pos: vec3<f32>, n: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    if (u.shadow.x < 0.5) {
        return 1.0;
    }
    // Slope-scaled bias along the light direction, scaled by world size: a shadow
    // texel is metres wide on km-scale terrain, so the old sub-metre offset left
    // lit surfaces self-shadowing (acne that darkened lit areas as strength rose).
    let ndl = max(dot(n, sun_dir), 0.0);
    let world_extent = max(u.world.x, u.world.y);
    let bias = u.shadow.y * world_extent * (1.0 + (1.0 - ndl) * 3.0);
    let light_clip = u.light_view_proj * vec4<f32>(world_pos + sun_dir * bias, 1.0);
    let ndc = light_clip.xyz / max(light_clip.w, 1e-4);
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (any(uv < vec2<f32>(0.001)) || any(uv > vec2<f32>(0.999)) || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    // Wider PCF kernel to hide ortho shadow texelation on large worlds.
    let soft = max(u.shadow.w, 1.5);
    let texel = soft / 2048.0;
    var vis = 0.0;
    vis += textureSampleCompare(shadow_map, shadow_samp, uv + vec2<f32>(-1.5 * texel, -1.5 * texel), ndc.z);
    vis += textureSampleCompare(shadow_map, shadow_samp, uv + vec2<f32>( 0.5 * texel, -1.5 * texel), ndc.z);
    vis += textureSampleCompare(shadow_map, shadow_samp, uv + vec2<f32>( 1.5 * texel, -0.5 * texel), ndc.z);
    vis += textureSampleCompare(shadow_map, shadow_samp, uv + vec2<f32>(-1.5 * texel,  0.5 * texel), ndc.z);
    vis += textureSampleCompare(shadow_map, shadow_samp, uv + vec2<f32>( 0.0, 0.0), ndc.z);
    vis += textureSampleCompare(shadow_map, shadow_samp, uv + vec2<f32>( 1.5 * texel,  1.5 * texel), ndc.z);
    vis += textureSampleCompare(shadow_map, shadow_samp, uv + vec2<f32>(-0.5 * texel,  1.5 * texel), ndc.z);
    vis += textureSampleCompare(shadow_map, shadow_samp, uv + vec2<f32>( 1.5 * texel, -1.5 * texel), ndc.z);
    vis += textureSampleCompare(shadow_map, shadow_samp, uv + vec2<f32>(-1.5 * texel,  1.5 * texel), ndc.z);
    // Shadow darkness from the raster shadow-strength control: 0 = no darkening,
    // 1 = fully black in shadow. (0.65 reproduces the old fixed 0.35 floor.)
    return mix(1.0 - u.raster.y, 1.0, vis / 9.0);
}

/// Trace against the full heightfield rather than only nearby screen pixels.
/// Vegetation density contributes a stochastic canopy proxy, allowing forests
/// to cast broad, naturally broken shadows without requiring every card in a BVH.
fn terrain_visibility(origin: vec3<f32>, direction: vec3<f32>, max_distance: f32, seed: f32) -> f32 {
    let texel_world = max(u.world.x / max(u.grid.x, 1.0), u.world.y / max(u.grid.y, 1.0));
    let start = max(texel_world * 1.35, 0.65);
    for (var step = 0u; step < 36u; step = step + 1u) {
        let s = (f32(step) + 0.65) / 36.0;
        let distance = start + (max_distance - start) * s * s;
        let ray = origin + direction * distance;
        let uv = vec2<f32>(ray.x / max(u.world.x, 1e-3), ray.z / max(u.world.y, 1e-3));
        if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
            return 1.0;
        }
        let terrain_h = sample_height_bilinear(uv);
        if (ray.y <= terrain_h + max(texel_world * 0.06, 0.12)) {
            return 0.0;
        }
        let density = clamp(sample_map(vegetation_tex, uv), 0.0, 1.0);
        let canopy_height = 2.0 + density * 15.0;
        let canopy_noise = hash21(floor(ray.xz * 0.21) + vec2<f32>(seed, seed * 0.37));
        if (density > 0.12 && canopy_noise < density && ray.y <= terrain_h + canopy_height) {
            return 0.0;
        }
    }
    return 1.0;
}

fn jittered_sun(sun_dir: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    let seed = u.render.x;
    let r1 = hash21(pixel + vec2<f32>(seed * 17.17, seed * 3.11));
    let r2 = hash21(pixel.yx + vec2<f32>(seed * 5.73, seed * 29.41));
    let radius = sqrt(r1) * 0.018;
    let angle = r2 * 6.2831853;
    let reference = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(sun_dir.y) > 0.96);
    let tangent = normalize(cross(reference, sun_dir));
    let bitangent = cross(sun_dir, tangent);
    return normalize(sun_dir + tangent * cos(angle) * radius + bitangent * sin(angle) * radius);
}

fn stochastic_sky_visibility(origin: vec3<f32>, n: vec3<f32>, pixel: vec2<f32>) -> f32 {
    let seed = u.render.x;
    let r1 = hash21(pixel + vec2<f32>(seed * 11.31, seed * 47.03));
    let r2 = hash21(pixel.yx + vec2<f32>(seed * 31.71, seed * 7.19));
    let phi = r1 * 6.2831853;
    let radial = sqrt(r2);
    let local = vec3<f32>(cos(phi) * radial, sqrt(max(1.0 - r2, 0.0)), sin(phi) * radial);
    let reference = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.96);
    let tangent = normalize(cross(reference, n));
    let bitangent = cross(n, tangent);
    let direction = normalize(tangent * local.x + n * local.y + bitangent * local.z);
    let extent = max(u.world.x, u.world.y);
    return terrain_visibility(origin, direction, extent * 0.16, seed + 19.0);
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn turbo_colormap(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    // Compact approximation of Google Turbo for debug viz.
    let r = clamp(0.1357 + x * (4.370 - x * (8.514 - x * 4.251)), 0.0, 1.0);
    let g = clamp(0.0914 + x * (2.185 + x * (0.219 - x * 1.870)), 0.0, 1.0);
    let b = clamp(0.3366 + x * (1.972 - x * (5.999 - x * 3.803)), 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn apply_contours(color: vec3<f32>, height: f32) -> vec3<f32> {
    if (u.viz.y < 0.5) {
        return color;
    }
    let interval = max(u.viz.z, 1.0);
    let band = fract(height / interval);
    let width = fwidth(height / interval);
    let line = 1.0 - smoothstep(0.0, max(width * 1.5, 0.02), min(band, 1.0 - band));
    return mix(color, color * 0.22, clamp(line, 0.0, 1.0) * 0.85);
}

@fragment
fn fs_wireframe(i: VsOut) -> @location(0) vec4<f32> {
    if (i.face < 0.5
        && (i.terrain_uv.x < 0.0 || i.terrain_uv.x > 1.0 || i.terrain_uv.y < 0.0 || i.terrain_uv.y > 1.0)) {
        discard;
    }
    return vec4<f32>(0.12, 0.14, 0.16, 0.85);
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    // Coarse clipmap rings / fallback leave a hole so finer coverage is never overdrawn.
    if (u.viz.w > 0.5) {
        let center = u.stream8.zw;
        let d = max(abs(i.world_pos.x - center.x), abs(i.world_pos.z - center.y));
        if (d < u.viz.w) {
            discard;
        }
    }
    if (u.stream.x > 0.5 && u.stream5.y > 0.5 && i.face < 0.5) {
        // terra-variant:infinite
        let resolved = resolve_height_infinite(i.world_pos.xz);
        // terra-variant:bounded
        let resolved = resolve_height_streamed(clamp(i.terrain_uv, vec2<f32>(0.0), vec2<f32>(1.0)));
        // terra-variant:end
        if (resolved.sample_class == STREAM_EXACT) {
            return vec4<f32>(0.10, 0.85, 0.25, 1.0);
        }
        if (resolved.sample_class == STREAM_ANCESTOR) {
            return vec4<f32>(0.95, 0.65, 0.10, 1.0);
        }
        if (resolved.sample_class == STREAM_BLEND) {
            return vec4<f32>(0.15, 0.55, 0.95, 1.0);
        }
        // Root-required mode makes a terminal sample intentionally conspicuous.
        return select(
            vec4<f32>(1.0, 0.0, 0.8, 1.0),
            vec4<f32>(0.75, 0.2, 0.95, 1.0),
            u.stream5.x > 0.5,
        );
    }
    // terra-variant:infinite
        let resolved = resolve_height_infinite(i.world_pos.xz);
        if (resolved.sample_class == STREAM_TERMINAL) {
            discard;
        }
        var n = normalize(i.normal);
        if (dot(n, n) < 1.0e-6) {
            n = vec3<f32>(0.0, 1.0, 0.0);
        }
        let light = normalize(-u.light_dir.xyz);
        let ndl = max(dot(n, light), 0.0);
        let span = max(u.world.w - u.world.z, 1.0e-3);
        let h = clamp((i.world_pos.y - u.world.z) / span, 0.0, 1.0);
        let low = vec3<f32>(0.29, 0.30, 0.23);
        let high = vec3<f32>(0.50, 0.49, 0.45);
        var color = mix(low, high, h) * (0.34 + 0.66 * ndl);
        let distance_m = length(i.world_pos - u.eye.xyz);
        let fog_amount = clamp(1.0 - exp(-distance_m * max(u.fog.x, 1.0e-6)), 0.0, u.fog.z);
        color = mix(color, vec3<f32>(0.48, 0.55, 0.64), fog_amount * u.raster.z);
        return vec4<f32>(aces_tonemap(color * max(u.eye.w, 0.1)), 1.0);
    }
    // terra-variant:bounded
    // Slab sides / underside — World Creator–style light cliff faces.
    if (i.face > 0.5) {
        var n = i.normal;
        if (dot(n, n) < 1e-6) {
            n = vec3<f32>(0.0, 1.0, 0.0);
        } else {
            n = normalize(n);
        }
        let light = normalize(-u.light_dir.xyz);
        let ndl = clamp(dot(n, light), 0.0, 1.0);
        let cliff = vec3<f32>(0.78, 0.80, 0.82);
        let ambient = 0.28;
        let lit = cliff * (ambient + (1.0 - ambient) * ndl) * u.eye.w;
        return vec4<f32>(lit, 1.0);
    }

    // Outside the authored heightfield — leave sky.
    if (i.terrain_uv.x < 0.0 || i.terrain_uv.x > 1.0 || i.terrain_uv.y < 0.0 || i.terrain_uv.y > 1.0) {
        discard;
    }

    let pixel_normal = textureSample(normal_tex, normal_samp, clamp(i.terrain_uv, vec2<f32>(0.0), vec2<f32>(1.0))).xyz;
    var n = mix(i.normal, pixel_normal, 0.92);
    if (dot(n, n) < 1e-6) {
        n = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        n = normalize(n);
    }

    let h_span = max(u.world.w - u.world.z, 1e-3);
    let h_norm = clamp((i.world_pos.y - u.world.z) / h_span, 0.0, 1.0);
    // sin(slope angle) preserves separation across gentle and steep terrain.
    let slope = sqrt(clamp(1.0 - n.y * n.y, 0.0, 1.0));
    let slope_deg = degrees(asin(clamp(slope, 0.0, 1.0)));

    // Viewport analysis modes (Height / Slope / Flow) — early-out before lit path.
    let mode = u32(u.viz.x + 0.5);
    if (mode == 1u) {
        var color = turbo_colormap(h_norm);
        color = apply_contours(color, i.world_pos.y);
        return vec4<f32>(color, 1.0);
    }
    if (mode == 2u) {
        var color = turbo_colormap(clamp(slope_deg / 60.0, 0.0, 1.0));
        color = apply_contours(color, i.world_pos.y);
        return vec4<f32>(color, 1.0);
    }
    if (mode == 3u) {
        let flow_raw = max(sample_map(flow_tex, i.terrain_uv), 0.0);
        let flow_n = clamp(log2(1.0 + flow_raw) / 8.0, 0.0, 1.0);
        var color = turbo_colormap(flow_n);
        color = apply_contours(color, i.world_pos.y);
        return vec4<f32>(color, 1.0);
    }

    // Height + slope tint — neutral grey-brown, avoids washed-out white highs.
    let low = vec3<f32>(0.29, 0.30, 0.23);
    let mid = vec3<f32>(0.39, 0.37, 0.30);
    let high = vec3<f32>(0.50, 0.49, 0.45);
    let height_albedo = mix(mix(low, mid, smoothstep(0.0, 0.55, h_norm)), high, smoothstep(0.55, 1.0, h_norm));
    let rock_tint = vec3<f32>(0.43, 0.42, 0.39);
    let base_tint = mix(height_albedo, rock_tint, smoothstep(0.38, 0.88, slope) * 0.76);

    let material_sample = sample_map(materials_tex, i.terrain_uv);
    let mdim = textureDimensions(materials_tex);
    let mtexel = vec2<f32>(1.0 / max(f32(mdim.x), 1.0), 1.0 / max(f32(mdim.y), 1.0));
    // Cross-fade categorical authoring IDs over neighboring texels. This keeps
    // the compact map format while presenting sand/soil/rock as splat blends.
    let ms0 = material_surface(material_sample);
    let ms1 = material_surface(sample_map(materials_tex, i.terrain_uv + vec2<f32>(mtexel.x, 0.0)));
    let ms2 = material_surface(sample_map(materials_tex, i.terrain_uv - vec2<f32>(mtexel.x, 0.0)));
    let ms3 = material_surface(sample_map(materials_tex, i.terrain_uv + vec2<f32>(0.0, mtexel.y)));
    let ms4 = material_surface(sample_map(materials_tex, i.terrain_uv - vec2<f32>(0.0, mtexel.y)));
    let blended_surface = ms0 * 0.50 + (ms1 + ms2 + ms3 + ms4) * 0.125;
    let mat_albedo = blended_surface.rgb;
    var roughness = blended_surface.a;
    var metalness = material_metalness(material_sample) * 0.50
        + (material_metalness(sample_map(materials_tex, i.terrain_uv + vec2<f32>(mtexel.x, 0.0)))
        + material_metalness(sample_map(materials_tex, i.terrain_uv - vec2<f32>(mtexel.x, 0.0)))
        + material_metalness(sample_map(materials_tex, i.terrain_uv + vec2<f32>(0.0, mtexel.y)))
        + material_metalness(sample_map(materials_tex, i.terrain_uv - vec2<f32>(0.0, mtexel.y)))) * 0.125;
    // Prefer materials when present; otherwise keep the readable height tint.
    let has_materials = material_sample > 1e-4;
    var albedo = mix(base_tint, mat_albedo, select(0.35, 0.96, has_materials));

    // Artist biome placement colours (World Design overlay).
    let biome_tint = textureSample(placement_tint_tex, albedo_samp, clamp(i.terrain_uv, vec2<f32>(0.0), vec2<f32>(1.0)));
    let biome_w = clamp(biome_tint.a * u.render.w, 0.0, 0.85);
    albedo = mix(albedo, biome_tint.rgb, biome_w);

    // Triplanar albedo maps when MaterialRule.albedo_path loaded into the array.
    // Sample unconditionally (uniform CF) then weight — avoids NVIDIA compiler
    // stack blowups on divergent textureSample.
    let tex_layer = material_tex_layer(material_sample);
    let tex_col = triplanar_albedo(i.world_pos, n, tex_layer, 0.085);
    let tex_w = select(0.0, 0.92, tex_layer >= 0 && has_materials);
    albedo = mix(albedo, albedo * tex_col, tex_w);

    let macro_a = value_noise2(i.world_pos.xz * 0.0016 + vec2<f32>(7.0, -13.0));
    let macro_b = value_noise2(i.world_pos.xz * 0.0085 + vec2<f32>(31.0, 5.0));
    let macro_variation = mix(macro_a, macro_b, 0.34);
    // Continuous variation avoids square cells and shimmer while panning.
    albedo *= mix(0.93, 1.07, macro_variation);

    // B2: world-space micro-detail (±12%) scaled by roughness — breaks plastic close-ups.
    let micro_a = value_noise2(i.world_pos.xz * 0.55);
    let micro_b = value_noise2(i.world_pos.xz * 1.85 + vec2<f32>(19.0, 7.0));
    let micro = mix(micro_a, micro_b, 0.45);
    albedo *= mix(1.0 - 0.055 * roughness, 1.0 + 0.055 * roughness, micro);

    // Scree / talus: darken steep faces + high-frequency speck.
    let scree = smoothstep(0.48, 0.92, slope);
    let scree_col = vec3<f32>(0.36, 0.33, 0.29);
    albedo = mix(albedo, scree_col, scree * 0.38);
    let scree_speck = smoothstep(0.58, 0.86, value_noise2(i.world_pos.xz * 0.31 + vec2<f32>(4.0, 17.0)));
    albedo = mix(albedo, scree_col * 0.88, scree * scree_speck * 0.16);

    // Ridge highlight without fwidth (derivatives + heavy CF can stack-overflow NV).
    let ridge = smoothstep(0.55, 0.95, slope) * (1.0 - scree * 0.5);
    albedo = mix(albedo, albedo * 1.10, ridge * 0.2);

    // Flow channels: log-ish remap of drainage accumulation → dark wet gullies.
    let flow_raw = max(sample_map(flow_tex, i.terrain_uv), 0.0);
    let flow_n = clamp(log2(1.0 + flow_raw) / 8.0, 0.0, 1.0);
    let channel = smoothstep(0.15, 0.75, flow_n) * (1.0 - slope * 0.55);
    let channel_col = vec3<f32>(0.22, 0.24, 0.20);
    albedo = mix(albedo, channel_col, channel * 0.55);
    var wetness = clamp(sample_map(wetness_tex, i.terrain_uv), 0.0, 1.0);
    wetness = max(wetness, channel * 0.55);

    // Climate: rainfall darkens / wets low slopes; temperature warm/cool bias.
    let rainfall = clamp(sample_map(rainfall_tex, i.terrain_uv), 0.0, 1.0);
    let temperature = clamp(sample_map(temperature_tex, i.terrain_uv), 0.0, 1.0);
    let rain_wet = rainfall * (1.0 - slope * 0.65) * 0.45;
    wetness = max(wetness, rain_wet);
    albedo *= mix(vec3<f32>(1.0), vec3<f32>(0.88, 0.90, 0.92), rainfall * 0.35 * (1.0 - slope));
    let warm = vec3<f32>(1.06, 0.98, 0.90);
    let cool = vec3<f32>(0.90, 0.94, 1.06);
    albedo *= mix(cool, warm, temperature);

    // Lakes / fluid sim: dark blue-green tint when wetness is high.
    let water_col = vec3<f32>(0.06, 0.20, 0.30);
    let water_blend = smoothstep(0.05, 0.42, wetness);
    albedo = mix(albedo, water_col, water_blend * 0.92);
    roughness = mix(roughness, 0.07, wetness * 0.88);
    metalness *= 1.0 - water_blend;
    albedo *= 1.0 - wetness * 0.16;

    let density = clamp(sample_map(vegetation_tex, i.terrain_uv), 0.0, 1.0);
    let canopy = smoothstep(0.05, 0.45, density);
    let speck = step(1.0 - density * 0.65, hash21(floor(i.world_pos.xz * 0.85)));
    let veg_col = vec3<f32>(0.12, 0.38, 0.11);
    var veg_w = max(speck * density, canopy * 0.55) * (1.0 - scree * 0.7) * (1.0 - water_blend);

    // Snow caps from climate snow map — suppress vegetation.
    let snow = clamp(sample_map(snow_tex, i.terrain_uv), 0.0, 1.0);
    let snow_w = smoothstep(0.12, 0.55, snow) * (1.0 - water_blend);
    let snow_col = vec3<f32>(0.86, 0.90, 0.94);
    albedo = mix(albedo, snow_col, snow_w * 0.92);
    roughness = mix(roughness, 0.55, snow_w * 0.6);
    veg_w *= 1.0 - snow_w;
    albedo = mix(albedo, veg_col, veg_w);

    // --- Lighting / approximate GI ---
    let sun_dir = normalize(-u.light_dir.xyz);
    let sun_intensity = max(u.light_dir.w, 0.0);
    let ndl = max(dot(n, sun_dir), 0.0);
    let sun_color = vec3<f32>(1.0, 0.96, 0.88) * sun_intensity;

    // Hemispheric ambient (sky vs ground bounce) — softer fill for form readability.
    let sky_col = vec3<f32>(0.38, 0.44, 0.55);
    let ground_col = vec3<f32>(0.18, 0.16, 0.12);
    let hemi = mix(ground_col, sky_col, n.y * 0.5 + 0.5);

    // Single-bounce fill: albedo bleeds into ambient (terrain radiosity approx).
    let bounce = albedo * 0.22 * (0.40 + 0.60 * h_norm);

    let ao = height_ao(i.terrain_uv, i.world_pos.y);
    let map_shadow = sample_shadow_map(i.world_pos, n, sun_dir);
    // Keep ambient occlusion as fill; shadow map only darkens direct sun.
    var shadow_visibility = mix(ao, ao * map_shadow, 0.85);
    var sky_visibility = ao;
    if (u.render.y > 0.5) {
        let origin = i.world_pos + n * max(h_span * 0.00035, 0.35);
        let stochastic_sun = jittered_sun(sun_dir, i.position.xy);
        let ray_vis = terrain_visibility(
            origin,
            stochastic_sun,
            max(u.world.x, u.world.y) * 1.35,
            u.render.x,
        );
        // Combine stable shadow map with stochastic contact/softening.
        shadow_visibility = min(shadow_visibility, mix(ray_vis, ray_vis * ao, 0.35));
        sky_visibility = mix(0.28, 1.0, stochastic_sky_visibility(origin, n, i.position.xy));
    }
    let ambient = (hemi * 0.38 + bounce * mix(0.65, 1.25, sky_visibility)) * sky_visibility * u.raster.x;
    // Cook-Torrance GGX using authored roughness and metalness.
    let view_dir = normalize(u.eye.xyz - i.world_pos);
    let half_v = normalize(sun_dir + view_dir);
    let ndv = max(dot(n, view_dir), 1e-4);
    let ndh = max(dot(n, half_v), 0.0);
    let vdh = max(dot(view_dir, half_v), 0.0);
    let alpha = max(roughness * roughness, 0.002);
    let alpha2 = alpha * alpha;
    let ggx_denom = ndh * ndh * (alpha2 - 1.0) + 1.0;
    let distribution = alpha2 / max(3.14159265 * ggx_denom * ggx_denom, 1e-5);
    let geometry_k = (roughness + 1.0) * (roughness + 1.0) * 0.125;
    let geometry_v = ndv / max(ndv * (1.0 - geometry_k) + geometry_k, 1e-5);
    let geometry_l = ndl / max(ndl * (1.0 - geometry_k) + geometry_k, 1e-5);
    let fresnel_base = mix(vec3<f32>(0.04), albedo, metalness);
    let fresnel = fresnel_base + (vec3<f32>(1.0) - fresnel_base) * pow(1.0 - vdh, 5.0);
    let specular = distribution * geometry_v * geometry_l * fresnel
        / max(4.0 * ndv * max(ndl, 1e-4), 1e-4);
    let diffuse_weight = (vec3<f32>(1.0) - fresnel) * (1.0 - metalness);
    let direct = (diffuse_weight * albedo / 3.14159265 + specular)
        * sun_color * ndl * shadow_visibility;

    var color = albedo * ambient * (1.0 - metalness * 0.5) + direct;

    // Camera-distance + height atmospheric haze (sun-aware aerial perspective).
    let cam_dist = length(i.world_pos - u.eye.xyz);
    let height_haze = clamp((i.world_pos.y - u.world.z) / h_span, 0.0, 1.0);
    let fog_density = max(u.fog.x, 1e-6);
    let fog_amount = (1.0 - exp(-cam_dist * fog_density))
        * mix(1.0, 1.0 - height_haze * u.fog.y, 0.65);
    let sun_scatter = u.fog.w;
    let fog_cool = vec3<f32>(0.42, 0.48, 0.56);
    let fog_warm = vec3<f32>(0.62, 0.55, 0.42);
    let fog_hi = vec3<f32>(0.55, 0.62, 0.72);
    let fog_col = mix(mix(fog_cool, fog_warm, sun_scatter * 0.45), fog_hi, height_haze);
    color = mix(color, fog_col, clamp(fog_amount * u.raster.z, 0.0, max(u.fog.z, 0.05)));

    let exposure = max(u.eye.w, 0.1);
    color = aces_tonemap(color * exposure);
    color = apply_contours(color, i.world_pos.y);
    return vec4<f32>(color, 1.0);
}
// terra-variant:end

@fragment
fn fs_ocean(i: VsOut) -> @location(0) vec4<f32> {
    if (i.terrain_uv.x < 0.0 || i.terrain_uv.x > 1.0 || i.terrain_uv.y < 0.0 || i.terrain_uv.y > 1.0) {
        discard;
    }
    let floor_h = sample_height_uv(i.terrain_uv);
    let depth = u.grid.z - floor_h;
    if (depth <= 0.08) {
        discard;
    }

    let shallow = vec3<f32>(0.08, 0.58, 0.62);
    let shelf = vec3<f32>(0.025, 0.25, 0.38);
    let abyss = vec3<f32>(0.012, 0.075, 0.15);
    let shelf_t = smoothstep(1.5, 32.0, depth);
    let deep_t = smoothstep(35.0, 220.0, depth);
    var color = mix(mix(shallow, shelf, shelf_t), abyss, deep_t);

    let n = normalize(i.normal);
    let view_dir = normalize(u.eye.xyz - i.world_pos);
    let sun_dir = normalize(-u.light_dir.xyz);
    let fresnel = 0.025 + 0.975 * pow(1.0 - max(dot(n, view_dir), 0.0), 5.0);
    let sky_reflection = vec3<f32>(0.34, 0.49, 0.64);
    color = mix(color, sky_reflection, fresnel * 0.72);
    var ocean_visibility = 1.0;
    var ocean_sun = sun_dir;
    if (u.render.y > 0.5) {
        ocean_sun = jittered_sun(sun_dir, i.position.xy);
        ocean_visibility = terrain_visibility(
            i.world_pos + n * 0.3,
            ocean_sun,
            max(u.world.x, u.world.y) * 1.35,
            u.render.x + 53.0,
        );
        color *= mix(0.58, 1.0, ocean_visibility);
    }
    let half_v = normalize(view_dir + ocean_sun);
    let glint = pow(max(dot(n, half_v), 0.0), 180.0) * max(u.light_dir.w, 0.0) * ocean_visibility;
    color += vec3<f32>(1.0, 0.92, 0.72) * glint * 1.8;

    // Clear water exposes reef/sand near shore; deeper water quickly becomes opaque.
    let opacity = mix(0.28, 0.92, smoothstep(0.4, 42.0, depth));
    let cam_dist = length(i.world_pos - u.eye.xyz);
    let haze = clamp(1.0 - exp(-cam_dist * 0.00018), 0.0, 0.24);
    color = mix(color, vec3<f32>(0.34, 0.43, 0.52), haze);
    return vec4<f32>(aces_tonemap(color * max(u.eye.w, 0.1)), opacity);
}
