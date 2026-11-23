// GPU culling's occlusion test (gpu_culling.rs `BoxTest`), after
// gpu_culling.wgsl in one module. On compute Hi-Z backends (`test_boxes`), one
// compute thread per queued box tests its 8 corners against the reversed-Z
// min-depth Hi-Z pyramid (`hiz_depth`), updates hidden statistics, and directly
// emits visible phase-2 indirect draws into `draws` without scanning all
// sections (`count2`/`scan2`/`scatter2`). On the `Early` raster fallback
// (`vs_box` + `EARLY_FS_BOX` in gpu_culling.rs), boxes are drawn front faces
// only against phase 1's depth, followed by `count2`, `scan2` and `scatter2`.
@group(0) @binding(9) var<storage, read> box_list: array<u32>;
@group(0) @binding(10) var<storage, read> box_origins: array<vec4<u32>>;
@group(0) @binding(11) var<storage, read_write> box_runs: array<u32>;
@group(0) @binding(12) var<storage, read> box_sections: array<Section>;
@group(1) @binding(0) var hiz_depth: texture_2d<f32>;

// The corners (bit 0: x, bit 1: y, bit 2: z) of the box's 12 triangles,
// counter-clockwise seen from outside (cull.wgsl's `box_triangle_corner`).
// Packed four bits each, eight a word, read through a switch: a constant array indexed by the
// vertex would be a private array, which Adreno's Vulkan compiler (build 0676.41) fails to link.
fn box_triangle_corner(v: u32) -> u32 {
    var word: u32;
    switch v / 8u {
        case 0u: { word = 0x31640260u; }
        case 1u: { word = 0x05105717u; }
        case 2u: { word = 0x76237245u; }
        case 3u: { word = 0x54320130u; }
        default: { word = 0x00006747u; }
    }
    return (word >> (v % 8u * 4u)) & 0xfu;
}

struct BoxOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) id: u32,
}

@vertex
fn vs_box(@builtin(vertex_index) v: u32, @builtin(instance_index) t: u32) -> BoxOut {
    let id = box_list[params.occl.y + t];
    let o = bitcast<vec3<f32>>(box_origins[id].xyz);
    var out: BoxOut;
    out.id = id;
    out.position = params.view_proj * vec4(box_corner(o, box_sections[id].a.z, box_triangle_corner(v)), 1.0);
    return out;
}

// Conservative reversed-Z Hi-Z visibility test for one section box.
fn hiz_box_visible(o: vec3<f32>, b: u32) -> bool {
    var lo = vec2(1e30);
    var hi = vec2(-1e30);
    var max_z = -1e30;
    for (var c = 0u; c < 8u; c++) {
        let p = params.view_proj * vec4(box_corner(o, b, c), 1.0);
        if !(p.w > 1e-6 && p.z < p.w) {
            return true;
        }
        let inv_w = 1.0 / p.w;
        let ndc = p.xy * inv_w;
        let z = p.z * inv_w;
        lo = min(lo, ndc);
        hi = max(hi, ndc);
        max_z = max(max_z, z);
    }

    if hi.x < -1.0 || lo.x > 1.0 || hi.y < -1.0 || lo.y > 1.0 || max_z < 0.0 {
        return false;
    }

    let ndc_lo = max(lo, vec2(-1.0));
    let ndc_hi = min(hi, vec2(1.0));
    let uv_min = vec2(ndc_lo.x * 0.5 + 0.5, 0.5 - ndc_hi.y * 0.5);
    let uv_max = vec2(ndc_hi.x * 0.5 + 0.5, 0.5 - ndc_lo.y * 0.5);

    let max_mip = i32(textureNumLevels(hiz_depth)) - 1;
    let hiz_base = textureDimensions(hiz_depth, 0);
    let hiz_size = vec2<f32>(hiz_base);
    let rect = max((uv_max - uv_min) * hiz_size, vec2(0.0));
    let max_extent = max(rect.x, rect.y);
    var mip = clamp(i32(ceil(log2(max(max_extent, 1.0)))), 0, max_mip);

    // Derive mip dimensions from mip 0 by integer shift rather than dynamic
    // textureDimensions(hiz_depth, mip) to avoid extra descriptor queries and
    // non-uniform OpImageQuerySizeLod lane miscompiles on Mesa llvmpipe.
    if mip > 0 {
        let fine = mip - 1;
        let fine_size = vec2<i32>(max(hiz_base >> vec2<u32>(u32(fine)), vec2<u32>(1u)));
        let f_lo = clamp(vec2<i32>(floor(uv_min * vec2<f32>(fine_size))), vec2(0), fine_size - vec2(1));
        let f_hi = clamp(vec2<i32>(floor(uv_max * vec2<f32>(fine_size))), vec2(0), fine_size - vec2(1));
        if (f_hi.x - f_lo.x <= 1) && (f_hi.y - f_lo.y <= 1) {
            mip = fine;
        }
    }

    var mip_size = vec2<i32>(max(hiz_base >> vec2<u32>(u32(mip)), vec2<u32>(1u)));
    var lo_px = clamp(vec2<i32>(floor(uv_min * vec2<f32>(mip_size))), vec2(0), mip_size - vec2(1));
    var hi_px = clamp(vec2<i32>(floor(uv_max * vec2<f32>(mip_size))), vec2(0), mip_size - vec2(1));
    if ((hi_px.x - lo_px.x > 1) || (hi_px.y - lo_px.y > 1)) && mip < max_mip {
        mip += 1;
        mip_size = vec2<i32>(max(hiz_base >> vec2<u32>(u32(mip)), vec2<u32>(1u)));
        lo_px = clamp(vec2<i32>(floor(uv_min * vec2<f32>(mip_size))), vec2(0), mip_size - vec2(1));
        hi_px = clamp(vec2<i32>(floor(uv_max * vec2<f32>(mip_size))), vec2(0), mip_size - vec2(1));
    }

    let d00 = textureLoad(hiz_depth, vec2<i32>(lo_px.x, lo_px.y), mip).r;
    let d10 = textureLoad(hiz_depth, vec2<i32>(hi_px.x, lo_px.y), mip).r;
    let d01 = textureLoad(hiz_depth, vec2<i32>(lo_px.x, hi_px.y), mip).r;
    let d11 = textureLoad(hiz_depth, vec2<i32>(hi_px.x, hi_px.y), mip).r;
    let occluder_min_z = min(min(d00, d10), min(d01, d11));

    return !(max_z < occluder_min_z);
}

// Workgroup scratch for fused `test_boxes`:
// - `box_tally`: [0] hidden boxes, [1] hidden quads, [2] phase-2 draws emitted.
// - `box_page_n` / `box_page_base`: per-page draw batching for pages < FAST_PAGES.
// - `box_emit`: per-lane `(page << PAGE_SHIFT) | n` for fast-page intra-WG prefix sum.
var<workgroup> box_tally: array<atomic<u32>, 3>;
var<workgroup> box_page_n: array<atomic<u32>, 8>;
var<workgroup> box_page_base: array<u32, 8>;
var<workgroup> box_emit: array<u32, 64>;

// Fused compute Hi-Z occlusion test + direct Phase-2 indirect draw emission:
// 1 thread per queued box, using the main compute bind group (`@group(0)`) and
// dispatched indirectly via `HizTest::dispatch_args` (copied from `counts[BOX + 2..5]`
// before world pass 1 so `counts` stays read-write with no INDIRECT usage aliasing).
@compute @workgroup_size(64)
fn test_boxes(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if lane < 3u {
        atomicStore(&box_tally[lane], 0u);
    }
    if lane < FAST_PAGES {
        atomicStore(&box_page_n[lane], 0u);
    }
    box_emit[lane] = 0u;
    workgroupBarrier();

    let t = (group.y * ROW + group.x) * BLOCK + lane;
    let total_boxes = atomicLoad(&counts[BOX + 1u]);
    var emit_id = 0u;
    var emit_page = 0u;
    var emit_n = 0u;
    var slow_slot = 0u;
    if t < total_boxes {
        let id = atomicLoad(&counts[params.occl.y + t]);
        let o = bitcast<vec3<f32>>(origins[id].xyz);
        let b = sections[id].a.z;
        let at = id * RUN_WORDS;
        let w = runs[at];
        let n = w & 7u;
        if hiz_box_visible(o, b) {
            runs[at + SEEN_WORD] = params.occl.x;
            if (w & PHASE2) != 0u && n > 0u {
                let page = w >> PAGE_SHIFT;
                emit_id = id;
                emit_page = page;
                emit_n = n;
                if page < FAST_PAGES {
                    atomicAdd(&box_page_n[page], n);
                    box_emit[lane] = (page << PAGE_SHIFT) | n;
                } else {
                    let r = region(page);
                    slow_slot = atomicAdd(&counts[page_count(page, 1u)], n);
                    if slow_slot < r.y {
                        atomicAdd(&box_tally[2], min(n, r.y - slow_slot));
                    }
                    if slow_slot + n > r.y {
                        atomicMin(&counts[page_count(page, 1u)], r.y);
                    }
                }
            }
        } else if n > 0u {
            var quads = 0u;
            for (var k = 0u; k < n; k++) {
                quads += runs[at + 2u + 2u * k];
            }
            atomicAdd(&box_tally[0], 1u);
            atomicAdd(&box_tally[1], quads);
        }
    }
    workgroupBarrier();

    if lane < FAST_PAGES {
        let wg_n = atomicLoad(&box_page_n[lane]);
        if wg_n > 0u {
            let r = region(lane);
            let base = atomicAdd(&counts[page_count(lane, 1u)], wg_n);
            box_page_base[lane] = base;
            if base < r.y {
                atomicAdd(&box_tally[2], min(wg_n, r.y - base));
            }
            if base + wg_n > r.y {
                atomicMin(&counts[page_count(lane, 1u)], r.y);
            }
        }
    }
    workgroupBarrier();

    if lane == 0u {
        let h = atomicLoad(&box_tally[0]);
        if h > 0u {
            atomicAdd(&counts[HIDDEN], h);
        }
        let hq = atomicLoad(&box_tally[1]);
        if hq > 0u {
            atomicAdd(&counts[HIDDEN_QUADS], hq);
        }
        let d2 = atomicLoad(&box_tally[2]);
        if d2 > 0u {
            atomicAdd(&counts[DRAWS], d2);
            atomicAdd(&counts[DRAWN2], d2);
        }
    }

    if emit_n > 0u {
        var slot = slow_slot;
        if emit_page < FAST_PAGES {
            slot = box_page_base[emit_page];
            for (var other = 0u; other < lane; other++) {
                let h = box_emit[other];
                if h != 0u && (h >> PAGE_SHIFT) == emit_page {
                    slot += h & 7u;
                }
            }
        }
        let r = region(emit_page);
        let at = emit_id * RUN_WORDS;
        let base_vertex = sections[emit_id].a.x;
        for (var k = 0u; k < emit_n; k++) {
            if slot + k >= r.y {
                break;
            }
            let d = (r.x + slot + k) * 5u;
            draws[d] = runs[at + 2u + 2u * k] * 6u;
            draws[d + 1u] = 1u;
            draws[d + 2u] = 0u;
            draws[d + 3u] = base_vertex + runs[at + 1u + 2u * k] * 4u;
            draws[d + 4u] = emit_id;
        }
    }
}

var<workgroup> tally: array<atomic<u32>, 2>;

// After the `Early` raster test, one workgroup per block of ids: the tested boxes
// not seen and their quads (`QUADS` is already tallied in `cull`), and per
// (page, band) each block's draws that phase 2 makes (`drawn_in`'s condition).
@compute @workgroup_size(64)
fn count2(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if lane < 2u {
        atomicStore(&tally[lane], 0u);
    }
    if lane < FAST_BUCKETS {
        atomicStore(&wg_buckets[lane], 0u);
    }
    workgroupBarrier();
    let block = group.y * ROW + group.x;
    let id = block * BLOCK + lane;
    if id < params.ids {
        let at = id * RUN_WORDS;
        let w = runs[at];
        let n = w & 7u;
        if n > 0u {
            let saw = runs[at + SEEN_WORD] == params.occl.x;
            if (w & TESTED) != 0u && !saw {
                var quads = 0u;
                for (var k = 0u; k < n; k++) {
                    quads += runs[at + 2u + 2u * k];
                }
                atomicAdd(&tally[0], 1u);
                atomicAdd(&tally[1], quads);
            }
            if (w & PHASE2) != 0u && saw {
                let page = w >> PAGE_SHIFT;
                let band = (w >> BAND_SHIFT) & BAND_MASK;
                if page < FAST_PAGES {
                    atomicAdd(&wg_buckets[page * BANDS + band], n);
                } else {
                    atomicAdd(&counts[block_count(block, page, band, 1u)], n);
                }
            }
        }
    }
    workgroupBarrier();
    if lane == 0u {
        let h = atomicLoad(&tally[0]);
        if h > 0u {
            atomicAdd(&counts[HIDDEN], h);
        }
        let hq = atomicLoad(&tally[1]);
        if hq > 0u {
            atomicAdd(&counts[HIDDEN_QUADS], hq);
        }
    }
    if lane < FAST_BUCKETS {
        let count = atomicLoad(&wg_buckets[lane]);
        if count > 0u {
            let page = lane / BANDS;
            let band = lane % BANDS;
            let blocks = (params.ids + BLOCK - 1u) / BLOCK;
            atomicStore(&counts[block_base(page, band, 1u, blocks) + block], count);
        }
    }
}

@compute @workgroup_size(256)
fn scan2(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    scan_page(group.x, lane, 1u);
}

@compute @workgroup_size(64)
fn scatter2(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    scatter_block(group.y * ROW + group.x, lane, 1u);
}
