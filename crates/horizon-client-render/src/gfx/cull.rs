//! Occlusion culling of solid section draws, in two phases.
//!
//! Frustum and cave culling leave many sections that terrain in front hides
//! entirely (about half of the solid vertices drawn from the ground at render
//! distance 16, more at 32), and drawing fewer vertices is what helps an
//! integrated GPU. Phase 1 draws the sections that the last test read back
//! saw. Then, in the same pass, the box of every other section (and of a
//! share of phase 1's, to notice when they go out of sight) is drawn against
//! phase 1's depth with no writes but to the test's results. Phase 2, in a
//! second pass, draws the sections whose box was seen; the others' draws have
//! an instance count of 0.
//!
//! The picture stays the one without culling: the boxes around the mesh's
//! geometry (padded by a block) are drawn front faces
//! only, so a fragment is the box's nearest point on its pixel, and the
//! fragment shader runs only where the depth test passes (forced early), so a
//! section is left out only where all of its fragments would fail the depth
//! test against the final depth, which is only nearer. What the CPU knows of
//! the results only sorts sections between the phases: a stale result draws
//! a hidden section in phase 1, or tests a section that is in sight.
//!
//! Far terrain tiles are tested the same way, with their meshes' own
//! bounds for a box (grown by a block too, which keeps their faces off the
//! box's), and resident slots of their own (`TILE`).
//!
//! Every solid draw is an entry of one of the two lists, so the CPU records
//! as many draws as without culling: drivers that unroll an indirect
//! multi-draw on the CPU (Mesa's on Intel Gen9) pay per entry. Phase 2's
//! counts come from the GPU, which reads them from the draws themselves:
//! Vulkan's native multi-draw, DirectX 12's `ExecuteIndirect`, and Metal's,
//! which wgpu replays as one indirect draw per entry. Translucent draws are
//! not culled.
//!
//! Where compute shaders and `R32Float` storage textures are available
//! (Vulkan, Metal, DirectX 12), the test runs in compute between the phases:
//! phase 1's reversed-Z depth buffer is conservatively min-downsampled into a
//! Hi-Z mip pyramid (2x2 texels per step, plus the third overlapping texel
//! whenever a source dimension > 1 is odd so no pixel can be skipped), and
//! one compute thread per box projects its 8 corners once and tests its
//! nearest reversed-Z depth against the 2x2 Hi-Z footprint covering its
//! screen rectangle. Boxes containing the camera or crossing the near plane
//! count as seen conservatively.
//!
//! Devices without compute Hi-Z support (OpenGL, constrained adapters, or
//! explicit `HORIZON_GPU_CULL=shader` / `early` A/B checks) fall back to box
//! rasterization: forced early depth tests in phase 1's pass (`Early`), or a
//! separate render pass comparing against phase 1's depth in the fragment
//! shader (`InShader`, for backends lacking forced early depth tests).
use std::num::NonZeroU64;
use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

const WORDS: u64 = 5 * 4;
/// A tested section's box, around its origin: its cell grown by a block.
pub const SECTION_BOX: [[f32; 3]; 2] = [[-1.0; 3], [17.0; 3]];
/// Mesh bounds already contain `BOUNDS_PAD`; grow them to a block of padding
/// in total, keeping the box's faces off the geometry. Without mesh bounds,
/// keep the section cell grown by a block, as before.
pub fn section_box(bounds: Option<[glam::Vec3; 2]>) -> [[f32; 3]; 2] {
    bounds.map_or(SECTION_BOX, |[low, high]| {
        let pad = glam::Vec3::splat(1.0 - crate::mesher::SectionMesh::BOUNDS_PAD);
        [(low - pad).to_array(), (high + pad).to_array()]
    })
}
/// The bit that marks a far tile's slot (`Renderer::tile_cull_slot`): tiles
/// count their slots apart from the sections' resident ones.
pub const TILE: u32 = 1 << 31;
/// Phase 1's sections are each tested every `RETEST` frames.
const RETEST: u32 = 8;
/// Culling runs while the last result read back hid at least this share of
/// the frame's solid indices: below it the test and the second pass cost
/// more than they save. Else one frame in `PROBE` tests every section.
const MIN_SHARE: f32 = 0.1;
const PROBE: u32 = 64;
/// The gate, for frame `stamp`, with `share` the newest result's hidden share:
/// whether the frame culls, and whether it probes (tests every section). GPU
/// culling's frames take it too.
pub(super) fn gate(share: f32, stamp: u32) -> (bool, bool) {
    let probe = share < MIN_SHARE;
    (!probe || stamp.is_multiple_of(PROBE), probe)
}
/// On Metal, sections in view from which a frame culls. Apple's GPUs store
/// and reload the whole frame at each of the passes the test adds, a fixed
/// cost that the sections it hides repay only at high render distances: on
/// an M4 Pro at 1080p and render distance 12 (1,000–1,600 draws) culling
/// added about 0.05 ms of GPU time while hiding 15–30% of the solid
/// indices; at 32 (over 4,096 draws) it saved 0.25–1.6 ms. Those draws were
/// 2.2 to 2.5 per section; a section's draw count now depends on the sides
/// it shows the eye (`mesher::Sides`), so the bar counts sections.
pub const TILED_MIN_SECTIONS: usize = 1600;
/// Boxes over more than this share of the screen count as seen untested.
const BIG: f32 = 0.25;
/// Results read back at once, some frames late so the CPU never waits.
const READBACKS: usize = 3;
const FREE: u8 = 0;
const PENDING: u8 = 1;
const MAPPED: u8 = 2;

/// One test, as `cull.wgsl` reads it: visible index and phase 2 draws
/// (`place`), then its box's corners around the origin.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Test {
    data: [u32; 4],
    low: [f32; 4],
    high: [f32; 4],
}
const TEST_BYTES: u64 = std::mem::size_of::<Test>() as u64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    stamp: u32,
    big: f32,
    count: u32,
    pad: u32,
}

/// One frame's results on their way back.
struct Readback {
    buffer: wgpu::Buffer,
    capacity: u64,
    state: Arc<AtomicU8>,
    stamp: u32,
    /// The resident slot of each test, and its section's solid indices.
    slots: Vec<u32>,
    indices: Vec<u32>,
    /// The frame's solid indices.
    total: u64,
}

/// The bind group and the buffers from outside it was made with.
struct Group {
    origins: wgpu::Buffer,
    globals: wgpu::Buffer,
    group: wgpu::BindGroup,
}

/// Where the boxes' depth test runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthTest {
    /// Forced before the fragment shader, in phase 1's pass (Vulkan / GL fallback).
    Early,
    /// In the fragment shader, against phase 1's depth read as a texture,
    /// in a pass of its own (Metal / DX12 / Adreno fallback).
    InShader,
    /// Compute shader against a conservative reversed-Z min-depth Hi-Z pyramid
    /// built from phase 1's depth (Vulkan, Metal, DirectX 12).
    Compute,
}

/// Workgroup edge for the 2D 1-mip and 4-mip Hi-Z pyramid downsample passes.
const WORKGROUP_2D: u32 = 8;
/// Workgroup edge for the 5-mip tiled Hi-Z pyramid downsample pass.
const WORKGROUP_TILED5: u32 = 16;
/// Workgroup width for the 1D compute box test pass.
const WORKGROUP_1D: u32 = 64;
/// Workgroup width for the single-workgroup odd-aware Hi-Z tail passes.
const WORKGROUP_TAIL: u32 = 256;
/// Maximum X workgroups before wrapping to Y (WebGPU's 65535 limit).
const COMPUTE_ROW: u32 = 65535;
/// Maximum `dst0` texel count handled by a single-workgroup Hi-Z tail dispatch.
/// Ping-pong workgroup arrays hold `2048 + 1024` `f32` values = 12,288 bytes
/// (well within WebGPU's 16 KiB `max_compute_workgroup_storage_size` floor).
const HIZ_TAIL_MAX_TEXELS: u32 = 2048;
/// Required `max_compute_workgroup_storage_size` for `hiz_downsample_4` (`(64 + 16 + 4) * 4`).
const HIZ_TILED4_SHARED_BYTES: u32 = (64 + 16 + 4) * 4;
/// Required `max_compute_workgroup_storage_size` for `hiz_tail_4` / `hiz_tail_8`.
const HIZ_TAIL_SHARED_BYTES: u32 = (HIZ_TAIL_MAX_TEXELS + HIZ_TAIL_MAX_TEXELS / 2) * 4;
/// Copy-source usage enabled only in unit tests for GPU mip readback verification.
const HIZ_TEST_READ: wgpu::TextureUsages = if cfg!(test) {
    wgpu::TextureUsages::COPY_SRC
} else {
    wgpu::TextureUsages::empty()
};

/// The Vulkan entry point: the depth test forced early, which naga cannot
/// write for Metal (so it is not in `cull.wgsl`, which every backend takes).
const EARLY_TEST: &str = "
@fragment
@early_depth_test(force)
fn fs_box_early(in: Box) -> @location(0) vec4<f32> {
    see(in.test);
    return vec4(0.0);
}
";

/// Conservative reversed-Z min-depth Hi-Z source reduction helper shared by all
/// Hi-Z pyramid downsample shaders.
///
/// Each destination texel `(dx, dy)` covers UV `[dx/D, (dx+1)/D] x [dy/D, (dy+1)/D]`.
/// When a source dimension `S > 1` is odd (`S = 2*D + 1`), source texel `2*d + 2`
/// overlaps that UV interval too (`2*d/(2*D+1) <= d/D` and `(d+1)/D <= (2*d+3)/(2*D+1)`
/// for every `0 <= d < D`), so every odd-axis texel takes the minimum over 3 source
/// texels along that axis (3x2, 2x3, or 3x3).
const HIZ_REDUCE_SRC: &str = "
@group(0) @binding(0) var src: texture_2d<f32>;

fn reduce_src(src_size: vec2<u32>, dst_coord: vec2<i32>) -> f32 {
    let max_coord = vec2<i32>(src_size) - vec2<i32>(1);
    let base = dst_coord * 2;
    let x0 = min(base.x, max_coord.x);
    let x1 = min(base.x + 1, max_coord.x);
    let y0 = min(base.y, max_coord.y);
    let y1 = min(base.y + 1, max_coord.y);

    var d = min(
        min(
            textureLoad(src, vec2<i32>(x0, y0), 0).r,
            textureLoad(src, vec2<i32>(x1, y0), 0).r,
        ),
        min(
            textureLoad(src, vec2<i32>(x0, y1), 0).r,
            textureLoad(src, vec2<i32>(x1, y1), 0).r,
        ),
    );

    let odd_x = src_size.x > 1u && (src_size.x & 1u) != 0u;
    let odd_y = src_size.y > 1u && (src_size.y & 1u) != 0u;

    if odd_x {
        let x2 = min(base.x + 2, max_coord.x);
        d = min(
            d,
            min(
                textureLoad(src, vec2<i32>(x2, y0), 0).r,
                textureLoad(src, vec2<i32>(x2, y1), 0).r,
            ),
        );
    }
    if odd_y {
        let y2 = min(base.y + 2, max_coord.y);
        d = min(
            d,
            min(
                textureLoad(src, vec2<i32>(x0, y2), 0).r,
                textureLoad(src, vec2<i32>(x1, y2), 0).r,
            ),
        );
        if odd_x {
            let x2 = min(base.x + 2, max_coord.x);
            d = min(d, textureLoad(src, vec2<i32>(x2, y2), 0).r);
        }
    }

    return d;
}
";

/// Single-mip conservative reversed-Z Hi-Z downsample entry point (fallback for
/// constrained devices with < 4 storage textures and single odd-intermediate steps).
const HIZ_DOWNSAMPLE_BODY: &str = "
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn hiz_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let src_size = textureDimensions(src);
    let dst_size = max(src_size >> vec2<u32>(1u), vec2<u32>(1u));
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }
    let d = reduce_src(src_size, vec2<i32>(gid.xy));
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(d, 0.0, 0.0, 0.0));
}
";

/// Helper for even-chain multi-mip tiled passes: returns true when `size` has
/// another mip level and neither axis > 1 is odd (so 2x2 reduction needs zero apron).
const HIZ_TILED_COMMON: &str = "
fn can_halve_even(size: vec2<u32>) -> bool {
    return (size.x > 1u || size.y > 1u)
        && (size.x == 1u || (size.x & 1u) == 0u)
        && (size.y == 1u || (size.y & 1u) == 0u);
}
";

/// 2..=4-mip shared-memory tiled Hi-Z downsampler (`@workgroup_size(8, 8)`):
/// handles arbitrary odd `src` on `dst0`, then reduces in shared memory across
/// consecutive even-or-1 intermediate mips without cross-workgroup halos or VRAM round-trips.
const HIZ_TILED4_BODY: &str = "
@group(0) @binding(1) var dst0: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var dst1: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var dst2: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var dst3: texture_storage_2d<r32float, write>;

var<workgroup> tile0: array<f32, 64>;
var<workgroup> tile1: array<f32, 16>;
var<workgroup> tile2: array<f32, 4>;

@compute @workgroup_size(8, 8)
fn hiz_downsample_4(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let src_size = textureDimensions(src);
    let s0 = max(src_size >> vec2<u32>(1u), vec2<u32>(1u));
    let lx = lid.x;
    let ly = lid.y;
    if gid.x < s0.x && gid.y < s0.y {
        let d0 = reduce_src(src_size, vec2<i32>(gid.xy));
        textureStore(dst0, vec2<i32>(gid.xy), vec4<f32>(d0, 0.0, 0.0, 0.0));
        tile0[ly * 8u + lx] = d0;
    }
    workgroupBarrier();

    let do1 = can_halve_even(s0);
    let s1 = max(s0 >> vec2<u32>(1u), vec2<u32>(1u));
    if do1 && lx < 4u && ly < 4u {
        let p = wg.xy * 4u + vec2<u32>(lx, ly);
        if p.x < s1.x && p.y < s1.y {
            let x0 = lx * 2u;
            let x1 = select(x0 + 1u, x0, s0.x == 1u);
            let y0 = ly * 2u;
            let y1 = select(y0 + 1u, y0, s0.y == 1u);
            let d1 = min(
                min(tile0[y0 * 8u + x0], tile0[y0 * 8u + x1]),
                min(tile0[y1 * 8u + x0], tile0[y1 * 8u + x1]),
            );
            textureStore(dst1, vec2<i32>(p), vec4<f32>(d1, 0.0, 0.0, 0.0));
            tile1[ly * 4u + lx] = d1;
        }
    }
    workgroupBarrier();

    let do2 = do1 && can_halve_even(s1);
    let s2 = max(s1 >> vec2<u32>(1u), vec2<u32>(1u));
    if do2 && lx < 2u && ly < 2u {
        let p = wg.xy * 2u + vec2<u32>(lx, ly);
        if p.x < s2.x && p.y < s2.y {
            let x0 = lx * 2u;
            let x1 = select(x0 + 1u, x0, s1.x == 1u);
            let y0 = ly * 2u;
            let y1 = select(y0 + 1u, y0, s1.y == 1u);
            let d2 = min(
                min(tile1[y0 * 4u + x0], tile1[y0 * 4u + x1]),
                min(tile1[y1 * 4u + x0], tile1[y1 * 4u + x1]),
            );
            textureStore(dst2, vec2<i32>(p), vec4<f32>(d2, 0.0, 0.0, 0.0));
            tile2[ly * 2u + lx] = d2;
        }
    }
    workgroupBarrier();

    let do3 = do2 && can_halve_even(s2);
    let s3 = max(s2 >> vec2<u32>(1u), vec2<u32>(1u));
    if do3 && lx == 0u && ly == 0u {
        let p = wg.xy;
        if p.x < s3.x && p.y < s3.y {
            let x1 = select(1u, 0u, s2.x == 1u);
            let y1 = select(1u, 0u, s2.y == 1u);
            let d3 = min(
                min(tile2[0u], tile2[x1]),
                min(tile2[y1 * 2u], tile2[y1 * 2u + x1]),
            );
            textureStore(dst3, vec2<i32>(p), vec4<f32>(d3, 0.0, 0.0, 0.0));
        }
    }
}
";

/// 5-mip shared-memory tiled Hi-Z downsampler (`@workgroup_size(16, 16)`) for
/// devices with >= 8 storage textures per stage (e.g. 2560x1440 -> 80x45 in 1 dispatch).
const HIZ_TILED5_BODY: &str = "
@group(0) @binding(1) var dst0: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var dst1: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var dst2: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var dst3: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var dst4: texture_storage_2d<r32float, write>;

var<workgroup> tile0: array<f32, 256>;
var<workgroup> tile1: array<f32, 64>;
var<workgroup> tile2: array<f32, 16>;
var<workgroup> tile3: array<f32, 4>;

@compute @workgroup_size(16, 16)
fn hiz_downsample_5(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let src_size = textureDimensions(src);
    let s0 = max(src_size >> vec2<u32>(1u), vec2<u32>(1u));
    let lx = lid.x;
    let ly = lid.y;
    if gid.x < s0.x && gid.y < s0.y {
        let d0 = reduce_src(src_size, vec2<i32>(gid.xy));
        textureStore(dst0, vec2<i32>(gid.xy), vec4<f32>(d0, 0.0, 0.0, 0.0));
        tile0[ly * 16u + lx] = d0;
    }
    workgroupBarrier();

    let do1 = can_halve_even(s0);
    let s1 = max(s0 >> vec2<u32>(1u), vec2<u32>(1u));
    if do1 && lx < 8u && ly < 8u {
        let p = wg.xy * 8u + vec2<u32>(lx, ly);
        if p.x < s1.x && p.y < s1.y {
            let x0 = lx * 2u;
            let x1 = select(x0 + 1u, x0, s0.x == 1u);
            let y0 = ly * 2u;
            let y1 = select(y0 + 1u, y0, s0.y == 1u);
            let d1 = min(
                min(tile0[y0 * 16u + x0], tile0[y0 * 16u + x1]),
                min(tile0[y1 * 16u + x0], tile0[y1 * 16u + x1]),
            );
            textureStore(dst1, vec2<i32>(p), vec4<f32>(d1, 0.0, 0.0, 0.0));
            tile1[ly * 8u + lx] = d1;
        }
    }
    workgroupBarrier();

    let do2 = do1 && can_halve_even(s1);
    let s2 = max(s1 >> vec2<u32>(1u), vec2<u32>(1u));
    if do2 && lx < 4u && ly < 4u {
        let p = wg.xy * 4u + vec2<u32>(lx, ly);
        if p.x < s2.x && p.y < s2.y {
            let x0 = lx * 2u;
            let x1 = select(x0 + 1u, x0, s1.x == 1u);
            let y0 = ly * 2u;
            let y1 = select(y0 + 1u, y0, s1.y == 1u);
            let d2 = min(
                min(tile1[y0 * 8u + x0], tile1[y0 * 8u + x1]),
                min(tile1[y1 * 8u + x0], tile1[y1 * 8u + x1]),
            );
            textureStore(dst2, vec2<i32>(p), vec4<f32>(d2, 0.0, 0.0, 0.0));
            tile2[ly * 4u + lx] = d2;
        }
    }
    workgroupBarrier();

    let do3 = do2 && can_halve_even(s2);
    let s3 = max(s2 >> vec2<u32>(1u), vec2<u32>(1u));
    if do3 && lx < 2u && ly < 2u {
        let p = wg.xy * 2u + vec2<u32>(lx, ly);
        if p.x < s3.x && p.y < s3.y {
            let x0 = lx * 2u;
            let x1 = select(x0 + 1u, x0, s2.x == 1u);
            let y0 = ly * 2u;
            let y1 = select(y0 + 1u, y0, s2.y == 1u);
            let d3 = min(
                min(tile2[y0 * 4u + x0], tile2[y0 * 4u + x1]),
                min(tile2[y1 * 4u + x0], tile2[y1 * 4u + x1]),
            );
            textureStore(dst3, vec2<i32>(p), vec4<f32>(d3, 0.0, 0.0, 0.0));
            tile3[ly * 2u + lx] = d3;
        }
    }
    workgroupBarrier();

    let do4 = do3 && can_halve_even(s3);
    let s4 = max(s3 >> vec2<u32>(1u), vec2<u32>(1u));
    if do4 && lx == 0u && ly == 0u {
        let p = wg.xy;
        if p.x < s4.x && p.y < s4.y {
            let x1 = select(1u, 0u, s3.x == 1u);
            let y1 = select(1u, 0u, s3.y == 1u);
            let d4 = min(
                min(tile3[0u], tile3[x1]),
                min(tile3[y1 * 2u], tile3[y1 * 2u + x1]),
            );
            textureStore(dst4, vec2<i32>(p), vec4<f32>(d4, 0.0, 0.0, 0.0));
        }
    }
}
";

/// Ping-pong workgroup storage and exact odd-aware 3x3 reduction helpers for
/// single-workgroup Hi-Z tail passes (`dst0.x * dst0.y <= 2048` => `dst1 <= 1024`).
const HIZ_TAIL_COMMON: &str = "
var<workgroup> tail_a: array<f32, 2048>;
var<workgroup> tail_b: array<f32, 1024>;

fn reduce_tail_a(prev_size: vec2<u32>, dx: u32, dy: u32) -> f32 {
    let sw = prev_size.x;
    let sh = prev_size.y;
    let max_x = sw - 1u;
    let max_y = sh - 1u;
    let bx = dx * 2u;
    let by = dy * 2u;
    let x0 = min(bx, max_x);
    let x1 = min(bx + 1u, max_x);
    let y0 = min(by, max_y);
    let y1 = min(by + 1u, max_y);
    var d = min(
        min(tail_a[y0 * sw + x0], tail_a[y0 * sw + x1]),
        min(tail_a[y1 * sw + x0], tail_a[y1 * sw + x1]),
    );
    let odd_x = sw > 1u && (sw & 1u) != 0u;
    let odd_y = sh > 1u && (sh & 1u) != 0u;
    if odd_x {
        let x2 = min(bx + 2u, max_x);
        d = min(d, min(tail_a[y0 * sw + x2], tail_a[y1 * sw + x2]));
    }
    if odd_y {
        let y2 = min(by + 2u, max_y);
        d = min(d, min(tail_a[y2 * sw + x0], tail_a[y2 * sw + x1]));
        if odd_x {
            let x2 = min(bx + 2u, max_x);
            d = min(d, tail_a[y2 * sw + x2]);
        }
    }
    return d;
}

fn reduce_tail_b(prev_size: vec2<u32>, dx: u32, dy: u32) -> f32 {
    let sw = prev_size.x;
    let sh = prev_size.y;
    let max_x = sw - 1u;
    let max_y = sh - 1u;
    let bx = dx * 2u;
    let by = dy * 2u;
    let x0 = min(bx, max_x);
    let x1 = min(bx + 1u, max_x);
    let y0 = min(by, max_y);
    let y1 = min(by + 1u, max_y);
    var d = min(
        min(tail_b[y0 * sw + x0], tail_b[y0 * sw + x1]),
        min(tail_b[y1 * sw + x0], tail_b[y1 * sw + x1]),
    );
    let odd_x = sw > 1u && (sw & 1u) != 0u;
    let odd_y = sh > 1u && (sh & 1u) != 0u;
    if odd_x {
        let x2 = min(bx + 2u, max_x);
        d = min(d, min(tail_b[y0 * sw + x2], tail_b[y1 * sw + x2]));
    }
    if odd_y {
        let y2 = min(by + 2u, max_y);
        d = min(d, min(tail_b[y2 * sw + x0], tail_b[y2 * sw + x1]));
        if odd_x {
            let x2 = min(bx + 2u, max_x);
            d = min(d, tail_b[y2 * sw + x2]);
        }
    }
    return d;
}
";

/// Single-workgroup odd-aware Hi-Z tail pass writing up to 4 mips (`@workgroup_size(256)`).
const HIZ_TAIL4_BODY: &str = "
@group(0) @binding(1) var dst0: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var dst1: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var dst2: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var dst3: texture_storage_2d<r32float, write>;

@compute @workgroup_size(256)
fn hiz_tail_4(@builtin(local_invocation_index) lane: u32) {
    let src_size = textureDimensions(src);
    let s0 = max(src_size >> vec2<u32>(1u), vec2<u32>(1u));
    let n0 = s0.x * s0.y;
    for (var idx = lane; idx < n0; idx += 256u) {
        let dx = idx % s0.x;
        let dy = idx / s0.x;
        let d = reduce_src(src_size, vec2<i32>(i32(dx), i32(dy)));
        tail_a[idx] = d;
        textureStore(dst0, vec2<i32>(i32(dx), i32(dy)), vec4<f32>(d, 0.0, 0.0, 0.0));
    }
    workgroupBarrier();

    let do1 = s0.x > 1u || s0.y > 1u;
    let s1 = max(s0 >> vec2<u32>(1u), vec2<u32>(1u));
    if do1 {
        let n1 = s1.x * s1.y;
        for (var idx = lane; idx < n1; idx += 256u) {
            let dx = idx % s1.x;
            let dy = idx / s1.x;
            let d = reduce_tail_a(s0, dx, dy);
            tail_b[idx] = d;
            textureStore(dst1, vec2<i32>(i32(dx), i32(dy)), vec4<f32>(d, 0.0, 0.0, 0.0));
        }
    }
    workgroupBarrier();

    let do2 = do1 && (s1.x > 1u || s1.y > 1u);
    let s2 = max(s1 >> vec2<u32>(1u), vec2<u32>(1u));
    if do2 {
        let n2 = s2.x * s2.y;
        for (var idx = lane; idx < n2; idx += 256u) {
            let dx = idx % s2.x;
            let dy = idx / s2.x;
            let d = reduce_tail_b(s1, dx, dy);
            tail_a[idx] = d;
            textureStore(dst2, vec2<i32>(i32(dx), i32(dy)), vec4<f32>(d, 0.0, 0.0, 0.0));
        }
    }
    workgroupBarrier();

    let do3 = do2 && (s2.x > 1u || s2.y > 1u);
    let s3 = max(s2 >> vec2<u32>(1u), vec2<u32>(1u));
    if do3 {
        let n3 = s3.x * s3.y;
        for (var idx = lane; idx < n3; idx += 256u) {
            let dx = idx % s3.x;
            let dy = idx / s3.x;
            let d = reduce_tail_a(s2, dx, dy);
            textureStore(dst3, vec2<i32>(i32(dx), i32(dy)), vec4<f32>(d, 0.0, 0.0, 0.0));
        }
    }
}
";

/// Single-workgroup odd-aware Hi-Z tail pass writing up to 8 mips (`@workgroup_size(256)`),
/// reducing the entire tail (e.g. `60x33 -> 1x1` or `40x22 -> 1x1`) in 1 dispatch with 0 barriers.
const HIZ_TAIL8_BODY: &str = "
@group(0) @binding(1) var dst0: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var dst1: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var dst2: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var dst3: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var dst4: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var dst5: texture_storage_2d<r32float, write>;
@group(0) @binding(7) var dst6: texture_storage_2d<r32float, write>;
@group(0) @binding(8) var dst7: texture_storage_2d<r32float, write>;

@compute @workgroup_size(256)
fn hiz_tail_8(@builtin(local_invocation_index) lane: u32) {
    let src_size = textureDimensions(src);
    let s0 = max(src_size >> vec2<u32>(1u), vec2<u32>(1u));
    let n0 = s0.x * s0.y;
    for (var idx = lane; idx < n0; idx += 256u) {
        let dx = idx % s0.x;
        let dy = idx / s0.x;
        let d = reduce_src(src_size, vec2<i32>(i32(dx), i32(dy)));
        tail_a[idx] = d;
        textureStore(dst0, vec2<i32>(i32(dx), i32(dy)), vec4<f32>(d, 0.0, 0.0, 0.0));
    }
    workgroupBarrier();

    let do1 = s0.x > 1u || s0.y > 1u;
    let s1 = max(s0 >> vec2<u32>(1u), vec2<u32>(1u));
    if do1 {
        let n1 = s1.x * s1.y;
        for (var idx = lane; idx < n1; idx += 256u) {
            let dx = idx % s1.x;
            let dy = idx / s1.x;
            let d = reduce_tail_a(s0, dx, dy);
            tail_b[idx] = d;
            textureStore(dst1, vec2<i32>(i32(dx), i32(dy)), vec4<f32>(d, 0.0, 0.0, 0.0));
        }
    }
    workgroupBarrier();

    let do2 = do1 && (s1.x > 1u || s1.y > 1u);
    let s2 = max(s1 >> vec2<u32>(1u), vec2<u32>(1u));
    if do2 {
        let n2 = s2.x * s2.y;
        for (var idx = lane; idx < n2; idx += 256u) {
            let dx = idx % s2.x;
            let dy = idx / s2.x;
            let d = reduce_tail_b(s1, dx, dy);
            tail_a[idx] = d;
            textureStore(dst2, vec2<i32>(i32(dx), i32(dy)), vec4<f32>(d, 0.0, 0.0, 0.0));
        }
    }
    workgroupBarrier();

    let do3 = do2 && (s2.x > 1u || s2.y > 1u);
    let s3 = max(s2 >> vec2<u32>(1u), vec2<u32>(1u));
    if do3 {
        let n3 = s3.x * s3.y;
        for (var idx = lane; idx < n3; idx += 256u) {
            let dx = idx % s3.x;
            let dy = idx / s3.x;
            let d = reduce_tail_a(s2, dx, dy);
            tail_b[idx] = d;
            textureStore(dst3, vec2<i32>(i32(dx), i32(dy)), vec4<f32>(d, 0.0, 0.0, 0.0));
        }
    }
    workgroupBarrier();

    let do4 = do3 && (s3.x > 1u || s3.y > 1u);
    let s4 = max(s3 >> vec2<u32>(1u), vec2<u32>(1u));
    if do4 {
        let n4 = s4.x * s4.y;
        for (var idx = lane; idx < n4; idx += 256u) {
            let dx = idx % s4.x;
            let dy = idx / s4.x;
            let d = reduce_tail_b(s3, dx, dy);
            tail_a[idx] = d;
            textureStore(dst4, vec2<i32>(i32(dx), i32(dy)), vec4<f32>(d, 0.0, 0.0, 0.0));
        }
    }
    workgroupBarrier();

    let do5 = do4 && (s4.x > 1u || s4.y > 1u);
    let s5 = max(s4 >> vec2<u32>(1u), vec2<u32>(1u));
    if do5 {
        let n5 = s5.x * s5.y;
        for (var idx = lane; idx < n5; idx += 256u) {
            let dx = idx % s5.x;
            let dy = idx / s5.x;
            let d = reduce_tail_a(s4, dx, dy);
            tail_b[idx] = d;
            textureStore(dst5, vec2<i32>(i32(dx), i32(dy)), vec4<f32>(d, 0.0, 0.0, 0.0));
        }
    }
    workgroupBarrier();

    let do6 = do5 && (s5.x > 1u || s5.y > 1u);
    let s6 = max(s5 >> vec2<u32>(1u), vec2<u32>(1u));
    if do6 {
        let n6 = s6.x * s6.y;
        for (var idx = lane; idx < n6; idx += 256u) {
            let dx = idx % s6.x;
            let dy = idx / s6.x;
            let d = reduce_tail_b(s5, dx, dy);
            tail_a[idx] = d;
            textureStore(dst6, vec2<i32>(i32(dx), i32(dy)), vec4<f32>(d, 0.0, 0.0, 0.0));
        }
    }
    workgroupBarrier();

    let do7 = do6 && (s6.x > 1u || s6.y > 1u);
    let s7 = max(s6 >> vec2<u32>(1u), vec2<u32>(1u));
    if do7 {
        let n7 = s7.x * s7.y;
        for (var idx = lane; idx < n7; idx += 256u) {
            let dx = idx % s7.x;
            let dy = idx / s7.x;
            let d = reduce_tail_a(s6, dx, dy);
            textureStore(dst7, vec2<i32>(i32(dx), i32(dy)), vec4<f32>(d, 0.0, 0.0, 0.0));
        }
    }
}
";

fn hiz_downsample_source() -> String {
    [HIZ_REDUCE_SRC, HIZ_DOWNSAMPLE_BODY].concat()
}

fn hiz_tiled4_source() -> String {
    [HIZ_REDUCE_SRC, HIZ_TILED_COMMON, HIZ_TILED4_BODY].concat()
}

fn hiz_tiled5_source() -> String {
    [HIZ_REDUCE_SRC, HIZ_TILED_COMMON, HIZ_TILED5_BODY].concat()
}

fn hiz_tail4_source() -> String {
    [HIZ_REDUCE_SRC, HIZ_TAIL_COMMON, HIZ_TAIL4_BODY].concat()
}

fn hiz_tail8_source() -> String {
    [HIZ_REDUCE_SRC, HIZ_TAIL_COMMON, HIZ_TAIL8_BODY].concat()
}

/// Compute Hi-Z box occlusion test appended to `cull.wgsl` when `DepthTest::Compute`
/// is active: 1 thread per box (8 `mat4 * vec4` transforms instead of 324, zero
/// full-resolution fragment overdraw). `depth` at group 1 binding 0 holds the full
/// mip chain of the reversed-Z min-depth Hi-Z pyramid.
const COMPUTE_TEST: &str = "
fn mark_seen(t: u32) {
    results[t] = params.stamp;
    let test = tests[t].data;
    for (var k = 0u; k < (test.z & 0xffffu); k++) {
        phase2[(test.y + k) * 5u + 1u] = 1u;
    }
    for (var k = 0u; k < (test.z >> 16u); k++) {
        phase2[(test.w + k) * 5u + 1u] = 1u;
    }
}

@compute @workgroup_size(64)
fn cs_box_test(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let t = (group.y * 65535u + group.x) * 64u + lane;
    if t >= params.count {
        return;
    }
    let test = tests[t];
    let origin = origins[test.data.x].xyz;
    let low = test.low.xyz;
    let high = test.high.xyz;

    // Camera inside (or within a block of) the box: conservatively visible.
    if all(origin < vec3(1.0) - low) && all(origin > vec3(-1.0) - high) {
        mark_seen(t);
        return;
    }

    var lo = vec2(1e30);
    var hi = vec2(-1e30);
    var max_z = -1e30;
    for (var c = 0u; c < 8u; c++) {
        let p = globals.view_proj * vec4(corner(origin, low, high, c), 1.0);
        // Any corner behind the camera, on/past the near plane (reversed-Z: z/w >= 1),
        // or NaN cannot be bounded by projecting the 8 corners: conservatively visible.
        if !(p.w > 1e-6 && p.z < p.w) {
            mark_seen(t);
            return;
        }
        let inv_w = 1.0 / p.w;
        let ndc = p.xy * inv_w;
        let z = p.z * inv_w;
        lo = min(lo, ndc);
        hi = max(hi, ndc);
        max_z = max(max_z, z);
    }

    // Entire box lies strictly outside the screen or beyond the far plane (reversed-Z < 0).
    if hi.x < -1.0 || lo.x > 1.0 || hi.y < -1.0 || lo.y > 1.0 || max_z < 0.0 {
        return;
    }

    let ndc_lo = max(lo, vec2(-1.0));
    let ndc_hi = min(hi, vec2(1.0));
    let uv_min = vec2(ndc_lo.x * 0.5 + 0.5, 0.5 - ndc_hi.y * 0.5);
    let uv_max = vec2(ndc_hi.x * 0.5 + 0.5, 0.5 - ndc_lo.y * 0.5);

    let max_mip = i32(textureNumLevels(depth)) - 1;
    let hiz_base = textureDimensions(depth, 0);
    let hiz_size = vec2<f32>(hiz_base);
    let rect = max((uv_max - uv_min) * hiz_size, vec2(0.0));
    let max_extent = max(rect.x, rect.y);
    var mip = clamp(i32(ceil(log2(max(max_extent, 1.0)))), 0, max_mip);

    // Derive mip dimensions from mip 0 by integer shift rather than dynamic
    // textureDimensions(depth, mip): avoids extra descriptor queries and prevents
    // non-uniform OpImageQuerySizeLod lane miscompiles on Mesa llvmpipe.
    // Step one mip finer whenever the projected box still fits within a 2x2 footprint.
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

    let d00 = textureLoad(depth, vec2<i32>(lo_px.x, lo_px.y), mip).r;
    let d10 = textureLoad(depth, vec2<i32>(hi_px.x, lo_px.y), mip).r;
    let d01 = textureLoad(depth, vec2<i32>(lo_px.x, hi_px.y), mip).r;
    let d11 = textureLoad(depth, vec2<i32>(hi_px.x, hi_px.y), mip).r;
    let occluder_min_z = min(min(d00, d10), min(d01, d11));

    if !(max_z < occluder_min_z) {
        mark_seen(t);
    }
}
";

/// Mip dimensions for a Hi-Z pyramid built from a depth buffer of `depth_size`:
/// mip 0 is half resolution (`max(1, size / 2)`), halving with floor down to `1x1`,
/// matching wgpu's `TextureDescriptor` mip chain.
pub(crate) fn hiz_mip_dimensions(depth_size: [u32; 2]) -> Vec<[u32; 2]> {
    let mut w = (depth_size[0] / 2).max(1);
    let mut h = (depth_size[1] / 2).max(1);
    let mut dims = vec![[w, h]];
    while w > 1 || h > 1 {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        dims.push([w, h]);
    }
    dims
}

/// Device capabilities controlling which multi-mip Hi-Z downsample kernels are enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HizCaps {
    /// `max_storage_textures >= 4` and workgroup storage fits `hiz_downsample_4`.
    pub(crate) tiled4: bool,
    /// `tiled4` + 256-thread workgroups + 12 KiB LDS for `hiz_tail_4`.
    pub(crate) tail4: bool,
    /// `tail4` + `max_storage_textures >= 8` + 16x16 workgroups for `hiz_downsample_5` / `hiz_tail_8`.
    pub(crate) wide8: bool,
}

impl HizCaps {
    fn from_limits(limits: &wgpu::Limits) -> Self {
        let tiled4 = limits.max_storage_textures_per_shader_stage >= 4
            && limits.max_compute_workgroup_storage_size >= HIZ_TILED4_SHARED_BYTES;
        let tail4 = tiled4
            && limits.max_compute_workgroup_size_x >= WORKGROUP_TAIL
            && limits.max_compute_invocations_per_workgroup >= WORKGROUP_TAIL
            && limits.max_compute_workgroup_storage_size >= HIZ_TAIL_SHARED_BYTES;
        let wide8 = tail4
            && limits.max_storage_textures_per_shader_stage >= 8
            && limits.max_compute_workgroup_size_y >= WORKGROUP_TILED5;
        Self {
            tiled4,
            tail4,
            wide8,
        }
    }
}

/// Which Hi-Z downsample compute pipeline a planned dispatch step invokes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HizStepKind {
    /// 1-mip direct downsample (`hiz_downsample`, 8x8 workgroup).
    Single,
    /// 2..=4-mip shared-memory tiled downsample (`hiz_downsample_4`, 8x8 workgroup).
    Tiled4,
    /// 5-mip shared-memory tiled downsample (`hiz_downsample_5`, 16x16 workgroup).
    Tiled5,
    /// 2..=4-mip single-workgroup odd-aware tail (`hiz_tail_4`, 256-thread workgroup).
    Tail4,
    /// 5..=8-mip single-workgroup odd-aware tail (`hiz_tail_8`, 256-thread workgroup).
    Tail8,
}

/// One planned Hi-Z downsample dispatch covering `first_mip..first_mip + mip_count`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HizStepSpec {
    pub(crate) kind: HizStepKind,
    pub(crate) first_mip: usize,
    pub(crate) mip_count: usize,
    pub(crate) workgroups: [u32; 2],
}

/// Plans the minimum-dispatch sequence of multi-mip shared-memory Hi-Z reductions
/// for `mip_sizes` under `caps`, preserving exact conservative odd-dimension coverage.
pub(crate) fn plan_hiz_steps(mip_sizes: &[[u32; 2]], caps: HizCaps) -> Vec<HizStepSpec> {
    let mut steps = Vec::new();
    let n = mip_sizes.len();
    let mut i = 0;
    while i < n {
        let [nw, nh] = mip_sizes[i];
        let texels = u64::from(nw) * u64::from(nh);
        if caps.tail4 && (n - i) >= 2 && texels <= u64::from(HIZ_TAIL_MAX_TEXELS) {
            let max_out = if caps.wide8 { 8 } else { 4 };
            let take = (n - i).min(max_out);
            let kind = if caps.wide8 && take > 4 {
                HizStepKind::Tail8
            } else {
                HizStepKind::Tail4
            };
            steps.push(HizStepSpec {
                kind,
                first_mip: i,
                mip_count: take,
                workgroups: [1, 1],
            });
            i += take;
            continue;
        }
        if caps.tiled4 {
            let max_tiled = if caps.wide8 { 5 } else { 4 };
            let mut take = 1;
            while take < max_tiled && i + take < n {
                let [pw, ph] = mip_sizes[i + take - 1];
                let odd_x = pw > 1 && !pw.is_multiple_of(2);
                let odd_y = ph > 1 && !ph.is_multiple_of(2);
                if !odd_x && !odd_y {
                    take += 1;
                } else {
                    break;
                }
            }
            let (kind, edge) = match take {
                5 => (HizStepKind::Tiled5, WORKGROUP_TILED5),
                2..=4 => (HizStepKind::Tiled4, WORKGROUP_2D),
                _ => (HizStepKind::Single, WORKGROUP_2D),
            };
            steps.push(HizStepSpec {
                kind,
                first_mip: i,
                mip_count: take,
                workgroups: [nw.div_ceil(edge), nh.div_ceil(edge)],
            });
            i += take;
        } else {
            steps.push(HizStepSpec {
                kind: HizStepKind::Single,
                first_mip: i,
                mip_count: 1,
                workgroups: [nw.div_ceil(WORKGROUP_2D), nh.div_ceil(WORKGROUP_2D)],
            });
            i += 1;
        }
    }
    steps
}

struct HizStep {
    spec: HizStepSpec,
    group: wgpu::BindGroup,
}

/// One allocated reversed-Z min-depth Hi-Z pyramid matching a depth texture size.
struct HizPyramid {
    depth_size: [u32; 2],
    depth_view: wgpu::TextureView,
    level_views: Vec<wgpu::TextureView>,
    steps: Vec<HizStep>,
    read_group: wgpu::BindGroup,
}

/// Layouts, pipelines, and distinct 1x1 dummy storage targets for multi-mip Hi-Z passes.
struct HizPipelines {
    caps: HizCaps,
    single_layout: wgpu::BindGroupLayout,
    single_pipeline: wgpu::ComputePipeline,
    multi4_layout: Option<wgpu::BindGroupLayout>,
    tiled4_pipeline: Option<wgpu::ComputePipeline>,
    tail4_pipeline: Option<wgpu::ComputePipeline>,
    tiled5_layout: Option<wgpu::BindGroupLayout>,
    tiled5_pipeline: Option<wgpu::ComputePipeline>,
    tail8_layout: Option<wgpu::BindGroupLayout>,
    tail8_pipeline: Option<wgpu::ComputePipeline>,
    /// Distinct 1x1 `R32Float` storage views bound to unused `dst1..dst{M-1}` slots so
    /// no two `WriteOnly` storage texture bindings ever alias the same subresource.
    dummy_views: Vec<wgpu::TextureView>,
}

impl HizPipelines {
    fn make_step_group(
        &self,
        device: &wgpu::Device,
        spec: HizStepSpec,
        src: &wgpu::TextureView,
        level_views: &[wgpu::TextureView],
    ) -> wgpu::BindGroup {
        let (layout, slots) = match spec.kind {
            HizStepKind::Single => (&self.single_layout, 1usize),
            HizStepKind::Tiled4 | HizStepKind::Tail4 => (
                self.multi4_layout.as_ref().expect("Hi-Z multi4 layout"),
                4usize,
            ),
            HizStepKind::Tiled5 => (
                self.tiled5_layout.as_ref().expect("Hi-Z tiled5 layout"),
                5usize,
            ),
            HizStepKind::Tail8 => (
                self.tail8_layout.as_ref().expect("Hi-Z tail8 layout"),
                8usize,
            ),
        };
        let mut entries = Vec::with_capacity(1 + slots);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(src),
        });
        for slot in 0..slots {
            let view = if slot < spec.mip_count {
                &level_views[spec.first_mip + slot]
            } else {
                &self.dummy_views[slot - 1]
            };
            entries.push(wgpu::BindGroupEntry {
                binding: (slot + 1) as u32,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hi-Z downsample"),
            layout,
            entries: &entries,
        })
    }

    fn pipeline(&self, kind: HizStepKind) -> &wgpu::ComputePipeline {
        match kind {
            HizStepKind::Single => &self.single_pipeline,
            HizStepKind::Tiled4 => self.tiled4_pipeline.as_ref().expect("Hi-Z tiled4 pipeline"),
            HizStepKind::Tiled5 => self.tiled5_pipeline.as_ref().expect("Hi-Z tiled5 pipeline"),
            HizStepKind::Tail4 => self.tail4_pipeline.as_ref().expect("Hi-Z tail4 pipeline"),
            HizStepKind::Tail8 => self.tail8_pipeline.as_ref().expect("Hi-Z tail8 pipeline"),
        }
    }
}

/// Compute downsample pipelines and cached Hi-Z pyramid shared by `Cull` and `GpuCulling`.
/// `bind_depth` calls to keep the 1x1 zeroed Hi-Z fallback active before
/// retrying pyramid allocation at the same depth size after an out-of-memory
/// failure (starting delay and cap).
const HIZ_OOM_RETRY_MIN: u32 = 60;
const HIZ_OOM_RETRY_MAX: u32 = 240;

struct HizOomBackoff {
    size: [u32; 2],
    /// Remaining `bind_depth` calls at `size` that keep the fallback active.
    wait: u32,
    /// Delay applied if the next attempt at `size` also runs out of memory.
    next_wait: u32,
}

struct Hiz {
    pipelines: HizPipelines,
    read_layout: wgpu::BindGroupLayout,
    /// 1x1 zero-cleared `R32Float` fallback read group used if the full pyramid
    /// cannot be allocated under video-memory exhaustion (#13624): depth 0.0 is the
    /// reversed-Z far plane, so every in-frustum box stays conservatively visible.
    fallback_read_group: wgpu::BindGroup,
    pyramid: Option<HizPyramid>,
    oom_backoff: Option<HizOomBackoff>,
}

fn make_hiz_layout(device: &wgpu::Device, label: &str, dst_count: u32) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(1 + dst_count as usize);
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    });
    for slot in 0..dst_count {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 1 + slot,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::R32Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn make_hiz_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    source: &str,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        ..Default::default()
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

impl Hiz {
    fn new(device: &wgpu::Device) -> Result<Self, String> {
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let caps = HizCaps::from_limits(&device.limits());
        let single_layout = make_hiz_layout(device, "Hi-Z downsample", 1);
        let single_pipeline = make_hiz_pipeline(
            device,
            "Hi-Z downsample",
            &single_layout,
            &hiz_downsample_source(),
            "hiz_downsample",
        );

        let (multi4_layout, tiled4_pipeline, tail4_pipeline) = if caps.tiled4 {
            let layout = make_hiz_layout(device, "Hi-Z multi4", 4);
            let tiled4 = make_hiz_pipeline(
                device,
                "Hi-Z tiled4",
                &layout,
                &hiz_tiled4_source(),
                "hiz_downsample_4",
            );
            let tail4 = caps.tail4.then(|| {
                make_hiz_pipeline(
                    device,
                    "Hi-Z tail4",
                    &layout,
                    &hiz_tail4_source(),
                    "hiz_tail_4",
                )
            });
            (Some(layout), Some(tiled4), tail4)
        } else {
            (None, None, None)
        };

        let (tiled5_layout, tiled5_pipeline, tail8_layout, tail8_pipeline) = if caps.wide8 {
            let t5_layout = make_hiz_layout(device, "Hi-Z tiled5", 5);
            let t5_pipe = make_hiz_pipeline(
                device,
                "Hi-Z tiled5",
                &t5_layout,
                &hiz_tiled5_source(),
                "hiz_downsample_5",
            );
            let t8_layout = make_hiz_layout(device, "Hi-Z tail8", 8);
            let t8_pipe = make_hiz_pipeline(
                device,
                "Hi-Z tail8",
                &t8_layout,
                &hiz_tail8_source(),
                "hiz_tail_8",
            );
            (
                Some(t5_layout),
                Some(t5_pipe),
                Some(t8_layout),
                Some(t8_pipe),
            )
        } else {
            (None, None, None, None)
        };

        let dummy_count = if caps.wide8 {
            7
        } else if caps.tiled4 {
            3
        } else {
            0
        };
        let dummy_views: Vec<wgpu::TextureView> = (0..dummy_count)
            .map(|_| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("Hi-Z unused slot"),
                        size: wgpu::Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::R32Float,
                        usage: wgpu::TextureUsages::STORAGE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            })
            .collect();

        let read_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Hi-Z read"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let fallback = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Hi-Z fallback"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let fallback_read_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hi-Z fallback read"),
            layout: &read_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&fallback),
            }],
        });

        let memory = pollster::block_on(memory.pop()).map(|e| e.to_string());
        let internal = pollster::block_on(internal.pop()).map(|e| e.to_string());
        let validation = pollster::block_on(validation.pop()).map(|e| e.to_string());
        #[cfg(test)]
        let memory = memory.or_else(|| {
            let left = super::frame_targets::NO_ROOM.get().checked_sub(1)?;
            super::frame_targets::NO_ROOM.set(left);
            Some("no room (test)".to_owned())
        });
        if let Some(e) = memory.or(internal).or(validation) {
            return Err(e);
        }
        Ok(Self {
            pipelines: HizPipelines {
                caps,
                single_layout,
                single_pipeline,
                multi4_layout,
                tiled4_pipeline,
                tail4_pipeline,
                tiled5_layout,
                tiled5_pipeline,
                tail8_layout,
                tail8_pipeline,
                dummy_views,
            },
            read_layout,
            fallback_read_group,
            pyramid: None,
            oom_backoff: None,
        })
    }

    fn forget_depth(&mut self) {
        self.pyramid = None;
        self.oom_backoff = None;
    }

    fn note_oom(&mut self, size: [u32; 2]) {
        let wait = match &self.oom_backoff {
            Some(b) if b.size == size => b.next_wait,
            _ => HIZ_OOM_RETRY_MIN,
        };
        let next_wait = wait.saturating_mul(2).min(HIZ_OOM_RETRY_MAX);
        self.oom_backoff = Some(HizOomBackoff {
            size,
            wait,
            next_wait,
        });
    }

    fn bind_depth(&mut self, device: &wgpu::Device, depth: &wgpu::TextureView) {
        let texture = depth.texture();
        if texture.sample_count() != 1 || texture.width() == 0 || texture.height() == 0 {
            self.pyramid = None;
            self.oom_backoff = None;
            return;
        }
        let depth_size = [texture.width(), texture.height()];
        let mut rebind_failed = false;
        let pipelines = &self.pipelines;
        if let Some(p) = &mut self.pyramid {
            if p.depth_view == *depth {
                return;
            }
            if p.depth_size == depth_size {
                // Scope same-size depth rebind under OOM/validation and cache the
                // bind group only on success; on failure fall back to zero depth so
                // subsequent frames never reuse a failed or stale binding.
                if let Some(group) =
                    super::frame_targets::made(device, super::frame_targets::Kind::Buffers, || {
                        pipelines.make_step_group(device, p.steps[0].spec, depth, &p.level_views)
                    })
                {
                    p.steps[0].group = group;
                    p.depth_view = depth.clone();
                    return;
                }
                rebind_failed = true;
            }
        }
        if rebind_failed {
            self.pyramid = None;
            self.note_oom(depth_size);
            return;
        }
        let clear_backoff = match &mut self.oom_backoff {
            Some(backoff) if backoff.size == depth_size => {
                if backoff.wait > 0 {
                    backoff.wait -= 1;
                    return;
                }
                false
            }
            Some(_) => true,
            None => false,
        };
        if clear_backoff {
            self.oom_backoff = None;
        }
        // Release the previous pyramid before allocating the new one so old and new
        // video memory are never held together (#13624).
        self.pyramid = None;
        let mip_sizes = hiz_mip_dimensions(depth_size);
        let step_specs = plan_hiz_steps(&mip_sizes, self.pipelines.caps);
        let pipelines = &self.pipelines;
        let read_layout = &self.read_layout;
        self.pyramid =
            super::frame_targets::made(device, super::frame_targets::Kind::Buffers, || {
                let pyramid_tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Hi-Z depth pyramid"),
                    size: wgpu::Extent3d {
                        width: mip_sizes[0][0],
                        height: mip_sizes[0][1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: mip_sizes.len() as u32,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | HIZ_TEST_READ,
                    view_formats: &[],
                });
                let all_view = pyramid_tex.create_view(&Default::default());
                let level_views: Vec<wgpu::TextureView> = (0..mip_sizes.len() as u32)
                    .map(|mip| {
                        pyramid_tex.create_view(&wgpu::TextureViewDescriptor {
                            base_mip_level: mip,
                            mip_level_count: Some(1),
                            ..Default::default()
                        })
                    })
                    .collect();
                let steps: Vec<HizStep> = step_specs
                    .iter()
                    .copied()
                    .map(|spec| {
                        let src = if spec.first_mip == 0 {
                            depth
                        } else {
                            &level_views[spec.first_mip - 1]
                        };
                        let group = pipelines.make_step_group(device, spec, src, &level_views);
                        HizStep { spec, group }
                    })
                    .collect();
                let read_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Hi-Z pyramid read"),
                    layout: read_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&all_view),
                    }],
                });
                HizPyramid {
                    depth_size,
                    depth_view: depth.clone(),
                    level_views,
                    steps,
                    read_group,
                }
            });
        if self.pyramid.is_some() {
            self.oom_backoff = None;
        } else {
            self.note_oom(depth_size);
        }
    }

    fn downsample(&self, pass: &mut wgpu::ComputePass<'_>) {
        let Some(p) = &self.pyramid else {
            return;
        };
        let mut active: Option<HizStepKind> = None;
        for step in &p.steps {
            if active != Some(step.spec.kind) {
                pass.set_pipeline(self.pipelines.pipeline(step.spec.kind));
                active = Some(step.spec.kind);
            }
            pass.set_bind_group(0, &step.group, &[]);
            pass.dispatch_workgroups(step.spec.workgroups[0], step.spec.workgroups[1], 1);
        }
    }

    fn read_group(&self) -> &wgpu::BindGroup {
        self.pyramid
            .as_ref()
            .map_or(&self.fallback_read_group, |p| &p.read_group)
    }
}

/// The test's depth texture binding (`InShader`), for the view it was made with.
struct DepthGroup {
    view: wgpu::TextureView,
    group: wgpu::BindGroup,
}

pub struct Cull {
    pipeline: Option<wgpu::RenderPipeline>,
    compute_pipeline: Option<wgpu::ComputePipeline>,
    hiz: Option<Hiz>,
    layout: wgpu::BindGroupLayout,
    group: Option<Group>,
    depth_test: DepthTest,
    depth_layout: Option<wgpu::BindGroupLayout>,
    depth_group: Option<DepthGroup>,
    params: wgpu::Buffer,
    /// Phase 2's draws: the solid draws after phase 1's in the frame's list.
    pub phase2: wgpu::Buffer,
    phase2_capacity: u64,
    tests: wgpu::Buffer,
    tests_capacity: u64,
    /// This frame's tests: visible index, then its phase 2 draws in each
    /// solid pass (`place`): the first of the first pass's, both counts (the
    /// second's in the high half), the first of the second pass's; and its box.
    test_data: Vec<Test>,
    test_slots: Vec<u32>,
    test_indices: Vec<u32>,
    total: u64,
    /// The frame tests every section, as culling was off.
    probe: bool,
    results: wgpu::Buffer,
    readbacks: Vec<Readback>,
    next: usize,
    current: Option<usize>,
    /// Per resident slot: whether the last result read back saw it; and per
    /// far tile's slot (`TILE`).
    seen: Vec<bool>,
    seen_tiles: Vec<bool>,
    stamp: u32,
    /// Of the boxes the last test read back drew, how many it did not see
    /// and how many it drew, and the share of solid indices hidden.
    pub hidden: ([usize; 2], f32),
    /// Cull every frame, whatever the hidden share (A/B checks).
    pub always: bool,
    /// Frames with fewer sections in view than this do not cull.
    pub min_sections: usize,
}

fn buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

const TEST_READ: wgpu::BufferUsages = if cfg!(test) {
    wgpu::BufferUsages::COPY_SRC
} else {
    wgpu::BufferUsages::empty()
};
const PHASE2_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::INDIRECT
    .union(wgpu::BufferUsages::STORAGE)
    .union(wgpu::BufferUsages::COPY_DST)
    .union(TEST_READ);
const TESTS_USAGE: wgpu::BufferUsages =
    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST);
const RESULTS_USAGE: wgpu::BufferUsages =
    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC);
const READBACK_USAGE: wgpu::BufferUsages =
    wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST);

/// The first `n` values of a packed lookup in WGSL `source` (`fn name`, as cull.wgsl's
/// `box_triangle_corner`): four bits each, eight a word, its words in the order of its cases.
#[cfg(test)]
pub(super) fn packed_lookup(source: &str, name: &str, n: usize) -> Vec<u32> {
    let body = &source[source.find(&format!("fn {name}(")).expect("the lookup")..];
    let body = &body[..body.find("return").expect("its return")];
    let words: Vec<u32> = body
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter_map(|t| t.strip_prefix("0x")?.strip_suffix('u'))
        .map(|hex| u32::from_str_radix(hex, 16).expect("a hex word"))
        .collect();
    (0..n)
        .map(|v| (words[v / 8] >> (v % 8 * 4)) & 0xf)
        .collect()
}

/// Qualcomm's PCI vendor id, which wgpu reports for Adreno GPUs.
const ADRENO: u32 = 0x5143;

/// Mesa's open-source Nouveau driver on OpenGL (NVC0/Fermi through Pascal):
/// its Gallium driver (`nvc0`) suffers from kernel pushbuf rejections and
/// channel kills when fragment shaders write to storage buffers alongside early
/// depth tests and indirect execution (#15633).
pub(crate) fn is_nouveau(info: &wgpu::AdapterInfo) -> bool {
    info.backend == wgpu::Backend::Gl
        && (info.driver_info.contains("Mesa") || info.driver.contains("Mesa"))
        && (info.name.starts_with("NV")
            || info.name.to_lowercase().contains("nouveau")
            || info.driver.to_lowercase().contains("nouveau"))
}

/// Whether the adapter and device can run the compute Hi-Z pyramid and box test.
pub(super) fn compute_hiz_supported(adapter: &wgpu::Adapter, device: &wgpu::Device) -> bool {
    let info = adapter.get_info();
    // OpenGL / WebGL backends lack reliable `textureLoad` on `Depth32Float` and `R32Float`
    // storage textures; keep them on the raster fallback.
    if !matches!(
        info.backend,
        wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Dx12
    ) {
        return false;
    }
    if !adapter
        .get_downlevel_capabilities()
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    {
        return false;
    }
    let limits = device.limits();
    if limits.max_storage_buffers_per_shader_stage < 4
        || limits.max_storage_textures_per_shader_stage < 1
        || limits.max_sampled_textures_per_shader_stage < 1
        || limits.max_compute_workgroup_size_x < WORKGROUP_1D
        || limits.max_compute_workgroup_size_y < WORKGROUP_2D
        || limits.max_compute_invocations_per_workgroup < WORKGROUP_1D
    {
        return false;
    }
    let r32f = adapter.get_texture_format_features(wgpu::TextureFormat::R32Float);
    let depth32f = adapter.get_texture_format_features(wgpu::TextureFormat::Depth32Float);
    r32f.allowed_usages
        .contains(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING)
        && r32f
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY)
        && depth32f
            .allowed_usages
            .contains(wgpu::TextureUsages::TEXTURE_BINDING)
}

impl Cull {
    /// The device feature culling needs, where the adapter has it.
    pub fn features(adapter: &wgpu::Adapter) -> wgpu::Features {
        adapter.features() & wgpu::Features::SHADER_EARLY_DEPTH_TEST
    }
    /// The raster occlusion backend this adapter and device support (`Early` on
    /// Vulkan/GL with early depth tests, `InShader` on Metal/DX12/Adreno), if any.
    pub fn supported_raster(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Option<DepthTest> {
        if !device
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
        {
            return None;
        }
        let info = adapter.get_info();
        let depth_test = match info.backend {
            // Adreno's Vulkan driver (the AYANEO Pocket DS's Adreno 740, build 0676.41) now and
            // then leaves sections out with the early test, the world flashing black or
            // see-through as a block breaks; the test in the shader, as on Metal, holds.
            wgpu::Backend::Vulkan if info.vendor == ADRENO => DepthTest::InShader,
            // Mesa's Nouveau OpenGL driver crashes with kernel pushbuf rejections and SIGSEGV
            // when fragment shaders write to storage buffers with early depth tests and indirect
            // execution (#15633).
            wgpu::Backend::Gl if is_nouveau(&info) => return None,
            wgpu::Backend::Vulkan | wgpu::Backend::Gl
                if device
                    .features()
                    .contains(wgpu::Features::SHADER_EARLY_DEPTH_TEST) =>
            {
                DepthTest::Early
            }
            // DirectX 12 has no forced early depth test in wgpu either: the test runs in the
            // shader there too.
            wgpu::Backend::Metal | wgpu::Backend::Dx12 => DepthTest::InShader,
            _ => return None,
        };
        adapter
            .get_downlevel_capabilities()
            .flags
            .contains(
                wgpu::DownlevelFlags::VERTEX_STORAGE
                    | wgpu::DownlevelFlags::FRAGMENT_WRITABLE_STORAGE,
            )
            .then_some(depth_test)
    }
    /// Where the device can run the test, if it can: compute Hi-Z on Vulkan,
    /// Metal and DirectX 12 when compute + `R32Float` storage + depth texture
    /// reads are supported; otherwise raster fallback (`Early` on Vulkan/GL with
    /// early depth tests, `InShader` on Metal/DX12/Adreno). All paths require
    /// `INDIRECT_FIRST_INSTANCE` for phase 2's indirect draws.
    pub fn supported(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Option<DepthTest> {
        if !device
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
        {
            return None;
        }
        if compute_hiz_supported(adapter, device) {
            return Some(DepthTest::Compute);
        }
        Self::supported_raster(adapter, device)
    }
    /// Builds an occlusion culler trying `primary` first, and when `primary` is
    /// [`DepthTest::Compute`] and compute or Hi-Z setup fails, falling back to
    /// the adapter's supported raster backend (`Early`, then `InShader` where
    /// depth sampling is supported) instead of disabling section occlusion.
    pub fn pick(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        color: wgpu::TextureFormat,
        primary: DepthTest,
    ) -> Option<Self> {
        if let Some(cull) = Self::new(device, color, primary) {
            return Some(cull);
        }
        if primary != DepthTest::Compute {
            return None;
        }
        let raster = Self::supported_raster(adapter, device)?;
        if let Some(cull) = Self::new(device, color, raster) {
            return Some(cull);
        }
        if raster == DepthTest::Early
            && adapter.get_info().backend != wgpu::Backend::Gl
            && adapter
                .get_texture_format_features(wgpu::TextureFormat::Depth32Float)
                .allowed_usages
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        {
            return Self::new(device, color, DepthTest::InShader);
        }
        None
    }
    /// Its test runs in phase 1's pass, the depth test forced early (Vulkan / GL fallback).
    pub fn early(&self) -> bool {
        self.depth_test == DepthTest::Early
    }
    /// Layout for binding the full mip chain of the Hi-Z pyramid in compute (`@binding(0)`),
    /// shared with GPU culling's compute occlusion test when `DepthTest::Compute` is active.
    pub fn hiz_read_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.hiz.as_ref().map(|h| &h.read_layout)
    }
    /// Bind group for reading the current frame's Hi-Z pyramid (or the 1x1 zeroed fallback on
    /// OOM), for GPU culling's compute occlusion test.
    pub fn hiz_read_group(&self) -> Option<&wgpu::BindGroup> {
        self.hiz.as_ref().map(Hiz::read_group)
    }
    /// Records the reversed-Z min-depth Hi-Z pyramid downsampling into an active
    /// compute pass so callers can coalesce Hi-Z generation with box tests.
    pub fn downsample_hiz(&self, pass: &mut wgpu::ComputePass<'_>) {
        if let Some(hiz) = &self.hiz {
            hiz.downsample(pass);
        }
    }

    /// Builds the reversed-Z min-depth Hi-Z pyramid from phase 1's depth, for GPU culling's
    /// compute occlusion test.
    #[cfg(test)]
    pub fn build_hiz(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timing: Option<&mut super::pack_timing::PassTiming>,
    ) {
        let Some(hiz) = &self.hiz else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Hi-Z depth pyramid"),
            timestamp_writes: timing.and_then(|t| t.compute_writes("Hi-Z pyramid")),
        });
        hiz.downsample(&mut pass);
    }
    /// `color` is the world pass's colour format. `None` where the device will not
    /// make the pass, the error caught rather than raised (a raised error is
    /// fatal): a driver can refuse it while `wgpu::Limits` promise the storage
    /// blocks it binds, as Mesa's r600 OpenGL driver reports no vertex shader
    /// storage blocks at all (#17994).
    pub fn new(
        device: &wgpu::Device,
        color: wgpu::TextureFormat,
        depth_test: DepthTest,
    ) -> Option<Self> {
        match Self::try_new(device, color, depth_test) {
            Ok(cull) => Some(cull),
            Err(e) => {
                log::warn!("Horizon renderer: occlusion culling unavailable ({e})");
                None
            }
        }
    }

    pub(super) fn try_new(
        device: &wgpu::Device,
        color: wgpu::TextureFormat,
        depth_test: DepthTest,
    ) -> Result<Self, String> {
        let hiz = if depth_test == DepthTest::Compute {
            Some(Hiz::new(device)?)
        } else {
            None
        };
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty,
            count: None,
        };
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let (pipeline, compute_pipeline, layout, depth_layout) = if let Some(hiz) = &hiz {
            let compute = wgpu::ShaderStages::COMPUTE;
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Section culling (compute)"),
                entries: &[
                    entry(0, compute, storage(true)),
                    entry(1, compute, storage(false)),
                    entry(2, compute, storage(false)),
                    entry(3, compute, storage(true)),
                    entry(4, compute, uniform),
                    entry(5, compute, uniform),
                ],
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Section culling (compute)"),
                source: wgpu::ShaderSource::Wgsl(source(depth_test).into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Section culling (compute)"),
                bind_group_layouts: &[Some(&layout), Some(&hiz.read_layout)],
                ..Default::default()
            });
            let compute_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Section culling (compute)"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("cs_box_test"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            (None, Some(compute_pipeline), layout, None)
        } else {
            let both = wgpu::ShaderStages::VERTEX_FRAGMENT;
            let vertex = wgpu::ShaderStages::VERTEX;
            let fragment = wgpu::ShaderStages::FRAGMENT;
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Section culling"),
                entries: &[
                    entry(0, both, storage(true)),
                    entry(1, fragment, storage(false)),
                    entry(2, fragment, storage(false)),
                    entry(3, vertex, storage(true)),
                    entry(4, vertex, uniform),
                    entry(5, both, uniform),
                ],
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Section culling"),
                source: wgpu::ShaderSource::Wgsl(source(depth_test).into()),
            });
            let depth_layout = (depth_test == DepthTest::InShader).then(|| {
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Section culling depth"),
                    entries: &[entry(
                        0,
                        fragment,
                        wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                    )],
                })
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Section culling"),
                bind_group_layouts: &[Some(&layout), depth_layout.as_ref()],
                ..Default::default()
            });
            // In the shader (`InShader`): no colour target, and the depth
            // attached read only (a pass needs an attachment), its test left to
            // the shader.
            let color_target = [Some(wgpu::ColorTargetState {
                format: color,
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
            })];
            let (fragment_entry, targets, depth_compare) = match depth_test {
                DepthTest::Early => (
                    "fs_box_early",
                    &color_target[..],
                    wgpu::CompareFunction::GreaterEqual,
                ),
                DepthTest::InShader => ("fs_box", &[][..], wgpu::CompareFunction::Always),
                DepthTest::Compute => unreachable!(),
            };
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Section culling"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_box"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                // The world pass's reversed depth test, without writes.
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(depth_compare),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fragment_entry),
                    compilation_options: Default::default(),
                    targets,
                }),
                multiview_mask: None,
                cache: None,
            });
            (Some(pipeline), None, layout, depth_layout)
        };
        let memory = pollster::block_on(memory.pop());
        let internal = pollster::block_on(internal.pop());
        let validation = pollster::block_on(validation.pop());
        if let Some(e) = memory.or(internal).or(validation) {
            return Err(e.to_string());
        }
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let cull = Self {
            pipeline,
            compute_pipeline,
            hiz,
            layout,
            group: None,
            depth_test,
            depth_layout,
            depth_group: None,
            params: buffer(
                device,
                "Section culling parameters",
                std::mem::size_of::<Params>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            phase2: buffer(device, "Section draws, phase 2", WORDS, PHASE2_USAGE),
            phase2_capacity: 1,
            tests: buffer(device, "Section tests", TEST_BYTES, TESTS_USAGE),
            tests_capacity: 1,
            test_data: Vec::new(),
            test_slots: Vec::new(),
            test_indices: Vec::new(),
            total: 0,
            probe: false,
            results: buffer(device, "Section test results", 4, RESULTS_USAGE),
            readbacks: (0..READBACKS)
                .map(|_| Readback {
                    buffer: buffer(device, "Section test readback", 4, READBACK_USAGE),
                    capacity: 1,
                    state: Arc::new(AtomicU8::new(FREE)),
                    stamp: 0,
                    slots: Vec::new(),
                    indices: Vec::new(),
                    total: 0,
                })
                .collect(),
            next: 0,
            current: None,
            seen: Vec::new(),
            seen_tiles: Vec::new(),
            stamp: 0,
            hidden: ([0; 2], 1.0),
            always: false,
            min_sections: 0,
        };
        let memory = pollster::block_on(memory.pop());
        let validation = pollster::block_on(validation.pop());
        if let Some(e) = memory.or(validation) {
            return Err(e.to_string());
        }
        Ok(cull)
    }

    /// Start a frame: take in the results that came back, and tell whether
    /// the frame culls.
    pub fn begin(&mut self, device: &wgpu::Device) -> bool {
        let _ = device.poll(wgpu::PollType::Poll);
        for i in 0..READBACKS {
            let r = &mut self.readbacks[(self.next + i) % READBACKS];
            if r.state.load(Ordering::Acquire) != MAPPED {
                continue;
            }
            let mapped = r.buffer.slice(..r.slots.len() as u64 * 4);
            if let Ok(data) = mapped.get_mapped_range() {
                // The mapping isn't guaranteed to be 4-byte aligned (a Vega 11
                // on Windows panicked in cast_slice), so copy the words out.
                let results: Vec<u32> = bytemuck::pod_collect_to_vec(&data);
                let (mut hidden, mut indices) = (0, 0u64);
                for (&result, &n) in results.iter().zip(&r.indices) {
                    if result != r.stamp {
                        hidden += 1;
                        indices += u64::from(n);
                    }
                }
                let share = indices as f32 / r.total.max(1) as f32;
                self.hidden = ([hidden, results.len()], share);
                for (&slot, &result) in r.slots.iter().zip(&results) {
                    let (seen, slot) = if slot & TILE != 0 {
                        (&mut self.seen_tiles, (slot & !TILE) as usize)
                    } else {
                        (&mut self.seen, slot as usize)
                    };
                    if slot >= seen.len() {
                        seen.resize(slot + 1, false);
                    }
                    seen[slot] = result == r.stamp;
                }
            }
            r.buffer.unmap();
            r.state.store(FREE, Ordering::Release);
        }
        self.stamp = self.stamp.wrapping_add(1).max(1);
        self.test_data.clear();
        self.test_slots.clear();
        self.test_indices.clear();
        self.total = 0;
        let (culls, probe) = match self.always {
            true => (true, false),
            false => gate(self.hidden.1, self.stamp),
        };
        self.probe = probe;
        culls
    }

    /// Whether phase 1 draws the section in resident `slot` (or the far tile
    /// in its `TILE` slot).
    pub fn first(&self, slot: u32) -> bool {
        if slot & TILE != 0 {
            return self
                .seen_tiles
                .get((slot & !TILE) as usize)
                .copied()
                .unwrap_or(false);
        }
        self.seen.get(slot as usize).copied().unwrap_or(false)
    }

    /// A section phase 1 draws, `index` in the visible list, with `indices`
    /// solid indices and the box `bounds` around its origin: tested in turn.
    pub fn drawn(&mut self, index: usize, slot: u32, indices: u32, bounds: [[f32; 3]; 2]) {
        self.total += u64::from(indices);
        if indices > 0 && (self.probe || slot.wrapping_add(self.stamp).is_multiple_of(RETEST)) {
            self.push(index, bounds, slot, indices);
        }
    }

    /// A section left to phase 2, with `draws` draws there: its test, whose
    /// draws `place` gives once phase 2's list is in its final order.
    pub fn deferred(
        &mut self,
        index: usize,
        slot: u32,
        draws: usize,
        indices: u32,
        bounds: [[f32; 3]; 2],
    ) -> Option<u32> {
        self.total += u64::from(indices);
        (draws > 0).then(|| {
            self.push(index, bounds, slot, indices);
            self.test_data.len() as u32 - 1
        })
    }

    /// `test`'s draws in solid pass `pass` (0 or 1): `draws` from phase 2's
    /// `first`. Each pass draws its own part of the section's quads.
    pub fn place(&mut self, test: u32, pass: u32, first: u32, draws: u32) {
        debug_assert!(pass < 2 && draws < 1 << 16);
        let t = &mut self.test_data[test as usize].data;
        t[1 + 2 * pass as usize] = first;
        t[2] |= draws << (16 * pass);
    }

    fn push(&mut self, index: usize, bounds: [[f32; 3]; 2], slot: u32, indices: u32) {
        let [low, high] = bounds.map(|[x, y, z]| [x, y, z, 0.0]);
        self.test_data.push(Test {
            data: [index as u32, 0, 0, 0],
            low,
            high,
        });
        self.test_slots.push(slot);
        self.test_indices.push(indices);
    }

    /// Stage this frame's inputs: phase 2's draws, with instance counts of 0,
    /// and the tests, which read the frame's camera-relative `origins` and
    /// view projection (the start of `globals`).
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        uploads: &mut super::upload::Uploads,
        encoder: &mut wgpu::CommandEncoder,
        phase2: &[[u32; 5]],
        origins: &wgpu::Buffer,
        globals: &wgpu::Buffer,
    ) {
        let draws = phase2.len() as u64;
        if draws > self.phase2_capacity {
            self.phase2_capacity = draws.next_power_of_two();
            self.phase2 = buffer(
                device,
                "Section draws, phase 2",
                self.phase2_capacity * WORDS,
                PHASE2_USAGE,
            );
            self.group = None;
        }
        let tests = self.test_data.len() as u64;
        if tests > self.tests_capacity {
            self.tests_capacity = tests.next_power_of_two();
            self.tests = buffer(
                device,
                "Section tests",
                self.tests_capacity * TEST_BYTES,
                TESTS_USAGE,
            );
            self.results = buffer(
                device,
                "Section test results",
                self.tests_capacity * 4,
                RESULTS_USAGE,
            );
            self.group = None;
        }
        if draws > 0 {
            uploads.buffer(encoder, &self.phase2, 0, bytemuck::cast_slice(phase2));
        }
        if tests > 0 {
            uploads.buffer(
                encoder,
                &self.tests,
                0,
                bytemuck::cast_slice(&self.test_data),
            );
        }
        let params = Params {
            stamp: self.stamp,
            big: BIG,
            count: self.test_data.len() as u32,
            pad: 0,
        };
        uploads.buffer(encoder, &self.params, 0, bytemuck::bytes_of(&params));
        if !self
            .group
            .as_ref()
            .is_some_and(|g| g.origins == *origins && g.globals == *globals)
        {
            self.group = Some(self.make_group(device, origins, globals));
        }
    }

    /// In the world pass after phase 1 (`Early`): draw the tested boxes
    /// against its depth. Leaves the pass's pipeline and group 0 changed.
    pub fn test(&self, pass: &mut wgpu::RenderPass<'_>) {
        let (Some(pipeline), Some(Group { group, .. })) = (&self.pipeline, &self.group) else {
            return;
        };
        if self.depth_test != DepthTest::Early || self.test_data.is_empty() {
            return;
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..36, 0..self.test_data.len() as u32);
    }

    /// Lets the depth `bind_depth` bound go, before the window's is made again.
    pub fn forget_depth(&mut self) {
        self.depth_group = None;
        if let Some(hiz) = &mut self.hiz {
            hiz.forget_depth();
        }
    }

    /// Before the world pass (`Compute` and `InShader`): the binding of the depth
    /// the world is drawn with this frame (and cached Hi-Z pyramid for `Compute`).
    pub fn bind_depth(&mut self, device: &wgpu::Device, depth: &wgpu::TextureView) {
        if let Some(hiz) = &mut self.hiz {
            hiz.bind_depth(device, depth);
            return;
        }
        let Some(layout) = &self.depth_layout else {
            return;
        };
        if self.depth_group.as_ref().is_none_or(|g| g.view != *depth) {
            self.depth_group = Some(DepthGroup {
                view: depth.clone(),
                group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Section culling depth"),
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(depth),
                    }],
                }),
            });
        }
    }

    /// Between the phases (`Compute` and `InShader`): test the boxes against
    /// phase 1's depth, bound by `bind_depth`.
    pub fn test_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timing: Option<&mut super::pack_timing::PassTiming>,
    ) {
        if self.test_data.is_empty() {
            return;
        }
        if let (Some(Group { group, .. }), Some(compute_pipeline), Some(hiz)) =
            (&self.group, &self.compute_pipeline, &self.hiz)
        {
            let wg = (self.test_data.len() as u32).div_ceil(WORKGROUP_1D);
            let (x, y) = (wg.min(COMPUTE_ROW), wg.div_ceil(COMPUTE_ROW));
            let mut build = |pass: &mut wgpu::ComputePass<'_>| {
                hiz.downsample(pass);
            };
            let mut boxes = |pass: &mut wgpu::ComputePass<'_>| {
                pass.set_pipeline(compute_pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.set_bind_group(1, hiz.read_group(), &[]);
                pass.dispatch_workgroups(x, y, 1);
            };
            super::pack_timing::compute_parts(
                encoder,
                timing,
                "Occlusion test",
                &mut [("hiz pyramid", &mut build), ("occlusion boxes", &mut boxes)],
            );
            return;
        }
        let (Some(pipeline), Some(Group { group, .. }), Some(DepthGroup { view, group: depth })) =
            (&self.pipeline, &self.group, &self.depth_group)
        else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Section culling"),
            timestamp_writes: timing.and_then(|t| t.writes("Occlusion test")),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: None,
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.set_bind_group(1, depth, &[]);
        pass.draw(0..36, 0..self.test_data.len() as u32);
    }

    /// After the world passes: copy the results for reading back, when a
    /// readback buffer is free.
    pub fn end(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let tests = self.test_data.len() as u64;
        let r = &mut self.readbacks[self.next];
        if tests == 0 || r.state.load(Ordering::Acquire) != FREE {
            return;
        }
        if tests > r.capacity {
            let capacity = tests.next_power_of_two();
            // Without room (#13624) the old buffer is kept and the frame reads nothing back.
            let label = "Section test readback";
            let Some(grown) =
                super::frame_targets::buffer(device, label, capacity * 4, READBACK_USAGE)
            else {
                return;
            };
            r.capacity = capacity;
            r.buffer = grown;
        }
        encoder.copy_buffer_to_buffer(&self.results, 0, &r.buffer, 0, tests * 4);
        r.stamp = self.stamp;
        r.slots.clear();
        r.slots.extend_from_slice(&self.test_slots);
        r.indices.clear();
        r.indices.extend_from_slice(&self.test_indices);
        r.total = self.total;
        self.current = Some(self.next);
    }

    /// The frame's results to read back, started by calling it right after the frame's
    /// submission (on the submit thread, when the frame goes there).
    pub fn submitted(&mut self) -> Option<impl FnOnce() + Send + 'static> {
        let i = self.current.take()?;
        let r = &self.readbacks[i];
        let state = r.state.clone();
        state.store(PENDING, Ordering::Release);
        let (buffer, bytes) = (r.buffer.clone(), r.slots.len() as u64 * 4);
        self.next = (i + 1) % READBACKS;
        Some(move || {
            buffer
                .slice(..bytes)
                .map_async(wgpu::MapMode::Read, move |result| {
                    state.store(
                        if result.is_ok() { MAPPED } else { FREE },
                        Ordering::Release,
                    );
                });
        })
    }

    fn make_group(
        &self,
        device: &wgpu::Device,
        origins: &wgpu::Buffer,
        globals: &wgpu::Buffer,
    ) -> Group {
        fn whole(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Section culling"),
            layout: &self.layout,
            entries: &[
                whole(0, &self.tests),
                whole(1, &self.results),
                whole(2, &self.phase2),
                whole(3, origins),
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: globals,
                        offset: 0,
                        size: NonZeroU64::new(64),
                    }),
                },
                whole(5, &self.params),
            ],
        });
        Group {
            origins: origins.clone(),
            globals: globals.clone(),
            group,
        }
    }
}

/// The test's shader for `depth_test`.
fn source(depth_test: DepthTest) -> String {
    let common = include_str!("cull.wgsl");
    match depth_test {
        DepthTest::Early => format!("{common}{EARLY_TEST}"),
        DepthTest::InShader => common.to_owned(),
        DepthTest::Compute => format!("{common}{COMPUTE_TEST}"),
    }
}

/// Whether the renderer should initialize its occlusion culling pass (`cull`).
///
/// Software adapters spend more time running the culling pass than drawing
/// the sections it removes. Keep the pass available for explicit A/B checks,
/// while leaving hardware adapters on the existing default.
pub(super) fn occlusion_culling_requested(info: &wgpu::AdapterInfo, env: Option<&str>) -> bool {
    env != Some("0") && (env.is_some() || info.device_type != wgpu::DeviceType::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Culling while the newest result hid enough; else one frame in `PROBE`
    /// tests every section. GPU culling's frames take the same gate.
    #[test]
    fn the_gate_culls_while_enough_is_hidden_and_probes_else() {
        assert_eq!(gate(MIN_SHARE, 3), (true, false));
        assert_eq!(gate(MIN_SHARE / 2.0, 3), (false, true));
        assert_eq!(gate(MIN_SHARE / 2.0, PROBE), (true, true));
    }

    fn translate_compute_entry(
        source: &str,
        entry_name: &str,
        capabilities: naga::valid::Capabilities,
    ) {
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&module)
            .unwrap_or_else(|e| panic!("{entry_name}: {e:?}"));
        let stage = naga::ShaderStage::Compute;
        let mut hlsl = String::new();
        naga::back::hlsl::Writer::new(
            &mut hlsl,
            &naga::back::hlsl::Options::default(),
            &naga::back::hlsl::PipelineOptions {
                entry_point: Some((stage, entry_name.to_owned())),
            },
        )
        .write(&module, &info, None)
        .unwrap_or_else(|e| panic!("{entry_name} HLSL: {e}"));
        assert!(!hlsl.is_empty());

        let (msl, _) = naga::back::msl::write_string(
            &module,
            &info,
            &naga::back::msl::Options {
                fake_missing_bindings: true,
                ..Default::default()
            },
            &naga::back::msl::PipelineOptions {
                entry_point: Some((stage, entry_name.to_owned())),
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{entry_name} MSL: {e}"));
        assert!(!msl.is_empty());

        let spv = naga::back::spv::write_vec(
            &module,
            &info,
            &naga::back::spv::Options::default(),
            Some(&naga::back::spv::PipelineOptions {
                shader_stage: stage,
                entry_point: entry_name.to_owned(),
            }),
        )
        .unwrap_or_else(|e| panic!("{entry_name} SPIR-V: {e}"));
        assert!(!spv.is_empty());
    }

    #[test]
    fn shaders_validate() {
        let no_early =
            naga::valid::Capabilities::all() - naga::valid::Capabilities::EARLY_DEPTH_TEST;
        for (depth_test, capabilities) in [
            (DepthTest::Early, naga::valid::Capabilities::all()),
            // What wgpu's Metal backend allows: no early depth test.
            (DepthTest::InShader, no_early),
            (DepthTest::Compute, no_early),
        ] {
            let src = source(depth_test);
            let module = naga::front::wgsl::parse_str(&src).unwrap();
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
                .validate(&module)
                .unwrap_or_else(|e| panic!("{depth_test:?}: {e:?}"));
        }
        for (source, entry) in [
            (hiz_downsample_source(), "hiz_downsample"),
            (hiz_tiled4_source(), "hiz_downsample_4"),
            (hiz_tiled5_source(), "hiz_downsample_5"),
            (hiz_tail4_source(), "hiz_tail_4"),
            (hiz_tail8_source(), "hiz_tail_8"),
        ] {
            translate_compute_entry(&source, entry, naga::valid::Capabilities::empty());
        }
        translate_compute_entry(&source(DepthTest::Compute), "cs_box_test", no_early);
    }

    #[test]
    fn hiz_mip_dimensions_cover_even_odd_and_degenerate_sizes() {
        for (depth, expected_mip0, expected_levels) in [
            ([1, 1], [1, 1], 1),
            ([1, 5], [1, 2], 2),
            ([5, 1], [2, 1], 2),
            ([2, 2], [1, 1], 1),
            ([3, 5], [1, 2], 2),
            ([1280, 720], [640, 360], 10),
            ([1366, 768], [683, 384], 10),
            ([1920, 1080], [960, 540], 10),
            ([2560, 1440], [1280, 720], 11),
            ([3840, 2160], [1920, 1080], 11),
        ] {
            let dims = hiz_mip_dimensions(depth);
            assert_eq!(dims[0], expected_mip0, "mip 0 for {depth:?}");
            assert_eq!(dims.len(), expected_levels, "levels for {depth:?}");
            assert_eq!(dims.last().copied(), Some([1, 1]), "tail for {depth:?}");
            // Matches WebGPU/wgpu's floor-halving mip dimensions at every level.
            let max_dim = expected_mip0[0].max(expected_mip0[1]);
            let expected_wgpu_mips = (u32::BITS - max_dim.leading_zeros()) as usize;
            assert_eq!(
                dims.len(),
                expected_wgpu_mips,
                "wgpu mip count for {depth:?}"
            );
            for (mip, &[w, h]) in dims.iter().enumerate() {
                assert_eq!(w, (expected_mip0[0] >> mip).max(1));
                assert_eq!(h, (expected_mip0[1] >> mip).max(1));
            }
        }
    }

    /// Reference CPU implementation of `HIZ_DOWNSAMPLE_SHADER` for one level.
    fn cpu_hiz_downsample(src: &[f32], src_size: [usize; 2]) -> (Vec<f32>, [usize; 2]) {
        let [sw, sh] = src_size;
        let dw = (sw / 2).max(1);
        let dh = (sh / 2).max(1);
        let odd_x = sw > 1 && (sw & 1) != 0;
        let odd_y = sh > 1 && (sh & 1) != 0;
        let mut dst = vec![0.0f32; dw * dh];
        for dy in 0..dh {
            let y0 = (dy * 2).min(sh - 1);
            let y1 = (dy * 2 + 1).min(sh - 1);
            let y2 = (dy * 2 + 2).min(sh - 1);
            for dx in 0..dw {
                let x0 = (dx * 2).min(sw - 1);
                let x1 = (dx * 2 + 1).min(sw - 1);
                let x2 = (dx * 2 + 2).min(sw - 1);
                let mut d = src[y0 * sw + x0]
                    .min(src[y0 * sw + x1])
                    .min(src[y1 * sw + x0])
                    .min(src[y1 * sw + x1]);
                if odd_x {
                    d = d.min(src[y0 * sw + x2]).min(src[y1 * sw + x2]);
                }
                if odd_y {
                    d = d.min(src[y2 * sw + x0]).min(src[y2 * sw + x1]);
                    if odd_x {
                        d = d.min(src[y2 * sw + x2]);
                    }
                }
                dst[dy * dw + dx] = d;
            }
        }
        (dst, [dw, dh])
    }

    /// Reference CPU implementation of the full Hi-Z mip pyramid.
    fn cpu_build_hiz(depth: &[f32], depth_size: [usize; 2]) -> Vec<(Vec<f32>, [usize; 2])> {
        let mut pyramid = Vec::new();
        let (mut cur, mut size) = cpu_hiz_downsample(depth, depth_size);
        loop {
            let done = size == [1, 1];
            pyramid.push((cur, size));
            if done {
                break;
            }
            let prev = pyramid.last().unwrap();
            let (next, next_size) = cpu_hiz_downsample(&prev.0, prev.1);
            cur = next;
            size = next_size;
        }
        pyramid
    }

    /// Reference CPU implementation of `cs_box_test`'s conservative 2x2 Hi-Z query.
    fn cpu_query_hiz(
        pyramid: &[(Vec<f32>, [usize; 2])],
        uv_min: [f32; 2],
        uv_max: [f32; 2],
    ) -> f32 {
        let max_mip = (pyramid.len() - 1) as i32;
        let [w0, h0] = pyramid[0].1;
        let rect_w = ((uv_max[0] - uv_min[0]) * w0 as f32).max(0.0);
        let rect_h = ((uv_max[1] - uv_min[1]) * h0 as f32).max(0.0);
        let max_extent = rect_w.max(rect_h);
        let mut mip = (max_extent.max(1.0).log2().ceil() as i32).clamp(0, max_mip);

        let bounds_at = |m: i32| -> ([usize; 2], [usize; 2]) {
            let [mw, mh] = pyramid[m as usize].1;
            let x0 = ((uv_min[0] * mw as f32).floor() as i32).clamp(0, mw as i32 - 1) as usize;
            let x1 = ((uv_max[0] * mw as f32).floor() as i32).clamp(0, mw as i32 - 1) as usize;
            let y0 = ((uv_min[1] * mh as f32).floor() as i32).clamp(0, mh as i32 - 1) as usize;
            let y1 = ((uv_max[1] * mh as f32).floor() as i32).clamp(0, mh as i32 - 1) as usize;
            ([x0, y0], [x1, y1])
        };

        if mip > 0 {
            let ([fx0, fy0], [fx1, fy1]) = bounds_at(mip - 1);
            if fx1 - fx0 <= 1 && fy1 - fy0 <= 1 {
                mip -= 1;
            }
        }
        let (mut lo, mut hi) = bounds_at(mip);
        if (hi[0] - lo[0] > 1 || hi[1] - lo[1] > 1) && mip < max_mip {
            mip += 1;
            (lo, hi) = bounds_at(mip);
        }
        assert!(hi[0] - lo[0] <= 1 && hi[1] - lo[1] <= 1);
        let (tex, [mw, _]) = &pyramid[mip as usize];
        tex[lo[1] * mw + lo[0]]
            .min(tex[lo[1] * mw + hi[0]])
            .min(tex[hi[1] * mw + lo[0]])
            .min(tex[hi[1] * mw + hi[0]])
    }

    #[test]
    fn hiz_cpu_conservative_min_never_misses_single_pixel_holes_on_odd_sizes() {
        for [w, h] in [[1, 1], [2, 3], [3, 5], [5, 7], [7, 9], [15, 13], [17, 19]] {
            // Test every single pixel as a sky hole (depth 0.25) inside a 0.9 wall,
            // including odd right/bottom borders and odd interior straddle texels.
            for py in 0..h {
                for px in 0..w {
                    let mut depth = vec![0.9f32; w * h];
                    depth[py * w + px] = 0.25;
                    let pyramid = cpu_build_hiz(&depth, [w, h]);
                    let cu = (px as f32 + 0.5) / w as f32;
                    let cv = (py as f32 + 0.5) / h as f32;
                    // Point query on the pixel center plus enclosing boxes of multiple scales.
                    for span in [0.0f32, 0.05, 0.2, 0.45, 1.0] {
                        let u0 = (cu - span * 0.4).max(0.0);
                        let u1 = (cu + span * 0.6).min(1.0);
                        let v0 = (cv - span * 0.3).max(0.0);
                        let v1 = (cv + span * 0.7).min(1.0);
                        let min_z = cpu_query_hiz(&pyramid, [u0, v0], [u1, v1]);
                        assert!(
                            min_z <= 0.25,
                            "false cull on {w}x{h} hole=({px},{py}) span={span}: got {min_z}"
                        );
                    }
                }
            }
        }
        // Solid wall (0.9 everywhere) accurately culls a box at 0.5 and keeps 0.9+ and NaN.
        let occluded = |max_z: f32, occluder_min_z: f32| max_z < occluder_min_z;
        let solid = vec![0.9f32; 33 * 25];
        let pyramid = cpu_build_hiz(&solid, [33, 25]);
        let occ = cpu_query_hiz(&pyramid, [0.2, 0.2], [0.7, 0.8]);
        assert_eq!(occ, 0.9);
        assert!(occluded(0.5, occ), "box behind wall must be occluded");
        assert!(!occluded(0.9, occ), "exact depth tie must remain visible");
        assert!(
            !occluded(0.95, occ),
            "box in front of wall must remain visible"
        );
        assert!(
            !occluded(f32::NAN, occ),
            "NaN depth must stay conservatively visible"
        );
    }

    /// The box's 12 triangles, as cull.wgsl and gpu_occlusion.wgsl draw them: two per face,
    /// counter-clockwise seen from outside, corners by bit (x 1, y 2, z 4).
    #[test]
    fn the_packed_box_corners_are_the_box_triangles() {
        let triangles = [
            0, 6, 2, 0, 4, 6, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4, 2, 7, 3, 2, 6, 7, 0, 3, 1, 0, 2,
            3, 4, 5, 7, 4, 7, 6,
        ];
        for source in [
            include_str!("cull.wgsl"),
            include_str!("gpu_occlusion.wgsl"),
        ] {
            assert_eq!(packed_lookup(source, "box_triangle_corner", 36), triangles);
        }
    }

    #[test]
    fn nouveau_is_detected_on_opengl_and_other_adapters_kept() {
        let mut info = wgpu::AdapterInfo {
            name: "NVC3".into(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::Other,
            device_pci_bus_id: String::new(),
            driver: "4.3 (Core Profile)".into(),
            driver_info: "Mesa 26.2.3-arch1.1".into(),
            backend: wgpu::Backend::Gl,
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            transient_saves_memory: None,
            limit_bucket: None,
        };
        assert!(is_nouveau(&info));

        // Other Nouveau chipsets (Fermi, Kepler, Maxwell, Pascal)
        for name in ["NVC0", "NVD9", "NVE4", "NV117", "NV130"] {
            info.name = name.into();
            assert!(is_nouveau(&info));
        }

        // Intel on Mesa OpenGL is kept (crocus / iris)
        info.name = "Mesa Intel(R) HD Graphics 4400 (HSW GT2)".into();
        assert!(!is_nouveau(&info));

        // AMD on Mesa OpenGL is kept (radeonsi)
        info.name = "AMD Radeon RX 580 (radeonsi, ...)".into();
        assert!(!is_nouveau(&info));

        // Proprietary NVIDIA on OpenGL does not have Mesa in driver_info
        info.name = "NVIDIA GeForce RTX 3060".into();
        info.driver_info = "NVIDIA 550.54.14".into();
        assert!(!is_nouveau(&info));

        // Vulkan is not OpenGL
        info.name = "NVC3".into();
        info.driver_info = "Mesa 26.2.3-arch1.1".into();
        info.backend = wgpu::Backend::Vulkan;
        assert!(!is_nouveau(&info));
    }

    /// A device whose storage buffers cannot hold the pass's bindings gets no pass
    /// rather than a panic (#17994). What it proves is the caught refusal and the
    /// `None`, not the way r600 refused: the driver there links no pipeline at all
    /// while its limits promise the bindings, which only a driver that refuses
    /// reproduces.
    #[test]
    fn a_device_short_of_storage_buffers_gets_no_culling_pass() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(adapter) = crate::gfx::test_gpu::adapter() else {
            return;
        };
        // Where the device has the storage buffers the pass is made, so the
        // refusal below is the shortfall and not something else on this machine.
        // An adapter that refuses the pass with its own limits (the r600 case)
        // leaves that half unproven instead of failing the test.
        let (device, _queue) = crate::gfx::test_gpu::device(&adapter);
        if Cull::new(
            &device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            DepthTest::InShader,
        )
        .is_none()
        {
            eprintln!("skipped: the adapter does not make the pass with its own limits");
            return;
        }
        drop(device);
        // The device is asked for two storage buffers per stage: the pass binds
        // three to its fragment stage and two to its vertex stage, so its layout
        // cannot be made.
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 2,
            ..Default::default()
        };
        let (device, _queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_limits: limits,
                ..Default::default()
            }))
            .expect("the device with two storage buffers per stage");
        assert!(
            Cull::new(
                &device,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                DepthTest::InShader
            )
            .is_none(),
            "the pass is sidelined where the device is short of storage buffers"
        );
        assert!(
            Cull::new(
                &device,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                DepthTest::Compute
            )
            .is_none(),
            "the compute Hi-Z pass is sidelined where the device is short of storage buffers"
        );
    }

    /// Minimal depth-only rect shader for GPU Hi-Z occlusion tests.
    const DEPTH_RECT_SHADER: &str = "
struct Rect {
    bounds: vec4<f32>,
    depth: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}
@group(0) @binding(0) var<uniform> rect: Rect;
@vertex fn vs(@builtin(vertex_index) v: u32) -> @builtin(position) vec4<f32> {
    let x = select(rect.bounds.x, rect.bounds.z, v == 1u || v == 2u || v == 5u);
    let y = select(rect.bounds.y, rect.bounds.w, v == 2u || v == 4u || v == 5u);
    return vec4<f32>(x, y, rect.depth, 1.0);
}
";

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct DepthRect {
        bounds: [f32; 4],
        depth: f32,
        pad: [f32; 3],
    }

    fn read_u32_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &wgpu::Buffer,
        words: usize,
    ) -> Vec<u32> {
        let bytes = (words * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull test readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| res.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = slice.get_mapped_range().unwrap();
        let out = bytemuck::pod_collect_to_vec(&data);
        drop(data);
        staging.unmap();
        out
    }

    #[test]
    fn compute_hiz_gpu_verdicts_visible_occluded_near_plane_and_odd_sizes() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(adapter) = crate::gfx::test_gpu::adapter() else {
            return;
        };
        if !crate::gfx::test_gpu::supports_depth_load(&adapter) {
            return;
        }
        let (device, queue) = crate::gfx::test_gpu::device(&adapter);
        if !compute_hiz_supported(&adapter, &device) {
            eprintln!("skipped: adapter lacks compute Hi-Z support");
            return;
        }
        let mut cull = Cull::try_new(&device, wgpu::TextureFormat::Rgba8Unorm, DepthTest::Compute)
            .unwrap_or_else(|e| panic!("Cull::new(Compute) failed on qualifying adapter: {e}"));
        cull.always = true;

        // Depth-only rect writer used to populate synthetic reversed-Z depth buffers.
        let rect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("depth rect"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let rect_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("depth rect"),
            source: wgpu::ShaderSource::Wgsl(DEPTH_RECT_SHADER.into()),
        });
        let rect_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("depth rect"),
            bind_group_layouts: &[Some(&rect_layout)],
            ..Default::default()
        });
        let rect_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth rect"),
            layout: Some(&rect_pl),
            vertex: wgpu::VertexState {
                module: &rect_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });
        let rect_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("depth rect"),
            size: std::mem::size_of::<DepthRect>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rect_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("depth rect"),
            layout: &rect_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: rect_buf.as_entire_binding(),
            }],
        });

        let make_depth =
            |w: u32, h: u32, clear: f32, draw: Option<DepthRect>| -> wgpu::TextureView {
                let view = device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("test depth"),
                        size: wgpu::Extent3d {
                            width: w,
                            height: h,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Depth32Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&Default::default());
                if let Some(rect) = draw {
                    queue.write_buffer(&rect_buf, 0, bytemuck::bytes_of(&rect));
                }
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("fill depth"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(clear),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    if draw.is_some() {
                        pass.set_pipeline(&rect_pipeline);
                        pass.set_bind_group(0, &rect_group, &[]);
                        pass.draw(0..6, 0..1);
                    }
                }
                queue.submit([encoder.finish()]);
                view
            };

        type TestBox = ([f32; 3], [[f32; 3]; 2]);
        let mut uploads = super::super::upload::Uploads::new(&device);
        let run_case = |cull: &mut Cull,
                        uploads: &mut super::super::upload::Uploads,
                        depth: &wgpu::TextureView,
                        view_proj: glam::Mat4,
                        boxes: &[TestBox]|
         -> (u32, Vec<u32>, Vec<u32>) {
            assert!(cull.begin(&device));
            let stamp = cull.stamp;
            let origins_data: Vec<[f32; 4]> = boxes
                .iter()
                .map(|([x, y, z], _)| [*x, *y, *z, 0.0])
                .collect();
            let origins = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test origins"),
                size: (origins_data.len().max(1) * 16) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&origins, 0, bytemuck::cast_slice(&origins_data));
            let globals = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test globals"),
                size: 64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&globals, 0, bytemuck::bytes_of(&view_proj.to_cols_array()));

            let mut phase2 = Vec::with_capacity(boxes.len());
            for (i, &(_, bounds)) in boxes.iter().enumerate() {
                let test = cull
                    .deferred(i, i as u32, 1, 6, bounds)
                    .expect("deferred test");
                cull.place(test, 0, i as u32, 1);
                phase2.push([6, 0, 0, 0, i as u32]);
            }
            cull.bind_depth(&device, depth);
            let mut encoder = device.create_command_encoder(&Default::default());
            cull.upload(&device, uploads, &mut encoder, &phase2, &origins, &globals);
            cull.test_pass(&mut encoder, None);
            uploads.finish(&encoder);
            queue.submit([encoder.finish()]);

            let results = read_u32_buffer(&device, &queue, &cull.results, boxes.len());
            let draws = read_u32_buffer(&device, &queue, &cull.phase2, boxes.len() * 5);
            let instances: Vec<u32> = draws.as_chunks::<5>().0.iter().map(|d| d[1]).collect();
            (stamp, results, instances)
        };

        // Case 1: Even 64x36 reversed-Z perspective with a right-half wall at z = 0.002.
        let depth_even = make_depth(
            64,
            36,
            0.0,
            Some(DepthRect {
                bounds: [0.0, -1.0, 1.0, 1.0],
                depth: 0.002,
                pad: [0.0; 3],
            }),
        );
        let proj = glam::Mat4::perspective_rh(70f32.to_radians(), 64.0 / 36.0, 256.0, 0.05);
        let unit = [[-2.0; 3], [2.0; 3]];
        let small = [[-1.0; 3], [1.0; 3]];
        let (stamp, results, instances) = run_case(
            &mut cull,
            &mut uploads,
            &depth_even,
            proj,
            &[
                // 0: occluded behind right-half wall (max_z ~ 0.00085 < 0.002)
                ([24.0, 0.0, -50.0], unit),
                // 1: visible in front of right-half wall (max_z ~ 0.00536 > 0.002)
                ([4.0, 0.0, -10.0], small),
                // 2: visible over clear sky (0.0) on left half
                ([-16.0, 0.0, -50.0], unit),
                // 3: camera inside inflated box -> always visible
                ([0.0, 0.0, 0.0], small),
                // 4: camera outside box, one corner behind camera (near-plane crossing) -> always visible
                ([5.0, 0.0, -1.0], unit),
                // 5: strictly off-screen to the right -> culled
                ([200.0, 0.0, -50.0], unit),
            ],
        );
        let verdicts: Vec<bool> = results.iter().map(|&r| r == stamp).collect();
        assert_eq!(verdicts, vec![false, true, true, true, true, false]);
        assert_eq!(instances, vec![0, 1, 1, 1, 1, 0]);

        // Case 2: Odd 67x39 depth buffer filled to 0.875 (exact dyadic f32 so -10.0 + 0.875 + 10.0
        // roundtrips bit-exact for the depth-tie box) with a single-pixel sky hole (0.0) at the odd
        // bottom-right corner (66, 38). Uses a z+10 translation matrix so boxes at origin z = -10
        // map directly to NDC without triggering the 1-block camera-inside guard.
        let depth_odd = make_depth(
            67,
            39,
            0.875,
            Some(DepthRect {
                bounds: [1.0 - 2.0 / 67.0, -1.0, 1.0, -1.0 + 2.0 / 39.0],
                depth: 0.0,
                pad: [0.0; 3],
            }),
        );
        let clip_mat = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 10.0));
        let (stamp, results, instances) = run_case(
            &mut cull,
            &mut uploads,
            &depth_odd,
            clip_mat,
            &[
                // 0: occluded behind 0.875 wall (away from the corner hole)
                ([0.0, 0.0, -10.0], [[-0.5, 0.0, 0.25], [0.0, 0.5, 0.5]]),
                // 1: bit-exact depth tie (0.875 >= 0.875) -> visible
                ([0.0, 0.0, -10.0], [[-0.5, 0.0, 0.25], [0.0, 0.5, 0.875]]),
                // 2: broad box overlapping the 1-pixel sky hole at odd corner (66, 38) -> visible
                ([0.0, 0.0, -10.0], [[0.8, -1.0, 0.125], [1.0, -0.8, 0.25]]),
                // 3: tight single-pixel box directly on odd corner (66, 38) at mip 0 -> visible
                (
                    [0.0, 0.0, -10.0],
                    [
                        [1.0 - 1.5 / 67.0, -1.0, 0.125],
                        [1.0, -1.0 + 1.5 / 39.0, 0.25],
                    ],
                ),
            ],
        );
        let verdicts: Vec<bool> = results.iter().map(|&r| r == stamp).collect();
        assert_eq!(
            verdicts,
            vec![false, true, true, true],
            "odd 67x39 hole verdict"
        );
        assert_eq!(instances, vec![0, 1, 1, 1]);

        // Case 3: OOM during pyramid allocation falls back to the 1x1 zeroed texture,
        // backs off repeated attempts at the same size, retries immediately when size
        // changes, and recovers at the same size once the bounded delay elapses.
        let occluded_odd = [([0.0, 0.0, -10.0], [[-0.5, 0.0, 0.2], [0.0, 0.5, 0.5]])];
        let occluded_even = [([24.0, 0.0, -50.0], unit)];
        cull.forget_depth();
        super::super::frame_targets::NO_ROOM.set(1);
        let (stamp, results, instances) =
            run_case(&mut cull, &mut uploads, &depth_odd, clip_mat, &occluded_odd);
        assert_eq!(super::super::frame_targets::NO_ROOM.get(), 0);
        super::super::frame_targets::begin_frame();
        assert!(cull.hiz.as_ref().is_some_and(|h| h.pyramid.is_none()));
        assert_eq!(results, vec![stamp]);
        assert_eq!(instances, vec![1]);

        // Same size during backoff: no retry attempt (`NO_ROOM` stays unspent) and
        // fallback remains active.
        super::super::frame_targets::NO_ROOM.set(1);
        let (stamp, results, instances) =
            run_case(&mut cull, &mut uploads, &depth_odd, clip_mat, &occluded_odd);
        assert_eq!(
            super::super::frame_targets::NO_ROOM.get(),
            1,
            "same-size bind during OOM backoff must not retry pyramid allocation"
        );
        super::super::frame_targets::NO_ROOM.set(0);
        assert!(cull.hiz.as_ref().is_some_and(|h| h.pyramid.is_none()));
        assert_eq!(results, vec![stamp]);
        assert_eq!(instances, vec![1]);

        // Size change clears backoff and retries immediately.
        let (stamp, results, instances) =
            run_case(&mut cull, &mut uploads, &depth_even, proj, &occluded_even);
        assert!(cull.hiz.as_ref().is_some_and(|h| h.pyramid.is_some()));
        assert_ne!(results[0], stamp);
        assert_eq!(instances, vec![0]);

        // Same-size recovery after the bounded retry delay elapses.
        super::super::frame_targets::NO_ROOM.set(1);
        let (stamp, results, instances) =
            run_case(&mut cull, &mut uploads, &depth_odd, clip_mat, &occluded_odd);
        assert_eq!(super::super::frame_targets::NO_ROOM.get(), 0);
        super::super::frame_targets::begin_frame();
        assert!(cull.hiz.as_ref().is_some_and(|h| h.pyramid.is_none()));
        assert_eq!(results, vec![stamp]);
        assert_eq!(instances, vec![1]);
        for frame in 0..HIZ_OOM_RETRY_MIN {
            if frame + 1 == HIZ_OOM_RETRY_MIN {
                super::super::frame_targets::NO_ROOM.set(1);
            }
            cull.bind_depth(&device, &depth_odd);
            assert!(cull.hiz.as_ref().is_some_and(|h| h.pyramid.is_none()));
        }
        assert_eq!(
            super::super::frame_targets::NO_ROOM.get(),
            1,
            "final backoff frame must not retry allocation"
        );
        super::super::frame_targets::NO_ROOM.set(0);
        let (stamp, results, instances) =
            run_case(&mut cull, &mut uploads, &depth_odd, clip_mat, &occluded_odd);
        assert!(
            cull.hiz.as_ref().is_some_and(|h| h.pyramid.is_some()),
            "pyramid allocation must recover at the same size after backoff expires"
        );
        assert_ne!(results[0], stamp);
        assert_eq!(instances, vec![0]);

        // Case 4: Same-size depth rebind under OOM is error-scoped, drops the stale
        // pyramid instead of caching a failed bind group, keeps the zero-depth fallback
        // active on the next frame, and recovers cleanly once reset.
        let depth_odd_rebind = make_depth(67, 39, 0.875, None);
        super::super::frame_targets::NO_ROOM.set(1);
        let (stamp, results, instances) = run_case(
            &mut cull,
            &mut uploads,
            &depth_odd_rebind,
            clip_mat,
            &occluded_odd,
        );
        assert_eq!(super::super::frame_targets::NO_ROOM.get(), 0);
        super::super::frame_targets::begin_frame();
        assert!(cull.hiz.as_ref().is_some_and(|h| h.pyramid.is_none()));
        assert_eq!(results, vec![stamp]);
        assert_eq!(instances, vec![1]);

        // Next frame with the same target never reuses a failed binding and stays backed off.
        super::super::frame_targets::NO_ROOM.set(1);
        let (stamp, results, instances) = run_case(
            &mut cull,
            &mut uploads,
            &depth_odd_rebind,
            clip_mat,
            &occluded_odd,
        );
        assert_eq!(super::super::frame_targets::NO_ROOM.get(), 1);
        super::super::frame_targets::NO_ROOM.set(0);
        assert!(cull.hiz.as_ref().is_some_and(|h| h.pyramid.is_none()));
        assert_eq!(results, vec![stamp]);
        assert_eq!(instances, vec![1]);

        cull.forget_depth();
        let (stamp, results, instances) = run_case(
            &mut cull,
            &mut uploads,
            &depth_odd_rebind,
            clip_mat,
            &occluded_odd,
        );
        assert!(cull.hiz.as_ref().is_some_and(|h| h.pyramid.is_some()));
        assert_ne!(results[0], stamp);
        assert_eq!(instances, vec![0]);
    }

    #[test]
    fn compute_hiz_setup_oom_falls_back_to_supported_raster_backend() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(adapter) = crate::gfx::test_gpu::adapter() else {
            return;
        };
        let features = (adapter.features() & wgpu::Features::INDIRECT_FIRST_INSTANCE)
            | Cull::features(&adapter);
        if !features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
            eprintln!("skipped: adapter lacks INDIRECT_FIRST_INSTANCE");
            return;
        }
        let (device, _queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features: features,
                ..Default::default()
            }))
            .expect("GPU test device creation failed");
        if !compute_hiz_supported(&adapter, &device) {
            eprintln!("skipped: adapter lacks compute Hi-Z support");
            return;
        }
        let Some(raster) = Cull::supported_raster(&adapter, &device) else {
            eprintln!("skipped: adapter lacks raster occlusion fallback");
            return;
        };

        // Simulated OOM during optional Hi-Z pipeline/fallback texture creation causes
        // Cull::new(Compute) to fail cleanly instead of panicking, and Cull::pick falls
        // back to the adapter's supported raster mode (`Early` or `InShader`).
        super::super::frame_targets::NO_ROOM.set(1);
        let fallback = Cull::pick(
            &adapter,
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            DepthTest::Compute,
        )
        .expect("Cull::pick must fall back to supported raster backend when Compute setup fails");
        assert_eq!(super::super::frame_targets::NO_ROOM.get(), 0);
        assert_eq!(fallback.depth_test, raster);
        assert!(fallback.hiz_read_layout().is_none());

        // Without OOM, Compute remains the primary choice.
        let compute = Cull::pick(
            &adapter,
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            DepthTest::Compute,
        )
        .expect("Cull::pick(Compute) must succeed on qualifying adapter");
        assert_eq!(compute.depth_test, DepthTest::Compute);
        assert!(compute.hiz_read_layout().is_some());
    }

    /// Exact CPU simulation of `hiz_downsample_4` (`edge = 8, max_mips = 4`) and
    /// `hiz_downsample_5` (`edge = 16, max_mips = 5`), poisoning shared tiles with
    /// `f32::NAN` so any unwritten read panics immediately.
    fn sim_tiled_step(
        src: &[f32],
        src_size: [usize; 2],
        edge: usize,
        max_mips: usize,
        dsts: &mut [Vec<f32>],
    ) {
        let [sw, sh] = src_size;
        let s0 = [(sw / 2).max(1), (sh / 2).max(1)];
        let reduce = |dx: usize, dy: usize| -> f32 {
            let mx = sw - 1;
            let my = sh - 1;
            let bx = dx * 2;
            let by = dy * 2;
            let x0 = bx.min(mx);
            let x1 = (bx + 1).min(mx);
            let y0 = by.min(my);
            let y1 = (by + 1).min(my);
            let mut d = src[y0 * sw + x0]
                .min(src[y0 * sw + x1])
                .min(src[y1 * sw + x0].min(src[y1 * sw + x1]));
            let ox = sw > 1 && !sw.is_multiple_of(2);
            let oy = sh > 1 && !sh.is_multiple_of(2);
            if ox {
                let x2 = (bx + 2).min(mx);
                d = d.min(src[y0 * sw + x2].min(src[y1 * sw + x2]));
            }
            if oy {
                let y2 = (by + 2).min(my);
                d = d.min(src[y2 * sw + x0].min(src[y2 * sw + x1]));
                if ox {
                    let x2 = (bx + 2).min(mx);
                    d = d.min(src[y2 * sw + x2]);
                }
            }
            d
        };
        let gx_n = s0[0].div_ceil(edge);
        let gy_n = s0[1].div_ceil(edge);
        let can_even = |s: [usize; 2]| {
            (s[0] > 1 || s[1] > 1)
                && (s[0] == 1 || s[0].is_multiple_of(2))
                && (s[1] == 1 || s[1].is_multiple_of(2))
        };
        for gy in 0..gy_n {
            for gx in 0..gx_n {
                let mut tiles: Vec<Vec<f32>> = (0..max_mips - 1)
                    .map(|m| vec![f32::NAN; (edge >> m) * (edge >> m)])
                    .collect();
                for ly in 0..edge {
                    for lx in 0..edge {
                        let px = gx * edge + lx;
                        let py = gy * edge + ly;
                        if px < s0[0] && py < s0[1] {
                            let d0 = reduce(px, py);
                            dsts[0][py * s0[0] + px] = d0;
                            tiles[0][ly * edge + lx] = d0;
                        }
                    }
                }
                let mut prev = s0;
                let mut active = true;
                for m in 1..max_mips {
                    let do_m = active && can_even(prev);
                    let cur = [(prev[0] / 2).max(1), (prev[1] / 2).max(1)];
                    let cur_edge = edge >> m;
                    let prev_edge = edge >> (m - 1);
                    if do_m {
                        for ly in 0..cur_edge {
                            for lx in 0..cur_edge {
                                let px = gx * cur_edge + lx;
                                let py = gy * cur_edge + ly;
                                if px < cur[0] && py < cur[1] {
                                    let x0 = lx * 2;
                                    let x1 = if prev[0] == 1 { x0 } else { x0 + 1 };
                                    let y0 = ly * 2;
                                    let y1 = if prev[1] == 1 { y0 } else { y0 + 1 };
                                    let pt = &tiles[m - 1];
                                    let vals = [
                                        pt[y0 * prev_edge + x0],
                                        pt[y0 * prev_edge + x1],
                                        pt[y1 * prev_edge + x0],
                                        pt[y1 * prev_edge + x1],
                                    ];
                                    assert!(vals.iter().all(|v| !v.is_nan()));
                                    let dm = vals[0].min(vals[1]).min(vals[2].min(vals[3]));
                                    dsts[m][py * cur[0] + px] = dm;
                                    if m < max_mips - 1 {
                                        tiles[m][ly * cur_edge + lx] = dm;
                                    }
                                }
                            }
                        }
                    }
                    active = do_m;
                    prev = cur;
                }
            }
        }
    }

    /// Exact CPU simulation of `hiz_tail_4` / `hiz_tail_8` ping-pong workgroup reduction.
    fn sim_tail_step(src: &[f32], src_size: [usize; 2], max_mips: usize, dsts: &mut [Vec<f32>]) {
        let mut tail_a = vec![f32::NAN; HIZ_TAIL_MAX_TEXELS as usize];
        let mut tail_b = vec![f32::NAN; (HIZ_TAIL_MAX_TEXELS / 2) as usize];
        let (first, s0) = cpu_hiz_downsample(src, src_size);
        let n0 = s0[0] * s0[1];
        assert!(n0 <= tail_a.len());
        tail_a[..n0].copy_from_slice(&first);
        dsts[0].copy_from_slice(&first);
        let mut prev = s0;
        let mut active = true;
        for m in 1..max_mips {
            let do_m = active && (prev[0] > 1 || prev[1] > 1);
            let cur = [(prev[0] / 2).max(1), (prev[1] / 2).max(1)];
            if do_m {
                let (next, next_size) = if m.is_multiple_of(2) {
                    cpu_hiz_downsample(&tail_b[..prev[0] * prev[1]], prev)
                } else {
                    cpu_hiz_downsample(&tail_a[..prev[0] * prev[1]], prev)
                };
                assert_eq!(next_size, cur);
                assert!(next.iter().all(|v| !v.is_nan()));
                let nm = cur[0] * cur[1];
                if m.is_multiple_of(2) {
                    tail_a[..nm].copy_from_slice(&next);
                } else {
                    tail_b[..nm].copy_from_slice(&next);
                }
                dsts[m].copy_from_slice(&next);
            }
            active = do_m;
            prev = cur;
        }
    }

    #[test]
    fn hiz_multimip_plan_reduces_dispatches_and_matches_reference_bit_exact() {
        let desktop = HizCaps {
            tiled4: true,
            tail4: true,
            wide8: true,
        };
        let core4 = HizCaps {
            tiled4: true,
            tail4: true,
            wide8: false,
        };
        let tiled_only = HizCaps {
            tiled4: true,
            tail4: false,
            wide8: false,
        };
        let single_only = HizCaps {
            tiled4: false,
            tail4: false,
            wide8: false,
        };

        // Concrete dispatch count reductions on standard resolutions.
        assert_eq!(
            plan_hiz_steps(&hiz_mip_dimensions([1920, 1080]), desktop).len(),
            3
        );
        assert_eq!(
            plan_hiz_steps(&hiz_mip_dimensions([2560, 1440]), desktop).len(),
            2
        );
        assert_eq!(
            plan_hiz_steps(&hiz_mip_dimensions([3840, 2160]), desktop).len(),
            3
        );
        assert_eq!(
            plan_hiz_steps(&hiz_mip_dimensions([1280, 720]), desktop).len(),
            2
        );
        assert_eq!(
            plan_hiz_steps(&hiz_mip_dimensions([1920, 1080]), core4).len(),
            4
        );
        assert_eq!(
            plan_hiz_steps(&hiz_mip_dimensions([2560, 1440]), core4).len(),
            4
        );
        assert_eq!(
            plan_hiz_steps(&hiz_mip_dimensions([67, 39]), desktop).len(),
            1
        );
        assert_eq!(
            plan_hiz_steps(&hiz_mip_dimensions([1, 1]), desktop).len(),
            1
        );

        let all_caps = [single_only, tiled_only, core4, desktop];
        // Structural plan invariants across every small size and large/odd/degenerate sizes.
        for [w, h] in (1..=33u32)
            .flat_map(|w| (1..=33u32).map(move |h| [w, h]))
            .chain([
                [64, 36],
                [67, 39],
                [129, 65],
                [240, 135],
                [320, 180],
                [1366, 768],
                [1919, 1079],
                [1920, 1080],
                [2560, 1440],
                [3840, 2160],
                [1, 2049],
                [2049, 1],
                [4097, 1],
            ])
        {
            let mips = hiz_mip_dimensions([w, h]);
            for caps in all_caps {
                let steps = plan_hiz_steps(&mips, caps);
                let mut cursor = 0usize;
                for step in &steps {
                    assert_eq!(step.first_mip, cursor, "{w}x{h} {caps:?}");
                    assert!(step.mip_count >= 1, "{w}x{h} {caps:?}");
                    assert!(step.workgroups[0] >= 1 && step.workgroups[1] >= 1);
                    if matches!(step.kind, HizStepKind::Tail4 | HizStepKind::Tail8) {
                        let [mw, mh] = mips[step.first_mip];
                        assert!(u64::from(mw) * u64::from(mh) <= u64::from(HIZ_TAIL_MAX_TEXELS));
                        assert_eq!(step.workgroups, [1, 1]);
                    }
                    cursor += step.mip_count;
                }
                assert_eq!(cursor, mips.len(), "{w}x{h} {caps:?}");
            }
        }

        // Bit-exact equivalence of simulated multi-mip reductions against cpu_build_hiz.
        for [w, h] in [
            [1usize, 1],
            [2, 3],
            [3, 5],
            [5, 7],
            [7, 9],
            [15, 13],
            [17, 19],
            [33, 25],
            [64, 36],
            [67, 39],
            [129, 65],
            [130, 67],
            [240, 135],
            [320, 180],
            [1, 2049],
            [2049, 1],
        ] {
            let mut seed = ((w as u64) << 32) ^ (h as u64) ^ 0x9e37_79b9_7f4a_7c15;
            let depth: Vec<f32> = (0..w * h)
                .map(|_| {
                    seed ^= seed >> 12;
                    seed ^= seed << 25;
                    seed ^= seed >> 27;
                    ((seed.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32) / 16_777_216.0
                })
                .collect();
            let reference = cpu_build_hiz(&depth, [w, h]);
            let mips = hiz_mip_dimensions([w as u32, h as u32]);
            for caps in all_caps {
                let steps = plan_hiz_steps(&mips, caps);
                let mut produced: Vec<Vec<f32>> = mips
                    .iter()
                    .map(|&[mw, mh]| vec![f32::NAN; mw as usize * mh as usize])
                    .collect();
                for step in steps {
                    let (src_slice, src_size) = if step.first_mip == 0 {
                        (depth.as_slice(), [w, h])
                    } else {
                        let [pw, ph] = mips[step.first_mip - 1];
                        (
                            produced[step.first_mip - 1].as_slice(),
                            [pw as usize, ph as usize],
                        )
                    };
                    let src_copy = src_slice.to_vec();
                    let out_slice = &mut produced[step.first_mip..step.first_mip + step.mip_count];
                    match step.kind {
                        HizStepKind::Single => {
                            let (next, _) = cpu_hiz_downsample(&src_copy, src_size);
                            out_slice[0].copy_from_slice(&next);
                        }
                        HizStepKind::Tiled4 => {
                            sim_tiled_step(&src_copy, src_size, 8, 4, out_slice);
                        }
                        HizStepKind::Tiled5 => {
                            sim_tiled_step(&src_copy, src_size, 16, 5, out_slice);
                        }
                        HizStepKind::Tail4 => {
                            sim_tail_step(&src_copy, src_size, 4, out_slice);
                        }
                        HizStepKind::Tail8 => {
                            sim_tail_step(&src_copy, src_size, 8, out_slice);
                        }
                    }
                }
                for (mip, (got, (expected, _))) in produced.iter().zip(&reference).enumerate() {
                    let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                    let exp_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(got_bits, exp_bits, "{w}x{h} {caps:?} mip {mip}");
                }
            }
        }
    }

    /// Read back every mip of `HizPyramid`'s `R32Float` texture and compare bit-for-bit
    /// against `cpu_build_hiz`.
    fn read_hiz_mip(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tex: &wgpu::Texture,
        mip: u32,
        size: [u32; 2],
    ) -> Vec<f32> {
        let row_bytes = size[0] * 4;
        let padded_row = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let total_bytes = u64::from(padded_row) * u64::from(size[1]);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hi-Z mip readback"),
            size: total_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| res.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = slice.get_mapped_range().unwrap();
        let mut out = Vec::with_capacity((size[0] * size[1]) as usize);
        for y in 0..size[1] as usize {
            let start = y * padded_row as usize;
            let row: &[f32] = bytemuck::cast_slice(&data[start..start + row_bytes as usize]);
            out.extend_from_slice(row);
        }
        drop(data);
        staging.unmap();
        out
    }

    #[test]
    fn compute_hiz_multimip_gpu_pyramid_matches_cpu_reference_on_odd_and_even_sizes() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(adapter) = crate::gfx::test_gpu::adapter() else {
            return;
        };
        if !crate::gfx::test_gpu::supports_depth_load(&adapter) {
            return;
        }
        // Run against both WebGPU default limits (4 storage textures -> Tiled4 + Tail4)
        // and the adapter's full limits (8+ storage textures -> Tiled5 + Tail8).
        let mut descriptors = vec![("default-4tex", wgpu::DeviceDescriptor::default())];
        if adapter.limits().max_storage_textures_per_shader_stage >= 8 {
            descriptors.push((
                "adapter-8tex",
                wgpu::DeviceDescriptor {
                    required_limits: adapter.limits(),
                    ..Default::default()
                },
            ));
        }

        for (tier, desc) in descriptors {
            let Ok((device, queue)) = pollster::block_on(adapter.request_device(&desc)) else {
                continue;
            };
            if !compute_hiz_supported(&adapter, &device) {
                continue;
            }
            let Some(mut cull) =
                Cull::new(&device, wgpu::TextureFormat::Rgba8Unorm, DepthTest::Compute)
            else {
                continue;
            };
            let rect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Hi-Z multimip test rect"),
                source: wgpu::ShaderSource::Wgsl(DEPTH_RECT_SHADER.into()),
            });
            let rect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
            let rect_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&rect_layout)],
                ..Default::default()
            });
            let rect_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Hi-Z multimip test rect"),
                layout: Some(&rect_pl),
                vertex: wgpu::VertexState {
                    module: &rect_shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: None,
                multiview_mask: None,
                cache: None,
            });
            let rect_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<DepthRect>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let rect_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &rect_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rect_buf.as_entire_binding(),
                }],
            });

            // Cases: (width, height, single-pixel sky hole coordinate (x, y)).
            let cases = [
                (67u32, 39u32, 66u32, 38u32), // odd x odd corner hole (Tail4/Tail8)
                (67, 39, 66, 19),             // odd right edge hole
                (67, 39, 33, 38),             // odd bottom edge hole
                (135, 79, 134, 78),           // odd -> odd intermediates + tail
                (257, 135, 256, 134),         // odd src -> even/odd intermediates + tail
                (320, 180, 161, 91),          // even top mips (Tiled4/Tiled5) -> odd mip -> tail
                (320, 180, 319, 179),         // bottom-right pixel in multi-mip tiled pass
            ];
            for (w, h, hx, hy) in cases {
                let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Hi-Z multimip depth"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let depth_view = depth_tex.create_view(&Default::default());
                // Tight 0.5-pixel quad centered on (hx + 0.5, hy + 0.5) so exactly
                // pixel (hx, hy) is rasterized with zero boundary tie ambiguity.
                let x0 = ((hx as f32 + 0.25) / w as f32) * 2.0 - 1.0;
                let x1 = ((hx as f32 + 0.75) / w as f32) * 2.0 - 1.0;
                let y1 = 1.0 - ((hy as f32 + 0.25) / h as f32) * 2.0;
                let y0 = 1.0 - ((hy as f32 + 0.75) / h as f32) * 2.0;
                let hole = DepthRect {
                    bounds: [x0, y0, x1, y1],
                    depth: 0.0,
                    pad: [0.0; 3],
                };
                queue.write_buffer(&rect_buf, 0, bytemuck::bytes_of(&hole));
                cull.bind_depth(&device, &depth_view);

                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear + single-pixel hole"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(0.875),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    rp.set_pipeline(&rect_pipeline);
                    rp.set_bind_group(0, &rect_group, &[]);
                    rp.draw(0..6, 0..1);
                }
                cull.build_hiz(&mut encoder, None);
                queue.submit([encoder.finish()]);

                let mut expected_depth = vec![0.875f32; (w * h) as usize];
                expected_depth[(hy * w + hx) as usize] = 0.0;
                let expected_pyr = cpu_build_hiz(&expected_depth, [w as usize, h as usize]);
                let mip_sizes = hiz_mip_dimensions([w, h]);
                let p = cull
                    .hiz
                    .as_ref()
                    .and_then(|hiz| hiz.pyramid.as_ref())
                    .expect("Hi-Z pyramid allocated");
                assert!(
                    p.steps.len() < mip_sizes.len(),
                    "{tier} {w}x{h} should reduce dispatch count: {} vs {}",
                    p.steps.len(),
                    mip_sizes.len()
                );
                let pyramid_tex = p.level_views[0].texture();
                for (mip, (&size, (expected, _))) in mip_sizes.iter().zip(&expected_pyr).enumerate()
                {
                    let got = read_hiz_mip(&device, &queue, pyramid_tex, mip as u32, size);
                    let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                    let exp_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(
                        got_bits, exp_bits,
                        "{tier} {w}x{h} hole ({hx},{hy}) mip {mip} ({size:?})"
                    );
                }
            }
        }
    }
}
