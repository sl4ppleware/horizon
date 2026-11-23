// Entities, dropped items and particles: camera-relative vertices textured
// from the shared 2D atlas, lit through the same lightmap as terrain.
struct Globals {
    view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    fog: vec4<f32>,
    sky_color: vec4<f32>,
    misc: vec4<f32>,
    inverse_view: mat4x4<f32>,
    sunrise: vec4<f32>,
    sun: vec4<f32>,
    // x: sky fog end, y: cloud fog end
    fog_ends: vec4<f32>,
    // Atlas rects (u0, v0, width, height) of end_sky.png and end_portal.png.
    end_portal_sprites: array<vec4<f32>, 2>,
    // x: vanilla's GameTime ((game time % 24000 + partial tick) / 24000)
    // y: Nether cardinal lighting, z: how many dynamic_lights are set,
    // w: the glint scroll clock (`Glint::scroll_ticks`).
    time: vec4<f32>,
    crumbling_sprite: vec4<f32>,
    // Selection outline parameters (read by terrain.wgsl).
    outline_params: vec4<f32>,
    // Horizon Extras' dynamic lights: camera-relative position and level (dynamic_light.rs).
    dynamic_lights: array<vec4<f32>, 16>,
    colored_lights: array<vec4<f32>, 32>,
    colored_count: vec4<u32>,
    emissive_layers: array<vec4<u32>, 128>,
    emissive_count: vec4<u32>,
    sunrise_fan: array<vec4<f32>, 32>,
    filtering: vec4<f32>,
    // HDR output: z 1 with it on; w: apply the no-pack HDR gain; HDR_WRITE_LIGHT independently
    // keeps float alpha as coverage. x and y are unused.
    hdr_light: vec4<f32>,
    // What `hdr_written` reads (hdr_light.wgsl): the display's headroom, the look and debug.
    hdr_settings: vec4<f32>,
    // The eye fog is measured from, camera-relative: zero but in a mod's world view.
    view_origin: vec4<f32>,
    // A mod's world view's own, zero outside one (gfx/world_views.rs's `ViewCamera`): x -1 when
    // mirrored (its screen runs right to left), else 1; y the clip-space depth its clip plane's
    // entities come forward (`fs_entity_clipped`), 0 without one; z 1 with a clip plane.
    world_view: vec4<f32>,
}
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;
@group(0) @binding(3) var lightmap: texture_2d<f32>;
@group(0) @binding(4) var lightmap_sampler: sampler;
// enchanted_glint_item on its own repeating texture (vanilla's GlintSampler).
@group(0) @binding(5) var glint_tex: texture_2d<f32>;
@group(0) @binding(6) var glint_sampler: sampler;
// The armour layers' GlintSampler: vanilla samples enchanted_glint_armor with
// the same repeating, linearly filtered sampler as the item glint.
@group(0) @binding(7) var glint_armor_tex: texture_2d<f32>;
// Compact vertex-to-fragment interface for the general entity/celestial/swirl pipelines
// (6 locations = 80 bytes/vertex instead of 14 locations = 176 bytes): drops weather_rect
// (moved to WeatherOut), drops smooth overlay_color (recomputed in the fragment only when
// an overlay tint is active), drops proj.zw (`in.clip.w` is 1 / clip_out.w), drops
// back_color.a (always color.a) and light_color.a (lightmap alpha is 1.0), and packs
// overlay, cull, tier and 16-bit wrap coordinates into one flat vec2<u32>.
// Fragment inputs stay well within OpenGL's 15-location limit (#8195).
struct EntityOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    // xy: atlas UV; zw: glint UV (`GLINT_TEXTURING` for 7/8/33, model UV for 36, dissolve mask for 38).
    @location(1) uv_glint: vec4<f32>,
    // xy: spherical/cylindrical fog distances; zw: End-portal projected xy (`clip.xy * 0.5 + clip.w * 0.5`).
    @location(2) fog_proj: vec4<f32>,
    // xyz: back_color.rgb (alpha matches color.a); w: light_color.r.
    @location(3) back_light_r: vec4<f32>,
    // xy: light_color.gb (lightmap alpha is always 1.0).
    @location(4) light_gb: vec2<f32>,
    // x: overlay (0..15) | cull (16) | HDR tier (17..24);
    // y: packed 16-bit wrap coords (light.x | (light.y << 8u) | (light.z << 16u) | (light.w << 24u)).
    @location(5) @interpolate(flat) packed_meta: vec2<u32>,
}
// Compact interface for clouds (`vs_cloud` -> `fs_cloud` / `fs_oit_cloud_*`, 2 locations = 20 bytes):
// clouds always use the solid-white `horizon/white` sprite, no normal shading and no lightmap.
struct CloudOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) fog_dist: f32,
}
// Compact interface for entity shadows (`vs_entity_shadow` -> `fs_entity_shadow` / `fs_oit_shadow_*`,
// 3 locations = 32 bytes): lightmap colour is premultiplied into `color` in the vertex stage.
struct ShadowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) fog_distance: vec2<f32>,
}
// Compact interface for weather streaks (`vs_weather` -> `fs_weather` / `fs_oit_weather_*`,
// 4 locations = 48 bytes): lightmap colour is premultiplied into `color` in the vertex stage.
struct WeatherOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) fog_distance: vec2<f32>,
    @location(3) @interpolate(flat) weather_rect: vec4<f32>,
}
// Compact interface for glowing entity silhouettes (`vs_outline` -> `fs_outline`,
// 2 locations = 24 bytes): outline vertices carry no normal, light or fog.
struct OutlineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}
// Unpacked vertex fields for native shader pack wrappers (`gfx/native_entity.rs`,
// `gfx/weather.rs`): never crosses a vanilla vertex-to-fragment stage boundary.
struct Out {
    clip: vec4<f32>,
    uv: vec2<f32>,
    color: vec4<f32>,
    fog_distance: vec2<f32>,
    overlay_color: vec4<f32>,
    overlay: u32,
    wrap: vec2<f32>,
    proj: vec4<f32>,
    back_color: vec4<f32>,
    light_color: vec4<f32>,
    glint_uv: vec2<f32>,
    cull: u32,
    tier: u32,
}
// Shaded entity/celestial output plus the already-fetched atlas texel and fog factor,
// so `entity_written_shade` and `fs_celestial` compute HDR marks without a second atlas fetch
// or a second `linear_fog` evaluation.
struct EntityShade {
    color: vec4<f32>,
    texel: vec4<f32>,
    fogged: f32,
}
fn entity_pack_flags(overlay: u32, cull: u32, tier: u32) -> u32 {
    return (overlay & 0xffffu) | ((cull & 1u) << 16u) | ((tier & 0xffu) << 17u);
}
fn entity_overlay(flags: u32) -> u32 {
    return flags & 0xffffu;
}
fn entity_cull(flags: u32) -> u32 {
    return (flags >> 16u) & 1u;
}
fn entity_tier(flags: u32) -> u32 {
    return (flags >> 17u) & 0xffu;
}
fn entity_pack_wrap(light: vec4<u32>) -> u32 {
    return light.x | (light.y << 8u) | (light.z << 16u) | (light.w << 24u);
}
fn entity_unpack_wrap(packed: u32) -> vec2<f32> {
    return vec2(f32(packed & 0xffffu), f32(packed >> 16u)) / 65535.0;
}
fn entity_light_rgb(in: EntityOut) -> vec3<f32> {
    return vec3(in.back_light_r.w, in.light_gb.x, in.light_gb.y);
}
// Shipped light.glsl: pre-normalized directional light vectors (0.2, 1.0, -0.7) and
// (-0.2, +-1.0, 0.7); normals are already transformed and RGBA8_SNORM packed.
const ENTITY_LIGHT_0 = vec3<f32>(0.16169041, 0.80845207, -0.56591648);
const ENTITY_LIGHT_1_OVERWORLD = vec3<f32>(-0.16169041, 0.80845207, 0.56591648);
const ENTITY_LIGHT_1_NETHER = vec3<f32>(-0.16169041, -0.80845207, 0.56591648);
fn entity_light(normal: vec3<f32>, nether: bool) -> f32 {
    let l1 = select(ENTITY_LIGHT_1_OVERWORLD, ENTITY_LIGHT_1_NETHER, nether);
    return min(1.0, (max(0.0, dot(ENTITY_LIGHT_0, normal)) + max(0.0, dot(l1, normal))) * 0.6 + 0.4);
}
// The OverlayTexture texel entity.vsh fetches (constants from overlay.rs):
// the hurt red for overlays 1 and 8, the white row at column 15 for the
// primed TNT flash (5) and at column overlay - 16 for 16..=31 (a creeper's
// swelling). Alpha 1 changes nothing.
fn overlay_texel(overlay: u32) -> vec4<f32> {
    if overlay == 1u || overlay == 8u {
        // PvP & HUD's hit colour (pvp.rs `Hit::texel`) rides in colored_count.y; 0 is vanilla's.
        let hit = globals.colored_count.y;
        return select(HURT_OVERLAY, unpack4x8unorm(hit), hit != 0u);
    }
    if overlay == 5u { return vec4(1.0, 1.0, 1.0, WHITE_OVERLAY_ALPHA[15]); }
    if overlay >= 16u && overlay < 32u { return vec4(1.0, 1.0, 1.0, WHITE_OVERLAY_ALPHA[overlay - 16u]); }
    return vec4(1.0);
}
// Horizon Extras' dynamic lights (dynamic_light.rs): the block light, in the vertex's units
// (16 per level), a vertex at camera-relative `pos` gets from the frame's light sources, one
// level less per block. time.z of them are set: none when the extra is off, so no loop runs.
fn dynamic_block_light(pos: vec3<f32>) -> f32 {
    let count = min(u32(globals.time.z), 16u);
    var level = 0.0;
    for (var i = 0u; i < count; i++) {
        let l = globals.dynamic_lights[i];
        level = max(level, l.w - distance(pos, l.xyz));
    }
    return level * 16.0;
}
fn native_colored_light(pos: vec3<f32>) -> vec3<f32> {
    var rgb = vec3(0.0);
    for (var i = 0u; i < min(globals.colored_count.x, 16u); i++) {
        let at = globals.colored_lights[i * 2u];
        let color = globals.colored_lights[i * 2u + 1u];
        rgb += color.rgb * max(0.0, 1.0 - distance(pos, at.xyz) / at.w) * color.a;
    }
    return min(rgb, vec3(1.0));
}
// Celestial bodies, clouds, beacon beams, End portals and breaking cracks are unlit.
fn entity_is_unlit(overlay: u32) -> bool {
    return overlay == 2u || overlay == 4u || overlay == 6u || overlay == 9u || overlay == 10u
        || overlay == 11u || overlay == 12u;
}
fn entity_sample_light(position: vec3<f32>, light: vec4<u32>) -> vec4<f32> {
    let l = vec2(max(f32(light.y), dynamic_block_light(position)), f32(light.x));
    var lm = textureSampleLevel(lightmap, lightmap_sampler,
        clamp(l / 256.0 + 0.5 / 16.0, vec2(0.5 / 16.0), vec2(15.5 / 16.0)), 0.0);
    if globals.colored_count.x > 0u {
        lm = vec4(min(lm.rgb + native_colored_light(position), vec3(1.0)), lm.a);
    }
    return lm;
}
fn entity_out(position: vec3<f32>, uv: vec2<f32>, raw_overlay: u32, color: vec4<f32>,
    light: vec4<u32>, normal: vec4<f32>, glint_uv: vec2<f32>) -> EntityOut {
    let overlay = raw_overlay & 0xffffu;
    let tier = (raw_overlay >> 16u) & 0xffu;
    let clip = globals.view_proj * vec4(position, 1.0);
    let proj_xy = clip.xy * 0.5 + clip.w * 0.5;
    let unlit = entity_is_unlit(overlay);
    var light_rgb = vec3(1.0);
    if !unlit {
        light_rgb = entity_sample_light(position, light).rgb;
    }
    // Normal w: 0 keeps the baked colour, +-127 shades by the normal, and a small positive 1
    // keeps the baked colour but culls back faces (entitySolid models such as the decorated pot).
    // -126/127 shades like -127 and also culls (entityCutoutCull models such as the bat, and the
    // entitySolid decorated pot).
    let cull = select(0u, 1u, (normal.w > 0.0 && normal.w < 0.5) || (normal.w < 0.0 && normal.w > -0.996));
    var c = color;
    var back_rgb = c.rgb;
    if abs(normal.w) >= 0.5 {
        let nether = globals.time.y != 0.0;
        c = vec4(c.rgb * entity_light(normal.xyz, nether), c.a);
        if cull == 0u && normal.w < 0.0 {
            back_rgb = back_rgb * entity_light(-normal.xyz, nether);
        } else {
            back_rgb = c.rgb;
        }
    }
    // Sky geometry (clouds, celestial bodies) is exempt from render-distance fog.
    var fog_dist = vec2(0.0);
    if overlay == 4u {
        fog_dist.x = length(position);
    } else if overlay < 2u || overlay > 4u {
        let eye = position - globals.view_origin.xyz;
        fog_dist = vec2(length(eye), max(length(eye.xz), abs(eye.y)));
    }
    return EntityOut(
        clip,
        c,
        vec4(uv, glint_uv),
        vec4(fog_dist, proj_xy),
        vec4(back_rgb, light_rgb.r),
        light_rgb.gb,
        vec2(entity_pack_flags(overlay, cull, tier), entity_pack_wrap(light)),
    );
}
fn entity_vertex(position: vec3<f32>, uv: vec2<f32>,
    overlay: u32, color: vec4<f32>, light: vec4<u32>, normal: vec4<f32>) -> Out {
    var out: Out;
    out.clip = globals.view_proj * vec4(position, 1.0);
    out.proj = vec4(out.clip.xy * 0.5 + out.clip.w * 0.5, out.clip.zw);
    out.uv = uv;
    let unlit = entity_is_unlit(overlay);
    out.light_color = vec4(1.0);
    if !unlit {
        out.light_color = entity_sample_light(position, light);
    }
    // Overlay 6 (beacon beams): the light bytes carry the sprite's v range; 11 and 12
    // (cracks): the destroy stage sprite's origin.
    out.wrap = vec2(f32(light.x | (light.y << 8u)), f32(light.z | (light.w << 8u))) / 65535.0;
    // minecraft_mix_light: the vertex colour times the cardinal light only.
    var c = color;
    var back = c;
    out.cull = select(0u, 1u, (normal.w > 0.0 && normal.w < 0.5) || (normal.w < 0.0 && normal.w > -0.996));
    if abs(normal.w) >= 0.5 {
        let nether = globals.time.y != 0.0;
        c = vec4(c.rgb * entity_light(normal.xyz, nether), c.a);
        if out.cull == 0u && normal.w < 0.0 {
            back = vec4(back.rgb * entity_light(-normal.xyz, nether), back.a);
        } else {
            back = c;
        }
    }
    out.color = c;
    out.back_color = back;
    out.overlay_color = overlay_texel(overlay);
    // Sky geometry (clouds, celestial bodies) is exempt from render-distance fog.
    out.fog_distance = vec2(0.0);
    if overlay == 4u {
        out.fog_distance.x = length(position);
    } else if overlay < 2u || overlay > 4u {
        let eye = position - globals.view_origin.xyz;
        out.fog_distance = vec2(length(eye), max(length(eye.xz), abs(eye.y)));
    }
    out.overlay = overlay;
    out.glint_uv = vec2(0.0);
    out.tier = 0u;
    return out;
}
@vertex fn vs_entity(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>,
    @location(2) overlay: u32, @location(3) color: vec4<f32>, @location(4) light: vec4<u32>,
    @location(5) normal: vec4<f32>, @location(6) glint_uv: vec2<f32>) -> EntityOut {
    return entity_out(position, uv, overlay, color, light, normal, glint_uv);
}
// Retained bolt-local geometry, translated using the same f64 camera
// subtraction as the former CPU path before adding each local f32 position.
// The bolt offset is instance attribute 7: 6 is the entity vertex's glint UV.
@vertex fn vs_lightning(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>,
    @location(2) overlay: u32, @location(3) color: vec4<f32>, @location(4) light: vec4<u32>, @location(5) normal: vec4<f32>,
    @location(7) offset: vec3<f32>) -> EntityOut {
    return entity_out(offset + position, uv, overlay, color, light, normal, vec2(0.0));
}
struct CloudInfo {
    offset: vec4<f32>,
    colors: array<vec4<f32>, 4>,
}
@group(1) @binding(0) var<uniform> cloud: CloudInfo;
@vertex fn vs_cloud(@location(0) position: vec3<f32>, @location(4) light: vec4<u32>) -> CloudOut {
    let pos = position + cloud.offset.xyz;
    return CloudOut(
        globals.view_proj * vec4(pos, 1.0),
        cloud.colors[light.x],
        length(pos),
    );
}
@vertex fn vs_entity_shadow(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>,
    @location(3) color: vec4<f32>, @location(4) light: vec4<u32>) -> ShadowOut {
    let eye = position - globals.view_origin.xyz;
    let light_color = entity_sample_light(position, light);
    return ShadowOut(
        globals.view_proj * vec4(position, 1.0),
        uv,
        color * light_color,
        vec2(length(eye), max(length(eye.xz), abs(eye.y))),
    );
}
@vertex fn vs_outline(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>,
    @location(3) color: vec4<f32>) -> OutlineOut {
    return OutlineOut(globals.view_proj * vec4(position, 1.0), uv, color);
}
@vertex fn vs_water_mask(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return globals.view_proj * vec4(position, 1.0);
}
// rendertype_end_portal.fsh: the layer colours.
const PORTAL_COLORS = array<vec3<f32>, 16>(
    vec3(0.022087, 0.098399, 0.110818),
    vec3(0.011892, 0.095924, 0.089485),
    vec3(0.027636, 0.101689, 0.100326),
    vec3(0.046564, 0.109883, 0.114838),
    vec3(0.064901, 0.117696, 0.097189),
    vec3(0.063761, 0.086895, 0.123646),
    vec3(0.084817, 0.111994, 0.166380),
    vec3(0.097489, 0.154120, 0.091064),
    vec3(0.106152, 0.131144, 0.195191),
    vec3(0.097721, 0.110188, 0.187229),
    vec3(0.133516, 0.138278, 0.148582),
    vec3(0.070006, 0.243332, 0.235792),
    vec3(0.196766, 0.142899, 0.214696),
    vec3(0.047281, 0.315338, 0.321970),
    vec3(0.204675, 0.390010, 0.302066),
    vec3(0.080955, 0.314821, 0.661491),
);
// textureProj on a repeating texture that lives in the atlas at `rect`.
fn portal_sample(rect: vec4<f32>, p: vec4<f32>) -> vec3<f32> {
    let st = fract(p.xy / p.w);
    return textureSampleLevel(atlas, atlas_sampler, rect.xy + st * rect.zw, 0.0).rgb;
}
// texProj0 * end_portal_layer(layer): the row vector through mat4(scale * rotate),
// translate and SCALE_TRANSLATE in turn.
fn portal_layer(p: vec4<f32>, layer: f32) -> vec4<f32> {
    let a = radians((layer * layer * 4321.0 + layer * 9.0) * 2.0);
    let c = cos(a);
    let s = sin(a);
    let k = (4.5 - layer / 4.0) * 2.0;
    var v = vec4(k * (c * p.x - s * p.y), k * (s * p.x + c * p.y), p.z, p.w);
    v = vec4(v.x + 17.0 / layer * v.w, v.y + (2.0 + layer / 1.5) * (globals.time.x * 1.5) * v.w, v.z, v.w);
    return vec4(0.5 * v.x + 0.25 * v.w, 0.5 * v.y + 0.25 * v.w, v.z, v.w);
}
fn linear_fog(d: f32, start: f32, end: f32) -> f32 {
    if d <= start { return 0.0; }
    if d >= end { return 1.0; }
    return (d - start) / (end - start);
}
// TextureTransform.setupGlintTexturing(0.16), ARMOR_ENTITY_GLINT_TEXTURING.
// globals.time.w is (Util.getMillis() * glintSpeed * 8) modulo 330000, the
// common period of both offsets. cos/sin of 0.17453292 rad (10 deg).
const ARMOR_GLINT_COS: f32 = 0.98480773;
const ARMOR_GLINT_SIN: f32 = 0.17364818;
fn armor_glint_uv(model: vec2<f32>) -> vec2<f32> {
    let ticks = globals.time.w;
    let offset0 = (ticks % 110000.0) / 110000.0;
    let offset1 = (ticks % 30000.0) / 30000.0;
    let scaled = model * 0.16;
    return vec2(scaled.x * ARMOR_GLINT_COS - scaled.y * ARMOR_GLINT_SIN - offset0,
                scaled.x * ARMOR_GLINT_SIN + scaled.y * ARMOR_GLINT_COS + offset1);
}
// PvP & HUD's glint colour (pvp.rs `Tweaks::glint_texel`, 0 for vanilla's): the sample's
// brightest channel over the chosen colour, so the shimmer keeps its pattern and strength.
fn tinted_glint(glint: vec4<f32>, packed: u32) -> vec4<f32> {
    if packed == 0u { return glint; }
    let tint = unpack4x8unorm(packed).rgb;
    return vec4(max(glint.r, max(glint.g, glint.b)) * tint, glint.a);
}
// RenderPipelines.GLINT: the source colour is scaled by itself, so `GlintAlpha`
// is squared along with the sample.
fn glint_over(color: vec4<f32>, glint: vec4<f32>) -> vec4<f32> {
    let alpha = globals.fog_ends.w;
    let g = alpha * tinted_glint(glint, globals.colored_count.z);
    return vec4(color.rgb + g.rgb * g.rgb, max(color.a, alpha));
}
// CRUMBLING_UV_BEGIN
fn crumbling_uv(uv: vec2<f32>, packed_origin: vec2<f32>, sprite_size: vec2<f32>, dimensions: vec2<u32>) -> vec2<f32> {
    let size = vec2<f32>(dimensions);
    // Undo stage_light's 16-bit quantization before adding local texels. Otherwise
    // different stages can fall on opposite sides of a nearest-sampling boundary.
    let origin = round(packed_origin * size);
    let extent = max(round(sprite_size * size), vec2(1.0));
    let local = min(floor(fract(uv) * extent), extent - vec2(1.0));
    return (origin + local + vec2(0.5)) / size;
}
// CRUMBLING_UV_END
// clouds.fsh: alpha fades out towards the cloud fog end, with zero atlas fetches.
fn cloud_color(in: CloudOut) -> vec4<f32> {
    return vec4(in.color.rgb, in.color.a * (1.0 - linear_fog(in.fog_dist, 0.0, globals.fog_ends.y)));
}
@fragment fn fs_cloud(in: CloudOut) -> @location(0) vec4<f32> {
    let color = cloud_color(in);
    if HDR_WRITE_LIGHT && globals.hdr_light.w != 0.0 {
        return vec4(hdr_written(color.rgb, 1.0, globals.hdr_settings), color.a);
    }
    return frame_stored(color);
}
@fragment fn fs_entity(in: EntityOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return entity_written_shade(in, entity_shade(in, front));
}
// Takes entity_shade's result rather than calling it: FXC (DX12) fails with "argument pulled into
// unrelated predicate" when entity_shade's discards sit a function call below the entry point.
fn entity_written_shade(in: EntityOut, shade: EntityShade) -> vec4<f32> {
    let overlay = entity_overlay(in.packed_meta.x);
    if HDR_WRITE_LIGHT && globals.hdr_light.w != 0.0 && overlay != 11u && overlay != 12u {
        let tier = entity_tier(in.packed_meta.x);
        var mark = 1.0;
        if tier != 0u {
            mark = hdr_entity_mark_from(tier, shade.texel, in.color, shade.fogged);
        }
        return vec4(hdr_written(shade.color.rgb, mark, globals.hdr_settings), shade.color.a);
    }
    if FRAME_LINEAR && (overlay == 11u || overlay == 12u) {
        // The cracks scale what is under them by twice their colour (CRUMBLING_BLEND). Vanilla's
        // product is in gamma, so under light it is (2c)^2.2: they give half of that.
        return vec4(pow(2.0 * max(shade.color.rgb, vec3(0.0)), vec3(2.2)) * 0.5, shade.color.a);
    }
    return frame_stored(shade.color);
}
// Companion for native shader pack wrappers (`gfx/native_entity.rs`).
fn entity_written_color(in: Out, color: vec4<f32>) -> vec4<f32> {
    if HDR_WRITE_LIGHT && globals.hdr_light.w != 0.0 && in.overlay != 11u && in.overlay != 12u {
        var mark = 1.0;
        if in.tier != 0u {
            mark = hdr_entity_mark(in);
        }
        return vec4(hdr_written(color.rgb, mark, globals.hdr_settings), color.a);
    }
    if FRAME_LINEAR && (in.overlay == 11u || in.overlay == 12u) {
        return vec4(pow(2.0 * max(color.rgb, vec3(0.0)), vec3(2.2)) * 0.5, color.a);
    }
    return frame_stored(color);
}
// `fs_entity` in a clipped world view (gfx/world_views.rs's `ENTITY_MARGIN`), whose pipeline has
// no constant bias: after clipping, so the plane cuts entities where it cuts terrain, each pixel
// comes forward by `world_view.y` over its clip w (`in.clip.w` is 1 / clip_out.w), a share of its distance.
struct ClippedEntity {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}
@fragment fn fs_entity_clipped(in: EntityOut, @builtin(front_facing) front: bool) -> ClippedEntity {
    let color = entity_written_shade(in, entity_shade(in, front));
    return ClippedEntity(color, min(in.clip.z + globals.world_view.y * in.clip.w, 1.0));
}
@fragment fn fs_entity_unmarked(in: EntityOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return frame_stored(entity_shade(in, front).color);
}
fn entity_shade(in: EntityOut, front: bool) -> EntityShade {
    let overlay = entity_overlay(in.packed_meta.x);
    let cull = entity_cull(in.packed_meta.x);
    var uv = in.uv_glint.xy;
    if overlay == 6u {
        let wrap = entity_unpack_wrap(in.packed_meta.y);
        uv.y = wrap.x + fract(uv.y) * wrap.y;
    } else if overlay == 11u || overlay == 12u {
        // Each crack's stage sprite: its origin in the light bytes, its size in glint_uv (breaking::sprite).
        uv = crumbling_uv(uv, entity_unpack_wrap(in.packed_meta.y), in.uv_glint.zw, textureDimensions(atlas));
    }
    let texel = textureSample(atlas, atlas_sampler, uv);
    var tint = vec4(select(in.back_light_r.xyz, in.color.rgb, front), in.color.a);
    if overlay == 38u {
        // RenderPipelines.ENTITY_CUTOUT_DISSOLVE (entity.fsh's DISSOLVE): gone where the vertex
        // alpha is below the mask's (its atlas UV in glint_uv), the rest drawn opaque.
        if tint.a < textureSampleLevel(atlas, atlas_sampler, in.uv_glint.zw, 0.0).a { discard; }
        tint.a = 1.0;
    }
    var color = texel * tint;
    let fog_dist = in.fog_proj.xy;
    if overlay == 9u || overlay == 10u {
        // rendertype_end_portal: end_sky and PORTAL_LAYERS (15 for portals, 16
        // for gateways) layers of end_portal.png projected from the screen.
        let proj = vec4(in.fog_proj.zw * in.clip.w, 0.0, 1.0);
        let layers = select(15, 16, overlay == 10u);
        var c = portal_sample(globals.end_portal_sprites[0], proj) * PORTAL_COLORS[0];
        for (var i = 0; i < layers; i++) {
            c += portal_sample(globals.end_portal_sprites[1], portal_layer(proj, f32(i + 1))) * PORTAL_COLORS[i];
        }
        color = vec4(c, 1.0);
    } else if overlay == 4u {
        // clouds.fsh: alpha fades out towards the cloud fog end.
        let c = vec4(color.rgb, color.a * (1.0 - linear_fog(fog_dist.x, 0.0, globals.fog_ends.y)));
        var fogged = 0.0;
        if HDR_WRITE_LIGHT && globals.hdr_light.w != 0.0 && entity_tier(in.packed_meta.x) != 0u {
            fogged = max(linear_fog(fog_dist.x, globals.fog.x, globals.fog.y),
                         linear_fog(fog_dist.y, globals.fog.z, globals.fog.w)) * globals.fog_color.a;
        }
        return EntityShade(c, texel, fogged);
    }
    if color.a < 0.1 || (cull != 0u && !front) { discard; }
    // entity.fsh calculateFinalColor: the overlay (hurt red, TNT white) over the
    // lit texture, then the lightmap, then the glint and fog.
    if overlay == 1u || overlay == 5u || overlay == 8u || (overlay >= 16u && overlay < 32u) {
        let oc = overlay_texel(overlay);
        color = vec4(mix(oc.rgb, color.rgb, oc.a), color.a);
    }
    color = vec4(color.rgb * entity_light_rgb(in), color.a);
    if overlay == 7u || overlay == 8u || overlay == 33u {
        // 33: a translucent item with the glint.
        // Enchantment glint: item.fsh's GLINT path, sampled from
        // enchanted_glint_item at the `GLINT_TEXTURING` UV.
        color = glint_over(color, textureSample(glint_tex, glint_sampler, in.uv_glint.zw));
    } else if overlay == 36u {
        // Enchanted armour (mobs::render::ARMOR_GLINT): enchanted_glint_armor at the scrolling model UV.
        color = glint_over(
            color,
            textureSample(glint_armor_tex, glint_sampler, armor_glint_uv(in.uv_glint.zw)),
        );
    }
    let f = max(linear_fog(fog_dist.x, globals.fog.x, globals.fog.y),
                linear_fog(fog_dist.y, globals.fog.z, globals.fog.w));
    let fogged = f * globals.fog_color.a;
    return EntityShade(vec4(mix(color.rgb, globals.fog_color.rgb, fogged), color.a), texel, fogged);
}
// Companion for native shader pack wrappers (`gfx/native_entity.rs`).
fn entity_color(in: Out, front: bool) -> vec4<f32> {
    var uv = in.uv;
    if in.overlay == 6u { uv.y = in.wrap.x + fract(in.uv.y) * in.wrap.y; }
    else if in.overlay == 11u || in.overlay == 12u {
        uv = crumbling_uv(in.uv, in.wrap, in.glint_uv, textureDimensions(atlas));
    }
    let texel = textureSample(atlas, atlas_sampler, uv);
    var tint = select(in.back_color, in.color, front);
    if in.overlay == 38u {
        if tint.a < textureSampleLevel(atlas, atlas_sampler, in.glint_uv, 0.0).a { discard; }
        tint.a = 1.0;
    }
    var color = texel * tint;
    if in.overlay == 9u || in.overlay == 10u {
        let layers = select(15, 16, in.overlay == 10u);
        var c = portal_sample(globals.end_portal_sprites[0], in.proj) * PORTAL_COLORS[0];
        for (var i = 0; i < layers; i++) {
            c += portal_sample(globals.end_portal_sprites[1], portal_layer(in.proj, f32(i + 1))) * PORTAL_COLORS[i];
        }
        color = vec4(c, 1.0);
    } else if in.overlay == 4u {
        return vec4(color.rgb, color.a * (1.0 - linear_fog(in.fog_distance.x, 0.0, globals.fog_ends.y)));
    }
    if color.a < 0.1 || (in.cull != 0u && !front) { discard; }
    color = vec4(mix(in.overlay_color.rgb, color.rgb, in.overlay_color.a), color.a);
    color *= in.light_color;
    if in.overlay == 7u || in.overlay == 8u || in.overlay == 33u {
        color = glint_over(color, textureSample(glint_tex, glint_sampler, in.glint_uv));
    } else if in.overlay == 36u {
        color = glint_over(
            color,
            textureSample(glint_armor_tex, glint_sampler, armor_glint_uv(in.glint_uv)),
        );
    }
    let f = max(linear_fog(in.fog_distance.x, globals.fog.x, globals.fog.y),
                linear_fog(in.fog_distance.y, globals.fog.z, globals.fog.w));
    return vec4(mix(color.rgb, globals.fog_color.rgb, f * globals.fog_color.a), color.a);
}

// rendertype_entity_shadow.fsh: the shadow texture times the vertex colour (with lightmap
// premultiplied in `vs_entity_shadow`), then fog, with no alpha cutout, so a faint shadow
// in dim light still draws.
@fragment fn fs_entity_shadow(in: ShadowOut) -> @location(0) vec4<f32> {
    return frame_stored(shadow_color(in));
}
fn shadow_color(in: ShadowOut) -> vec4<f32> {
    let color = textureSample(atlas, atlas_sampler, in.uv) * in.color;
    let f = max(linear_fog(in.fog_distance.x, globals.fog.x, globals.fog.y),
                linear_fog(in.fog_distance.y, globals.fog.z, globals.fog.w));
    return vec4(mix(color.rgb, globals.fog_color.rgb, f * globals.fog_color.a), color.a);
}

// rendertype_outline.fsh: the vertex colour (the team colour; outline vertices carry no
// normal, so no cardinal light) wherever the texture is not fully clear.
@fragment fn fs_outline(in: OutlineOut) -> @location(0) vec4<f32> {
    if textureSample(atlas, atlas_sampler, in.uv).a == 0.0 { discard; }
    return vec4(in.color.rgb, 1.0);
}

// HDR's mark (hdr_look.wgsl `hdr_mark`) of a light-giving particle's or entity's pixel, the
// way terrain.wgsl works out a light source's, reusing the already-fetched texel and fog factor.
fn hdr_entity_mark_from(tier: u32, texel: vec4<f32>, color: vec4<f32>, fogged: f32) -> f32 {
    // What gives no light stays unmarked.
    if texel.a < 0.1 || tier == 0u { return 1.0; }
    // A layer that fades (the warden's) shines by what's left of it.
    let y = brightness(texel.rgb * color.rgb) * color.a * (1.0 - fogged);
    return hdr_mark(tier, y);
}
fn hdr_entity_mark(in: Out) -> f32 {
    if in.tier == 0u { return 1.0; }
    var uv = in.uv;
    if in.overlay == 6u { uv.y = in.wrap.x + fract(in.uv.y) * in.wrap.y; }
    let texel = textureSampleLevel(atlas, atlas_sampler, uv, 0.0);
    // Fog hides a light source as it hides everything else.
    let fogged = max(linear_fog(in.fog_distance.x, globals.fog.x, globals.fog.y),
        linear_fog(in.fog_distance.y, globals.fog.z, globals.fog.w)) * globals.fog_color.a;
    return hdr_entity_mark_from(in.tier, texel, in.color, fogged);
}

fn celestial_shade(in: EntityOut) -> EntityShade {
    let overlay = entity_overlay(in.packed_meta.x);
    let fog_dist = in.fog_proj.xy;
    // RenderPipelines.DRAGON_RAYS / rendertype_lightning.fsh: untextured
    // vertex colour, with fog fading both RGB and alpha before additive blending.
    if overlay == 37u {
        let fog = max(linear_fog(fog_dist.x, globals.fog.x, globals.fog.y),
                      linear_fog(fog_dist.y, globals.fog.z, globals.fog.w));
        return EntityShade(in.color * (1.0 - fog), vec4(1.0), fog * globals.fog_color.a);
    }
    let texel = textureSample(atlas, atlas_sampler, in.uv_glint.xy);
    let base = texel * in.color;
    let color = vec4(base.rgb * entity_light_rgb(in), base.a);
    if overlay == 45u {
        // A Polytone custom particle drawn additively: the fog over its colour, as vanilla's
        // particle.fsh, under the additive blend (no alpha cutout: its faint texels add up).
        let f = max(linear_fog(fog_dist.x, globals.fog.x, globals.fog.y),
                    linear_fog(fog_dist.y, globals.fog.z, globals.fog.w));
        let fogged = f * globals.fog_color.a;
        return EntityShade(vec4(mix(color.rgb, globals.fog_color.rgb, fogged), color.a), texel, fogged);
    }
    var fogged = 0.0;
    if HDR_WRITE_LIGHT && globals.hdr_light.w != 0.0 && entity_tier(in.packed_meta.x) != 0u {
        fogged = max(linear_fog(fog_dist.x, globals.fog.x, globals.fog.y),
                     linear_fog(fog_dist.y, globals.fog.z, globals.fog.w)) * globals.fog_color.a;
    }
    return EntityShade(color, texel, fogged);
}
fn celestial_color(in: EntityOut) -> vec4<f32> {
    return celestial_shade(in).color;
}
@fragment fn fs_celestial(in: EntityOut) -> @location(0) vec4<f32> {
    let shade = celestial_shade(in);
    if HDR_WRITE_LIGHT && globals.hdr_light.w != 0.0 {
        let tier = entity_tier(in.packed_meta.x);
        var mark = 1.0;
        if tier != 0u {
            mark = hdr_entity_mark_from(tier, shade.texel, in.color, shade.fogged);
        }
        return vec4(hdr_written(shade.color.rgb, mark, globals.hdr_settings), shade.color.a);
    }
    return frame_stored(shade.color);
}

// The sky's celestial pass (sun, moon, the End sky): fs_celestial's light,
// premultiplied for an additive (one, one) blend. On the float frame, where the world is
// lifted, a texel past its tier's window shines at the mark (hdr_mark) of its tier, so the
// sun's and moon's discs shine; the dark rest of their quads and the dim End sky stay
// unmarked.
fn sky_celestial_color(in: EntityOut, color: vec4<f32>) -> vec4<f32> {
    let added = color.rgb * color.a;
    // The sun at the brightest tier, the moon at its own (tier); geometry that sets
    // none (the End sky, the End flash) stays brightest.
    let raw_tier = entity_tier(in.packed_meta.x);
    let tier = select(raw_tier, 4u, raw_tier == 0u);
    let y = brightness(added);
    // Only what's past the tier's window takes the mark: a soft edge keeps the sky's light,
    // or it would come out darker than the sky around it.
    let full = hdr_lifts_fully(tier, y);
    if HDR_WRITE_LIGHT && globals.hdr_light.w != 0.0 {
        var mark = 1.0;
        if full { mark = hdr_mark(tier, y); }
        return vec4(hdr_written(added, mark, globals.hdr_settings), color.a);
    }
    // Premultiplied after the decode: the light the texel adds, at its coverage.
    if HDR_WRITE_LIGHT { return vec4(frame_stored(color).rgb * color.a, color.a); }
    return vec4(added, 0.0);
}
@fragment fn fs_sky_celestial(in: EntityOut) -> @location(0) vec4<f32> {
    return sky_celestial_color(in, celestial_color(in));
}

// The sun in Horizon's lifted world (gfx/celestial.rs): its disc shows vanilla's colour, the sky
// plus the sun clipped at white channel by channel, with white at S: the scene's peak, and on a
// display above it the display's, up to a cap (hdr_look.wgsl `hdr_sun_peak` of the display's
// headroom, `hdr_settings.x`). The sky under it is vanilla's, at paper white, so in the frame's
// encoding that is min(S * sun + S * frame, S): this pass blends src + frame * alpha, then
// fs_sky_sun_clip takes the min. Added in the frame's gamma instead, sky and sun pass S and the
// tone curve's hold flattens the disc in the sky's hue. w is the very bright tier's lift by the
// texel's brightness (hdr_mark's): from 0 (sun.png's faint halo, the moon) to 1 both passes go
// smoothly from fs_sky_celestial's add to the clipped disc.
fn sun_weight(in: EntityOut, added: vec3<f32>) -> f32 {
    let w = hdr_lift(4u, brightness(added));
    return select(0.0, w, entity_tier(in.packed_meta.x) == 4u);
}
@fragment fn fs_sky_sun(in: EntityOut) -> @location(0) vec4<f32> {
    let color = celestial_color(in);
    let added = color.rgb * color.a;
    let w = sun_weight(in, added);
    let peak = hdr_sun_peak(globals.hdr_settings.x);
    let sun = hdr_written_sun(added, globals.hdr_settings);
    return vec4(mix(sky_celestial_color(in, color).rgb, sun, w), mix(1.0, pow(peak, 1.0 / 2.2), w));
}
// The clip at S in the frame's encoding where w is 1, none (half float's largest) where it is 0.
@fragment fn fs_sky_sun_clip(in: EntityOut) -> @location(0) vec4<f32> {
    let color = celestial_color(in);
    let peak = pow(hdr_sun_peak(globals.hdr_settings.x), 1.0 / 2.2);
    return vec4(vec3(peak / max(sun_weight(in, color.rgb * color.a), peak / 65504.0)), 1.0);
}

// RenderPipelines.ENERGY_SWIRL (entity.fsh with EMISSIVE and NO_OVERLAY): the sprite, scrolled by
// the UV and repeated within itself, times the layer colour 0xFF808080, unlit; the pipeline adds
// it to the frame. The light bytes carry the sprite's origin and the colour bytes its size, both
// as breaking::stage_light packs them.
fn energy_swirl_base(in: EntityOut) -> vec4<f32> {
    let bytes = round(in.color * 255.0);
    let size = vec2(bytes.x + bytes.y * 256.0, bytes.z + bytes.w * 256.0) / 65535.0;
    let color = textureSample(atlas, atlas_sampler, crumbling_uv(in.uv_glint.xy, entity_unpack_wrap(in.packed_meta.y), size, textureDimensions(atlas)));
    if color.a < 0.1 { discard; }
    return vec4(color.rgb * (128.0 / 255.0), color.a);
}
fn energy_swirl_fog(in: EntityOut) -> f32 {
    let fog_dist = in.fog_proj.xy;
    let f = max(linear_fog(fog_dist.x, globals.fog.x, globals.fog.y),
                linear_fog(fog_dist.y, globals.fog.z, globals.fog.w));
    return f * globals.fog_color.a;
}
@fragment fn fs_energy_swirl(in: EntityOut) -> @location(0) vec4<f32> {
    let color = energy_swirl_base(in);
    return frame_stored(vec4(mix(color.rgb, globals.fog_color.rgb, energy_swirl_fog(in)), color.a));
}

// RenderPipelines.WATER_MASK: depth only (the pipeline masks the colour off).
@fragment fn fs_water_mask() -> @location(0) vec4<f32> {
    return vec4(0.0);
}

// WEATHER_MATH_BEGIN
// Unsigned 64-bit helpers (low word first). Snow adds a random double to a
// float drift before casting to float. Retaining the 53-bit base avoids early
// rounding, including ties, without requiring optional GPU float64 support.
fn weather_shr_jam(v: vec2<u32>, n: u32) -> vec2<u32> {
    if n == 0u { return v; }
    if n < 32u {
        return vec2((v.x >> n) | (v.y << (32u - n)) | select(0u, 1u, (v.x << (32u - n)) != 0u), v.y >> n);
    }
    if n == 32u { return vec2(v.y | select(0u, 1u, v.x != 0u), 0u); }
    if n < 64u {
        return vec2((v.y >> (n - 32u)) | select(0u, 1u, v.x != 0u || (v.y << (64u - n)) != 0u), 0u);
    }
    return vec2(select(0u, 1u, any(v != vec2(0u))), 0u);
}
fn weather_add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
fn weather_sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}
fn weather_double_plus_float(base: vec2<u32>, delta: f32) -> f32 {
    // The random base is nonnegative and either zero or >= 2^-53.
    if all(base == vec2(0u)) { return select(delta, 0.0, delta == 0.0); }
    let db = bitcast<u32>(delta);
    let mag = db & 0x7fffffffu;
    let be = i32((base.y >> 20u) & 2047u) - 1023;
    var de = i32(mag >> 23u) - 127;
    var dm = (mag & 0x7fffffu) | 0x800000u;
    if mag < 0x800000u {
        dm = mag;
        if dm == 0u { de = -200; }
        else {
            let shift = countLeadingZeros(dm) - 8u;
            dm <<= shift;
            de = -126 - i32(shift);
        }
    }
    let e = max(be, de);
    // Leading significand bit at 61, with room for addition and guard bits.
    var a = vec2(base.x << 9u, ((base.y & 0xfffffu) | 0x100000u) << 9u | (base.x >> 23u));
    var b = vec2(0u, dm << 6u);
    a = weather_shr_jam(a, u32(e - be));
    b = weather_shr_jam(b, u32(e - de));
    var sign = 0u;
    var value: vec2<u32>;
    if (db >> 31u) == 0u { value = weather_add64(a, b); }
    else if a.y > b.y || (a.y == b.y && a.x >= b.x) { value = weather_sub64(a, b); }
    else { value = weather_sub64(b, a); sign = 0x80000000u; }
    if all(value == vec2(0u)) { return 0.0; }
    let leading = select(31u - countLeadingZeros(value.x), 63u - countLeadingZeros(value.y), value.y != 0u);
    var exponent = e + i32(leading) - 61;
    // Normalize the leading bit to 62. After jamming, bit 38 is the guard
    // bit and bits 0..37 distinguish a tie from a value above the midpoint.
    let shift = 62u - leading;
    if shift >= 32u { value = vec2(0u, value.x << (shift - 32u)); }
    else if shift != 0u { value = vec2(value.x << shift, (value.y << shift) | (value.x >> (32u - shift))); }
    // Vanilla first rounds the addition to double, then casts to float.
    // Round at bit 10 (53 retained bits) before the 24-bit rounding below;
    // otherwise a double-rounding midpoint can select the wrong float.
    let double_guard = (value.x & 512u) != 0u;
    let double_sticky = (value.x & 511u) != 0u;
    let double_odd = (value.x & 1024u) != 0u;
    value.x &= 0xfffffc00u;
    if double_guard && (double_sticky || double_odd) {
        value = weather_add64(value, vec2(1024u, 0u));
    }
    if (value.y & 0x80000000u) != 0u {
        value = weather_shr_jam(value, 1u);
        exponent += 1;
    }
    var mantissa = value.y >> 7u;
    let guard = (value.y & 64u) != 0u;
    let sticky = (value.y & 63u) != 0u || value.x != 0u;
    if guard && (sticky || (mantissa & 1u) != 0u) { mantissa += 1u; }
    if mantissa >= 0x1000000u { mantissa >>= 1u; exponent += 1; }
    return bitcast<f32>(sign | (u32(exponent + 127) << 23u) | (mantissa & 0x7fffffu));
}
fn weather_phase(animation: vec4<f32>, bases: vec4<u32>, snow: u32, time: vec4<f32>) -> vec2<f32> {
    if snow == 0u {
        let ticks = time.x + animation.y;
        let t = ticks + time.z;
        let step = -t / 32.0;
        let offset = step * animation.x;
        return vec2(0.0, offset % 32.0);
    }
    let t = time.x + time.z;
    let u_time = t * 0.01;
    let du = u_time * animation.z;
    let v_time = t * animation.w;
    let dv = v_time * 0.001;
    let u = weather_double_plus_float(bases.xy, du);
    let v = weather_double_plus_float(bases.zw, dv);
    let fall_time = time.y + time.z;
    let fall = -fall_time / 512.0;
    return vec2(u, fall + v);
}
// Atlas sampler is nearest with no mipmaps. Clamp the integer texel inside
// the sprite even when a fractional coordinate rounds up at its far edge.
fn weather_texel(uv: vec2<f32>, rect: vec4<f32>, dimensions: vec2<u32>) -> vec2<i32> {
    let size = vec2<f32>(dimensions);
    let origin = vec2<i32>(round(rect.xy * size));
    let extent = max(vec2<i32>(round(rect.zw * size)), vec2(1));
    let local = min(vec2<i32>(floor(fract(uv) * vec2<f32>(extent))), extent - vec2(1));
    return origin + local;
}
// WEATHER_MATH_END

struct WeatherTime { values: vec4<f32> }
@group(1) @binding(1) var<uniform> weather_time: WeatherTime;
@vertex fn vs_weather(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>,
    @location(2) rect: vec4<f32>, @location(3) animation: vec4<f32>,
    @location(4) bases: vec4<u32>, @location(5) color: vec4<f32>,
    @location(6) light: vec4<u32>, @location(7) snow: u32) -> WeatherOut {
    let phase = weather_phase(animation, bases, snow, weather_time.values);
    let eye = position - globals.view_origin.xyz;
    let light_color = entity_sample_light(position, light);
    return WeatherOut(
        globals.view_proj * vec4(position, 1.0),
        uv + phase,
        color * light_color,
        vec2(length(eye), max(length(eye.xz), abs(eye.y))),
        rect,
    );
}
@fragment fn fs_weather(in: WeatherOut) -> @location(0) vec4<f32> {
    return frame_stored(weather_color(in));
}
fn weather_color(in: WeatherOut) -> vec4<f32> {
    // Repeat within this sprite, never across the atlas. The atlas uses nearest
    // filtering and no mipmaps, so fractional UVs have no seam derivatives.
    let texel = weather_texel(in.uv, in.weather_rect, textureDimensions(atlas));
    let color = textureLoad(atlas, texel, 0) * in.color;
    if color.a < 0.1 { discard; }
    let f = max(linear_fog(in.fog_distance.x, globals.fog.x, globals.fog.y),
                linear_fog(in.fog_distance.y, globals.fog.z, globals.fog.w));
    return vec4(mix(color.rgb, globals.fog_color.rgb, f * globals.fog_color.a), color.a);
}
