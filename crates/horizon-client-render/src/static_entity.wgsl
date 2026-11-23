// Static block-entity meshes use section-local positions and the shared
// entity vertex/fragment helpers. The shader subtracts integer block origins
// before converting to f32; CPU supplies the camera's fractional block offset.
struct StaticEntityUniform {
    camera_block: vec4<i32>,
    camera_fraction: vec4<f32>,
}
@group(1) @binding(0) var<uniform> static_entity: StaticEntityUniform;

@vertex fn vs_static_entity(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>,
    @location(2) overlay: u32, @location(3) color: vec4<f32>, @location(4) light: vec4<u32>,
    @location(5) normal: vec4<f32>, @location(6) glint_uv: vec2<f32>,
    @location(7) section_origin: vec3<i32>) -> EntityOut {
    let block_delta = section_origin - static_entity.camera_block.xyz;
    let camera_relative = vec3<f32>(block_delta) + position - static_entity.camera_fraction.xyz;
    return entity_out(camera_relative, uv, overlay, color, light, normal, glint_uv);
}
