// Improved transparency's cloud stages (`clouds.fsh` under OIT_*), appended to
// entity.wgsl after oit.wgsl. The clouds' vertex shader and draw are the
// classic ones; only the fragment colour goes through the three stages.

// The cloud's depth bounds see the unfaded alpha; transmittance and accumulate
// fade out towards the cloud fog end (`cloud_color`), with zero atlas fetches.
@fragment fn fs_oit_cloud_bounds(in: CloudOut) -> @location(0) vec4<f32> {
    let alpha = in.color.a;
    if alpha < 0.01 { discard; }
    let linear = oit_linear_depth(in.clip.z);
    let opaque = select(0.0, in.clip.z, alpha > OIT_FULLY_OPAQUE_ALPHA);
    return vec4(-linear, linear, in.clip.z, opaque);
}
@fragment fn fs_oit_cloud_transmittance(in: CloudOut) -> OitCoefficients {
    let color = cloud_color(in);
    return oit_add_transmittance(1.0 - color.a, in.clip.z, in.clip.xy);
}
@fragment fn fs_oit_cloud_accumulate(in: CloudOut) -> @location(0) vec4<f32> {
    var color = cloud_color(in);
    if HDR_WRITE_LIGHT && globals.hdr_light.w != 0.0 {
        color = vec4(hdr_written(color.rgb, 1.0, globals.hdr_settings), color.a);
    }
    let seen = oit_sample_transmittance(vec2<i32>(in.clip.xy),
        oit_normalize_depth(in.clip.z, in.clip.xy), 1.0 - color.a);
    return vec4(color.rgb * color.a, color.a) * seen;
}
