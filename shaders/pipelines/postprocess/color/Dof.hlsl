#include "core/common/Bindless.hlsl"
#include "core/common/Common.hlsl"
BINDLESS_BINDINGS(3, 2, 4, 5)
#include "materials/Material.hlsli"
#include "shared/raster/ShaderParameters.h"

[[vk::push_constant]] ConstantBuffer<Moer::DofPipelineBindlessParam> param;

float4 main(float2 uv : TEXCOORD0) : SV_TARGET {
    // uv 即 屏幕坐标，值域为[0, 1]，表示了不同的像素

    float3 color = TextureHandle(param.input_color_tex).Sample2D<float4>(uv).rgb;

    // 深度：值域为 [0, 1]，近处为1，无限远处为0
    // - 另外，MoerEngine使用了Reverse-Z技术。如果你接触过其他渲染器，你会发现此处0和1的对应关系跟通常渲染器不同
    float depth = TextureHandle(param.depth_tex).Sample2D<float>(uv);

    float coc = 0.0f;
    float focus_range = max(param.focus_plane_range, 1e-3);

    if (depth <= 1e-4) {
        // 如果当前像素是天空（无限远）
        coc = 1e10; // 一个非常大的值，表示完全虚化
    } else {
        // 正常情况

        // 线性化深度：值域为 [near_clip, far_clip]，单位为世界坐标距离
        float linearized_depth = param.near_clip * param.far_clip / (param.far_clip + (1.0 - depth) * (param.near_clip - param.far_clip));

        // 焦平面距离
        float focus_depth = clamp(param.focus_plane_distance, param.near_clip, param.far_clip);
        // 距离差值
        float signed_delta = linearized_depth - focus_depth;

        float coc_magnitude = max(abs(signed_delta) - focus_range, 0.0) / focus_range * param.dof_intensity;

        // 焦平面附近 coc = 0，背景 coc > 0，前景 coc < 0
        coc = sign(signed_delta) * coc_magnitude;
    }

    // 根据coc计算模糊半径，最大为6像素
    int blur_radius = min(max((int)round(abs(coc)), 0), 6);

    // 根据coc虚化像素
    float3 blur_color = color;
    float sampled_depth = 0;
    if (blur_radius > 0) {
        blur_color = 0.0;
        float sample_count = 0.0;
        // 模糊，本质上就是获取周围像素的颜色，然后求平均值（卷积）
        // 下面的代码就是在以当前像素为中心，半径为blur_radius的范围内，采样颜色并求平均值
        [loop]
        for (int y = -blur_radius; y <= blur_radius; ++y) {
            [loop]
            for (int x = -blur_radius; x <= blur_radius; ++x) {
                sampled_depth = TextureHandle(param.depth_tex).Sample2D<float>(saturate(uv + float2(x, y) * param.resolution_inv));
                float coc_sampled = 0.0;
                if (sampled_depth <= 1e-4) {
                    coc_sampled = 1e10;
                } else {
                     // 线性化深度：值域为 [near_clip, far_clip]，单位为世界坐标距离
                    float linearized_depth = param.near_clip * param.far_clip / (param.far_clip + (1.0 - sampled_depth) * (param.near_clip - param.far_clip));

                    // 焦平面距离
                    float focus_depth = clamp(param.focus_plane_distance, param.near_clip, param.far_clip);
                    // 距离差值
                    float signed_delta = linearized_depth - focus_depth;

                    float coc_magnitude = max(abs(signed_delta) - focus_range, 0.0) / focus_range * param.dof_intensity;

                    // 焦平面附近 coc = 0，背景 coc > 0，前景 coc < 0
                    coc_sampled = sign(signed_delta) * coc_magnitude;
                }

                if (coc_sampled * coc < 0) {
                    continue;
                }

                float2 offset = float2(x, y) * param.resolution_inv;
                blur_color += TextureHandle(param.input_color_tex).Sample2D<float4>(saturate(uv + offset)).rgb;
                sample_count += 1.0;
            }
        }

        if (sample_count > 0) blur_color /= sample_count;
        else blur_color = color;
    }
    color = blur_color;

    // 可视化焦平面
    if (param.b_visualize_focus_plan != 0) {
        if (abs(coc) <= focus_range) {
            color = color; // do nothing
        } else if (coc > 0.0) {
            color = lerp(color, float3(0.0, 0.0, 1.0), 0.7);
        } else {
            color = lerp(color, float3(0.0, 1.0, 0.0), 0.7);
        }
    }
    else if (param.b_visualize_blur_radius != 0) {
        float blur_strength = saturate((float)blur_radius / 6.0);
        color = float3(blur_strength, blur_strength, blur_strength);
    }

    return float4(color, 1.0);
}