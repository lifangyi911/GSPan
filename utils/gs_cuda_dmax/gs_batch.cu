#include <stdio.h>   
#include <cmath>

#define PI 3.1415926536
#define PI2 6.283153072

__global__ void _gs_render_batch_cuda(
    const float *sigmas, // [B, S, 3]
    const float *coords, // [B, S, 2]
    const float *colors, // [B, S, 4]
    float *rendered_img, // [B, H, W, 4]
    const int b, const int s, const int h, const int w, const int c, const float dmax
) {
    int curs = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y; // 关键：获取当前是 Batch 中的哪一张图

    if (curs >= s || batch_idx >= b) return;

    // 计算当前 batch 的偏移量
    const float* b_sigmas = sigmas + batch_idx * s * 3;
    const float* b_coords = coords + batch_idx * s * 2;
    const float* b_colors = colors + batch_idx * s * c;
    float* b_rendered_img = rendered_img + batch_idx * h * w * c;

    float sigma_x = b_sigmas[curs * 3 + 0];
    float sigma_y = b_sigmas[curs * 3 + 1];
    float rho = b_sigmas[curs * 3 + 2];
    float x = b_coords[curs * 2 + 0];
    float y = b_coords[curs * 2 + 1];

    // 数值保护
    float rho_sq = rho * rho;
    if (rho_sq > 0.999f) rho_sq = 0.999f;

    float negative_half_inv_rho = -0.5f / (1.0f - rho_sq);
    float inv_sx2 = 1.0f / (sigma_x * sigma_x);
    float inv_sy2 = 1.0f / (sigma_y * sigma_y);
    float two_rho_sx_sy = 2.0f * rho / (sigma_x * sigma_y);

    for (int hi = 0; hi < h; hi++) {
        float curh_f = 2.0f * hi / (h - 1) - 1.0f;
        float d_y = curh_f - y;
        if (abs(d_y) > dmax) continue;

        for (int wi = 0; wi < w; wi++) {
            float curw_f = 2.0f * wi / (w - 1) - 1.0f;
            float d_x = curw_f - x;
            if (abs(d_x) > dmax) continue;

            float v = (inv_sx2 * d_x * d_x - two_rho_sx_sy * d_x * d_y + inv_sy2 * d_y * d_y) * negative_half_inv_rho;
            v = exp(v);

            for (int ci = 0; ci < c; ci++) {
                atomicAdd(&b_rendered_img[(hi * w + wi) * c + ci], v * b_colors[curs * c + ci]);
            }
        }
    }
}


void _gs_render_batch(
        const float *sigmas,
        const float *coords,
        const float *colors,
        float *rendered_img,
	const int b,
	const int s, 
	const int h, 
	const int w,
	const int c,
	const float dmax
	) {

        int threads=64;
        dim3 grid((s + threads - 1) / threads, b); // Y 维度为 Batch Size
        dim3 block(threads);
        _gs_render_batch_cuda<<<grid, block>>>(sigmas, coords, colors, rendered_img, b, s, h, w, c, dmax);
}

__global__ void _gs_render_backward_batch_cuda(
    const float *sigmas,   // [B, S, 3]
    const float *coords,   // [B, S, 2]
    const float *colors,   // [B, S, C]
    const float *grads,    // [B, H, W, C] (来自 Loss 的梯度)
    float *grads_sigmas,   // [B, S, 3] (输出梯度)
    float *grads_coords,   // [B, S, 2] (输出梯度)
    float *grads_colors,   // [B, S, C] (输出梯度)
    const int b, const int s, const int h, const int w, const int c, const float dmax
) {
    int curs = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y; // 关键：确定当前是 batch 中的第几张图

    if (curs >= s || batch_idx >= b) return;

    // --- 计算当前 Batch 的指针偏移 ---
    const float* b_sigmas = sigmas + batch_idx * s * 3;
    const float* b_coords = coords + batch_idx * s * 2;
    const float* b_colors = colors + batch_idx * s * c;
    const float* b_grads  = grads  + batch_idx * h * w * c;

    float* b_grads_sigmas = grads_sigmas + batch_idx * s * 3;
    float* b_grads_coords = grads_coords + batch_idx * s * 2;
    float* b_grads_colors = grads_colors + batch_idx * s * c;

    // --- 读取当前 GS 参数 ---
    float sigma_x = b_sigmas[curs*3+0];
    float sigma_y = b_sigmas[curs*3+1];
    float rho = b_sigmas[curs*3+2];
    float x = b_coords[curs*2+0];
    float y = b_coords[curs*2+1];

    // 数值保护
    float rho_sq = rho * rho;
    if(rho_sq > 0.999f) rho_sq = 0.999f;

    float w1 = -0.5f / (1.0f - rho_sq);
    float w2 = 1.0f / (sigma_x * sigma_x);
    float w3 = 1.0f / (sigma_x * sigma_y);
    float w4 = 1.0f / (sigma_y * sigma_y);
    float od_sx = 1.0f / sigma_x;
    float od_sy = 1.0f / sigma_y;

    // 局部梯度累加器
    float l_gsx = 0, l_gsy = 0, l_gsigx = 0, l_gsigy = 0, l_grho = 0;

    for (int hi = 0; hi < h; hi++) {
        float curh_f = 2.0f * hi / (h - 1) - 1.0f;
        float d_y = curh_f - y;
        if (abs(d_y) > dmax) continue;

        for (int wi = 0; wi < w; wi++) {
            float curw_f = 2.0f * wi / (w - 1) - 1.0f;
            float d_x = curw_f - x;
            if (abs(d_x) > dmax) continue;

            float d_val = w2 * d_x * d_x - 2.0f * rho * w3 * d_x * d_y + w4 * d_y * d_y;
            float v = exp(w1 * d_val);

            float v_2_w1 = v * 2.0f * w1;
            float dv_dx = v_2_w1 * (-w2 * d_x + rho * w3 * d_y);
            float dv_dy = v_2_w1 * (-w4 * d_y + rho * w3 * d_x);
            float dv_dsigx = v_2_w1 * od_sx * (w3 * rho * d_x * d_y - w2 * d_x * d_x);
            float dv_dsigy = v_2_w1 * od_sy * (w3 * rho * d_x * d_y - w4 * d_y * d_y);
            float dv_drho = -v_2_w1 * (2.0f * w1 * rho * d_val + w3 * d_x * d_y);

            for (int ci = 0; ci < c; ci++) {
                float g_out = b_grads[(hi * w + wi) * c + ci]; // 读取当前 Batch 图的梯度
                
                // 1. 颜色梯度
                b_grads_colors[curs * c + ci] += v * g_out;

                // 2. 参数梯度累加
                float g_temp = g_out * b_colors[curs * c + ci];
                l_gsx += g_temp * dv_dx;
                l_gsy += g_temp * dv_dy;
                l_gsigx += g_temp * dv_dsigx;
                l_gsigy += g_temp * dv_dsigy;
                l_grho += g_temp * dv_drho;
            }
        }
    }
    // 写回当前 Batch 的梯度数组
    b_grads_coords[curs*2+0] = l_gsx;
    b_grads_coords[curs*2+1] = l_gsy;
    b_grads_sigmas[curs*3+0] = l_gsigx;
    b_grads_sigmas[curs*3+1] = l_gsigy;
    b_grads_sigmas[curs*3+2] = l_grho;
}

void _gs_render_backward_batch(
        const float *sigmas,
        const float *coords,
        const float *colors,
	const float *grads, // (h, w, c)
	float *grads_sigmas,
	float *grads_coords,
	float *grads_colors,
	const int b,
	const int s, 
	const int h, 
	const int w,
	const int c,
	const float dmax
	) {

        int threads = 128; // 一个 Block 放 128 线程
		dim3 block(threads);
		dim3 grid((s + threads - 1) / threads, b); 
        _gs_render_backward_batch_cuda<<<grid, block>>>(sigmas, coords, colors, grads, grads_sigmas, grads_coords, grads_colors, b, s, h, w, c, dmax);
}

