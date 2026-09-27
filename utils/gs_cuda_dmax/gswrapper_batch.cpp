#include "gs_batch.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// --- Forward 包装函数 ---
void gs_render_batch(
        torch::Tensor &sigmas,      // [B, S, 3]
        torch::Tensor &coords,      // [B, S, 2]
        torch::Tensor &colors,      // [B, S, C]
        torch::Tensor &rendered_img, // [B, H, W, C]
        const int b,                 // batch size
        const int s,                 // gs num per sample
        const int h,                 // height
        const int w,                 // width
        const int c,                 // channels
        const float dmax
        ){
      
        CHECK_INPUT(sigmas);
        CHECK_INPUT(coords);
        CHECK_INPUT(colors);
        CHECK_INPUT(rendered_img);

        // 确保在输入张量所在的 GPU 设备上运行
        const at::cuda::OptionalCUDAGuard device_guard(device_of(sigmas));

        // 调用刚才在 .cu 文件中定义的批量渲染函数
        _gs_render_batch(
            (const float *) sigmas.data_ptr(),
            (const float *) coords.data_ptr(),
            (const float *) colors.data_ptr(),
            (float *) rendered_img.data_ptr(),
            b, s, h, w, c, dmax);
}

// --- Backward 包装函数 ---
void gs_render_backward_batch(
        torch::Tensor &sigmas,
        torch::Tensor &coords,
        torch::Tensor &colors,
        torch::Tensor &grads,        // [B, H, W, C]
        torch::Tensor &grads_sigmas, // [B, S, 3]
        torch::Tensor &grads_coords, // [B, S, 2]
        torch::Tensor &grads_colors, // [B, S, C]
        const int b,
        const int s,
        const int h,
        const int w,
        const int c,
        const float dmax
        ){

        CHECK_INPUT(sigmas);
        CHECK_INPUT(coords);
        CHECK_INPUT(colors);
        CHECK_INPUT(grads);
        CHECK_INPUT(grads_sigmas);
        CHECK_INPUT(grads_coords);
        CHECK_INPUT(grads_colors);

        const at::cuda::OptionalCUDAGuard device_guard(device_of(sigmas));

        // 调用批量反向传播函数
        _gs_render_backward_batch(
            (const float *) sigmas.data_ptr(),
            (const float *) coords.data_ptr(),
            (const float *) colors.data_ptr(),
            (const float *) grads.data_ptr(),
            (float *) grads_sigmas.data_ptr(),
            (float *) grads_coords.data_ptr(),
            (float *) grads_colors.data_ptr(),
            b, s, h, w, c, dmax);
}

// --- 绑定到 Python 接口 ---
PYBIND11_MODULE( TORCH_EXTENSION_NAME, m) {
        m.def( "gs_render_batch",
                &gs_render_batch,
                "cuda forward batch wrapper");
        m.def( "gs_render_backward_batch",
                &gs_render_backward_batch,
                "cuda backward batch wrapper");
}