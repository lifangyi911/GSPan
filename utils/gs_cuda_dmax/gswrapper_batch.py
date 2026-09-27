import os
import torch
from torch.utils.cpp_extension import load
from torch.autograd import Function
from torch.autograd.function import once_differentiable

build_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'build')
os.makedirs(build_path, exist_ok=True)

file_path = os.path.split(os.path.abspath(__file__))[0]

import gscuda_batch
GSWrapper = gscuda_batch

class GSCUDA_batch(Function):
   
        @staticmethod
        def forward(ctx, sigmas, coords, colors, rendered_img, dmax):
            ctx.save_for_backward(sigmas, coords, colors)
            ctx.dmax = dmax
            _, h, w, c = rendered_img.shape
            b, s, _ = sigmas.shape
            GSWrapper.gs_render_batch(sigmas, coords, colors, rendered_img, b, s, h, w, c, dmax)
            return rendered_img

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            sigmas, coords, colors = ctx.saved_tensors
            dmax = ctx.dmax
            b, s, _ = sigmas.shape
            _, h, w, c = grad_output.shape
            grads_sigmas = torch.zeros_like(sigmas)
            grads_coords = torch.zeros_like(coords)
            grads_colors = torch.zeros_like(colors)
            GSWrapper.gs_render_backward_batch(sigmas, coords, colors, grad_output.contiguous(), grads_sigmas, grads_coords, grads_colors, b, s, h, w, c, dmax)
            return (grads_sigmas, grads_coords, grads_colors, None, None)

def gaussiansplatting_render(sigmas, coords, colors, image_size,dmax=100):
    sigmas = sigmas.contiguous() # (gs num, 3)
    coords = coords.contiguous() # (gs num, 2)
    colors = colors.contiguous() # (gs num, c)
    h, w = image_size[:2]
    c = colors.shape[-1]
    rendered_img = torch.zeros(h, w, c).to(colors.device).to(torch.float32)
    return GSCUDA.apply(sigmas, coords, colors, rendered_img, dmax)

