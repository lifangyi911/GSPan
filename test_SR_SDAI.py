# 在 2 张显卡上运行
# torchrun --nproc_per_node=2 test_SR_SDAI.py --dataset_name qb --batch_size 4
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import random
import math
import os
from os.path import join
import time
import sys
from dataset_h5 import *
from functions import *
import pansharp_metrics as mtc
from datetime import datetime

import openpyxl  # 保存到Excel表
from tqdm import tqdm
import argparse
from model import GSPan

import h5py
from hqnr_torch_fast import EfficientHQNR
import torch.nn.functional as F
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
from wald_utilities import wald_protocol_v1, wald_protocol_v2
from utils.gaussian_splatting import generate_2D_gaussian_splatting_step, generate_2D_gaussian_splatting_batch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='Args for FusionNet pansharpening')
parser.add_argument('-seed', type=int, default=2, help='seed')
parser.add_argument('--output_nc', type=int, default=8, help='output image channels')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size') #default 16 for training, 1 for testing
parser.add_argument('--num_epochs', type=int, default=400, help='training epochs')
parser.add_argument('--lr', type=float, default=0.0004, help='output image channels')
parser.add_argument('--log_freq', type=int, default=10, help='screen output')
parser.add_argument('--patience', type=int, default=250, help='early stopping patience')
parser.add_argument('--dataset_name', type=str, default='wv3_4K', help='读取h5文件')
parser.add_argument('--gpuid', type=str, default='1', help='screen output')
parser.add_argument('--version', type=str, default=r'11',help='method version')
parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume from (e.g. .../net_ckpt_epoch200.pth)')
args = parser.parse_args()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
############## arguments
sensor = 'WV3'
if sensor=='GF2':
    bit = 10
else:
    bit = 11
spectral_num = args.output_nc
img_range = 2 ** bit   # 1.0  # 2047.0
disk_name = 'data7a'  # QB nvme4a, GF2: data7a, WV4-bands: nvme4b


method = 'GS'

estimation_ratio=1
dataset = r'/raid/dataset/data_lifangyi/data/pansharpen/%s' % args.dataset_name
model_dir = r'/raid/dataset/data_lifangyi/apnn'
eval_ckpt_path = r'/raid/dataset/data_lifangyi/apnn/logs_%s_%s/%s_ckpt_epoch300.pth' % (args.dataset_name, method,method)
test_sample_dir = r'/raid/dataset/data_lifangyi/apnn/results/%s/%s_results/test_sample_v%s' % (args.dataset_name,method,args.version)
origin_test_sample_dir = r'/raid/dataset/data_lifangyi/apnn/results/%s/%s_results/SDAIx%s_test_sample_v%s' % (args.dataset_name, method,estimation_ratio, args.version)
record_dir = r'/raid/dataset/data_lifangyi/apnn/results/record_v%s' % (args.version)
logs_dir = r"./logs_%s_%s/" % (args.dataset_name, method)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(test_sample_dir):
    os.makedirs(test_sample_dir)
if not os.path.exists(origin_test_sample_dir):
    os.makedirs(origin_test_sample_dir)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)


print(f'Method: {method}, batch_size: {args.batch_size}, Version: {args.version}')


mode = "a" if args.resume else "w"
f = open("./logs_%s_%s/log_v%s.txt" % (args.dataset_name, method, args.version), 'a')


## Device configuration
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print(f'Using device: {device}')
if args.resume and local_rank == 0:
    f.write(f"\n\n{'=' * 20} RESUME TRAINING AT {datetime.now()} {'=' * 20}\n")

# 初始化损失函数
FastHQNR = EfficientHQNR(sensor=sensor, ratio=4, channels=4, device='cuda')

############## Loading datasets
print('===> Loading datasets')


test_low_set = MultiExmTest_h5(os.path.join(dataset, 'test_%s_multiExm1.h5' % args.dataset_name),img_scale=img_range)
test_low_loader = DataLoader(test_low_set, batch_size=1, num_workers=1, shuffle=False, pin_memory=False, drop_last=False)

test_full_set = MultiExmTest_h5(os.path.join(dataset, 'test_%s_OrigScale_multiExm1.h5'  % args.dataset_name),img_scale=img_range)
test_full_loader = DataLoader(test_full_set, batch_size=1, num_workers=1, shuffle=False, pin_memory=False, drop_last=False)

class SSIMLoss(nn.Module):
    def __init__(self, loss_weight = 1.0, data_range=1):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range

    def forward(self, x, y):
        from pytorch_msssim import ssim
        loss = 1 - ssim(x, y, data_range=self.data_range,win_size=7, size_average=True)
        return self.loss_weight * loss


class SAMLoss(nn.Module):
    def __init__(self, eps=1e-8, loss_weight = 0.01):
        super(SAMLoss, self).__init__()
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, img1, img2):
        """
        img1, img2: [B, C, H, W] 格式
        """
        # 1. 计算点积: sum(x * y)
        dot_product = torch.sum(img1 * img2, dim=1)

        # 2. 计算模长: sqrt(sum(x^2))
        img1_norm = torch.sqrt(torch.sum(img1 ** 2, dim=1) + self.eps)
        img2_norm = torch.sqrt(torch.sum(img2 ** 2, dim=1) + self.eps)

        # 3. 计算余弦相似度
        cos_sim = dot_product / (img1_norm * img2_norm + self.eps)

        # 4. 这里的 Loss 是 1 - cos_sim
        # 余弦值越接近 1，表示角度越小，光谱越一致
        sam_loss = 1.0 - torch.mean(cos_sim)

        return self.loss_weight * sam_loss
criterionL1 = nn.L1Loss().to(device)
criterionSmoothL1 = nn.SmoothL1Loss(beta = 0.5).to(device)
criterionMSE = nn.MSELoss().to(device)
criterionSSIM = SSIMLoss(loss_weight = 0.02).to(device)
criterionSAM = SAMLoss(eps = 1e-8,loss_weight=0.01).to(device)
net = model = GSPan(spectral_num=spectral_num,shuffle_scale1 = 2, shuffle_scale2 = 1).to(device)


if local_rank == 0:
    print(args)
    print('')
    print(optimizer)
    print('')

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def _circular_pad2d(x, pad_left, pad_right, pad_top, pad_bottom):
    """
    x: [B, C, H, W]
    自定义 circular padding，避免某些小尺寸图像下 F.pad(mode='circular') 的限制。
    """
    _, _, h, w = x.shape

    iy = torch.arange(-pad_top, h + pad_bottom, device=x.device) % h
    ix = torch.arange(-pad_left, w + pad_right, device=x.device) % w

    return x.index_select(2, iy).index_select(3, ix)


def _filter_cdf23_circular(x, coeff):
    """
    对每个通道独立做 23-tap separable circular filtering。
    x: [B, C, H, W]
    coeff: [23]
    """
    c = x.shape[1]
    p = coeff.numel() // 2

    coeff = coeff.to(device=x.device, dtype=x.dtype)

    # vertical filter: [C, 1, 23, 1]
    kv = coeff.view(1, 1, -1, 1).repeat(c, 1, 1, 1)

    # horizontal filter: [C, 1, 1, 23]
    kh = coeff.view(1, 1, 1, -1).repeat(c, 1, 1, 1)

    # MATLAB 代码等价于先垂直方向滤波，再水平方向滤波
    x = _circular_pad2d(x, 0, 0, p, p)
    x = F.conv2d(x, kv, groups=c)

    x = _circular_pad2d(x, p, p, 0, 0)
    x = F.conv2d(x, kh, groups=c)

    return x


def interp23tap_nchw(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """
    PyTorch implementation of MATLAB interp23tap.

    Args:
        x: input tensor, shape [B, C, H, W]
        ratio: scale factor, should be power of 2, e.g. 2, 4, 8

    Returns:
        interpolated tensor, shape [B, C, H*ratio, W*ratio]
    """
    if x.ndim != 4:
        raise ValueError("x must have shape [B, C, H, W].")

    ratio = int(ratio)

    if not _is_power_of_two(ratio):
        raise ValueError("Only resize factors that are powers of 2 are supported.")

    base = 2.0 * torch.tensor(
        [
            0.5,
            0.305334091185,
            0.0,
            -0.072698593239,
            0.0,
            0.021809577942,
            0.0,
            -0.005192756653,
            0.0,
            0.000807762146,
            0.0,
            -0.000060081482,
        ],
        device=x.device,
        dtype=x.dtype,
    )

    coeff = torch.cat([torch.flip(base[1:], dims=[0]), base], dim=0)  # [23]

    # 更合理的写法：ratio=4 时做两次 2x；ratio=8 时做三次 2x
    num_steps = int(math.log2(ratio))

    out = x
    first = True

    for _ in range(num_steps):
        b, c, h, w = out.shape
        up = out.new_zeros(b, c, h * 2, w * 2)

        if first:
            # MATLAB: I1LRU(2:2:end, 2:2:end, :) = I_Interpolated
            # MATLAB 是 1-based，所以对应 Python 的 1::2
            up[:, :, 1::2, 1::2] = out
            first = False
        else:
            # MATLAB: I1LRU(1:2:end, 1:2:end, :) = I_Interpolated
            # 对应 Python 的 0::2
            up[:, :, 0::2, 0::2] = out

        out = _filter_cdf23_circular(up, coeff)

    return out
def mtf_downsample_matlab_nearest(x, scale, hqnr_calculator, pad_mode="replicate"):
    """
    尽量模拟 MATLAB:
        I_MS_LP = MTF(x, sensor, ratio)
        I_MS_LP_D = imresize(I_MS_LP, 1/ratio, 'nearest')

    Args:
        x: [B, C, H, W]
        scale: downsampling ratio, e.g. 4
        hqnr_calculator: EfficientHQNR, 用其中的 mtf_kernel
        pad_mode: 建议和 MATLAB MTF 函数保持一致。
                  如果不确定，replicate 通常比 zero padding 更安全。
    """
    b, c, h, w = x.shape

    kernel = hqnr_calculator.mtf_kernel.to(device=x.device, dtype=x.dtype)


    pad = kernel.shape[-1] // 2

    # 不建议用 zero padding；MATLAB 的 MTF 通常不会希望边缘变暗
    x_pad = F.pad(x, (pad, pad, pad, pad), mode=pad_mode)

    blurred = F.conv2d(
        x_pad,
        kernel,
        padding=0,
        groups=c
    )

    # 模拟 MATLAB imresize(..., 1/scale, 'nearest') 的中心采样相位
    # scale=4 时，对应 zero-based index: 2, 6, 10, ...
    start = scale // 2

    y = blurred[:, :, start::scale, start::scale]

    return y


import numpy as np
import torch
import torch.nn.functional as F


def get_pan_gnyq(sensor: str) -> float:
    """
    Match MTF_PAN_new.m.
    """
    if sensor == "QB":
        return 0.15
    elif sensor in ["IKONOS", "Ikonos"]:
        return 0.17
    elif sensor in ["GeoEye1", "WV4"]:
        return 0.16
    elif sensor == "WV2":
        return 0.11
    elif sensor in ["WV3", "WV3-4bands"]:
        return 0.14
    elif sensor == "GF2":
        return 0.15
    else:
        return 0.15


def fspecial_gauss_np(size, sigma):
    """
    MATLAB fspecial('gaussian', N, sigma) 的近似实现。
    """
    if isinstance(size, int):
        size = (size, size)

    m, n = [(ss - 1.0) / 2.0 for ss in size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    sumh = h.sum()
    if sumh != 0:
        h = h / sumh

    return h


def fir_filter_wind_2d_matlab_like(Hd, win_1d):
    """
    近似 MATLAB fwind1(Hd, kaiser(N))。

    注意：
    - MATLAB fwind1 的严格数值复现较麻烦；
    - 这里采用频域模板 -> 空域冲激响应 -> 2D Kaiser window -> DC normalization；
    - 对训练中的 MTF degradation 来说已经足够稳定。
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = np.real(h)

    win_2d = np.outer(win_1d, win_1d)
    h = h * win_2d

    # 保持 DC 增益为 1，避免整体亮度漂移
    h_sum = h.sum()
    if abs(h_sum) > 1e-12:
        h = h / h_sum

    return h


def gen_pan_mtf_kernel(sensor="WV3", ratio=4, N=41, device="cuda", dtype=torch.float32):
    """
    Generate PAN MTF kernel according to MTF_PAN_new.m.
    Return shape: [1, 1, N, N]
    """
    GNyq = get_pan_gnyq(sensor)
    fcut = 1.0 / float(ratio)

    alpha = np.sqrt(((N - 1) * (fcut / 2.0)) ** 2 / (-2.0 * np.log(GNyq)))

    H = fspecial_gauss_np((N, N), alpha)
    Hd = H / H.max()

    win = np.kaiser(N, beta=5.0)  # MATLAB kaiser(N) 默认 beta 约等于 5
    h = fir_filter_wind_2d_matlab_like(Hd, win)

    kernel = torch.from_numpy(h).to(device=device, dtype=dtype)
    kernel = kernel.view(1, 1, N, N)

    return kernel


def pan_mtf_filter(input_pan, sensor="WV3", ratio=4, pad_mode="replicate"):
    """
    PAN MTF filtering only.

    Args:
        input_pan: [B, 1, H, W]
    """
    if input_pan.ndim != 4 or input_pan.shape[1] != 1:
        raise ValueError(f"input_pan should be [B, 1, H, W], got {input_pan.shape}")

    kernel = gen_pan_mtf_kernel(
        sensor=sensor,
        ratio=ratio,
        N=41,
        device=input_pan.device,
        dtype=input_pan.dtype
    )

    pad = kernel.shape[-1] // 2
    x_pad = F.pad(input_pan, (pad, pad, pad, pad), mode=pad_mode)

    return F.conv2d(x_pad, kernel, padding=0)


def pan_mtf_downsample(
        input_pan,
        ratio=4,
        sensor="WV3",
        out_size=None,
        pad_mode="replicate",
        resize_mode="bicubic",
):
    """
    Approximate:
        I_PAN = MTF_PAN_new(PAN, sensor, ratio);
        I_PAN = imresize(I_PAN, 1/ratio);

    Args:
        input_pan: [B, 1, H, W]
        ratio: downsample ratio
        sensor: sensor name, e.g. 'WV3'
        out_size: optional target size, e.g. input_lr.shape[2:]
        pad_mode: should be 'replicate' to match MATLAB imfilter(...,'replicate')
        resize_mode: MATLAB default imresize is closer to bicubic than nearest

    Returns:
        downsampled PAN: [B, 1, H/ratio, W/ratio] or [B, 1, out_h, out_w]
    """
    pan_lp = pan_mtf_filter(
        input_pan,
        sensor=sensor,
        ratio=ratio,
        pad_mode=pad_mode
    )

    if out_size is not None:
        # 推荐显式指定尺寸，避免 rounding 差异
        try:
            pan_d = F.interpolate(
                pan_lp,
                size=out_size,
                mode=resize_mode,
                align_corners=False,
                antialias=True
            )
        except TypeError:
            pan_d = F.interpolate(
                pan_lp,
                size=out_size,
                mode=resize_mode,
                align_corners=False
            )
    else:
        try:
            pan_d = F.interpolate(
                pan_lp,
                scale_factor=1.0 / ratio,
                mode=resize_mode,
                align_corners=False,
                antialias=True
            )
        except TypeError:
            pan_d = F.interpolate(
                pan_lp,
                scale_factor=1.0 / ratio,
                mode=resize_mode,
                align_corners=False
            )

    return pan_d
def load_checkpoint(model, checkpoint_path):
    """通用权重加载函数，处理 DDP 的 module. 前缀"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 如果是完整的 checkpoint 字典
    if 'net' in checkpoint:
        state_dict = checkpoint['net']
    else:
        state_dict = checkpoint

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # 如果权重里有 module. 而当前模型没有，或者反之，进行处理
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # 如果 model 本身是 DDP 包装的，需要加载到 model.module
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    print(f"Successfully loaded weights from {checkpoint_path}")

def net_low_eval(data_loader, model, eval_ckpt_path, f):
    load_checkpoint(model, eval_ckpt_path)
    model.eval()
    print("len(data_loader)", len(data_loader))

    ERGAS, SAM, Q2n, SCC = [], [], [], []
    with torch.no_grad():
        for index, batch in enumerate(data_loader,1):
            input_pan = batch['pan'].to(device)
            input_lr_u = batch['lms'].to(device)
            input_lr = batch['ms'].to(device)
            target = batch['gt'].to(device)

            batch_gs_parameters = net(input_lr_u, input_pan)
            gt_size = target.shape[2:]  # [h_gt, w_gt]

            psh = generate_2D_gaussian_splatting_batch(sr_size=gt_size, gs_parameters=batch_gs_parameters,
                                                       scale=1,
                                                       sample_coords=None,
                                                       scale_modify=[1, 1],
                                                       default_step_size=1.2,
                                                       cuda_rendering=True,
                                                       mode='scale',
                                                       if_dmax=True,
                                                       spectral_num=spectral_num,
                                                       dmax_mode='fix',
                                                       dmax=0.1)

            input_lr_u = F.interpolate(input_lr_u, size=target.shape[2:], mode='bicubic')
            psh = psh + input_lr_u
            psh = psh * img_range

            psh = trim_image(psh, L=0, R=2 ** bit-1)

            target, psh = target.detach().cpu().numpy(), psh.detach().cpu().numpy()
            target, psh = np.transpose(target, (0, 2, 3, 1)), np.transpose(psh, (0, 2, 3, 1))
            target, psh = target[0], psh[0]

            ERGAS.append(mtc.ERGAS(target, psh, 4))
            SAM.append(mtc.SAM(target, psh, eps=0.0005))
            Q2n.append(mtc.Q2n(target, psh, q_block_size=32, q_shift=32))
            SCC.append(mtc.SCC(target, psh))


    print("SAM, SCC, ERGAS, Q2n")
    print("%.4lf±%.4lf, %.4lf±%.4lf, %.4lf±%.4lf, %.4lf±%.4lf " %
          (np.mean(SAM), np.std(SAM), np.mean(SCC), np.std(SCC), np.mean(ERGAS), np.std(ERGAS), np.mean(Q2n), np.std(Q2n)))

    print("SAM, SCC, ERGAS, Q2n", file=f, flush=True)
    print("%.4lf±%.4lf, %.4lf±%.4lf, %.4lf±%.4lf, %.4lf±%.4lf " %
          (np.mean(SAM), np.std(SAM), np.mean(SCC), np.std(SCC), np.mean(ERGAS), np.std(ERGAS), np.mean(Q2n), np.std(Q2n)), file=f, flush=True)

    wb_low_eval_metric = openpyxl.Workbook()
    ws_low_eval_metric = wb_low_eval_metric.create_sheet('sheet1',0)

    metrics_name_list = ["SAM", "SCC", "ERGAS", "Q2n"]
    metrics_list = ["%.4lf±%.4lf" % (np.mean(SAM), np.std(SAM)) , "%.4lf±%.4lf" % (np.mean(SCC), np.std(SCC)), "%.4lf±%.4lf" % (np.mean(ERGAS), np.std(ERGAS)), "%.4lf±%.4lf" % (np.mean(Q2n), np.std(Q2n))]

    for i in range(len(metrics_name_list)):
        ws_low_eval_metric.cell(row=1, column=i + 1).value = metrics_name_list[i]
        ws_low_eval_metric.cell(row=2, column=i + 1).value = metrics_list[i]

    wb_low_eval_metric.save('%s/low_eval_metric_record.xlsx' % record_dir)


def _exact_int(value):
    result = round(value)
    if not math.isfinite(value) or abs(value - result) > 1e-7:
        raise ValueError(f"Non-integer spatial coordinate/size: {value}")
    return int(result)


def mtf_resize_ms_fractional(ms, ratio):
    """MTF then center-based nearest sampling, including fractional ratios.

    Construct kernels directly: EfficientHQNR's block size setup is not
    designed for a fractional ratio. Integer 2/4 sampling matches the old
    start=ratio//2, step=ratio convention. No extra antialias filter is added.
    """
    from hqnr_torch_fast import genMTF
    if ratio == 1:
        return ms
    h, w = ms.shape[-2:]
    oh, ow = _exact_int(h / ratio), _exact_int(w / ratio)
    kernel_np = np.moveaxis(genMTF(ratio, "WV3"), -1, 0)[:, None]
    kernel = torch.as_tensor(kernel_np, device=ms.device, dtype=ms.dtype)
    if kernel.shape[0] != ms.shape[1]:
        raise ValueError("WV3 MTF kernel and MS band count mismatch")
    pad = kernel.shape[-1] // 2
    blurred = F.conv2d(
        F.pad(ms, (pad, pad, pad, pad), mode="replicate"),
        kernel, groups=ms.shape[1],
    )
    iy = torch.floor((torch.arange(oh, device=ms.device, dtype=torch.float64) + 0.5) * ratio).long()
    ix = torch.floor((torch.arange(ow, device=ms.device, dtype=torch.float64) + 0.5) * ratio).long()
    return blurred.index_select(-2, iy).index_select(-1, ix)


@torch.no_grad()
def prepare_sdai_3840(pan, ms, estimation_ratio):
    """Return CPU PAN3840, MS960, shared base3840, PAN_est, LMS_est.

    Specifically for registered WV3 PAN4096/MS1024 input pairs.
    Never crop the five settings differently or upsample GT as a base.
    """
    if estimation_ratio not in (1, 1.5, 2, 3, 4):
        raise ValueError("Supported ratios: 1, 1.5, 2, 3, 4")
    if pan.ndim != 4 or ms.ndim != 4:
        raise ValueError("Expected BCHW inputs")
    if pan.shape[1:] != (1, 4096, 4096) or ms.shape[1:] != (8, 1024, 1024):
        raise ValueError("Expected registered PAN4096 and 8-band MS1024")
    if pan.shape[0] != ms.shape[0]:
        raise ValueError("Batch sizes differ")
    pan_fr = pan.detach().cpu().float()[..., 128:3968, 128:3968].contiguous()
    ms_fr = ms.detach().cpu().float()[..., 32:992, 32:992].contiguous()
    base_fr = interp23tap_nchw(ms_fr, 4)
    if estimation_ratio == 1:
        pan_est, ms_est = pan_fr, ms_fr
    else:
        n = _exact_int(3840 / estimation_ratio)
        pan_est = pan_mtf_downsample(
            pan_fr, ratio=estimation_ratio, sensor="WV3",
            out_size=(n, n), pad_mode="replicate", resize_mode="bicubic",
        )
        ms_est = mtf_resize_ms_fractional(ms_fr, estimation_ratio)
    lms_est = interp23tap_nchw(ms_est, 4)
    if lms_est.shape[-2:] != pan_est.shape[-2:]:
        raise ValueError("Estimation input sizes do not match")
    return pan_fr, ms_fr, base_fr, pan_est, lms_est
# ============ PAN 退化质量量化：pan_est vs pan_fr PSNR（最小实现） ============
def quantify_pan_psnr(pan_fr, pan_est, estimation_ratio, peak=None, upsample="bicubic"):
    """pan_est 相对 pan_fr 的退化 PSNR。

    口径：pan_est 先上采样回与 pan_fr 相同的网格（3840×3840）再逐像素比较；
    ratio=1 时二者是同一张图，PSNR = inf（无退化基线）。

    Args:
        pan_fr:  [B,1,3840,3840]，prepare_sdai_3840 返回的第 1 个量。
        pan_est: [B,1,n,n]，n = 3840/estimation_ratio，prepare_sdai_3840 返回的第 4 个量。
        estimation_ratio: 1 / 1.5 / 2 / 3 / 4。
        peak: PSNR 峰值信号。默认 1.0（dataset 已除以 img_scale=2047 归一化到 [0,1]）；
              若传原始 11-bit 域数据，请显式传 2047。
        upsample: "bicubic"（跨倍率统一口径，推荐）或 "interp23tap"
                  （仅 2/4 倍率可用，与 MS 侧 lms_est 插值一致）。

    Returns:
        float PSNR（dB），并打印一行结果。
    """
    if pan_fr.ndim != 4 or pan_est.ndim != 4:
        raise ValueError("Expected BCHW tensors")
    if pan_fr.shape[1] != 1 or pan_est.shape[1] != 1:
        raise ValueError("Expected single-band PAN")

    a = pan_fr.detach().cpu().float()
    e = pan_est.detach().cpu().float()
    h, w = a.shape[-2:]

    if e.shape[-2:] == (h, w):              # ratio=1：同一张图，PSNR = inf
        ref = e
    else:
        if abs(e.shape[-2] - 3840 / estimation_ratio) > 1e-6:
            raise ValueError("pan_est 尺寸与 estimation_ratio 不匹配")
        if upsample == "bicubic":
            ref = F.interpolate(e, size=(h, w), mode="bicubic", align_corners=False)
        elif upsample == "interp23tap":
            r = int(estimation_ratio)
            if not _is_power_of_two(r):
                raise ValueError("interp23tap 仅支持 2/4 倍率，1.5/3 请用 'bicubic'")
            ref = interp23tap_nchw(e, r)
        else:
            raise ValueError("upsample 仅支持 'bicubic' / 'interp23tap'")

    if peak is None:
        peak = 1.0

    mse = ((a - ref) ** 2).mean().item()
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(peak * peak / mse)
    print(f"[PAN 退化] ratio={estimation_ratio}  PSNR={psnr:.4f} dB  "
          f"MSE={mse:.6e}  (pan_fr {tuple(a.shape[-2:])} <- pan_est {tuple(e.shape[-2:])})")
    return psnr

@torch.no_grad()
def split_and_joint_pansharpen(
    pan, lms, split_size, spectral_num, overlap_size,
    model, img_range, scale=1,
    default_step_size=1.2, dmax=1.0, base=None,
):
    """Original padded sliding-window structure, with exact scaled stitching.

    scale is OUTPUT/ESTIMATION, allowed: 1, 1.5, 2, 3, 4.
    The server renderer scale remains 1, as in the original FR call.
    Returns CPU float32, with the target-resolution base already added.
    """
    if scale not in (1, 1.5, 2, 3, 4):
        raise ValueError("Unsupported scale")
    if split_size <= 0 or not 0 <= overlap_size < split_size:
        raise ValueError("Require 0 <= overlap_size < split_size")
    if split_size % 8:
        raise ValueError("split_size must be divisible by the model window size 8")
    b, c, h_raw, w_raw = lms.shape
    if pan.shape != (b, 1, h_raw, w_raw) or c != spectral_num:
        raise ValueError("Input shapes do not match")
    target_h, target_w = _exact_int(h_raw * scale), _exact_int(w_raw * scale)
    if base is None:
        if scale != 1:
            raise ValueError("Provide the shared target-resolution MS base")
        base = lms
    if base.shape != (b, c, target_h, target_w):
        raise ValueError("base shape does not match target")

    stride = split_size - overlap_size
    tile_nums_h = max(1, math.ceil((h_raw - overlap_size) / stride))
    tile_nums_w = max(1, math.ceil((w_raw - overlap_size) / stride))
    pad_h = tile_nums_h * stride + overlap_size - h_raw
    pad_w = tile_nums_w * stride + overlap_size - w_raw
    tile_out_size = _exact_int(split_size * scale)
    overlap_out = _exact_int(overlap_size * scale)
    stride_out = _exact_int(stride * scale)
    h_out = _exact_int((h_raw + pad_h) * scale)
    w_out = _exact_int((w_raw + pad_w) * scale)

    pad_mode = "reflect" if pad_h < h_raw and pad_w < w_raw else "replicate"
    pan_pad = F.pad(pan.detach().cpu(), (0, pad_w, 0, pad_h), mode=pad_mode)
    lms_pad = F.pad(lms.detach().cpu(), (0, pad_w, 0, pad_h), mode=pad_mode)
    output_canvas = torch.zeros((b, c, h_out, w_out), dtype=torch.float32)
    count_map = torch.zeros((1, 1, h_out, w_out), dtype=torch.float32)
    window = torch.ones((1, 1, tile_out_size, tile_out_size), dtype=torch.float32)
    for i in range(overlap_out):
        v = (i + 1) / overlap_out
        window[:, :, i, :] *= v
        window[:, :, -(i + 1), :] *= v
        window[:, :, :, i] *= v
        window[:, :, :, -(i + 1)] *= v

    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    was_training = model.training
    model.eval()
    try:
        for hi in range(tile_nums_h):
            for wi in range(tile_nums_w):
                hs, ws = hi * stride, wi * stride
                tile_pan = pan_pad[..., hs:hs + split_size, ws:ws + split_size].to(device=device, dtype=dtype)
                tile_lms = lms_pad[..., hs:hs + split_size, ws:ws + split_size].to(device=device, dtype=dtype)
                params = model(tile_lms, tile_pan)
                tile_out = generate_2D_gaussian_splatting_batch(
                    sr_size=(tile_out_size, tile_out_size), gs_parameters=params,
                    scale=1/scale, sample_coords=None, scale_modify=[1, 1],
                    default_step_size=default_step_size, cuda_rendering=True,
                    mode="scale", spectral_num=spectral_num,
                    if_dmax=True, dmax_mode="fix", dmax=dmax,
                )
                if tile_out.shape != (b, c, tile_out_size, tile_out_size):
                    raise ValueError(f"Unexpected renderer output shape {tile_out.shape}")
                oy, ox = hi * stride_out, wi * stride_out
                output_canvas[..., oy:oy + tile_out_size, ox:ox + tile_out_size] += tile_out.float().cpu() * window
                count_map[..., oy:oy + tile_out_size, ox:ox + tile_out_size] += window
                del params, tile_pan, tile_lms, tile_out
        if not torch.all(count_map > 0).item():
            raise RuntimeError("Uncovered pixels in tiled output")
        output_canvas.div_(count_map)
        result = output_canvas[..., :target_h, :target_w]
        result.add_(base.detach().cpu().float())
        return result
    finally:
        model.train(was_training)


def net_full_eval(data_loader_rr,data_loader_fr, model, eval_ckpt_path, f):
    load_checkpoint(model, eval_ckpt_path)
    model.eval()

    HQNR_set, D_lambda_k_set, D_s_set = [], [], []
    print("len(data_loader_rr)", len(data_loader_rr))
    scale = 4
    FastHQNR = EfficientHQNR(sensor=sensor, ratio=scale, channels=4, device='cuda')
    with torch.no_grad():
        psnr=0
        for index, (batch_rr, batch_fr) in enumerate(zip(data_loader_rr, data_loader_fr)):


            # 只读取原始FR的PAN、MS
            # 准备过程在CPU执行，减少GPU显存占用
            pan_raw = batch_fr["pan"]
            ms_raw = batch_fr["ms"]

            (
                pan_fr,  # [1, 1, 3840, 3840]
                ms_fr,  # [1, 8, 960, 960]
                base_fr,  # [1, 8, 3840, 3840]
                pan_est,
                lms_est,
            ) = prepare_sdai_3840(
                pan_raw,
                ms_raw,
                estimation_ratio,
            )
            # 指标也必须使用对应的3840区域
            input_pan = pan_fr.to(device)
            input_lr = ms_fr.to(device)
            input_lr_u = base_fr.to(device)
            input_pan_l = wald_protocol_v2(input_lr, input_pan, ratio=scale, sensor=sensor,
                                           channels=args.output_nc)

            psnr+=quantify_pan_psnr(pan_fr, pan_est, estimation_ratio)


            print(f'pan_fr:{pan_fr.shape},pan_est:{pan_est.shape},lms_est:{lms_est.shape}')
            start = time.time()
            output = split_and_joint_pansharpen(
                pan=pan_est,
                lms=lms_est,
                split_size=512,
                spectral_num=spectral_num,
                overlap_size=32,
                model=model,
                img_range=img_range,
                scale=estimation_ratio,
                default_step_size=1.2,
                dmax=0.025,
                base=base_fr,
            )

            # 输出已经加过MS基底
            output = output.clamp(0, 1).to(device)


            print('==>图片生成单张时长： {:.2f}h\n'.format((time.time() - start) / 3600), file=f, flush=True)
            print('==>图片生成单张时长： {:.2f}s\n'.format((time.time() - start)))
            ## old and slow version

            # new and fast version
            FastHQNR_value,D_lambda_k_value,D_s_value = FastHQNR(output, input_lr_u, input_lr, input_pan, input_pan_l)
            HQNR_value = FastHQNR_value

            HQNR_set.append(HQNR_value.item()), D_lambda_k_set.append(D_lambda_k_value.item()), D_s_set.append(D_s_value.item())
            print('Test %d' % index, 'D_spe',D_lambda_k_value.item(),'D_spa',D_s_value.item(),'HQNR',HQNR_value.item())

            ##  将 每个 Fused FR image 保存为 一个 mat 文件

            # 清理缓存
            import gc
            torch.cuda.empty_cache()
            gc.collect()
        print(f'psnr_everage:{psnr/8}')
    print("D_lambda_k, D_s, HQNR")
    print("%.4lf±%.4lf, %.4lf±%.4lf, %.4lf±%.4lf" %
          (np.mean(D_lambda_k_set), np.std(D_lambda_k_set), np.mean(D_s_set), np.std(D_s_set), np.mean(HQNR_set), np.std(HQNR_set)))
    print("D_lambda_k, D_s, HQNR", file=f, flush=True)
    print("%.4lf±%.4lf, %.4lf±%.4lf, %.4lf±%.4lf" %
          (np.mean(D_lambda_k_set), np.std(D_lambda_k_set), np.mean(D_s_set), np.std(D_s_set), np.mean(HQNR_set), np.std(HQNR_set)), file=f, flush=True)

    wb_full_eval_metric = openpyxl.Workbook()
    ws_full_eval_metric = wb_full_eval_metric.create_sheet('sheet1',0)

    metrics_name_list = ["D_lambda_k", "D_s", "HQNR"]
    metrics_list = ["%.4lf±%.4lf" % (np.mean(D_lambda_k_set), np.std(D_lambda_k_set)),
                    "%.4lf±%.4lf" % (np.mean(D_s_set), np.std(D_s_set)), "%.4lf±%.4lf" % (np.mean(HQNR_set), np.std(HQNR_set))]

    for i in range(len(metrics_name_list)):
        ws_full_eval_metric.cell(row=1, column=i + 1).value = metrics_name_list[i]
        ws_full_eval_metric.cell(row=2, column=i + 1).value = metrics_list[i]

    wb_full_eval_metric.save('%s/full_eval_metric_record.xlsx' % record_dir)

if "__main__" == __name__:
    print('Begin Time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # Train

    ## Eval.
    print('Evaluating...')
    start_totol = time.time()
    net_full_eval(test_low_loader, test_full_loader, net, eval_ckpt_path, f)

    print('==>总时长： {:.2f}h\n'.format((time.time() - start_totol) / 3600), file=f, flush=True)
    print('==>总时长： {:.2f}s\n'.format((time.time() - start_totol) ))
    print('End Time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ========== 1. 模型参数量统计 ==========
    from thop import profile, clever_format

    net.eval()
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\n{'=' * 50}")
    print(f"Model Statistics")
    print(f"{'=' * 50}")
    print(f"Total parameters: {total_params / 1e6:.4f}M ({total_params:,})")
    print(f"Trainable parameters: {trainable_params / 1e6:.4f}M ({trainable_params:,})")
    print(f"Non-trainable: {(total_params - trainable_params) / 1e6:.2f}M")
