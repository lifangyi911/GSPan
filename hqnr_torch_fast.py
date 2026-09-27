##################################
# AI Optimize
import numpy as np

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class EfficientHQNR(nn.Module):
    def __init__(self, sensor='QB', ratio=4, block_size=32, alpha=1, beta=1, q=1, 
                 device='cuda', channels=4):
        super().__init__()
        self.sensor = sensor
        self.ratio = ratio
        self.block_size = block_size
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.channels = channels
        
        # 预计算MTF核
        self.register_buffer('mtf_kernel', self._precompute_mtf_kernel())
        
        # 预定义块处理滤波器
        self._setup_block_filters()
        
    def _precompute_mtf_kernel(self):
        """预计算MTF卷积核"""
        mtf_kernel_np = genMTF(self.ratio, self.sensor)
        mtf_kernel_np = np.moveaxis(mtf_kernel_np, -1, 0)
        mtf_kernel_np = np.expand_dims(mtf_kernel_np, axis=1)
        mtf_kernel = torch.from_numpy(mtf_kernel_np).float().to('cuda')
        
        # 扩展为多通道
        return mtf_kernel
    
    def _setup_block_filters(self):
        """预定义块处理相关的卷积核"""
        # 用于块统计量计算的卷积核
        S = self.block_size
        self.sum_filter = torch.ones((1, 1, S, S), device=self.mtf_kernel.device)
        
        # 用于低分辨率块处理的卷积核
        S_low = S // self.ratio
        self.sum_filter_low = torch.ones((1, 1, S_low, S_low), device=self.mtf_kernel.device)
    
    def _fast_mtf(self, x):
        """快速MTF处理 - 使用F.conv2d"""
        # x: [B, C, H, W]
        batch_size, channels, H, W = x.shape
        kernel_size = self.mtf_kernel.shape[2:]
        padding = kernel_size[0] // 2


        x_mtf = F.conv2d(x, self.mtf_kernel, 
                        padding=padding, 
                        groups=channels)

        return x_mtf
    
    def _vectorized_uqi(self, img1, img2, block_size):
        """向量化UQI计算 - 替代原来的blockproc+uqi"""
        # img1, img2: [B, C, H, W] 或 [B, 1, H, W]
        B, C, H, W = img1.shape
        
        # 使用卷积计算块统计量
        sum_filter = self.sum_filter if block_size == self.block_size else self.sum_filter_low
        N = block_size ** 2
        
        # 计算各统计量
        img1_sum = F.conv2d(img1, sum_filter.repeat(C, 1, 1, 1), padding=0, groups=C)
        img2_sum = F.conv2d(img2, sum_filter.repeat(C, 1, 1, 1), padding=0, groups=C)
        img1_sq_sum = F.conv2d(img1 * img1, sum_filter.repeat(C, 1, 1, 1), padding=0, groups=C)
        img2_sq_sum = F.conv2d(img2 * img2, sum_filter.repeat(C, 1, 1, 1), padding=0, groups=C)
        img12_sum = F.conv2d(img1 * img2, sum_filter.repeat(C, 1, 1, 1), padding=0, groups=C)
        
        # 计算UQI分量
        img12_sum_mul = img1_sum * img2_sum
        img12_sq_sum_mul = img1_sum * img1_sum + img2_sum * img2_sum
        
        numerator = 4 * (N * img12_sum - img12_sum_mul) * img12_sum_mul
        denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
        denominator = denominator1 * img12_sq_sum_mul
        
        # 处理边界条件
        quality_map = torch.ones_like(denominator)
        zeros = torch.zeros_like(denominator)
        
        # 条件1: denominator1 == 0 and img12_sq_sum_mul != 0
        cond1 = (denominator1 == zeros) & (img12_sq_sum_mul != zeros)
        quality_map[cond1] = 2 * img12_sum_mul[cond1] / img12_sq_sum_mul[cond1]
        
        # 条件2: denominator != 0
        cond2 = denominator != zeros
        quality_map[cond2] = numerator[cond2] / denominator[cond2]
        
        return quality_map.mean(dim=[2, 3])  # [B, C]
    
    def _compute_d_lambda(self, sr, ms):
        """批量计算光谱失真D_lambda"""
        # sr, ms: [B, C, H, W]
        batch_size, channels = sr.shape[0], sr.shape[1]
        
        # 批量MTF处理
        fused_degraded = self._fast_mtf(sr)
        
        # 批量计算UQI
        q_values = self._vectorized_uqi(ms, fused_degraded, self.block_size)  # [B, C]
        
        # 计算D_lambda
        q_avg = q_values.mean(dim=1)  # [B]
        d_lambda = 1 - q_avg
        
        return d_lambda.mean()  # 标量
    
    def _compute_d_s(self, sr, lrms, pan, pan_filt):
        """批量计算空间失真D_s"""
        # sr: [B, C, H, W], lrms: [B, C, H_low, W_low]
        # pan: [B, 1, H, W], pan_filt: [B, 1, H_low, W_low]
        batch_size, channels = sr.shape[0], sr.shape[1]
        
        # 扩展pan以匹配通道数
        pan_expanded = pan.repeat(1, channels, 1, 1)  # [B, C, H, W]
        pan_filt_expanded = pan_filt.repeat(1, channels, 1, 1)  # [B, C, H_low, W_low]
        
        # 批量计算高分辨率UQI
        q_high = self._vectorized_uqi(sr, pan_expanded, self.block_size)  # [B, C]
        
        # 批量计算低分辨率UQI
        q_low = self._vectorized_uqi(lrms, pan_filt_expanded, self.block_size // self.ratio)  # [B, C]
        
        # 计算D_s
        d_s_values = torch.abs(q_high - q_low) ** self.q  # [B, C]
        d_s = (d_s_values.mean(dim=1)) ** (1 / self.q)  # [B]
        
        return d_s.mean()  # 标量
    
    def forward(self, sr, ms, lrms, pan, pan_filt):
        """
        参数:
            sr: 超分结果 [B, C, H, W]
            ms: 多光谱图像 [B, C, H, W] 
            lrms: 低分辨率多光谱 [B, C, H//ratio, W//ratio]
            pan: 全色图像 [B, 1, H, W]
            pan_filt: 滤波后的全色图像 [B, 1, H//ratio, W//ratio]
        """
        # 输入验证
        assert sr.shape == ms.shape, f"SR shape {sr.shape} != MS shape {ms.shape}"
        assert pan.shape[1] == 1, "PAN should be single channel"
        
        # 计算D_lambda和D_s
        d_lambda = self._compute_d_lambda(sr, ms)
        d_s = self._compute_d_s(sr, lrms, pan, pan_filt)
        
        # 计算QNR
        qnr = (1 - d_lambda) ** self.alpha * (1 - d_s) ** self.beta
        
        return qnr, d_lambda, d_s

# 保留原有的辅助函数（不需要修改）
def genMTF(ratio, sensor, N=41):
    if sensor == 'QB':
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22])  # Bands Order: B,G,R,NIR
    elif sensor in ['Ikonos', 'IKONOS']:
        GNyq = np.asarray([0.26, 0.28, 0.29, 0.28])  # Bands Order: B,G,R,NIR
    elif sensor in ['GeoEye1', 'WV4']:
        GNyq = np.asarray([0.23, 0.23, 0.23, 0.23])  # Bands Order: B, G, R, NIR
    elif sensor == 'WV2':
        GNyq = 0.35 * np.ones(7)
        GNyq = np.append(GNyq, 0.27)
    elif sensor == 'WV3':
        GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]
    elif sensor == 'WV3-4bands':
        GNyq = [0.355, 0.360, 0.365, 0.335]
    else:
        GNyq = np.asarray([0.3, 0.3, 0.3, 0.3])
    
    # 这里需要NyquistFilterGenerator的实现
    h = NyquistFilterGenerator(GNyq, ratio, N)
    return h

def NyquistFilterGenerator(Gnyq, ratio, N):
    assert isinstance(Gnyq, (np.ndarray, list)), 'Error: GNyq must be a list or a ndarray'
    if isinstance(Gnyq, list):
        Gnyq = np.asarray(Gnyq)
    nbands = Gnyq.shape[0]

    kernel = np.zeros((N, N, nbands))  # generic kerenel (for normalization purpose)
    fcut = 1 / np.double(ratio)
    for j in range(nbands):
        alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(Gnyq[j])))
        H = fspecial_gauss((N, N), alpha)
        Hd = H / np.max(H)
        h = np.kaiser(N, 0.5)
        kernel[:, :, j] = np.real(fir_filter_wind(Hd, h))
    return kernel

def fspecial_gauss(size, sigma):
    # Function to mimic the 'fspecial' gaussian MATLAB function
    m, n = [(ss-1.)/2. for ss in size]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def fir_filter_wind(Hd, w):
    """
    compute fir filter with window method
    Hd:     desired freqeuncy response (2D)
    w:      window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = np.clip(h, a_min=0, a_max=np.max(h))
    h = h / np.sum(h)
    return h