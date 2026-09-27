from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

# 确保basicsr在Python路径中
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(project_root)

import warnings
import math
import copy
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_, kaiming_normal_


from einops import rearrange


import warnings
import math
import copy
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_, kaiming_normal_
from einops import rearrange
from torch.utils.checkpoint import checkpoint

import numpy as np



def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class EDSRNOUP(nn.Module):
    def __init__(self,
                 num_in_ch=5,
                 num_feat=64,
                 num_block=6,
                 res_scale=1):
        super(EDSRNOUP, self).__init__()

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)


    def forward(self, x):

        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        x = res + x

        return res


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out + res


def window_partition(x, window_size):
    # x is the feature from net_g
    b, c, h, w = x.shape
    windows = rearrange(x, 'b c (h_count dh) (w_count dw) -> (b h_count w_count) (dh dw) c', dh=window_size,
                        dw=window_size)

    return windows


def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.ReLU):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class WindowCrossAttn(nn.Module):
    def __init__(self, inchanel=64, dim=180, num_heads=6, window_size=12, num_gs_seed=2304):
        super(WindowCrossAttn, self).__init__()
        self.inchanel = inchanel
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_gs_seed = num_gs_seed
        self.num_gs_seed_sqrt = int(math.sqrt(num_gs_seed))

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        coords_source = (np.indices((self.num_gs_seed_sqrt, self.num_gs_seed_sqrt)) + 0.5) * self.window_size
        coords_tgt = (np.indices((self.window_size, self.window_size)) + 0.5) * self.num_gs_seed_sqrt
        coords_delta = coords_source.reshape(2, -1)[:, :, np.newaxis] - coords_tgt.reshape(2, -1)[:, np.newaxis, :]
        shape = coords_delta.shape
        indexed_coords_delta = coords_delta.flatten()
        indexed_coords_delta = list(map(list(set(sorted(indexed_coords_delta))).index, indexed_coords_delta))
        indexed_coords_delta = np.array(indexed_coords_delta).reshape(shape)

        indexed_coords_delta[0] *= indexed_coords_delta.max()
        indexed_coords_delta = torch.from_numpy(indexed_coords_delta.sum(0))
        self.register_buffer('relative_position_index', indexed_coords_delta)
        # assign closed feature bigger initial weight
        relative_position_bias_table = torch.zeros(
            (2 * max(self.num_gs_seed_sqrt, window_size) - 1) * (2 * max(self.num_gs_seed_sqrt, window_size) - 1),
            num_heads)
        self.relative_position_bias_table = nn.Parameter(relative_position_bias_table)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qhead = nn.Linear(dim, dim, bias=True)
        self.khead = nn.Linear(dim, dim, bias=True)
        self.vhead = nn.Linear(dim, dim, bias=True)

        self.proj = nn.Linear(dim, dim)

        # softmax
        self.softmax = nn.Softmax(dim=-1)

        # direction basis per head: shape (num_heads, 2, head_dim)
        # B_{h,0}, B_{h,1} are learnable basis vectors used as:
        # d_h(theta) = cos2θ * B_{h,0} + sin2θ * B_{h,1}

        # independent q_dir projection is NOT required here because modulation uses q/k directly,
        # but we keep a small linear if you want to map GS into same scale (optional)
        # (we'll not use q_dir_proj for projection to dir space; modulation uses q/k and dir_basis)
        # head-wise learnable alpha

    def forward(self, gs, feat, dir_windows=None):
        # gs shape: b*h_count*w_count, num_gs, c    the input gs here should already include pos embedding and scale embedding
        # feat shape: b*h_count*w_count, dh*dw, c    dh=dw=window_size
        b_, num_gs, c = gs.shape
        b_, n, c = feat.shape

        q = self.qhead(gs)  # b_, num_gs_, c
        q = q.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        k = self.khead(feat)  # b_, n_, c
        k = k.reshape(b_, n, self.num_heads, c // self.num_heads)
        k = k.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        v = self.vhead(feat)  # b_, n_, c
        v = v.reshape(b_, n, self.num_heads, c // self.num_heads)
        v = v.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        q = q * self.scale
        attn_logits = (q @ k.transpose(-2, -1))  # (B_win, H, Nq, Nk)


        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.num_gs_seed_sqrt * self.num_gs_seed_sqrt, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn_logits + relative_position_bias.unsqueeze(0)  # * 100
        attn = self.softmax(attn)
        # always focus self
        x = (attn @ v).transpose(1, 2).reshape(b_, num_gs, c)
        x = self.proj(x)

        return x


class WindowCrossAttnLayer(nn.Module):
    def __init__(self, dim=180, num_heads=6, window_size=12, shift_size=0, num_gs_seed=2308):
        super(WindowCrossAttnLayer, self).__init__()

        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.shift_size = shift_size
        self.window_size = window_size

        self.window_cross_attn = WindowCrossAttn(dim=dim, num_heads=num_heads, window_size=window_size,
                                                 num_gs_seed=num_gs_seed)

        self.mlp_crossattn_feature = MLP(in_features=dim, hidden_features=dim, out_features=dim)

    def forward(self, x, query_pos, feat, scale_embedding, dir_windows=None):
        # gs shape: b*h_count*w_count, num_gs, c
        # query_pos shape: b*h_count*w_count, num_gs, c
        # feat shape: b,c,h,w
        # scale_embedding shape: b*h_count*w_count, 1, c


        ###cross attention for Q,K,V
        resi = x
        x = self.norm3(x)
        if self.shift_size > 0:
            shift_feat = torch.roll(feat, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shift_feat = feat
        shift_feat = window_partition(shift_feat, self.window_size)  # b*h_count*w_count, dh*dw, c  dh=dw=window_size


        x = self.window_cross_attn(with_pos_embed(x, query_pos),shift_feat)
        x = resi + x

        ###FFN
        resi = x
        x = self.norm4(x)
        x = self.mlp_crossattn_feature(x)
        x = resi + x

        return x


class WindowCrossAttnBlock(nn.Module):
    def __init__(self, dim=180, window_size=12, num_heads=6, num_layers=4, num_gs_seed=2308):
        super(WindowCrossAttnBlock, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            WindowCrossAttnLayer(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                num_gs_seed=num_gs_seed) for i in range(num_layers)
        ])

    def forward(self, x, query_pos, feat, scale_embedding, dir_windows=None):
        resi = x
        x = self.norm(x)
        for block in self.blocks:
            x = block(x, query_pos, feat, scale_embedding, dir_windows=dir_windows)
        x = self.mlp(x)
        x = resi + x
        return x


class GSSelfAttn(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_gs_seed_sqrt=12):
        super(GSSelfAttn, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_gs_seed_sqrt = num_gs_seed_sqrt

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.proj = nn.Linear(dim, dim)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.num_gs_seed_sqrt - 1) * (2 * self.num_gs_seed_sqrt - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.num_gs_seed_sqrt)
        coords_w = torch.arange(self.num_gs_seed_sqrt)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.num_gs_seed_sqrt - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.num_gs_seed_sqrt - 1
        relative_coords[:, :, 0] *= 2 * self.num_gs_seed_sqrt - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer('relative_position_index', relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)

        self.qhead = nn.Linear(dim, dim, bias=True)
        self.khead = nn.Linear(dim, dim, bias=True)
        self.vhead = nn.Linear(dim, dim, bias=True)


    def forward(self, gs):
        # gs shape: b*h_count*w_count, num_gs, c
        # pos shape: b*h_count*w_count, num_gs, c
        b_, num_gs, c = gs.shape

        q = self.qhead(gs)
        q = q.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        k = self.khead(gs)
        k = k.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        k = k.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        v = self.vhead(gs)
        v = v.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        v = v.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # b_, num_heads, num_gs, n

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.num_gs_seed_sqrt * self.num_gs_seed_sqrt, self.num_gs_seed_sqrt * self.num_gs_seed_sqrt,
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = (attn @ v).transpose(1, 2).reshape(b_, num_gs, c)
        attn = self.proj(attn)

        return attn


class GSSelfAttnLayer(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_gs_seed_sqrt=12, shift_size=0):
        super(GSSelfAttnLayer, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)


        self.gs_self_attn = GSSelfAttn(dim=dim, num_heads=num_heads, num_gs_seed_sqrt=num_gs_seed_sqrt)

        self.mlp_selfattn = MLP(in_features=dim, hidden_features=dim, out_features=dim)

        self.num_gs_seed_sqrt = num_gs_seed_sqrt
        self.shift_size = shift_size


    def forward(self, gs, pos, h_count, w_count, scale_embedding):
        # gs shape:b*h_count*w_count, num_gs_seed, channel
        # pos shape: b*h_count*w_count, num_gs_seed, channel
        # scale_embedding shape: b*h_count*w_count, 1, channel


        resi = gs
        gs = self.norm1(gs)

        #### shift gs
        if self.shift_size > 0:
            shift_gs = rearrange(gs, '(b m n) (h w) c -> b (m h) (n w) c', m=h_count, n=w_count,
                                 h=self.num_gs_seed_sqrt, w=self.num_gs_seed_sqrt)
            shift_gs = torch.roll(shift_gs, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shift_gs = rearrange(shift_gs, 'b (m h) (n w) c -> (b m n) (h w) c', m=h_count, n=w_count,
                                 h=self.num_gs_seed_sqrt, w=self.num_gs_seed_sqrt)
        else:
            shift_gs = gs

        #### gs self attention
        gs = self.gs_self_attn(shift_gs)

        #### shift gs back
        if self.shift_size > 0:
            shift_gs = rearrange(gs, '(b m n) (h w) c -> b (m h) (n w) c', m=h_count, n=w_count,
                                 h=self.num_gs_seed_sqrt, w=self.num_gs_seed_sqrt)
            shift_gs = torch.roll(shift_gs, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            shift_gs = rearrange(shift_gs, 'b (m h) (n w) c -> (b m n) (h w) c', m=h_count, n=w_count,
                                 h=self.num_gs_seed_sqrt, w=self.num_gs_seed_sqrt)
        else:
            shift_gs = gs

        gs = shift_gs + resi

        # FFN
        resi = gs
        gs = self.norm2(gs)
        gs = self.mlp_selfattn(gs)
        gs = gs + resi
        return gs


class SSIA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        ) # 典型的Transformer FFN

    def forward(self, x_q, x_kv):
        # x_q: gs (B, N_gs, C), pos_q: gs_pos (B, N_gs, C)
        # x_kv: gs_ori (B, N_gs, C)

        # MHA Block
        resi = x_q
        q = self.norm_q(x_q)
        attn_out, _ = self.cross_attn(query=q, key=x_kv, value=x_kv)
        x_q = resi + attn_out # Residual connection for MHA

        # FFN Block
        resi = x_q
        x_q = self.ffn(self.norm_ffn(x_q))
        out = resi + x_q

        return out

class GSSelfAttnBlock(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_selfattn_layers=4, num_gs_seed_sqrt=12):
        super(GSSelfAttnBlock, self).__init__()

        self.num_gs_seed_sqrt = int(num_gs_seed_sqrt)

        self.mlp1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            GSSelfAttnLayer(
                dim=dim,
                num_heads=num_heads,
                num_gs_seed_sqrt=num_gs_seed_sqrt,
                shift_size=0 if i % 2 == 0 else num_gs_seed_sqrt // 2
            ) for i in range(num_selfattn_layers)
        ])
        self.blocks_ori = nn.ModuleList([
            GSSelfAttnLayer(
                dim=dim,
                num_heads=num_heads,
                num_gs_seed_sqrt=num_gs_seed_sqrt,
                shift_size=0 if i % 2 == 0 else num_gs_seed_sqrt // 2
            ) for i in range(num_selfattn_layers)
        ])

        self.streamfusion1 = SpectralReferentialCrossAttention(dim, num_heads)
        self.streamfusion2 = SpectralReferentialCrossAttention(dim, num_heads)


    def forward(self, gs, gs_ori, pos, h_count, w_count, scale_embedding):
        resi = gs
        resi_ori = gs_ori
        gs = self.norm1(gs)
        gs_ori = self.norm2(gs_ori)
        for block in self.blocks:
            gs = block(gs, pos, h_count, w_count, scale_embedding)
        for block in self.blocks_ori:
            gs_ori = block(gs_ori, pos, h_count, w_count, scale_embedding)
        gs_fuse = self.streamfusion1(x_q=gs, x_kv=gs_ori)

        gs_ori_fuse = self.streamfusion2(x_q=gs_ori, x_kv=gs)

        gs_fuse = self.mlp1(gs_fuse)
        gs_fuse = gs_fuse + resi
        gs_ori_fuse = self.mlp2(gs_ori_fuse)
        gs_ori_fuse = gs_ori_fuse + resi_ori


        return gs_fuse, gs_ori_fuse

class DSHIBlock(nn.Module): # Dual-stream Interaction Attention Block
    def __init__(self, channel,num_gs_seed_sqrt,window_size,num_heads,num_crossattn_layers,num_gs_seed,
                 num_crossattn_blocks,num_crossattn_ori_layers,num_crossattn_ori_blocks,num_selfattn_layers,num_selfattn_blocks,use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        self.window_crossattn_ori_blocks = nn.ModuleList([
            WindowCrossAttnBlock(dim=channel,
                                 window_size=window_size,
                                 num_heads=num_heads,
                                 num_layers=num_crossattn_layers,
                                 num_gs_seed=num_gs_seed) for i in range(num_crossattn_blocks)
        ])
        self.window_crossattn_blocks = nn.ModuleList([
            WindowCrossAttnBlock(dim=channel,
                                 window_size=window_size,
                                 num_heads=num_heads,
                                 num_layers=num_crossattn_ori_layers,
                                 num_gs_seed=num_gs_seed) for i in range(num_crossattn_ori_blocks)
        ])


        self.gs_selfattn_blocks = nn.ModuleList([
            GSSelfAttnBlock(dim=channel,
                            num_heads=num_heads,
                            num_selfattn_layers=num_selfattn_layers,
                            num_gs_seed_sqrt=num_gs_seed_sqrt
                            ) for i in range(num_selfattn_blocks)
        ])

    def forward(self, query, query_pos,query_context, feat, feat_context,scale_embedding,h,w):
        """
        x_content: 主流 query (B, N, C) - Local/RGB focus
        x_context: 辅流 query_context (B, N, C) - Global/Freq focus
        """


        for block in self.window_crossattn_blocks:
            if self.use_checkpoint:
                query = checkpoint(block, query, query_pos, feat, scale_embedding)
            else:
                query = block(query, query_pos, feat, scale_embedding)  # b*h_count*w_count, num_gs_seed, channel


        for block in self.window_crossattn_ori_blocks:
            if self.use_checkpoint:
                query_context = checkpoint(block, query_context, query_pos, feat_context, scale_embedding)
            else:
                query_context = block(query_context, query_pos, feat_context,
                                  scale_embedding)  # b*h_count*w_count, num_gs_seed, channel

        skip_query = query
        skip_query_context = query_context


        for i, block in enumerate(self.gs_selfattn_blocks):
            if self.use_checkpoint:
                query, query_context = checkpoint(block, query, query_context, query_pos, h // self.window_size,
                                              w // self.window_size, scale_embedding)
            else:
                query, query_context = block(query, query_context, query_pos, h // self.window_size, w // self.window_size,
                                         scale_embedding)

        query = query + skip_query
        query_context = query_context + skip_query_context

        return query, query_context

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GSPan(nn.Module):
    def __init__(self, spectral_num=4,sensor=None, inchannel=64, channel=60, num_heads=6, num_crossattn_blocks=1, num_crossattn_layers=2, num_crossattn_ori_blocks=1, num_crossattn_ori_layers=2,
                 num_selfattn_blocks = 4, num_selfattn_layers = 4 ,depths=[4,4],num_DSHIblocks=1,
                 num_gs_seed=64, gs_up_factor=1.0, window_size=8, img_range=1.0, shuffle_scale1 = 2, shuffle_scale2 = 1, use_checkpoint = False):
        super(STDSfea2gsV2B, self).__init__()
        self.channel = channel
        self.nhead = num_heads
        self.gs_up_factor = gs_up_factor
        self.num_gs_seed = num_gs_seed
        self.window_size = window_size
        self.num_selfattn_blocks = num_selfattn_blocks
        self.img_range = img_range
        self.use_checkpoint = use_checkpoint
        self.spectral_num = spectral_num
        self.sensor = sensor

        self.num_gs_seed_sqrt = int(math.sqrt(num_gs_seed))
        self.gs_up_factor_sqrt = int(math.sqrt(gs_up_factor))

        self.shuffle_scale1 = shuffle_scale1
        self.shuffle_scale2 = shuffle_scale2

        self.encoder_pan = EDSRNOUP(num_in_ch=2,num_feat=inchannel)
        self.encoder_lms = EDSRNOUP(num_in_ch=spectral_num,num_feat=inchannel)

        # shared gaussian embedding and its pos embedding

        # HF -> query 映射
        self.content_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel))
        self.direction_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel))
        self.hf_stats = {
            'base_feat_mean': 0.0,
            'hf_enhanced_mean': 0.0,
            'fusion_gate_mean': 0.0,

        }

        self.pos_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel), requires_grad=True)
        trunc_normal_(self.content_embedding, std=.02)
        trunc_normal_(self.direction_embedding, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)
        # ---- context-tensor / sampling helpers ----
        # project input (srcs) -> intensity for ST (if srcs is feature map)

        # small smoothing kernel implemented via avg pooling (or conv)
        # we'll use avg_pool2d in forward for smoothing

        # global scale for mapping lambda -> sigma (learnable)

        # residual scale for sigma and rho predictions (learnable small multipliers)

        # small eps for numerical stability


        self.img_feat_proj = nn.Sequential(
            nn.Conv2d(inchannel, channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )
        self.img_feat_context_proj = nn.Sequential(
            nn.Conv2d(inchannel, channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )


        self.DSHIblocks = nn.ModuleList([
            DSHIBlock( channel,self.num_gs_seed_sqrt,window_size,num_heads,num_crossattn_layers,
                     num_gs_seed,num_crossattn_blocks,num_crossattn_ori_layers,num_crossattn_ori_blocks,num_selfattn_layers,num_selfattn_blocks,use_checkpoint) for i in range(num_DSHIblocks)
        ])


        # GS sigma_x, sigma_y
        self.mlp_block_sigma = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(2 * gs_up_factor))
        )

        # GS rho
        self.mlp_block_rho = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(1 * gs_up_factor))
        )

        # GS alpha
        self.mlp_block_alpha = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(1 * gs_up_factor))
        )

        # GS RGB values
        self.mlp_block_rgb = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(self.spectral_num * gs_up_factor))
        )

        # GS mean_x, mean_y
        self.mlp_block_mean = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(2 * gs_up_factor))
        )
        self.conv_after_body_Fusion = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )


        self.UPNet = nn.Sequential(
            nn.Conv2d(channel, channel * self.shuffle_scale1 * self.shuffle_scale1, 3, 1, 1),
            nn.PixelShuffle(self.shuffle_scale1),
            nn.Conv2d(channel, channel * self.shuffle_scale2 * self.shuffle_scale2, 3, 1, 1),
            nn.PixelShuffle(self.shuffle_scale2)
        )

    @staticmethod
    def get_N_reference_points(h, w, device='cuda'):
        step_y = 1 / h
        step_x = 1 / w
        ref_y, ref_x = torch.meshgrid(torch.linspace(step_y / 2, 1 - step_y / 2, h, dtype=torch.float32, device=device),
                                      torch.linspace(step_x / 2, 1 - step_x / 2, w, dtype=torch.float32, device=device))
        reference_points = torch.stack((ref_x.reshape(-1), ref_y.reshape(-1)), -1)
        reference_points = reference_points[None, :, None]
        return reference_points


    def forward(self, lr_u, pan, scale=1):
        '''
        using deformable detr decoder for cross attention
        Args:
            query: (batch_size, num_query, dim)
            query_pos: (batch_size, num_query, dim)
            srcs: (batch_size, dim, h1, w1)
        '''

        pan_hp = pan - F.avg_pool2d(pan, kernel_size=5, stride=1, padding=2)
        pan_input = torch.cat([pan, pan_hp], dim=1)
        feat_pan = self.encoder_pan(pan_input)
        feat_lms = self.encoder_lms(lr_u)

        b, c, h, w = feat_lms.shape  ###srcs is pad to the size that could be divided by window_size


        query = self.content_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (w // self.window_size),1, 1)  # b, h_count*w_count, num_gs_seed, channel
        query = query.reshape(b * (h // self.window_size) * (w // self.window_size), -1,self.channel)  # b*h_count*w_count, num_gs_seed, channel

        feat = self.img_feat_proj(feat_lms)  # (b, channel, h, w)

        feat_context = self.img_feat_context_proj(feat_pan)

        query_context = self.direction_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (
                w // self.window_size), 1, 1)  # b, h_count*w_count, num_gs_seed, channel
        query_context = query_context.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                                  self.channel)  # b*h_count*w_count, num_gs_seed, channel


        scale_embedding = None

        query_pos = self.pos_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (
                w // self.window_size), 1, 1)  # b, h_count*w_count, num_gs_seed, channel

        query_pos = query_pos.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                                      self.channel)  # b*h_count*w_count, num_gs_seed, channel


        for i, block in enumerate(self.DSHIblocks):
            if self.use_checkpoint:
                query, query_context = checkpoint(block, query, query_pos,query_context, feat, feat_context,scale_embedding,h,w)
            else:
                query, query_context = block(query, query_pos,query_context, feat, feat_context,scale_embedding,h,w)

        query = rearrange(query, '(b m n) (h w) c -> b c (m h) (n w)',
                          m=h // self.window_size, n=w // self.window_size, h=self.num_gs_seed_sqrt)
        query_context = rearrange(query_context, '(b m n) (h w) c -> b c (m h) (n w)',
                                  m=h // self.window_size, n=w // self.window_size, h=self.num_gs_seed_sqrt)
        query_map = self.conv_after_body_Fusion(torch.cat([query, query_context], 1))
        # 3. UPNet 上采样
        query = self.UPNet(query_map)  # (B, C, H_up, W_up)

        query = query.permute(0, 2, 3, 1)


        query_sigma = self.mlp_block_sigma(query).reshape(b, -1, 2) # b, h_count*w_count*H*W, 2
        query_rho = self.mlp_block_rho(query).reshape(b, -1, 1)

        query_alpha = self.mlp_block_alpha(query).reshape(b, -1, 1)
        query_rgb = self.mlp_block_rgb(query).reshape(b, -1, self.spectral_num)
        query_mean = self.mlp_block_mean(query).reshape(b, -1, 2)
        # limit_cells = 6.0会崩溃
        if self.sensor == 'QB':
            limit_cells = 5.0 # qb有限制，其他没有
            query_mean = torch.tanh(query_mean) * limit_cells

        query_mean = query_mean / torch.tensor(
            [self.num_gs_seed_sqrt * (w // self.window_size) * self.shuffle_scale1 * self.shuffle_scale2,
             self.num_gs_seed_sqrt * (h // self.window_size) * self.shuffle_scale1 * self.shuffle_scale2])[
            None, None].to(query_mean.device)  # b, h_count*w_count*num_gs_seed, 2

        reference_offset = self.get_N_reference_points(
            self.num_gs_seed_sqrt * (h // self.window_size) * self.shuffle_scale1 * self.shuffle_scale2,
            self.num_gs_seed_sqrt * (w // self.window_size) * self.shuffle_scale1 * self.shuffle_scale2, pan.device)
        query_mean = query_mean + reference_offset.reshape(1, -1, 2)

        query = torch.cat([query_sigma, query_rho, query_alpha, query_rgb, query_mean],
                          dim=-1)  # b, h_count*w_count*num_gs_seed, 9

        return query


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 假设 float32 (4 bytes)
    size_mb = total * 4 / (1024 ** 2)
    return total, trainable, size_mb

