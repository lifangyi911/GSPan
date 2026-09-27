import numpy as np
from numpy.linalg import norm
import math
from math import ceil, floor, log2
from scipy.ndimage import laplace
from scipy.stats import pearsonr
import cv2

# 2023-10-05 更新

def SAM(x_true, x_pred, eps=0):
   """
   修订版：shenkq （从2015年综述文章的开源指标matlab指标代码改编，结果完全吻合）
   :param x_true: 高光谱图像：格式：(H, W, C)
   :param x_pred: 高光谱图像：格式：(H, W, C)
   :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度，输出的数值的单位是角度制的°
   """
   M, N = x_true.shape[0], x_true.shape[1]

   prod_scal = np.sum(x_true * x_pred, axis=2)  # 星乘表示矩阵内各对应位置相乘, shape：[H,W]
   norm_orig, norm_fusa = np.sum(x_true * x_true, axis=2), np.sum(x_pred * x_pred, axis=2)
   prod_norm = np.sqrt(norm_orig * norm_fusa)

   prod_scal, prod_norm = prod_scal.reshape(M * N, 1), prod_norm.reshape(M * N, 1)

   z = np.where(prod_norm != 0)
   prod_norm, prod_scal = prod_norm[z], prod_scal[z]   # 把分母中出现0的地方剔除掉

   res = np.arccos(np.clip(prod_scal / (prod_norm + eps),-1.0, 1.0))  # 每一个位置的弧度
   sam = np.mean(res)   # 取平均
   sam = sam * 180 / math.pi   # 转换成角度°
   return sam

def SAM_torch_new(x_true, x_pred, eps=0):
    prod_scal = torch.sum(x_true * x_pred, dim=1)

    norm_true = torch.norm(x_true, dim=1)
    norm_pred = torch.norm(x_pred, dim=1)
    prod_norm = norm_pred * norm_true

    a = torch.Tensor([1]).to(x_true.device, dtype=x_true.dtype)
    b = torch.Tensor([-1]).to(x_true.device, dtype=x_true.dtype)

    mask = prod_norm != 0
    prod_scal = prod_scal[mask]
    prod_norm = prod_norm[mask]

    res = prod_scal / (prod_norm + eps)

    res = torch.max(torch.min(res, a), b)

    sam = torch.mean(torch.acos(res))

    sam = sam * 180 / math.pi  # 转换成角度°

    return sam

def SCC(ms, ps):
    '''shenkq 更新：把sobel算子换成laplace算子'''
    ps_sobel = laplace(ps, mode='constant')   #laplace是二阶微分算子，用来提取锐化细节，SCC发明的论文里用的是laplace算子
    ms_sobel = laplace(ms, mode='constant')
    scc = 0.0
    for i in range(ms.shape[2]):
        a = (ps_sobel[:,:,i]).reshape(ms.shape[0]*ms.shape[1])
        b = (ms_sobel[:,:,i]).reshape(ms.shape[0]*ms.shape[1])
        scc += pearsonr(a, b)[0]

    return scc/ms.shape[2]

def CC(ms, ps):
    '''从2015年综述文章的开源指标matlab指标代码改编，结果完全吻合'''
    cc = 0.0
    for i in range(ms.shape[2]):
        a = (ps[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        b = (ms[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        cc += pearsonr(a, b)[0]

    return cc / ms.shape[2]

def Q4(ms, ps):
    '''从2015年综述文章的开源指标matlab指标代码改编，结果完全吻合'''
    def conjugate(a):
        sign = -1 * np.ones(a.shape)
        sign[0,:]=1
        return a*sign
    def product(a, b):
        a = a.reshape(a.shape[0],1)
        b = b.reshape(b.shape[0],1)
        R = np.dot(a, b.transpose())
        r = np.zeros(4)
        r[0] = R[0, 0] - R[1, 1] - R[2, 2] - R[3, 3]
        r[1] = R[0, 1] + R[1, 0] + R[2, 3] - R[3, 2]
        r[2] = R[0, 2] - R[1, 3] + R[2, 0] + R[3, 1]
        r[3] = R[0, 3] + R[1, 2] - R[2, 1] + R[3, 0]
        return r
    imps = np.copy(ps)
    imms = np.copy(ms)
    vec_ps = imps.reshape(imps.shape[1]*imps.shape[0], imps.shape[2])
    vec_ps = vec_ps.transpose(1,0)

    vec_ms = imms.reshape(imms.shape[1]*imms.shape[0], imms.shape[2])
    vec_ms = vec_ms.transpose(1,0)

    m1 = np.mean(vec_ps, axis=1)
    d1 = (vec_ps.transpose(1,0)-m1).transpose(1,0)
    s1 = np.mean(np.sum(d1*d1, axis=0))

    m2 = np.mean(vec_ms, axis=1)
    d2 = (vec_ms.transpose(1, 0) - m2).transpose(1, 0)
    s2 = np.mean(np.sum(d2 * d2, axis=0))

    Sc = np.zeros(vec_ms.shape)
    d2 = conjugate(d2)
    for i in range(vec_ms.shape[1]):
        Sc[:,i] = product(d1[:,i], d2[:,i])
    C = np.mean(Sc, axis=1)

    Q4 = 4 * np.sqrt(np.sum(m1*m1) * np.sum(m2*m2) * np.sum(C*C)) / (s1 + s2) / (np.sum(m1 * m1) + np.sum(m2 * m2))
    return Q4

def RMSE(ms, ps):
    '''输入都是单波段的，不是单独的指标，下面的ERGAS要调用这个函数'''
    d = (ms - ps)**2

    rmse = np.sqrt(np.sum(d)/(d.shape[0]*d.shape[1]))
    return rmse

def ERGAS_shen(ms, ps, ratio=4):
    '''从2015年综述文章的开源指标matlab指标代码改编，结果有微小出入'''
    m, n, d = ms.shape
    summed = 0.0
    for i in range(d):
        summed += (RMSE(ms[:,:,i], ps[:,:,i]))**2 / np.mean(ps[:,:,i])**2

    ergas = 100 * (1/ratio) *np.sqrt(summed/d)
    return ergas

def ERGAS(I1, I2, ratio):
    '''Gemine 2022 python version，与matlab完全一致'''
    I1 = I1.astype('float64')
    I2 = I2.astype('float64')

    Err = I1 - I2

    ERGAS_index = 0

    for iLR in range(I1.shape[2]):
        ERGAS_index = ERGAS_index + np.mean(Err[:, :, iLR] ** 2, axis=(0, 1)) / (
            np.mean(I1[:, :, iLR], axis=(0, 1))) ** 2

    ERGAS_index = (100 / ratio) * math.sqrt((1 / I1.shape[2]) * ERGAS_index)

    return np.squeeze(ERGAS_index)

def NRMSE_numpy(x_true, x_pred, norm_type = 'euclidean'):
    '''
    :param x_true: target image, shape like [H, W, C]
    :param x_pred: predict image, shape like [H, W, C]
    :param norm_type: {‘euclidean’, ‘min-max’, ‘mean’}  # 采用的是euclidean模式
    :return: normalized rmse value
    '''
    m, n, c = x_true.shape
    scores = []
    for i in range(c):
        if norm_type == 'euclidean':
            denorm = np.sqrt(np.mean(x_true[:,:,i] * x_true[:,:,i]))
        elif norm_type == 'min-max':
            denorm = x_true[:,:,i].max() - x_true[:,:,i].min()
        else:
            denorm = x_true[:,:,i].mean()

        scores.append(np.sqrt(np.mean((x_true[:, :, i] - x_pred[:, :, i]) ** 2)) / denorm)

    return np.mean(scores)

def UIQC(ms, ps):
    band = ms.shape[2]
    uiqc = 0.0
    for i in range(band):
        uiqc += Q(ms[:,:,i], ps[:,:,i])

    return uiqc/band

def Q(a, b):
    '''输入都是单波段的，不是单独的指标，上面的UIQC要调用这个函数'''
    a = a.reshape(a.shape[0]*a.shape[1])  # 直接延展成1维信号，照论文里的方式，应该要用滑窗去计算
    b = b.reshape(b.shape[0]*b.shape[1])
    temp=np.cov(a,b)   # a 和 b 的协方差矩阵
    d1 =  temp[0,0]
    cov = temp[0,1]
    d2 = temp[1,1]
    m1 = np.mean(a)   #
    m2 = np.mean(b)
    Q = 4*cov*m1*m2/(d1+d2)/(m1**2+m2**2)

    return Q

def D_lamda(ps, l_ms):
    '''
    ps：[H,W,C]
    l_ms:[H/4 ,W/4 ,C] 注意还没有上采样
    '''
    L = ps.shape[2]
    sum = 0.0
    for i in range(L):
        for j in range(L):
            if j!=i:
                sum += np.abs(Q(ps[:, :, i], ps[:, :, j]) - Q(l_ms[:, :, i], l_ms[:, :, j]))
    return sum/L/(L-1)

def D_s(ps, l_ms, pan):
    '''
    ps：[H,W,C]
    l_ms:[H/4 ,W/4 ,C]  注意还没有上采样
    pan:[H,W,C]
    '''
    L = ps.shape[2]
    l_pan = cv2.pyrDown(pan)
    l_pan = cv2.pyrDown(l_pan)
    sum = 0.0
    for i in range(L):
        sum += np.abs(Q(ps[:,:,i], pan) - Q(l_ms[:,:,i], l_pan))
    return sum/L

def QNR(ps, l_ms, pan):
    qnr = (1 - D_lamda(ps,l_ms)) * (1 - D_s(ps, l_ms, pan))
    return qnr

def normalize_block(im):
    """
        Auxiliary Function for Q2n computation.

        Parameters
        ----------
        im : Numpy Array
            Image on which calculate the statistics. Dimensions: H, W

        Return
        ------
        y : Numpy array
            The normalized version of im
        m : float
            The mean of im
        s : float
            The standard deviation of im

    """

    m = np.mean(im)
    s = np.std(im, ddof=1)

    if s == 0:
        s = 1e-10

    y = ((im - m) / s) + 1

    return y, m, s

def cayley_dickson_property_1d(onion1, onion2):
    """
        Cayley-Dickson construction for 1-D arrays.
        Auxiliary function for Q2n calculation.

        Parameters
        ----------
        onion1 : Numpy Array
            First 1-D array
        onion2 : Numpy Array
            Second 1-D array

        Return
        ------
        ris : Numpy array
            The result of Cayley-Dickson construction on the two arrays.
    """

    n = onion1.__len__()

    if n > 1:
        half_pos = int(n / 2)
        a = onion1[:half_pos]
        b = onion1[half_pos:]

        neg = np.ones(b.shape)
        neg[1:] = -1

        b = b * neg
        c = onion2[:half_pos]
        d = onion2[half_pos:]
        d = d * neg

        if n == 2:
            ris = np.concatenate([(a * c) - (d * b), (a * d) + (c * b)])
        else:
            ris1 = cayley_dickson_property_1d(a, c)

            ris2 = cayley_dickson_property_1d(d, b * neg)
            ris3 = cayley_dickson_property_1d(a * neg, d)
            ris4 = cayley_dickson_property_1d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.concatenate([aux1, aux2])
    else:
        ris = onion1 * onion2

    return ris

def cayley_dickson_property_2d(onion1, onion2):
    """
        Cayley-Dickson construction for 2-D arrays.
        Auxiliary function for Q2n calculation.

        Parameters
        ----------
        onion1 : Numpy Array
            First MultiSpectral img. Dimensions: H, W, Bands
        onion2 : Numpy Array
            Second MultiSpectral img. Dimensions: H, W, Bands

        Return
        ------
        ris : Numpy array
            The result of Cayley-Dickson construction on the two arrays.
    """

    dim3 = onion1.shape[-1]
    if dim3 > 1:
        half_pos = int(dim3 / 2)

        a = onion1[:, :, :half_pos]
        b = onion1[:, :, half_pos:]
        b = np.concatenate([np.expand_dims(b[:, :, 0], -1), -b[:, :, 1:]], axis=-1)

        c = onion2[:, :, :half_pos]
        d = onion2[:, :, half_pos:]
        d = np.concatenate([np.expand_dims(d[:, :, 0], -1), -d[:, :, 1:]], axis=-1)

        if dim3 == 2:
            ris = np.concatenate([(a * c) - (d * b), (a * d) + (c * b)], axis=-1)
        else:
            ris1 = cayley_dickson_property_2d(a, c)
            ris2 = cayley_dickson_property_2d(d,
                                              np.concatenate([np.expand_dims(b[:, :, 0], -1), -b[:, :, 1:]], axis=-1))
            ris3 = cayley_dickson_property_2d(np.concatenate([np.expand_dims(a[:, :, 0], -1), -a[:, :, 1:]], axis=-1),
                                              d)
            ris4 = cayley_dickson_property_2d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = np.concatenate([aux1, aux2], axis=-1)
    else:
        ris = onion1 * onion2

    return ris

def q_index_metric(im1, im2, size):
    """
        Q2n calculation on a window of dimension (size, size).
        Auxiliary function for Q2n calculation.

        Parameters
        ----------
        im1 : Numpy Array
            First MultiSpectral img. Dimensions: H, W, Bands
        im2 : Numpy Array
            Second MultiSpectral img. Dimensions: H, W, Bands
        size : int
            The size of the squared windows on which calculate the UQI index


        Return
        ------
        q : Numpy array
            The Q2n calculated on a window of dimension (size,size).
    """

    im1 = im1.astype(np.double)
    im2 = im2.astype(np.double)
    im2 = np.concatenate([np.expand_dims(im2[:, :, 0], -1), -im2[:, :, 1:]], axis=-1)

    depth = im1.shape[-1]
    for i in range(depth):
        im1[:, :, i], m, s = normalize_block(im1[:, :, i])
        if m == 0:
            if i == 0:
                im2[:, :, i] = im2[:, :, i] - m + 1
            else:
                im2[:, :, i] = -(-im2[:, :, i] - m + 1)
        else:
            if i == 0:
                im2[:, :, i] = ((im2[:, :, i] - m) / s) + 1
            else:
                im2[:, :, i] = -(((-im2[:, :, i] - m) / s) + 1)

    m1 = np.mean(im1, axis=(0, 1))
    m2 = np.mean(im2, axis=(0, 1))

    mod_q1m = np.sqrt(np.sum(m1 ** 2))
    mod_q2m = np.sqrt(np.sum(m2 ** 2))

    mod_q1 = np.sqrt(np.sum(im1 ** 2, axis=-1))
    mod_q2 = np.sqrt(np.sum(im2 ** 2, axis=-1))

    term2 = mod_q1m * mod_q2m
    term4 = mod_q1m ** 2 + mod_q2m ** 2
    temp = (size ** 2) / (size ** 2 - 1)
    int1 = temp * np.mean(mod_q1 ** 2)
    int2 = temp * np.mean(mod_q2 ** 2)
    int3 = temp * (mod_q1m ** 2 + mod_q2m ** 2)
    term3 = int1 + int2 - int3

    mean_bias = 2 * term2 / term4

    if term3 == 0:
        q = np.zeros((1, 1, depth), dtype='float64')
        q[:, :, -1] = mean_bias
    else:
        cbm = 2 / term3
        qu = cayley_dickson_property_2d(im1, im2)
        qm = cayley_dickson_property_1d(m1, m2)

        qv = temp * np.mean(qu, axis=(0, 1))
        q = qv - temp * qm
        q = q * mean_bias * cbm

    return q

def Q2n(outputs, labels, q_block_size=32, q_shift=32):
    """
        Q2n calculation on a window of dimension (size, size).  Giuseppe 论文
        Auxiliary function for Q2n calculation.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144
        [Garzelli09]        A. Garzelli and F. Nencini, "Hypercomplex quality assessment of multi/hyper-spectral images,"
                            IEEE Geoscience and Remote Sensing Letters, vol. 6, no. 4, pp. 662-665, October 2009.
        [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods",
                            IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.

        Parameters
        ----------
        outputs : Numpy Array
            The Fused image. Dimensions: H, W, Bands
        labels : Numpy Array
            The reference image. Dimensions: H, W, Bands
        q_block_size : int
            The windows size on which calculate the Q2n index
        q_shift : int
            The stride for Q2n index calculation

        Return
        ------
        q2n_index : float
            The Q2n index.
        q2n_index_map : Numpy Array
            The Q2n map, on a support of (q_block_size, q_block_size)
    """

    height, width, depth = labels.shape
    stepx = ceil(height / q_shift)
    stepy = ceil(width / q_shift)

    if stepy <= 0:
        stepx = 1
        stepy = 1

    est1 = (stepx - 1) * q_shift + q_block_size - height
    est2 = (stepy - 1) * q_shift + q_block_size - width

    if (est1 != 0) and (est2 != 0):
        labels = np.pad(labels, ((0, est1), (0, est2), (0, 0)), mode='reflect')
        outputs = np.pad(outputs, ((0, est1), (0, est2), (0, 0)), mode='reflect')

        outputs = outputs.astype(np.int16)
        labels = labels.astype(np.int16)

    height, width, depth = labels.shape

    if ceil(log2(depth)) - log2(depth) != 0:
        exp_difference = 2 ** (ceil(log2(depth))) - depth
        diff_zeros = np.zeros((height, width, exp_difference), dtype="float64")
        labels = np.concatenate([labels, diff_zeros], axis=-1)
        outputs = np.concatenate([outputs, diff_zeros], axis=-1)

    height, width, depth = labels.shape

    values = np.zeros((stepx, stepy, depth))
    for j in range(stepx):
        for i in range(stepy):
            values[j, i, :] = q_index_metric(
                labels[j * q_shift:j * q_shift + q_block_size, i * q_shift: i * q_shift + q_block_size, :],
                outputs[j * q_shift:j * q_shift + q_block_size, i * q_shift: i * q_shift + q_block_size, :],
                q_block_size
            )

    q2n_index_map = np.sqrt(np.sum(values ** 2, axis=-1))
    q2n_index = np.mean(q2n_index_map)

    return q2n_index.item()

