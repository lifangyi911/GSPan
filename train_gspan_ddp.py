# 在 2 张显卡上运行
# torchrun --nproc_per_node=2 train_gspan_ddp.py --dataset_name qb --batch_size 4
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import random
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
parser.add_argument('--batch_size', type=int, default=24, help='batch_size') #default 16 for training, 1 for testing
parser.add_argument('--num_epochs', type=int, default=500, help='training epochs')
parser.add_argument('--lr', type=float, default=0.0004, help='output image channels')
parser.add_argument('--log_freq', type=int, default=10, help='screen output')
parser.add_argument('--patience', type=int, default=250, help='early stopping patience')
parser.add_argument('--dataset_name', type=str, default='wv3_4K', help='读取h5文件')
parser.add_argument('--gpuid', type=str, default='0,1,2,3', help='screen output')
parser.add_argument('--version', type=str, default=r'1',help='method version')
parser.add_argument('--resume', type=str, default='/raid/dataset/data_lifangyi/apnn/logs_wv3_4K_GS_SR/GS_SR_ckpt_epoch300.pth', help='path to checkpoint to resume from (e.g. .../net_ckpt_epoch200.pth)')
args = parser.parse_args()


############## arguments
sensor = 'WV3'
if sensor=='GF2':
    bit = 10
else:
    bit = 11
spectral_num = args.output_nc
img_range = 2.0 ** bit   # 1.0  # 2047.0
disk_name = 'data7a'  # QB nvme4a, GF2: data7a, WV4-bands: nvme4b


method = 'GS_SR2'
dataset = r'/raid/dataset/data_lifangyi/data/pansharpen/%s' % args.dataset_name
model_dir = r'/raid/dataset/data_lifangyi/apnn'

test_sample_dir = r'/raid/dataset/data_lifangyi/apnn/results/%s/%s_results/test_sample_v%s' % (args.dataset_name,method,args.version)
origin_test_sample_dir = r'/raid/dataset/data_lifangyi/apnn/results/%s/%s_results/origin_test_sample_v%s' % (args.dataset_name,method,args.version)
record_dir = r'/raid/dataset/data_lifangyi/apnn/results//record_v1'
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


## Device configuration
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

dist.init_process_group(backend='nccl')
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
print(f'Using device: {device}')

if local_rank == 0:
    mode = "a" if args.resume else "w"
    f = open("./logs_%s_%s/log_v%s.txt" % (args.dataset_name, method, args.version), mode)
else:
    f = None  # 其他进程不打开文件
if args.resume and local_rank == 0:
    f.write(f"\n\n{'=' * 20} RESUME TRAINING AT {datetime.now()} {'=' * 20}\n")

# 初始化损失函数
FastHQNR = EfficientHQNR(sensor=sensor, ratio=4, channels=4, device='cuda')
hqnr_calculator = EfficientHQNR(
                sensor=sensor,
                ratio=4,
                channels=spectral_num,
                device=device
            )
############## Loading datasets
print('===> Loading datasets')

train_low_set = Dataset_Pro(os.path.join(dataset, 'train_%s.h5' % args.dataset_name),img_scale=img_range,augment=False)
train_sampler = DistributedSampler(train_low_set)
train_loader = DataLoader(train_low_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=2, pin_memory=False, drop_last=True)
print('len(train_loader)', len(train_loader))


criterionL1 = nn.L1Loss().to(device)
criterionSmoothL1 = nn.SmoothL1Loss(beta = 0.5).to(device)
criterionMSE = nn.MSELoss().to(device)

net = model = GSPan(spectral_num=spectral_num,sensor = sensor,shuffle_scale1 = 2, shuffle_scale2 = 1).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))


# # 1. 设定参数
warmup_epochs = 5      # 预热代数
T_max = args.num_epochs - warmup_epochs  # 余弦退火的总周期（总代数减去预热代数）
eta_min = 1e-7         # 最小学习率，通常设为一个很小的值
#
# 2. 定义 Warmup 调度器：从 0.1 * lr 线性增加到 lr
scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=warmup_epochs
)

# 3. 定义余弦退火调度器
# T_max 是到达 eta_min 所需的迭代次数
scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=T_max,
    eta_min=eta_min
)

# 4. 组合调度器
# milestones=[warmup_epochs] 表示在第 5 个 epoch 结束后切换到余弦退火
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[scheduler_warmup, scheduler_main],
    milestones=[warmup_epochs]
)

if local_rank == 0:
    print(args)
    print('')
    print(optimizer)
    print('')
net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)


def train(train_data_loader, valid_data_loader, num_epochs, patience, f):
    # 创建 log 目录
    if local_rank == 0:
        writer = SummaryWriter(log_dir=f'./logs_{args.dataset_name}_{method}/log_v{args.version}/')


    train_losses, valid_losses = [],[]
    avg_train_losses, avg_valid_losses = [],[]

    Q2n, ERGAS, SCC, SAM = [], [], [], []
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main_process = (rank == 0)
    for epoch in range(start_epoch, num_epochs+1):
        train_sampler.set_epoch(epoch)  # 保证数据打乱

        start = time.time()
        curr_lr = optimizer.param_groups[0]['lr']
        if local_rank == 0:

            print(f"Epoch {epoch} | Current LR: {curr_lr:.6f}")

        for i, batch in tqdm(enumerate(train_data_loader, 1), total=len(train_data_loader),disable=not is_main_process,   # 非主进程不显示
    dynamic_ncols=True):
            input_pan = batch['pan'].to(device)
            input_lr = batch['ms'].to(device)
            input_lr_u = batch['lms'].to(device)
            target = batch['gt'].to(device)

            net.train()
            net.zero_grad()
            optimizer.zero_grad()

            batch_gs_parameters = net(input_lr_u, input_pan )
            gt_size = target.shape[2:]  # [h_gt, w_gt]


            I_fake = generate_2D_gaussian_splatting_batch(sr_size=gt_size, gs_parameters=batch_gs_parameters,
                                                      scale=1,
                                                      sample_coords=None,
                                                      scale_modify = [1,1],
                                                      default_step_size = 1.2,
                                                      cuda_rendering=True,
                                                      mode = 'scale',
                                                      if_dmax = True,
                                                     spectral_num = spectral_num,
                                                      dmax_mode = 'fix',
                                                      dmax = 0.5)

            I_fake = I_fake + input_lr_u


            I_fake = trim_image(I_fake, L = 0, R = 1)

            loss = criterionL1(I_fake, target)

            loss.backward()


            optimizer.step()

            train_losses.append(loss.item())
        if local_rank == 0:
            elapsed = (time.time() - start)
            print(f'--> Epoch {epoch} Train Time: {elapsed:.2f} s  Current LR: {curr_lr:.6f}  train_loss: {loss.item():.6f} ')
            print(f'--> Epoch {epoch} Train Time: {elapsed:.2f} s  Current LR: {curr_lr:.6f}  train_loss: {loss.item():.6f} ' , file=f, flush=True)

        scheduler.step()

        if epoch % 10 == 0 and local_rank == 0:
            checkpoint = {
                'epoch': epoch,
                'net': net.module.state_dict(),  # 记得保存 .module
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            save_path = os.path.join(model_dir + r"/logs_%s_%s/" % (args.dataset_name, method), f'{method}_ckpt_epoch{epoch}.pth')
            torch.save(checkpoint, save_path)
            print(f'Checkpoint saved to {save_path}')
        if epoch % 5 == 0:
            net.eval()
            with torch.no_grad():
                for k, data in enumerate(valid_data_loader,1):
                    input_pan, input_lr, input_lr_u, target = data['pan'].to(device), data['ms'].to(device), data['lms'].to(device),data['gt'].to(device)


                    batch_gs_parameters = net(input_lr_u, input_pan )
                    gt_size = target.shape[2:]  # [h_gt, w_gt]

                    psh = generate_2D_gaussian_splatting_batch(sr_size=gt_size, gs_parameters=batch_gs_parameters,
                                                                  scale=1,
                                                                  sample_coords=None,
                                                                  scale_modify=[1, 1],
                                                                  default_step_size=1.2,
                                                                  cuda_rendering=True,
                                                                  mode='scale',
                                                                  if_dmax=True,
                                                                  dmax_mode='fix',
                                                               spectral_num=spectral_num,
                                                                  dmax=0.1)
                    psh = psh + input_lr_u

                    psh = psh * img_range

                    psh = trim_image(psh, L = 0, R = 2**bit - 1)

                    loss = criterionSmoothL1(psh, target)
                    valid_losses.append(loss.item())

                    # 计算RR metrics
                    target, psh = target.detach().cpu().numpy(), psh.detach().cpu().numpy()
                    target, psh = np.transpose(target, (0, 2, 3, 1)), np.transpose(psh, (0, 2, 3, 1))
                    target, psh = target[0], psh[0]

                    ERGAS.append(mtc.ERGAS(target, psh, 4))
                    SAM.append(mtc.SAM(target, psh, eps=0.0005))  #
                    Q2n.append(mtc.Q2n(target, psh, q_block_size=32, q_shift=32))
                    SCC.append(mtc.SCC(target, psh))


            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            if local_rank == 0:
                print_msg = (f'net Training [{epoch}/{num_epochs}] ' + f'train_loss: {train_loss:.6f} ' +
                            f'valid_loss: {valid_loss:.4f}')
                print(print_msg)
                print(print_msg, file=f, flush=True)

            # --- 写入 TensorBoard ---
            if local_rank == 0:
                writer.add_scalar('train loss', train_loss, epoch)
                writer.add_scalar('valid loss', valid_loss, epoch)

                print("Test | RRMetrics: SAM, SCC, ERGAS, Q2n")
                print("Test | RRMetrics: SAM, SCC, ERGAS, Q2n", file=f, flush=True)

                print("%.4lf±%.4lf, %.4lf±%.4lf, %.4lf±%.4lf, %.4lf±%.4lf" %
                    (np.mean(SAM), np.std(SAM), np.mean(SCC), np.std(SCC),np.mean(ERGAS), np.std(ERGAS),np.mean(Q2n), np.std(Q2n)))
                print("%.4lf±%.4lf, %.4lf±%.4lf, %.4lf±%.4lf, %.4lf±%.4lf" %
                    (np.mean(SAM), np.std(SAM), np.mean(SCC), np.std(SCC),np.mean(ERGAS), np.std(ERGAS),np.mean(Q2n), np.std(Q2n)), file=f, flush=True)

            train_losses = []
            valid_losses = []


if "__main__" == __name__:
    print('Begin Time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # Train
    start = time.time()
    train(train_loader, test_low_loader, args.num_epochs, args.patience, f)
    print('==>Train 时长： {:.2f}h\n'.format((time.time() - start) / 3600))
    print('==>Train 时长： {:.2f}h\n'.format((time.time() - start) / 3600), file=f, flush=True)
