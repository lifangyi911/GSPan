# GSPan

Official PyTorch implementation of **GSPan**: arbitrary-scale pansharpening built on
**continuous 2D Gaussian splatting**. The scene is represented by a set of 2D Gaussian
primitives predicted from the low-resolution multispectral image and the panchromatic
image; the high-resolution fusion result is obtained by differentiable Gaussian
splatting/rendering, which supports continuous (arbitrary) scaling and scale-decoupled
asymmetric inference (SDAI).

> The manuscript is currently under review. The code in this repository is the
> implementation used to produce all the quantitative and qualitative results in the paper.

## Dataset

All experiments are conducted on the **PanCollection** dataset (WorldView-2,
WorldView-3, QuickBird and Gaofen-2).

- **Dataset source: <https://github.com/liangjiandeng/PanCollection>**
- Please download the dataset from the link above and follow that repository for the
  simulation procedure, license and citation of the data.

The `.h5` files expected by this code follow the PanCollection / DLPan-Toolbox
convention (`gt` / `ms` / `lms` / `pan` keys), e.g. for `--dataset_name qb`:

```
<dataset>/
├── train_qb.h5                      # training set (reduced resolution)
├── valid_qb.h5                      # validation set (reduced resolution)
├── test_qb_multiExm1.h5             # reduced-resolution test set
└── test_qb_OrigScale_multiExm1.h5   # full-resolution test set
```

The data root directory is set by the `dataset` variable at the top of the scripts
(`train_gspan_ddp.py`, `test_gspan_ddp.py`, `test_SR_SDAI.py`) — **change it to your own
local path before running**. Likewise `--resume` / `eval_ckpt_path` and the result
directories in those scripts are absolute paths from the machine used for the paper and
have to be adapted locally.

## Repository structure

```
GSPan/
├── model.py                     # GSPan network (windowed cross-attention, Gaussian parameter heads)
├── train_gspan_ddp.py           # training (DistributedDataParallel)
├── test_gspan_ddp.py            # reduced-resolution (RR) + full-resolution (FR) evaluation
├── test_SR_SDAI.py              # large-scene / scale-decoupled asymmetric inference (SDAI)
├── dataset_h5.py                # .h5 dataset loaders (Dataset_Pro, MultiExmTest_h5)
├── utils/
│   ├── gaussian_splatting.py    # 2D Gaussian rendering (PyTorch and CUDA paths)
│   └── gs_cuda_dmax/            # custom CUDA extension (forward + backward of the splatting)
│       ├── gs.cu / gswrapper.cpp                 # single-image renderer  -> module `gscuda`
│       └── gs_batch.cu / gswrapper_batch.cpp     # batched renderer       -> module `gscuda_batch`
├── wald_utilities.py            # Wald protocol / MTF-based degradation utilities
├── pansharp_metrics.py          # reference-based metrics (SAM, ERGAS, Q2n, SCC, ...)
└── hqnr_torch_fast.py           # no-reference metric HQNR (D_lambda, D_s) in PyTorch
```

## Requirements

- Python 3.8+, PyTorch (with CUDA), torchvision
- `numpy`, `scipy`, `h5py`, `opencv-python`, `einops`, `tqdm`, `openpyxl`,
  `tensorboard`, `thop`

```bash
pip install numpy scipy h5py opencv-python einops tqdm openpyxl tensorboard thop
```

## Build the CUDA extension

The splatting renderer is a custom CUDA extension. Build it **before** training or
testing:

```bash
python setup_gscuda_batch.py build_ext --inplace
```

This produces the `gscuda_batch` module used by the batched renderer. The single-image
renderer (`utils/gs_cuda_dmax/gswrapper.py`) expects a second module named `gscuda`,
which is compiled from `gs.cu` + `gswrapper.cpp` in the same way.

## Training

```bash
# 2 GPUs
torchrun --nproc_per_node=2 train_gspan_ddp.py --dataset_name qb --batch_size 4

# single GPU
CUDA_VISIBLE_DEVICES=0 python train_gspan_ddp.py --dataset_name qb --batch_size 4 --gpuid 0
```

| argument | default | meaning |
| --- | --- | --- |
| `--dataset_name` | `wv3_4K` | dataset used to build the `.h5` file names |
| `--output_nc` | `8` | number of spectral bands (4 for QB/GF2/WV3, 8 for WV2) |
| `--batch_size` | `24` | batch size |
| `--num_epochs` | `500` | training epochs |
| `--lr` | `4e-4` | initial learning rate (warm-up + cosine annealing) |
| `--resume` | – | checkpoint to resume from |
| `--gpuid` | `0,1,2,3` | GPUs used by the run |

Note: the sensor configuration (`sensor`, `bit`, `spectral_num`) is defined near the top
of each script and must match the dataset being used.

## Evaluation

```bash
# reduced resolution + full resolution
python test_gspan_ddp.py --dataset_name qb --resume <path/to/ckpt.pth>

# full-resolution inference with scale-decoupled asymmetric inference (SDAI),
# tiled inference for large scenes
python test_SR_SDAI.py --dataset_name qb --resume <path/to/ckpt.pth>
```

Both scripts print the mean ± std of SAM / SCC / ERGAS / Q2n over the test scenes and
write the per-scene results to an `.xlsx` record file.

## Metrics

- **Reference-based** (`pansharp_metrics.py`): SAM, ERGAS, Q2n, SCC, CC, PSNR, ...
- **No-reference** (`hqnr_torch_fast.py`): HQNR = (1 − D_λ)(1 − D_s), computed in a
  batched/vectorized PyTorch implementation.

Metrics are computed in the original radiometric (DN) scale, i.e. on images multiplied
back by `img_range = 2**bit`.

## Citation

```bibtex
@article{gspan,
  title   = {GSPan: Arbitrary-Scale Pansharpening with Continuous 2D Gaussian Splatting},
  author  = {Anonymous},
  journal = {Submitted to Information Fusion},
  year    = {2026}
}
```

The bibliographic information will be updated once the paper is accepted.

## Acknowledgements

This code is built upon the publicly available
[PanCollection](https://github.com/liangjiandeng/PanCollection) /
[DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox) benchmarks for data and
evaluation protocols, and on the 2D Gaussian splatting CUDA renderer design of the
original 2DGS project. We thank the authors for releasing their code and data.
