# Minimal CUDA CNN

[繁體中文 README](README.zh-TW.md)

Minimal CUDA CNN is an experimental CUDA/C++ and Python project for training and debugging a small CIFAR-10 CNN without using a deep-learning framework for the handwritten implementation path.

The repository contains:

- CUDA/C++ kernels built into `cpp/libminimal_cuda_cnn.so`
- Python `ctypes` bindings for the shared object
- a handwritten CUDA CIFAR-10 training script
- a matched PyTorch baseline for comparison
- documentation for compiling and calling the shared object

CIFAR-10 dataset files, compiled `.so` files, `__pycache__`, and training checkpoints are local-only and ignored by Git.

## Quick Start

From a clean machine with CUDA, `nvcc`, Python, and NumPy available:

```bash
git clone https://github.com/s9213712/minimal_cuda_cnn.git
cd minimal_cuda_cnn
make -C cpp
python3 python/prepare_cifar10.py
python3 python/train_split.py
```

`prepare_cifar10.py` downloads and extracts the CIFAR-10 Python archive into `data/cifar-10-batches-py/`. The first training run builds on the shared object produced at `cpp/libminimal_cuda_cnn.so`.

## Current CIFAR-10 Experiment

Both the handwritten CUDA trainer and the PyTorch baseline use the same comparison conditions:

- Dataset: CIFAR-10 Python batches under `data/cifar-10-batches-py`
- Train source: `data_batch_1` through `data_batch_5`
- Split: `45000` train / `5000` validation
- Test: official `test_batch`
- Dataset split seed: `42`
- Initial weight seed: `42`
- Input normalization: CIFAR-10 channel mean/std
- Batch size: `64`
- Max epochs: `50`
- Early stopping patience: `8`
- Architecture: valid 3x3 conv stack, no padding
- Activation: LeakyReLU with alpha `0.1`
- Optimizer behavior: manual Momentum SGD updates, weight decay, gradient clipping, and LR plateau reduction
- Initial learning rates: `0.005` for conv1, other conv layers, and FC
- Momentum: `0.9`
- Conv gradient spatial normalization: currently disabled

Architecture:

```text
Conv(3->32, 3x3 valid) -> LeakyReLU
Conv(32->32, 3x3 valid) -> LeakyReLU
MaxPool(2x2)
Conv(32->64, 3x3 valid) -> LeakyReLU
Conv(64->64, 3x3 valid) -> LeakyReLU
MaxPool(2x2)
FC(1600->10)
```

Shape flow:

```text
32x32 -> 30x30 -> 28x28 -> 14x14 -> 12x12 -> 10x10 -> 5x5
```

## Python Layout

```text
python/
  train_split.py                 handwritten CUDA CIFAR-10 trainer
  train_split_torch_baseline.py  matched PyTorch baseline
  train_config.py                shared training/model constants
  cifar10_data.py                CIFAR-10 preparation, loading, split, normalization
  prepare_cifar10.py             standalone dataset preparation command
  model_init.py                  shared host-side initial weights
  cuda_backend.py                ctypes bindings and GPU helper functions
  model_forward.py               CUDA inference/evaluation helper
  model_weights.py               CUDA weight upload/checkpoint/cleanup helper
```

The split keeps `train_split.py` focused on the training loop and backward pass. Shared configuration and initialization are reused by both CUDA and PyTorch scripts so the two runs can be compared under the same conditions.

## Requirements

- NVIDIA GPU for the handwritten CUDA trainer
- CUDA toolkit
- Optional cuBLAS, usually installed with the CUDA toolkit. Enabled by default for speed, but the handwritten CUDA fallback remains available.
- Python 3
- NumPy
- PyTorch for `python/train_split_torch_baseline.py`

The CUDA implementation path does not depend on PyTorch. PyTorch is only used as a baseline/reference trainer.

## Build

Build the CUDA shared library:

```bash
cd minimal_cuda_cnn
make -C cpp
```

This produces:

```text
cpp/libminimal_cuda_cnn.so
```

By default, the shared object links cuBLAS. `gemm_forward` and the conv weight-gradient path in `conv_backward` use `cublasSgemm`.

To build the fast cuBLAS backend:

```bash
make -C cpp USE_CUBLAS=1
```

To build the from-scratch fallback without linking cuBLAS:

```bash
make -C cpp USE_CUBLAS=0
```

If your CUDA install or GPU architecture differs, edit `cpp/Makefile`:

```makefile
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CFLAGS = -O3 -Xcompiler -fPIC -arch=sm_86
USE_CUBLAS ?= 1
```

To verify that the shared object was built and exports the expected API:

```bash
make -C cpp check
```

## Data

The trainer can prepare CIFAR-10 automatically:

```bash
python3 python/prepare_cifar10.py
```

This downloads and extracts the CIFAR-10 Python archive into:

```text
data/cifar-10-batches-py/
  data_batch_1
  data_batch_2
  data_batch_3
  data_batch_4
  data_batch_5
  test_batch
```

If the machine cannot access the network, place the extracted CIFAR-10 Python batch files in that same directory manually.

The current experiment uses all five CIFAR-10 training batches, controlled in `python/train_config.py`:

```python
N_TRAIN = 45000
N_VAL = 5000
TRAIN_BATCH_IDS = (1, 2, 3, 4, 5)
```

## Run CUDA Trainer

```bash
cd minimal_cuda_cnn
python3 python/train_split.py
```

The CUDA trainer saves its best local checkpoint to:

```text
python/best_model_split.npz
```

That checkpoint is ignored by Git.

`train_split.py` now keeps the main FC backward and optimizer updates on GPU:

- `dense_forward` consumes `d_pool2_nchw` directly instead of doing `d_pool2_nchw -> CPU -> GPU`
- `dense_backward_full` produces FC weight/bias gradients and the pool gradient
- FC/Conv parameters use fused GPU momentum update with weight decay and gradient clipping
- pool gradients use GPU in-place clipping
- softmax, cross-entropy gradient, loss accumulation, and batch accuracy run in CUDA; Python only downloads scalar loss/correct counts
- conv forward GEMM uses cuBLAS when `USE_CUBLAS=1`, or the handwritten GEMM kernel when `USE_CUBLAS=0`
- conv backward weight gradients use im2col + cuBLAS when `USE_CUBLAS=1`; the handwritten fallback remains available when `USE_CUBLAS=0`
- fixed-shape per-batch GPU buffers are allocated once in `BatchWorkspace` and reused across the batch loop
- train/eval support remainder batches instead of dropping the final partial batch
- conv backward reuses the forward im2col buffers instead of recomputing im2col and allocating extra scratch space

2026-04-18 smoke test with `USE_CUBLAS=1`: `timeout 70s python3 -u python/train_split.py` reached epoch 6, with epochs taking about `8.6-14.8s` and epoch-6 validation accuracy at `73.14%`. This is not a full benchmark, but it confirms the previous `Batch 100/703` speed point dropped from roughly two minutes to a few seconds.

## Run PyTorch Baseline

```bash
cd minimal_cuda_cnn
python3 python/train_split_torch_baseline.py
```

If CUDA/NVML initialization crashes PyTorch in the current environment, force the baseline to CPU:

```bash
FORCE_CPU=1 python3 python/train_split_torch_baseline.py
```

The PyTorch baseline saves its best local checkpoint to:

```text
python/best_model_split_torch.pt
```

That checkpoint is ignored by Git.

Previous Momentum baseline result from the older `data_batch_1`-only run on 2026-04-18:

```text
Best validation accuracy: 63.26% at epoch 25
Official test accuracy:   62.16%
Early stopping:           epoch 33
Device used:              CPU
```

The current full-dataset configuration has not been fully benchmarked yet. PyTorch reported `torch.cuda.is_available() == False` in this environment, so the baseline run above used CPU. The script automatically uses CUDA if PyTorch can see a CUDA device.

## Shared Object Documentation

The shared object usage guide is split across:

```text
docs/USAGE.md
docs/01_project_files.md
docs/02_build_shared_library.md
docs/03_c_api_reference.md
docs/04_python_ctypes_mnist.md
docs/05_cpp_linking.md
docs/06_layout_and_debug.md
docs/train_mnist_so.py
```

These files describe the `.cu` and `.h` roles, how to compile the `.so`, and how to call it from Python or C++.

## Useful Checks

Syntax check:

```bash
python3 -m py_compile python/*.py docs/train_mnist_so.py
```

Quick PyTorch initialization equality check:

```bash
python3 -c "import sys, torch; sys.path.insert(0, 'python'); from model_init import init_weights; from train_split_torch_baseline import TorchCifarCnn, load_initial_weights; from train_config import INIT_SEED; m=TorchCifarCnn(); load_initial_weights(m, torch.device('cpu')); w=init_weights(INIT_SEED); print((m.conv1.weight.detach().flatten()-torch.from_numpy(w[0])).abs().max().item())"
```

Expected output:

```text
0.0
```

## Notes

This is still a research/debugging workspace. When changing CUDA kernels, run focused kernel-level checks first, then run `python/train_split.py` and compare against `python/train_split_torch_baseline.py`.
