# Minimal CUDA CNN

Minimal CUDA CNN is an experimental CUDA/C++ and Python project for building and debugging neural-network layers without using a deep-learning framework.

The project contains handwritten CUDA kernels, a shared library build, Python `ctypes` callers, CIFAR-10 data-loading experiments, and many focused debugging scripts for forward and backward passes.

## Scope

Implemented or partially explored components include:

- GPU memory helpers
- dense layers
- im2col and GEMM-style convolution paths
- convolution backward passes
- ReLU and leaky ReLU
- max pooling forward and backward variants
- layout conversion
- reorganize layers
- layer normalization and batch normalization experiments
- softmax/loss helpers
- SGD optimizer helpers
- AlexNet/ResNet-oriented experiments

This is a research and debugging workspace, not a polished framework API.

## Project Layout

```text
cpp/
  Makefile                 build shared CUDA library
  include/                 C++ headers
  src/                     CUDA/C++ kernels and layer implementations
python/
  train_*.py               training experiments
  test_*.py                focused kernel and pipeline tests
  debug_*.py               debugging scripts
data/
  cifar-10-batches-py/     local CIFAR-10 files
```

Large CIFAR-10 batch files and compiled `.so` files are intentionally ignored by Git.

## Requirements

- NVIDIA GPU
- CUDA toolkit
- Python 3
- NumPy

Some comparison/debug scripts may import PyTorch, but the CUDA kernels and core project code are handwritten. Scripts named with `pytorch` are reference checks, not the implementation path.

## Build

The Makefile currently points at CUDA 13.2 and targets `sm_86`:

```bash
cd /home/s92137/NN/minimal_cuda_cnn
make -C cpp
```

This produces:

```text
cpp/libminimal_cuda_cnn.so
```

If your CUDA install or GPU architecture differs, edit `cpp/Makefile`:

```makefile
CC = /usr/local/cuda-13.2/bin/nvcc
CFLAGS = -O3 -Xcompiler -fPIC -arch=sm_86
```

## Data

The local CIFAR-10 directory is:

```text
data/cifar-10-batches-py
```

The repository tracks only small metadata files. Download or copy the CIFAR-10 Python batch files locally before running training scripts that require them:

```text
data_batch_1
data_batch_2
data_batch_3
data_batch_4
data_batch_5
test_batch
```

## Usage

Build the CUDA shared library first:

```bash
make -C cpp
```

Run a small shared-library load test:

```bash
python3 python/test_so.py
```

Run one of the training experiments:

```bash
python3 python/train_real.py
```

Run focused kernel checks as needed:

```bash
python3 python/test_gemm.py
python3 python/test_softmax.py
python3 python/test_pool_2x2.py
```

## Notes

This directory contains many historical debugging scripts. When changing CUDA kernels, prefer small focused tests first, then run the larger training scripts after the kernel-level behavior is stable.
