# Python ctypes 與 MNIST 範例

本文說明如何用 Python `ctypes` 載入 `.so`，並用 MNIST 示範一個最小 CNN 訓練流程。

## 載入 `.so`

```python
import ctypes
import os
import numpy as np
from ctypes import c_float, c_int, c_void_p

ROOT = "/home/s92137/NN/minimal_cuda_cnn"
lib = ctypes.CDLL(os.path.join(ROOT, "cpp", "libminimal_cuda_cnn.so"))

lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_malloc.restype = c_void_p
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]

lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.cnhw_to_nchw.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.nchw_to_cnhw.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.leaky_relu_forward.argtypes = [c_void_p, c_float, c_int]
lib.leaky_relu_backward.argtypes = [c_void_p, c_void_p, c_float, c_int]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.dense_backward_full.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]
```

## Host/device helper

```python
def upload(arr):
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    ptr = lib.gpu_malloc(arr.nbytes)
    lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.nbytes)
    return ptr

def zeros(size):
    ptr = lib.gpu_malloc(size * 4)
    lib.gpu_memset(ptr, 0, size * 4)
    return ptr

def download(ptr, shape):
    out = np.empty(shape, dtype=np.float32)
    lib.gpu_memcpy_d2h(out.ctypes.data, ptr, out.nbytes)
    return out

def free_all(*ptrs):
    for ptr in ptrs:
        if ptr:
            lib.gpu_free(ptr)
```

## MNIST 示範網路

```text
Input 1x28x28
Conv(1->8, 3x3) -> LeakyReLU
MaxPool 2x2
FC(8*13*13 -> 10)
Softmax cross entropy
```

`torchvision.datasets.MNIST` 只用來讀資料，不使用 PyTorch 訓練。

## 訓練流程骨架

```python
BATCH = 64
OUT_C, C_IN, H, W = 8, 1, 28, 28
KH, KW = 3, 3
OH, OW = 26, 26
PH, PW = 13, 13
FC_IN = OUT_C * PH * PW
ALPHA = 0.1

def forward(x, d_w_conv, d_w_fc, d_b_fc):
    n = x.shape[0]
    d_x = upload(x)

    d_col = lib.gpu_malloc(C_IN * KH * KW * n * OH * OW * 4)
    d_conv = lib.gpu_malloc(OUT_C * n * OH * OW * 4)
    lib.im2col_forward(d_x, d_col, n, C_IN, H, W, KH, KW, OH, OW)
    lib.gemm_forward(d_w_conv, d_col, d_conv, OUT_C, n * OH * OW, C_IN * KH * KW)
    lib.leaky_relu_forward(d_conv, c_float(ALPHA), OUT_C * n * OH * OW)

    d_pool = lib.gpu_malloc(OUT_C * n * PH * PW * 4)
    d_idx = lib.gpu_malloc(OUT_C * n * PH * PW * 4)
    lib.maxpool_forward_store(d_pool, d_conv, d_idx, n, OUT_C, OH, OW)

    d_pool_nchw = lib.gpu_malloc(n * OUT_C * PH * PW * 4)
    lib.cnhw_to_nchw(d_pool, d_pool_nchw, n, OUT_C, PH, PW)

    d_logits = lib.gpu_malloc(n * 10 * 4)
    lib.dense_forward(d_pool_nchw, d_w_fc, d_b_fc, d_logits, n, FC_IN, 10)
    logits = download(d_logits, (n, 10))

    cache = (d_x, d_col, d_conv, d_pool, d_idx, d_pool_nchw, d_logits)
    return logits, cache
```

Backward 的主要順序：

```text
CPU softmax/cross entropy -> upload grad_logits
dense_backward_full
nchw_to_cnhw
maxpool_backward_use_idx
leaky_relu_backward
conv_backward
apply_sgd_update
```

## 完整檔案建議

可建立：

```text
python/train_mnist_so.py
```

完整程式可直接沿用這份骨架補上：

1. `torchvision.datasets.MNIST` 讀取 `x_train/y_train/x_test/y_test`。
2. He init 初始化 `w_conv` 與 `w_fc`。
3. 每個 batch 呼叫 `forward`。
4. 在 CPU 計算 softmax loss 與 `grad_logits`。
5. 呼叫 CUDA backward API 與 `apply_sgd_update`。
6. 每個 batch 結束 `free_all(...)`。

執行：

```bash
cd /home/s92137/NN/minimal_cuda_cnn
make -C cpp
python3 -u python/train_mnist_so.py
```

第一次執行 `torchvision.datasets.MNIST(..., download=True)` 需要網路下載資料。如果機器不能連網，請先把 MNIST 放到 `data/MNIST/raw/`。

## 快速驗證

```bash
cuda-memcheck python3 -u python/train_mnist_so.py
```

如果只想驗證 `.so` 函式本身，使用既有 sanity test：

```bash
python3 -u /tmp/so_function_check.py
cuda-memcheck python3 -u /tmp/so_function_check.py
```

