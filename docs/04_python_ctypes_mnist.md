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
lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.conv_backward_precol.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]
lib.apply_momentum_update.argtypes = [c_void_p, c_void_p, c_void_p, c_float, c_float, c_int]
lib.conv_update_fused.argtypes = [c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_float, c_float, c_int]
lib.clip_inplace.argtypes = [c_void_p, c_float, c_int]
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

完整可執行版本已放在 [train_mnist_so.py](train_mnist_so.py)。該檔案不用 `torchvision`，只用 NumPy 與 Python 標準函式庫解析 MNIST IDX gzip 檔。

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
conv_backward 或 conv_backward_precol
apply_momentum_update 或 conv_update_fused
```

如果 forward 階段保留了 `im2col_forward` 產生的 `d_col`，backward 建議使用 `conv_backward_precol`，避免重複 im2col 與配置 scratch buffer。`conv_backward` 仍可用於最小範例，因為它會在函式內自行建立 col buffer。

Momentum SGD 需要為每個 trainable buffer 準備同長度的 velocity buffer，訓練開始前設為 0，整個訓練期間保留：

```python
MOMENTUM = 0.9
d_v_conv = zeros(OUT_C * C_IN * KH * KW)
d_v_fc = zeros(10 * FC_IN)
d_v_bias = zeros(10)

lib.apply_momentum_update(d_w_conv, d_grad_conv, d_v_conv, c_float(lr), c_float(MOMENTUM), OUT_C * C_IN * KH * KW)
```

若需要 weight decay、gradient clipping 與 Momentum update 一次留在 GPU 端完成，可改用 `conv_update_fused`：

```python
lib.conv_update_fused(
    d_w_conv,
    d_grad_conv,
    d_v_conv,
    c_float(lr),
    c_float(MOMENTUM),
    c_float(weight_decay),
    c_float(clip_value),
    c_float(grad_normalizer),
    OUT_C * C_IN * KH * KW,
)
```

若只想做最小化測試，也可以繼續使用 `apply_sgd_update`。目前 CIFAR-10 trainer 預設使用 `conv_update_fused`，並用 `BatchWorkspace` 重用固定大小的 batch buffer。

## 完整範例檔

```text
docs/train_mnist_so.py
```

執行：

```bash
cd /home/s92137/NN/minimal_cuda_cnn
make -C cpp
python3 -u docs/train_mnist_so.py --download
```

第一次執行若本機沒有 MNIST，使用 `--download` 下載 gzip IDX 檔到 `data/mnist/`。如果機器不能連網，請先把四個 MNIST `.gz` 檔放到該目錄。

## 快速驗證

```bash
cuda-memcheck python3 -u docs/train_mnist_so.py
```

如果只想驗證 `.so` 函式本身，使用既有 sanity test：

```bash
python3 -u /tmp/so_function_check.py
cuda-memcheck python3 -u /tmp/so_function_check.py
```
