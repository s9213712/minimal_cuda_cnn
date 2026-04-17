#!/usr/bin/env python3
"""Debug pool backward with real data"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import pickle

so = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
lib = ctypes.CDLL(so)
lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_malloc.restype = c_void_p
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_relu.argtypes = [c_void_p, c_int]
lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_nchw.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

with open("minimal_cuda_cnn/data/cifar-10-batches-py/data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"][:2].astype(np.float32) / 255.0
x = x.reshape(-1, 3, 32, 32)

BATCH, C, H, W = 2, 3, 32, 32
KH, KW, OC = 3, 3, 32
outH, outW = H - KH + 1, W - KW + 1

np.random.seed(42)
w = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.05

d_x = lib.gpu_malloc(BATCH * C * H * W * 4)
d_w = lib.gpu_malloc(OC * C * KH * KW * 4)
lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
lib.gpu_memcpy_h2d(d_w, w.ctypes.data, OC * C * KH * KW * 4)

col_size = C * KH * KW * BATCH * outH * outW
d_col = lib.gpu_malloc(col_size * 4)
lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)

d_conv_raw = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.gemm_forward(d_w, d_col, d_conv_raw, OC, BATCH * outH * outW, C * KH * KW)
lib.apply_relu(d_conv_raw, OC * BATCH * outH * outW)

d_conv = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.reorganize_forward(d_conv_raw, d_conv, BATCH, OC, outH, outW)

d_pool = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
lib.apply_maxpool(d_conv, d_pool, BATCH, OC, outH, outW)

# Read conv output
h_conv = np.zeros((BATCH, OC, outH, outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv.ctypes.data, d_conv, BATCH * OC * outH * outW * 4)
print("Conv output (n=0, c=0):")
print(f"  max={h_conv[0,0].max():.4f}, at", np.unravel_index(h_conv[0,0].argmax(), h_conv[0,0].shape))

# Read pool output
h_pool = np.zeros((BATCH, OC, 15, 15), dtype=np.float32)
lib.gpu_memcpy_d2h(h_pool.ctypes.data, d_pool, BATCH * OC * 15 * 15 * 4)
print("\nPool output (n=0, c=0):")
print(f"  max={h_pool[0,0].max():.4f}, at", np.unravel_index(h_pool[0,0].argmax(), h_pool[0,0].shape))

# Create gradient of all ones for pool output
grad_pool_data = np.ones((BATCH, OC, 15, 15), dtype=np.float32)
d_pool_grad = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
lib.gpu_memcpy_h2d(d_pool_grad, grad_pool_data.ctypes.data, OC * BATCH * 15 * 15 * 4)

# Pool backward
d_conv_grad = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.gpu_memset(d_conv_grad, 0, OC * BATCH * outH * outW * 4)
lib.maxpool_backward_nchw(d_pool_grad, d_conv, d_conv_grad, BATCH, OC, outH, outW)

# Read conv gradient
h_conv_grad = np.zeros((BATCH, OC, outH, outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv_grad.ctypes.data, d_conv_grad, BATCH * OC * outH * outW * 4)
print("\nConv gradient after pool backward (n=0, c=0):")
print(f"  max={h_conv_grad[0,0].max():.4f}, sum={h_conv_grad[0,0].sum():.4f}")

lib.gpu_free(d_x)
lib.gpu_free(d_w)
lib.gpu_free(d_col)
lib.gpu_free(d_conv_raw)
lib.gpu_free(d_conv)
lib.gpu_free(d_pool)
lib.gpu_free(d_pool_grad)
lib.gpu_free(d_conv_grad)
print("\nDone!")
