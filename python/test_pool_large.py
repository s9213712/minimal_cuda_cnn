#!/usr/bin/env python3
"""Check pool with large conv output"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_int
import pickle

so = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
lib = ctypes.CDLL(so)
lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_malloc.restype = c_void_p
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_relu.argtypes = [c_void_p, c_int]
lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]

# Load CIFAR
with open("minimal_cuda_cnn/data/cifar-10-batches-py/data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"][:8].astype(np.float32) / 255.0
x = x.reshape(-1, 3, 32, 32)

BATCH, C, H, W = 8, 3, 32, 32
KH, KW, OC = 3, 3, 32
outH, outW = H - KH + 1, W - KW + 1

np.random.seed(42)
w = np.random.randn(OC * C * KH * KW).astype(np.float32) * 2.0

d_x = lib.gpu_malloc(BATCH * C * H * W * 4)
d_w = lib.gpu_malloc(OC * C * KH * KW * 4)
lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
lib.gpu_memcpy_h2d(d_w, w.ctypes.data, OC * C * KH * KW * 4)

col_size = C * KH * KW * BATCH * outH * outW
d_col = lib.gpu_malloc(col_size * 4)
lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)

d_conv = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.gemm_forward(d_w, d_col, d_conv, OC, BATCH * outH * outW, C * KH * KW)
lib.apply_relu(d_conv, OC * BATCH * outH * outW)

h_conv = np.zeros((BATCH, OC * outH * outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv.ctypes.data, d_conv, BATCH * OC * outH * outW * 4)
print(f"Conv output: min={h_conv.min():.4f}, max={h_conv.max():.4f}")
print(f"Conv output[0,:10]: {h_conv[0,:10]}")

# Pool
d_pool = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
lib.apply_maxpool(d_conv, d_pool, BATCH, OC, outH, outW)

h_pool = np.zeros((BATCH, OC * 15 * 15), dtype=np.float32)
lib.gpu_memcpy_d2h(h_pool.ctypes.data, d_pool, BATCH * OC * 15 * 15 * 4)
print(f"Pool output: min={h_pool.min():.4f}, max={h_pool.max():.4f}")
print(f"Pool output[0,:10]: {h_pool[0,:10]}")

lib.gpu_free(d_x)
lib.gpu_free(d_w)
lib.gpu_free(d_col)
lib.gpu_free(d_conv)
lib.gpu_free(d_pool)
print("Done!")
