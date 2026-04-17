#!/usr/bin/env python3
"""Test reorganize + pool"""
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
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]

with open("minimal_cuda_cnn/data/cifar-10-batches-py/data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"][:8].astype(np.float32) / 255.0
x = x.reshape(-1, 3, 32, 32)

BATCH, C, H, W, KH, KW, OC = 8, 3, 32, 32, 3, 3, 32
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

d_conv_raw = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.gemm_forward(d_w, d_col, d_conv_raw, OC, BATCH * outH * outW, C * KH * KW)
lib.apply_relu(d_conv_raw, OC * BATCH * outH * outW)

# Reorganize: (OC, N) -> (N, OC, H, W)
d_conv = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.reorganize_forward(d_conv_raw, d_conv, BATCH, OC, outH, outW)

# Check reorganized output
h_conv = np.zeros((BATCH, OC, outH, outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv.ctypes.data, d_conv, BATCH * OC * outH * outW * 4)
print("Reorganized conv:")
print("  shape:", h_conv.shape)
print("  n=0, c=0, max:", h_conv[0, 0].max())

# Pool
d_pool = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
lib.apply_maxpool(d_conv, d_pool, BATCH, OC, outH, outW)

h_pool = np.zeros((BATCH, OC, 15, 15), dtype=np.float32)
lib.gpu_memcpy_d2h(h_pool.ctypes.data, d_pool, BATCH * OC * 15 * 15 * 4)
print("\nPool output:")
print("  shape:", h_pool.shape)
print("  n=0, c=0, max:", h_pool[0, 0].max())
print("  n=0, c=0, [0,:]:", h_pool[0, 0, 0, :].tolist())

lib.gpu_free(d_x)
lib.gpu_free(d_w)
lib.gpu_free(d_col)
lib.gpu_free(d_conv_raw)
lib.gpu_free(d_conv)
lib.gpu_free(d_pool)
print("\nDone!")
