#!/usr/bin/env python3
"""Debug pool output in detail"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int

so = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
lib = ctypes.CDLL(so)
lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_malloc.restype = c_void_p
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.leaky_relu_forward.argtypes = [c_void_p, c_float, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

BATCH, C, H, W = 4, 3, 32, 32
KH, KW, OC = 3, 3, 32
outH, outW = H - KH + 1, W - KW + 1

x = np.random.randn(BATCH, C, H, W).astype(np.float32) * 0.05
w = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.05

d_x = lib.gpu_malloc(BATCH * C * H * W * 4)
d_w = lib.gpu_malloc(OC * C * KH * KW * 4)
lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
lib.gpu_memcpy_h2d(d_w, w.ctypes.data, OC * C * KH * KW * 4)

col_size = C * KH * KW * BATCH * outH * outW
d_col = lib.gpu_malloc(col_size * 4)
d_conv_raw = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
d_conv = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
d_pool = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
d_max_idx = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)

lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)
lib.gemm_forward(d_w, d_col, d_conv_raw, OC, BATCH * outH * outW, C * KH * KW)
lib.leaky_relu_forward(d_conv_raw, c_float(0.1), OC * BATCH * outH * outW)
lib.reorganize_forward(d_conv_raw, d_conv, BATCH, OC, outH, outW)

# Check conv values before pool
h_conv = np.zeros((BATCH, OC, outH, outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv.ctypes.data, d_conv, BATCH * OC * outH * outW * 4)
print("Conv (n=0, c=0) sample [0:3, 0:3]:")
print(h_conv[0, 0, 0:3, 0:3])
print("Conv stats:", h_conv.min(), h_conv.max(), h_conv.mean())

# Run pool
lib.maxpool_forward_store(d_pool, d_conv, d_max_idx, BATCH, OC, outH, outW)

# Check pool output
h_pool = np.zeros((BATCH, OC, 15, 15), dtype=np.float32)
lib.gpu_memcpy_d2h(h_pool.ctypes.data, d_pool, BATCH * OC * 15 * 15 * 4)
print("\nPool (n=0, c=0) sample [0:3, 0:3]:")
print(h_pool[0, 0, 0:3, 0:3])
print("Pool stats:", h_pool.min(), h_pool.max(), h_pool.mean())

# Check max indices
h_max_idx = np.zeros((BATCH, OC, 15, 15), dtype=np.int32)
lib.gpu_memcpy_d2h(h_max_idx.ctypes.data, d_max_idx, BATCH * OC * 15 * 15 * 4)
print("\nMax indices (n=0, c=0) sample [0:3, 0:3]:")
print(h_max_idx[0, 0, 0:3, 0:3])

lib.gpu_free(d_x)
lib.gpu_free(d_w)
lib.gpu_free(d_col)
lib.gpu_free(d_conv_raw)
lib.gpu_free(d_conv)
lib.gpu_free(d_pool)
lib.gpu_free(d_max_idx)
