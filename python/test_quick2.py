#!/usr/bin/env python3
"""Quick test of full flow with small batch"""
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
lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.leaky_relu_forward.argtypes = [c_void_p, c_float, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

BATCH, C, H, W = 4, 3, 32, 32
KH, KW, OC = 3, 3, 32
outH, outW = H - KH + 1, W - KW + 1

print("Allocating memory...")
d_x = lib.gpu_malloc(BATCH * C * H * W * 4)
d_w = lib.gpu_malloc(OC * C * KH * KW * 4)
col_size = C * KH * KW * BATCH * outH * outW
d_col = lib.gpu_malloc(col_size * 4)
d_conv_raw = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
d_conv = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
d_pool = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
d_max_idx = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)

print("Memory allocated")

x = np.random.randn(BATCH, C, H, W).astype(np.float32) * 0.05
w = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.05

lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
lib.gpu_memcpy_h2d(d_w, w.ctypes.data, OC * C * KH * KW * 4)

print("Running im2col...")
lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)
print("im2col done")

print("Running GEMM...")
lib.gemm_forward(d_w, d_col, d_conv_raw, OC, BATCH * outH * outW, C * KH * KW)
print("GEMM done")

print("Running LeakyReLU...")
lib.leaky_relu_forward(d_conv_raw, c_float(0.1), OC * BATCH * outH * outW)
print("LeakyReLU done")

print("Running reorganize...")
lib.reorganize_forward(d_conv_raw, d_conv, BATCH, OC, outH, outW)
print("reorganize done")

print("Running maxpool_forward_store...")
lib.maxpool_forward_store(d_pool, d_conv, d_max_idx, BATCH, OC, outH, outW)
print("maxpool_forward_store done")

print("Reading pool output...")
h_pool = np.zeros((BATCH, OC, 15, 15), dtype=np.float32)
lib.gpu_memcpy_d2h(h_pool.ctypes.data, d_pool, BATCH * OC * 15 * 15 * 4)
print("Pool output stats:", "min", h_pool.min(), "max", h_pool.max())

# Test backward
print("Testing backward...")
grad_pool = np.random.randn(BATCH, OC, 15, 15).astype(np.float32) * 0.01
d_pool_grad = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
lib.gpu_memcpy_h2d(d_pool_grad, grad_pool.ctypes.data, OC * BATCH * 15 * 15 * 4)

d_conv_grad = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.gpu_memset(d_conv_grad, 0, OC * BATCH * outH * outW * 4)
lib.maxpool_backward_use_idx(d_pool_grad, d_max_idx, d_conv_grad, BATCH, OC, outH, outW)

h_conv_grad = np.zeros((BATCH, OC, outH, outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv_grad.ctypes.data, d_conv_grad, BATCH * OC * outH * outW * 4)
print("Conv grad stats:", "min", h_conv_grad.min(), "max", h_conv_grad.max())

# Cleanup
lib.gpu_free(d_x)
lib.gpu_free(d_w)
lib.gpu_free(d_col)
lib.gpu_free(d_conv_raw)
lib.gpu_free(d_conv)
lib.gpu_free(d_pool)
lib.gpu_free(d_max_idx)
lib.gpu_free(d_pool_grad)
lib.gpu_free(d_conv_grad)
print("All freed! Test passed!")
