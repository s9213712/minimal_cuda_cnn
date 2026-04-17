#!/usr/bin/env python3
"""Test Conv2 SGD update in isolation"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os
import pickle

so = os.path.join(os.path.dirname(__file__), "..", "cpp", "libminimal_cuda_cnn.so")
lib = ctypes.CDLL(so)
lib.gpu_malloc.restype = c_void_p
lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.leaky_relu_forward.argtypes = [c_void_p, c_float, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]

BATCH = 64
LR = 0.001
C2_IN = 32; C2_OUT = 64; KH2 = 3; KW2 = 3
outH2 = 13; outW2 = 13; poolH2 = 6; poolW2 = 6
poolH1 = 15; poolW1 = 15

np.random.seed(42)
w2_orig = np.random.randn(C2_OUT * C2_IN * KH2 * KW2).astype(np.float32) * 0.05
h_pool1 = np.random.randn(BATCH, C2_IN, poolH1, poolW1).astype(np.float32) * 0.05
h_pool2_grad = np.random.randn(BATCH, C2_OUT, poolH2, poolW2).astype(np.float32) * 0.1
y = np.array([6, 9, 9, 4, 1, 1, 2, 7] + [3] * 56)

dw2 = lib.gpu_malloc(C2_OUT * C2_IN * KH2 * KW2 * 4)
dp1 = lib.gpu_malloc(BATCH * C2_IN * poolH1 * poolW1 * 4)
dp2g = lib.gpu_malloc(BATCH * C2_OUT * poolH2 * poolW2 * 4)
dc2g = lib.gpu_malloc(BATCH * C2_OUT * outH2 * outW2 * 4)
dw2g = lib.gpu_malloc(C2_OUT * C2_IN * KH2 * KW2 * 4)

lib.gpu_memcpy_h2d(dw2, w2_orig.ctypes.data, C2_OUT * C2_IN * KH2 * KW2 * 4)
lib.gpu_memcpy_h2d(dp1, h_pool1.ctypes.data, BATCH * C2_IN * poolH1 * poolW1 * 4)
lib.gpu_memcpy_h2d(dp2g, h_pool2_grad.ctypes.data, BATCH * C2_OUT * poolH2 * poolW2 * 4)

# Weight before
hw2b = np.zeros(C2_OUT * C2_IN * KH2 * KW2, dtype=np.float32)
lib.gpu_memcpy_d2h(hw2b.ctypes.data, dw2, C2_OUT * C2_IN * KH2 * KW2 * 4)
print(f"Weight before: min={hw2b.min():.6f}, max={hw2b.max():.6f}")

# Pool2 backward
lib.maxpool_backward_use_idx(dp2g, dp1, dc2g, BATCH, C2_OUT, outH2, outW2)
hdc2g = np.zeros(BATCH * C2_OUT * outH2 * outW2, dtype=np.float32)
lib.gpu_memcpy_d2h(hdc2g.ctypes.data, dc2g, BATCH * C2_OUT * outH2 * outW2 * 4)
print(f"dc2g: min={hdc2g.min():.6f}, max={hdc2g.max():.6f}")

# Conv backward
lib.gpu_memset(dw2g, 0, C2_OUT * C2_IN * KH2 * KW2 * 4)
lib.conv_backward(dc2g, dp1, dw2, dw2g, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2, C2_OUT)
hdw2g = np.zeros(C2_OUT * C2_IN * KH2 * KW2, dtype=np.float32)
lib.gpu_memcpy_d2h(hdw2g.ctypes.data, dw2g, C2_OUT * C2_IN * KH2 * KW2 * 4)
print(f"dw2g: min={hdw2g.min():.6f}, max={hdw2g.max():.6f}, nan={np.isnan(hdw2g).any()}")

# SGD update
print(f"Calling SGD with dw2={dw2}, dw2g={dw2g}, size={C2_OUT * C2_IN * KH2 * KW2}")
lib.apply_sgd_update(dw2, dw2g, c_float(LR), C2_OUT * C2_IN * KH2 * KW2)
print("SGD done")

# Weight after
hw2a = np.zeros(C2_OUT * C2_IN * KH2 * KW2, dtype=np.float32)
lib.gpu_memcpy_d2h(hw2a.ctypes.data, dw2, C2_OUT * C2_IN * KH2 * KW2 * 4)
print(f"Weight after: min={hw2a.min():.6f}, max={hw2a.max():.6f}")
print(f"NaN: {np.isnan(hw2a).any()}")
print(f"Inf: {np.isinf(hw2a).any()}")

lib.gpu_free(dw2)
lib.gpu_free(dp1)
lib.gpu_free(dp2g)
lib.gpu_free(dc2g)
lib.gpu_free(dw2g)
print("Done")
