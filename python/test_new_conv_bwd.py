#!/usr/bin/env python3
"""Quick verify: new conv_backward outputs non-zero input grad"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os

workspace = "/mnt/c/Users/user/.openclaw/workspace"
so = os.path.join(workspace, "NN/minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so")
lib = ctypes.CDLL(so)

lib.gpu_malloc.argtypes = [ctypes.c_size_t]; lib.gpu_malloc.restype = c_void_p
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]

def g2h(ptr, size):
    h = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, size * 4)
    return h

BATCH = 4; C = 32; H = 15; W = 15; KH = 3; KW = 3
outH = H - KH + 1; outW = W - KW + 1

np.random.seed(42)
inp = np.random.randn(BATCH*C*H*W).astype(np.float32) * 0.1
w = np.random.randn(C*KH*KW).astype(np.float32) * 0.05
d_conv2g = np.random.randn(C*BATCH*outH*outW).astype(np.float32) * 0.01

d_inp = lib.gpu_malloc(BATCH*C*H*W*4)
d_w = lib.gpu_malloc(C*KH*KW*4)
d_gout = lib.gpu_malloc(C*BATCH*outH*outW*4)
d_wg = lib.gpu_malloc(C*KH*KW*4)
d_gin = lib.gpu_malloc(BATCH*C*H*W*4)

lib.gpu_memcpy_h2d(d_inp, inp.ctypes.data, BATCH*C*H*W*4)
lib.gpu_memcpy_h2d(d_w, w.ctypes.data, C*KH*KW*4)
lib.gpu_memcpy_h2d(d_gout, d_conv2g.ctypes.data, C*BATCH*outH*outW*4)
lib.gpu_memset(d_wg, 0, C*KH*KW*4)
lib.gpu_memset(d_gin, 0, BATCH*C*H*W*4)

lib.conv_backward(d_gout, d_inp, d_w, d_wg, d_gin, BATCH, C, H, W, KH, KW, outH, outW, C)

h_wg = g2h(d_wg, C*KH*KW)
h_gin = g2h(d_gin, BATCH*C*H*W)

print(f"weight_grad: min={h_wg.min():.6f}, max={h_wg.max():.6f}, mean={h_wg.mean():.6f}")
print(f"input_grad:  min={h_gin.min():.6f}, max={h_gin.max():.6f}, mean={h_gin.mean():.6f}")
print(f"input_grad non-zero: {np.count_nonzero(h_gin)}/{h_gin.size}")

lib.gpu_free(d_inp); lib.gpu_free(d_w); lib.gpu_free(d_gout); lib.gpu_free(d_wg); lib.gpu_free(d_gin)
print("OK!")
