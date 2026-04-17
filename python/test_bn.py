#!/usr/bin/env python3
"""Quick BN forward+backward test"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os

so = "/mnt/c/Users/user/.openclaw/workspace/NN/minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
lib = ctypes.CDLL(so)
lib.gpu_malloc.argtypes = [ctypes.c_size_t]; lib.gpu_malloc.restype = c_void_p
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.batch_norm_forward.argtypes = [c_void_p]*8 + [c_float, c_float] + [c_int]*5
lib.batch_norm_backward.argtypes = [c_void_p]*8 + [c_float] + [c_int]*4

def g2h(ptr, size):
    h = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, size * 4)
    return h

N, C, H, W = 4, 8, 8, 8
x = np.random.randn(N*C*H*W).astype(np.float32)
gamma = np.ones(C, dtype=np.float32)
beta = np.zeros(C, dtype=np.float32)
running_mean = np.zeros(C, dtype=np.float32)
running_var = np.ones(C, dtype=np.float32)
dy = np.random.randn(N*C*H*W).astype(np.float32) * 0.1

d_x = lib.gpu_malloc(N*C*H*W*4)
d_out = lib.gpu_malloc(N*C*H*W*4)
d_rm = lib.gpu_malloc(C*4)
d_rv = lib.gpu_malloc(C*4)
d_sm = lib.gpu_malloc(C*4)
d_sv = lib.gpu_malloc(C*4)
d_gamma = lib.gpu_malloc(C*4)
d_beta = lib.gpu_malloc(C*4)
d_dy = lib.gpu_malloc(N*C*H*W*4)
d_dx = lib.gpu_malloc(N*C*H*W*4)
d_dgamma = lib.gpu_malloc(C*4)
d_dbeta = lib.gpu_malloc(C*4)

lib.gpu_memcpy_h2d(d_x, x.ctypes.data, N*C*H*W*4)
lib.gpu_memcpy_h2d(d_gamma, gamma.ctypes.data, C*4)
lib.gpu_memcpy_h2d(d_beta, beta.ctypes.data, C*4)
lib.gpu_memcpy_h2d(d_rm, running_mean.ctypes.data, C*4)
lib.gpu_memcpy_h2d(d_rv, running_var.ctypes.data, C*4)
lib.gpu_memcpy_h2d(d_dy, dy.ctypes.data, N*C*H*W*4)

lib.batch_norm_forward(d_out, d_x, d_rm, d_rv, d_sm, d_sv, d_gamma, d_beta, c_float(0.9), c_float(1e-5), N, C, H, W, 1)
h_out = g2h(d_out, N*C*H*W)
print(f"BN forward out: min={h_out.min():.4f}, max={h_out.max():.4f}, mean={h_out.mean():.4f}, std={h_out.std():.4f}")

lib.batch_norm_backward(d_dx, d_dy, d_x, d_sm, d_sv, d_gamma, d_dgamma, d_dbeta, c_float(1e-5), N, C, H, W)
h_dx = g2h(d_dx, N*C*H*W)
h_dgamma = g2h(d_dgamma, C)
h_dbeta = g2h(d_dbeta, C)
print(f"BN backward dx: min={h_dx.min():.4f}, max={h_dx.max():.4f}, non-zero={np.count_nonzero(h_dx)}/{h_dx.size}")
print(f"BN backward dgamma: {h_dgamma}")
print(f"BN backward dbeta: {h_dbeta}")

lib.gpu_free(d_x); lib.gpu_free(d_out); lib.gpu_free(d_rm); lib.gpu_free(d_rv)
lib.gpu_free(d_sm); lib.gpu_free(d_sv); lib.gpu_free(d_gamma); lib.gpu_free(d_beta)
lib.gpu_free(d_dy); lib.gpu_free(d_dx); lib.gpu_free(d_dgamma); lib.gpu_free(d_dbeta)
print("OK!")
