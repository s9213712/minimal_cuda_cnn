#!/usr/bin/env python3
"""Test FC with clean GPU state"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os

so = os.path.join(os.path.dirname(__file__), "..", "cpp", "libminimal_cuda_cnn.so")
lib = ctypes.CDLL(so)
lib.gpu_malloc.restype = c_void_p
lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]

BATCH = 64
FC_IN = 2304
LR = 0.001

np.random.seed(42)
# Same initialization as train scripts
fc_w_orig = np.random.randn(10 * FC_IN).astype(np.float32) * np.sqrt(2.0 / FC_IN) * 0.1
fc_b = np.zeros(10, dtype=np.float32)
# h_pool2 from actual network should be small values
h_pool2 = np.random.randn(BATCH, FC_IN).astype(np.float32) * 0.05
y = np.array([6, 9, 9, 4, 1, 1, 2, 7] + [3] * 56)

print(f"fc_w_orig: min={fc_w_orig.min():.4f} max={fc_w_orig.max():.4f}")
print(f"h_pool2: min={h_pool2.min():.4f} max={h_pool2.max():.4f}")

dfw = lib.gpu_malloc(10 * FC_IN * 4)
dfb = lib.gpu_malloc(40)
dfi = lib.gpu_malloc(BATCH * FC_IN * 4)
dfo = lib.gpu_malloc(BATCH * 10 * 4)

lib.gpu_memcpy_h2d(dfw, fc_w_orig.ctypes.data, 10 * FC_IN * 4)
lib.gpu_memcpy_h2d(dfb, fc_b.ctypes.data, 40)
lib.gpu_memcpy_h2d(dfi, h_pool2.ctypes.data, BATCH * FC_IN * 4)

lib.dense_forward(dfi, dfw, dfb, dfo, BATCH, FC_IN, 10)
h_out = np.zeros((BATCH, 10), dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, dfo, BATCH * 10 * 4)
print(f"FC out: min={h_out.min():.4f} max={h_out.max():.4f}")

lib.gpu_free(dfw); lib.gpu_free(dfb); lib.gpu_free(dfi); lib.gpu_free(dfo)
print("DONE")
