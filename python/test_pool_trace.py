#!/usr/bin/env python3
"""Trace pool with BATCH=8, OC=32"""
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
lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]

N, C, H, W = 8, 32, 30, 30

# Create pattern: only element [n=0, c=0, h=0, w=0] = 1, rest = 0
input_data = np.zeros((N * C * H * W), dtype=np.float32)
input_data[0] = 1.0  # First element

print("Input shape:", (N, C, H, W))
print("Input[0,0,0,0] = 1, all else 0")

d_in = lib.gpu_malloc(N * C * H * W * 4)
lib.gpu_memcpy_h2d(d_in, input_data.ctypes.data, N * C * H * W * 4)

d_out = lib.gpu_malloc(N * C * 15 * 15 * 4)
lib.apply_maxpool(d_in, d_out, N, C, H, W)

output = np.zeros((N * C * 15 * 15), dtype=np.float32)
lib.gpu_memcpy_d2h(output.ctypes.data, d_out, N * C * 15 * 15 * 4)

print("Output shape:", output.shape)
print("Non-zero output elements:", np.nonzero(output)[0].tolist())
print("Output[0] (first 20):", output[:20])
print("Output[0,0]:", output[0])
print("Max output:", output.max())

lib.gpu_free(d_in)
lib.gpu_free(d_out)
