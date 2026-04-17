#!/usr/bin/env python3
"""Test FC with tiny weights"""
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
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]

# Simple test: input all 0.1, weights all 0.01
BATCH, IN_F, OUT_F = 4, 128, 10

input_data = np.ones((BATCH, IN_F), dtype=np.float32) * 0.1
weights = np.ones((OUT_F, IN_F), dtype=np.float32) * 0.01
bias = np.zeros(OUT_F, dtype=np.float32)

print("Input:", input_data.shape, "min", input_data.min(), "max", input_data.max())
print("Weights:", weights.shape, "min", weights.min(), "max", weights.max())

d_in = lib.gpu_malloc(BATCH * IN_F * 4)
d_w = lib.gpu_malloc(OUT_F * IN_F * 4)
d_b = lib.gpu_malloc(OUT_F * 4)
d_out = lib.gpu_malloc(BATCH * OUT_F * 4)

lib.gpu_memcpy_h2d(d_in, input_data.ctypes.data, BATCH * IN_F * 4)
lib.gpu_memcpy_h2d(d_w, weights.ctypes.data, OUT_F * IN_F * 4)
lib.gpu_memcpy_h2d(d_b, bias.ctypes.data, OUT_F * 4)

lib.dense_forward(d_in, d_w, d_b, d_out, BATCH, IN_F, OUT_F)

output = np.zeros((BATCH, OUT_F), dtype=np.float32)
lib.gpu_memcpy_d2h(output.ctypes.data, d_out, BATCH * OUT_F * 4)

print("Output:", output.shape, "min", output.min(), "max", output.max())
print("Expected: 0.01 * 128 * 0.1 = 0.128 per element")
print("Output[0]:", output[0])

lib.gpu_free(d_in)
lib.gpu_free(d_w)
lib.gpu_free(d_b)
lib.gpu_free(d_out)
