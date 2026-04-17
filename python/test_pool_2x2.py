#!/usr/bin/env python3
"""Test maxpool with 2x2 input"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_int

so = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
lib = ctypes.CDLL(so)
lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_malloc.restype = c_void_p
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]

N, C, H, W = 1, 1, 2, 2

# Input: [[1, 2], [3, 4]]
input_data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
print("Input (NCHW):")
print(input_data[0, 0])

d_input = lib.gpu_malloc(4)
lib.gpu_memcpy_h2d(d_input, input_data.ctypes.data, 4)

d_output = lib.gpu_malloc(4)
lib.apply_maxpool(d_input, d_output, N, C, H, W)

h_output = np.zeros((1, 1, 1, 1), dtype=np.float32)
lib.gpu_memcpy_d2h(h_output.ctypes.data, d_output, 4)
print("Pool output (expected 4.0):", h_output[0, 0, 0, 0])

lib.gpu_free(d_input)
lib.gpu_free(d_output)
print("Done!")
