#!/usr/bin/env python3
"""Test maxpool with specific input"""
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

# Test with input [[1,2],[3,4]]
input_data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
print("Input shape:", input_data.shape)
print("Input values:", input_data.flatten())
print("Expected max: 4.0")

d_input = lib.gpu_malloc(16)  # 4 floats
lib.gpu_memcpy_h2d(d_input, input_data.ctypes.data, 16)

# Read back to verify
h_verify = np.zeros(4, dtype=np.float32)
lib.gpu_memcpy_d2h(h_verify.ctypes.data, d_input, 16)
print("Verified input on GPU:", h_verify)

# Pool
d_output = lib.gpu_malloc(4)
lib.apply_maxpool(d_input, d_output, 1, 1, 2, 2)

h_output = np.zeros(1, dtype=np.float32)
lib.gpu_memcpy_d2h(h_output.ctypes.data, d_output, 4)
print("Pool output:", h_output[0], "(expected 4.0)")

lib.gpu_free(d_input)
lib.gpu_free(d_output)
print("Done!")
