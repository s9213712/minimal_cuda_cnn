#!/usr/bin/env python3
"""Debug maxpool step by step"""
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

# Test with 4x4 input
N, C, H, W = 1, 1, 4, 4
input_data = np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]], dtype=np.float32)
print("Input 4x4:")
print(input_data[0, 0])

d_input = lib.gpu_malloc(N * C * H * W * 4)
lib.gpu_memcpy_h2d(d_input, input_data.ctypes.data, N * C * H * W * 4)

# Pool forward: 4x4 -> 2x2
d_output = lib.gpu_malloc(N * C * 2 * 2 * 4)
lib.apply_maxpool(d_input, d_output, N, C, H, W)

h_output = np.zeros((N, C, 2, 2), dtype=np.float32)
lib.gpu_memcpy_d2h(h_output.ctypes.data, d_output, N * C * 2 * 2 * 4)
print("\nPool output 2x2:")
print(h_output[0, 0])
print("Expected: [[6, 8], [14, 16]]")

lib.gpu_free(d_input)
lib.gpu_free(d_output)
print("\nDone!")
