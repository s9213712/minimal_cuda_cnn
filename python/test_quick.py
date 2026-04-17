#!/usr/bin/env python3
"""Quick test of maxpool_forward_store"""
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
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

N, C, H, W = 1, 1, 4, 4
input_data = np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]], dtype=np.float32)

d_input = lib.gpu_malloc(N * C * H * W * 4)
lib.gpu_memcpy_h2d(d_input, input_data.ctypes.data, N * C * H * W * 4)

d_output = lib.gpu_malloc(N * C * 2 * 2 * 4)
d_max_idx = lib.gpu_malloc(N * C * 2 * 2 * 4)

print("Testing maxpool_forward_store...")
lib.maxpool_forward_store(d_output, d_input, d_max_idx, N, C, H, W)
print("Success!")

h_output = np.zeros((N, C, 2, 2), dtype=np.float32)
lib.gpu_memcpy_d2h(h_output.ctypes.data, d_output, N * C * 2 * 2 * 4)
print("Pool output:", h_output[0, 0])

lib.gpu_free(d_input)
lib.gpu_free(d_output)
lib.gpu_free(d_max_idx)
print("Done!")
