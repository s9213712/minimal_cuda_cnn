#!/usr/bin/env python3
"""Test new maxpool with stored indices"""
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
lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

N, C, H, W = 1, 1, 4, 4

# Input: [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
input_data = np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]], dtype=np.float32)
print("Input:")
print(input_data[0, 0])

d_input = lib.gpu_malloc(N * C * H * W * 4)
lib.gpu_memcpy_h2d(d_input, input_data.ctypes.data, N * C * H * W * 4)

# Forward + store max indices
d_output = lib.gpu_malloc(N * C * (H//2) * (W//2) * 4)
d_max_idx = lib.gpu_malloc(N * C * (H//2) * (W//2) * 4)  # int32
lib.maxpool_forward_store(d_output, d_input, d_max_idx, N, C, H, W)

h_output = np.zeros((N, C, H//2, W//2), dtype=np.float32)
lib.gpu_memcpy_d2h(h_output.ctypes.data, d_output, N * C * (H//2) * (W//2) * 4)
print("\nPool output:")
print(h_output[0, 0])
print("Expected: [[6, 8], [14, 16]]")

h_max_idx = np.zeros((N, C, H//2, W//2), dtype=np.int32)
lib.gpu_memcpy_d2h(h_max_idx.ctypes.data, d_max_idx, N * C * (H//2) * (W//2) * 4)
print("\nMax indices:")
print(h_max_idx[0, 0])
print("Expected: [[5, 7], [13, 15]]")

# Backward
grad_out_data = np.ones((N, C, H//2, W//2), dtype=np.float32)
d_grad_out = lib.gpu_malloc(N * C * (H//2) * (W//2) * 4)
lib.gpu_memcpy_h2d(d_grad_out, grad_out_data.ctypes.data, N * C * (H//2) * (W//2) * 4)

d_grad_input = lib.gpu_malloc(N * C * H * W * 4)
lib.gpu_memset(d_grad_input, 0, N * C * H * W * 4)
lib.maxpool_backward_use_idx(d_grad_out, d_max_idx, d_grad_input, N, C, H, W)

h_grad_input = np.zeros((N, C, H, W), dtype=np.float32)
lib.gpu_memcpy_d2h(h_grad_input.ctypes.data, d_grad_input, N * C * H * W * 4)
print("\nGrad input after backward:")
print(h_grad_input[0, 0])
print("Expected: gradient only at max positions [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]")

lib.gpu_free(d_input)
lib.gpu_free(d_output)
lib.gpu_free(d_max_idx)
lib.gpu_free(d_grad_out)
lib.gpu_free(d_grad_input)
print("\nDone!")
