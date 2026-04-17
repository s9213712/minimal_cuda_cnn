#!/usr/bin/env python3
"""Test pool backward with known gradient"""
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
lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_nchw.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

N, C, H, W = 1, 1, 4, 4

# Input: [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
input_data = np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]], dtype=np.float32)
print("Input (NCHW):")
print(input_data[0, 0])

# After 2x2 maxpool with stride 2, output should be [[6, 8], [14, 16]]
d_input = lib.gpu_malloc(N * C * H * W * 4)
lib.gpu_memcpy_h2d(d_input, input_data.ctypes.data, N * C * H * W * 4)

d_output = lib.gpu_malloc(N * C * 2 * 2 * 4)
lib.apply_maxpool(d_input, d_output, N, C, H, W)

h_output = np.zeros((N, C, 2, 2), dtype=np.float32)
lib.gpu_memcpy_d2h(h_output.ctypes.data, d_output, N * C * 2 * 2 * 4)
print("\nPool output:")
print(h_output[0, 0])

# Now test backward
# grad_out = ones
grad_out_data = np.ones((N, C, 2, 2), dtype=np.float32)
d_grad_out = lib.gpu_malloc(N * C * 2 * 2 * 4)
lib.gpu_memcpy_h2d(d_grad_out, grad_out_data.ctypes.data, N * C * 2 * 2 * 4)

d_grad_input = lib.gpu_malloc(N * C * H * W * 4)
lib.maxpool_backward_nchw(d_grad_out, d_input, d_grad_input, N, C, 2, 2)

h_grad_input = np.zeros((N, C, H, W), dtype=np.float32)
lib.gpu_memcpy_d2h(h_grad_input.ctypes.data, d_grad_input, N * C * H * W * 4)
print("\nGrad input (should have 1s at max locations):")
print(h_grad_input[0, 0])

lib.gpu_free(d_input)
lib.gpu_free(d_output)
lib.gpu_free(d_grad_out)
lib.gpu_free(d_grad_input)
print("\nDone!")
