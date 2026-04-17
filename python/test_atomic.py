#!/usr/bin/env python3
"""Test atomicAdd on float"""
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

# Simple test: 2x2 input, 1x1 output (max of all)
N, C, H, W = 1, 1, 2, 2

# Input: [[1, 2], [3, 4]]
input_data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
print("Input:", input_data[0, 0])

d_input = lib.gpu_malloc(4)
lib.gpu_memcpy_h2d(d_input, input_data.ctypes.data, 4)

# Pool forward: should output 4
d_output = lib.gpu_malloc(4)
lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.apply_maxpool(d_input, d_output, N, C, H, W)

h_output = np.zeros((1, 1, 1, 1), dtype=np.float32)
lib.gpu_memcpy_d2h(h_output.ctypes.data, d_output, 4)
print("Pool output:", h_output[0, 0, 0, 0])

# Pool backward: grad=1
grad_out_data = np.ones((1, 1, 1, 1), dtype=np.float32)
d_grad_out = lib.gpu_malloc(4)
lib.gpu_memcpy_h2d(d_grad_out, grad_out_data.ctypes.data, 4)

# Init grad_input to 0
d_grad_input = lib.gpu_malloc(4)
lib.gpu_memset(d_grad_input, 0, 4)

# Backward
lib.maxpool_backward_nchw.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_nchw(d_grad_out, d_input, d_grad_input, N, C, 1, 1)

h_grad_input = np.zeros((1, 1, 2, 2), dtype=np.float32)
lib.gpu_memcpy_d2h(h_grad_input.ctypes.data, d_grad_input, 4)
print("Grad input after backward:", h_grad_input[0, 0])
print("Expected: gradient at max location [1,1]=4, so [[0,0],[0,1]]")

lib.gpu_free(d_input)
lib.gpu_free(d_output)
lib.gpu_free(d_grad_out)
lib.gpu_free(d_grad_input)
