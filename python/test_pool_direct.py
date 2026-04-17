#!/usr/bin/env python3
"""Direct pool test with known data"""
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

# Create known pattern: input[0,0] = 1, others = 0
# After 2x2 maxpool with stride 2, output[0,0] should be 1
N, C, H, W = 1, 1, 4, 4
input_data = np.zeros((N * C * H * W), dtype=np.float32)
input_data[0] = 1.0  # First element = 1

print("Input (flattened):", input_data)
print("Input reshaped (1x1x4x4):")
print("  [[", input_data[0:4], "]")
print("   [", input_data[4:8], "]")
print("   [", input_data[8:12], "]")
print("   [", input_data[12:16], "]]")

d_in = lib.gpu_malloc(N * C * H * W * 4)
lib.gpu_memcpy_h2d(d_in, input_data.ctypes.data, N * C * H * W * 4)

d_out = lib.gpu_malloc(N * C * 2 * 2 * 4)
lib.apply_maxpool(d_in, d_out, N, C, H, W)

output = np.zeros((N * C * 2 * 2), dtype=np.float32)
lib.gpu_memcpy_d2h(output.ctypes.data, d_out, N * C * 2 * 2 * 4)
print("\nOutput (flattened):", output)
print("Output reshaped (1x1x2x2):")
print("  [[", output[0:2].tolist(), "]")
print("   [", output[2:4].tolist(), "]]")
print("Expected: [[1.0, 0.0], [0.0, 0.0]]")

lib.gpu_free(d_in)
lib.gpu_free(d_out)
