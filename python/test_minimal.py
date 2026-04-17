#!/usr/bin/env python3
"""Test minimal GPU operations"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os

so = os.path.join(os.path.dirname(__file__), "..", "cpp", "libminimal_cuda_cnn.so")
lib = ctypes.CDLL(so)
lib.gpu_malloc.restype = c_void_p
lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]

print("Testing small GEMM...")
# Simple 2x3 @ 3x2 = 2x2
A = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape(2, 3)
B = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape(3, 2)
C = np.zeros((2, 2), dtype=np.float32)

d_A = lib.gpu_malloc(6 * 4)
d_B = lib.gpu_malloc(6 * 4)
d_C = lib.gpu_malloc(4 * 4)
lib.gpu_memcpy_h2d(d_A, A.ctypes.data, 6 * 4)
lib.gpu_memcpy_h2d(d_B, B.ctypes.data, 6 * 4)
lib.gpu_memcpy_h2d(d_C, C.ctypes.data, 4 * 4)

print(f"A ptr: {d_A}, B ptr: {d_B}, C ptr: {d_C}")
print(f"A: {A.flatten()}")
print(f"B: {B.flatten()}")

lib.gemm_forward(d_A, d_B, d_C, 2, 2, 3)

h_C = np.zeros(4, dtype=np.float32)
lib.gpu_memcpy_d2h(h_C.ctypes.data, d_C, 4 * 4)
print(f"C result: {h_C}")
print(f"C reshaped: {h_C.reshape(2, 2)}")
print(f"Expected: {A @ B}")

lib.gpu_free(d_A)
lib.gpu_free(d_B)
lib.gpu_free(d_C)
print("Small GEMM OK")

print("\nTesting large conv weights...")
np.random.seed(42)
C1_IN = 3; C1_OUT = 32; KH1 = 3; KW1 = 3
w1 = np.random.randn(C1_OUT * C1_IN * KH1 * KW1).astype(np.float32) * 0.05
print(f"CPU w1: min={w1.min():.6f}, max={w1.max():.6f}")

dw1 = lib.gpu_malloc(C1_OUT * C1_IN * KH1 * KW1 * 4)
print(f"dw1 ptr: {dw1}")

lib.gpu_memcpy_h2d(dw1, w1.ctypes.data, C1_OUT * C1_IN * KH1 * KW1 * 4)

h_w1_check = np.zeros(C1_OUT * C1_IN * KH1 * KW1, dtype=np.float32)
lib.gpu_memcpy_d2h(h_w1_check.ctypes.data, dw1, C1_OUT * C1_IN * KH1 * KW1 * 4)
print(f"GPU w1 after copy: min={h_w1_check.min():.6f}, max={h_w1_check.max():.6f}")
print(f"Match: {np.allclose(w1, h_w1_check)}")

lib.gpu_free(dw1)
print("Large conv weights OK")
