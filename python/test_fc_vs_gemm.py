#!/usr/bin/env python3
"""FC forward using test_minimal.py style - just GEMM, no complex stuff"""
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
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]

BATCH = 64
FC_IN = 2304
OUT_F = 10

# Simple test: FC is just GEMM after all
np.random.seed(42)
fc_w = np.random.randn(OUT_F * FC_IN).astype(np.float32) * 0.05
fc_b = np.zeros(OUT_F, dtype=np.float32)
h_pool2 = np.random.randn(BATCH * FC_IN).astype(np.float32) * 0.05

print(f"fc_w: min={fc_w.min():.4f} max={fc_w.max():.4f}")
print(f"h_pool2: min={h_pool2.min():.4f} max={h_pool2.max():.4f}")

dfw = lib.gpu_malloc(OUT_F * FC_IN * 4)
dfb = lib.gpu_malloc(OUT_F * 4)
dfi = lib.gpu_malloc(BATCH * FC_IN * 4)
dfo = lib.gpu_malloc(BATCH * OUT_F * 4)

lib.gpu_memcpy_h2d(dfw, fc_w.ctypes.data, OUT_F * FC_IN * 4)
lib.gpu_memcpy_h2d(dfb, fc_b.ctypes.data, OUT_F * 4)
lib.gpu_memcpy_h2d(dfi, h_pool2.ctypes.data, BATCH * FC_IN * 4)

# Test using GEMM directly (this works in test_minimal.py)
# GEMM: d_A @ d_B = d_C with dimensions (rows_a, cols_b, cols_a)
d_A = dfi  # (64, 2304)
d_B = dfw  # (10, 2304) -> treated as (2304, 10) by column-major GEMM
d_C = dfo  # (64, 10)

print(f"Calling gemm_forward with d_A={d_A}, d_B={d_B}, d_C={d_C}")
print(f"GEMM: rows_a={BATCH}, cols_b={OUT_F}, cols_a={FC_IN}")

lib.gemm_forward(d_A, d_B, d_C, BATCH, OUT_F, FC_IN)

h_out = np.zeros(BATCH * OUT_F, dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, dfo, BATCH * OUT_F * 4)
print(f"GEMM result: min={h_out.min():.4f} max={h_out.max():.4f}")
print(f"GEMM result reshaped (64,10):\\n{h_out.reshape(BATCH, OUT_F)[:5]}")

# NOW test dense_forward
h_out2 = np.zeros(BATCH * OUT_F, dtype=np.float32)
lib.gpu_memcpy_d2h(h_out2.ctypes.data, dfo, BATCH * OUT_F * 4)
print(f"Before dense_forward: min={h_out2.min():.4f}")

lib.dense_forward(dfi, dfw, dfb, dfo, BATCH, FC_IN, OUT_F)

h_out2 = np.zeros(BATCH * OUT_F, dtype=np.float32)
lib.gpu_memcpy_d2h(h_out2.ctypes.data, dfo, BATCH * OUT_F * 4)
print(f"dense_forward result: min={h_out2.min():.4f} max={h_out2.max():.4f}")

lib.gpu_free(dfw); lib.gpu_free(dfb); lib.gpu_free(dfi); lib.gpu_free(dfo)
print("DONE")
