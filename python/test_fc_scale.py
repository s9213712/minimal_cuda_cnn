#!/usr/bin/env python3
"""Test FC output scale"""
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
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]

BATCH = 64
FC_IN = 2304

h_fc_in = np.random.randn(BATCH, FC_IN).astype(np.float32) * 0.1
fc_w = np.random.randn(10, FC_IN).astype(np.float32) * 0.05
fc_b = np.zeros(10, dtype=np.float32)

d_fc_in = lib.gpu_malloc(BATCH * FC_IN * 4)
d_fc_w = lib.gpu_malloc(10 * FC_IN * 4)
d_fc_b = lib.gpu_malloc(40)
d_fc_out = lib.gpu_malloc(BATCH * 10 * 4)
lib.gpu_memcpy_h2d(d_fc_in, h_fc_in.ctypes.data, BATCH * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10 * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 40)
lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)

h_out = np.zeros((BATCH, 10), dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH * 10 * 4)
print(f"FC out: min={h_out.min():.4f}, max={h_out.max():.4f}, mean={h_out.mean():.4f}")

h_sh = h_out - h_out.max(axis=1, keepdims=True)
exp_o = np.exp(h_sh)
print(f"Exp: min={exp_o.min():.4f}, max={exp_o.max():.4f}")
probs = exp_o / exp_o.sum(axis=1, keepdims=True)
print(f"Probs: min={probs.min():.4f}, max={probs.max():.4f}")

lib.gpu_free(d_fc_in)
lib.gpu_free(d_fc_w)
lib.gpu_free(d_fc_b)
lib.gpu_free(d_fc_out)
