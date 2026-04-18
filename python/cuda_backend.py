"""ctypes bindings and GPU helper functions for libminimal_cuda_cnn.so."""

import ctypes
import os
from ctypes import c_float, c_int, c_void_p

import numpy as np

from train_config import KH, KW, LEAKY_ALPHA


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SO_PATH = os.path.join(ROOT, "cpp", "libminimal_cuda_cnn.so")
lib = ctypes.CDLL(SO_PATH)

lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_malloc.restype = c_void_p
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.leaky_relu_forward.argtypes = [c_void_p, c_float, c_int]
lib.leaky_relu_backward.argtypes = [c_void_p, c_void_p, c_float, c_int]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]
lib.nchw_to_cnhw.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.cnhw_to_nchw.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.conv_backward.argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
]


def g2h(ptr, size):
    h = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, size * 4)
    return h


def gpu_zeros(size):
    ptr = lib.gpu_malloc(size * 4)
    lib.gpu_memset(ptr, 0, size * 4)
    return ptr


def upload(arr):
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    ptr = lib.gpu_malloc(arr.size * 4)
    lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.size * 4)
    return ptr


def cnhw_to_nchw_alloc(d_cnhw, n, c, h, w):
    d_nchw = lib.gpu_malloc(n * c * h * w * 4)
    lib.cnhw_to_nchw(d_cnhw, d_nchw, n, c, h, w)
    return d_nchw


def nchw_to_cnhw_alloc(d_nchw, n, c, h, w):
    d_cnhw = lib.gpu_malloc(n * c * h * w * 4)
    lib.nchw_to_cnhw(d_nchw, d_cnhw, n, c, h, w)
    return d_cnhw


def conv_forward(d_input_nchw, d_weight, n, in_c, in_h, in_w, out_c):
    out_h, out_w = in_h - KH + 1, in_w - KW + 1
    col_size = in_c * KH * KW * n * out_h * out_w
    raw_size = out_c * n * out_h * out_w
    d_col = lib.gpu_malloc(col_size * 4)
    d_raw = lib.gpu_malloc(raw_size * 4)
    lib.im2col_forward(d_input_nchw, d_col, n, in_c, in_h, in_w, KH, KW, out_h, out_w)
    lib.gemm_forward(d_weight, d_col, d_raw, out_c, n * out_h * out_w, in_c * KH * KW)
    lib.leaky_relu_forward(d_raw, c_float(LEAKY_ALPHA), raw_size)
    return d_col, d_raw, out_h, out_w


def maxpool_forward(d_input_cnhw, n, c, h, w):
    out_h, out_w = h // 2, w // 2
    out_size = c * n * out_h * out_w
    d_pool = lib.gpu_malloc(out_size * 4)
    d_idx = lib.gpu_malloc(out_size * 4)
    lib.maxpool_forward_store(d_pool, d_input_cnhw, d_idx, n, c, h, w)
    return d_pool, d_idx, out_h, out_w


def update_conv(d_weight, d_grad, lr, size, name, weight_decay, clip_value, log_grad=False):
    h_grad = g2h(d_grad, size).reshape(-1)
    h_weight = g2h(d_weight, size).reshape(-1)
    h_grad = h_grad + weight_decay * h_weight
    if log_grad:
        print(f"    {name} grad_abs_mean={np.mean(np.abs(h_grad)):.6e} grad_abs_max={np.max(np.abs(h_grad)):.6e}")
    h_grad_clip = np.clip(h_grad, -clip_value, clip_value).astype(np.float32)
    lib.gpu_memcpy_h2d(d_grad, h_grad_clip.ctypes.data, size * 4)
    lib.apply_sgd_update(d_weight, d_grad, c_float(lr), size)
