#!/usr/bin/env python3
"""Full gradient debug"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import pickle

so = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
lib = ctypes.CDLL(so)
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
lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.reorganize_backward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.maxpool_backward_nchw.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

with open("minimal_cuda_cnn/data/cifar-10-batches-py/data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"][:64].astype(np.float32) / 255.0
    y = np.array(batch[b"labels"][:64])
x = x.reshape(-1, 3, 32, 32)

BATCH, C, H, W = 64, 3, 32, 32
KH, KW, OC = 3, 3, 32
outH, outW = H - KH + 1, W - KW + 1
FC_IN = OC * 15 * 15

np.random.seed(42)
w_conv = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.3
fc_w = np.random.randn(10 * FC_IN).astype(np.float32) * 0.1
fc_b = np.zeros(10, dtype=np.float32)

# Forward
d_x = lib.gpu_malloc(BATCH * C * H * W * 4)
d_w_conv = lib.gpu_malloc(OC * C * KH * KW * 4)
d_fc_w = lib.gpu_malloc(10 * FC_IN * 4)
d_fc_b = lib.gpu_malloc(10 * 4)
lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
lib.gpu_memcpy_h2d(d_w_conv, w_conv.ctypes.data, OC * C * KH * KW * 4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10 * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 10 * 4)

col_size = C * KH * KW * BATCH * outH * outW
d_col = lib.gpu_malloc(col_size * 4)
d_conv_raw = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)
lib.gemm_forward(d_w_conv, d_col, d_conv_raw, OC, BATCH * outH * outW, C * KH * KW)
lib.leaky_relu_forward(d_conv_raw, c_float(0.02), OC * BATCH * outH * outW)

d_conv = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.reorganize_forward(d_conv_raw, d_conv, BATCH, OC, outH, outW)

d_pool = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
lib.apply_maxpool(d_conv, d_pool, BATCH, OC, outH, outW)

h_pool = np.zeros((BATCH, FC_IN), dtype=np.float32)
lib.gpu_memcpy_d2h(h_pool.ctypes.data, d_pool, BATCH * FC_IN * 4)

d_fc_in = lib.gpu_malloc(BATCH * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_in, h_pool.ctypes.data, BATCH * FC_IN * 4)

d_fc_out = lib.gpu_malloc(BATCH * 10 * 4)
lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)

h_out = np.zeros((BATCH, 10), dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH * 10 * 4)

h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
exp_out = np.exp(h_out_shifted)
probs = exp_out / exp_out.sum(axis=1, keepdims=True)
print("Prob stats:", "min", probs.min(), "max", probs.max(), "mean", probs.mean())
print("Sample probs:", probs[0])
print("True labels:", y[0:4])

# Backward
labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
labels_onehot[np.arange(BATCH), y] = 1.0
d_loss = probs - labels_onehot

print("\nFC grad stats:", "min", d_loss.min(), "max", d_loss.max())

# FC weight gradient
grad_fc_w = d_loss.T @ h_pool
print("FC weight grad stats:", "min", grad_fc_w.min(), "max", grad_fc_w.max())

# FC -> Pool gradient
fc_w_reshaped = fc_w.reshape(10, FC_IN)
grad_pool = d_loss @ fc_w_reshaped
grad_pool = np.clip(grad_pool, -1.0, 1.0)
print("Pool grad stats:", "min", grad_pool.min(), "max", grad_pool.max())

# Pool backward
d_pool_grad = lib.gpu_malloc(OC * BATCH * 15 * 15 * 4)
lib.gpu_memcpy_h2d(d_pool_grad, grad_pool.flatten().ctypes.data, OC * BATCH * 15 * 15 * 4)

d_conv_grad = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.gpu_memset(d_conv_grad, 0, OC * BATCH * outH * outW * 4)
lib.maxpool_backward_nchw(d_pool_grad, d_conv, d_conv_grad, BATCH, OC, outH, outW)

h_conv_grad = np.zeros((BATCH, OC, outH, outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv_grad.ctypes.data, d_conv_grad, BATCH * OC * outH * outW * 4)
print("Conv grad after pool:", "min", h_conv_grad.min(), "max", h_conv_grad.max())

# Leaky ReLU backward
lib.leaky_relu_backward(d_conv, d_conv_grad, c_float(0.02), OC * BATCH * outH * outW)
lib.gpu_memcpy_d2h(h_conv_grad.ctypes.data, d_conv_grad, BATCH * OC * outH * outW * 4)
print("Conv grad after leaky relu:", "min", h_conv_grad.min(), "max", h_conv_grad.max())

# Reorganize backward
d_conv_raw_grad = lib.gpu_malloc(OC * BATCH * outH * outW * 4)
lib.reorganize_backward(d_conv_grad, d_conv_raw_grad, BATCH, OC, outH, outW)

h_conv_raw_grad = np.zeros((OC, BATCH * outH * outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv_raw_grad.ctypes.data, d_conv_raw_grad, OC * BATCH * outH * outW * 4)
print("Conv raw grad:", "min", h_conv_raw_grad.min(), "max", h_conv_raw_grad.max())

# Conv backward
d_conv_grad_w = lib.gpu_malloc(OC * C * KH * KW * 4)
lib.conv_backward(d_conv_raw_grad, d_x, d_w_conv, d_conv_grad_w,
                 BATCH, C, H, W, KH, KW, outH, outW, OC)

h_conv_grad_w = np.zeros((OC * C * KH * KW,), dtype=np.float32)
lib.gpu_memcpy_d2h(h_conv_grad_w.ctypes.data, d_conv_grad_w, OC * C * KH * KW * 4)
print("Conv weight grad:", "min", h_conv_grad_w.min(), "max", h_conv_grad_w.max())

lib.gpu_free(d_x)
lib.gpu_free(d_w_conv)
lib.gpu_free(d_fc_w)
lib.gpu_free(d_fc_b)
lib.gpu_free(d_col)
lib.gpu_free(d_conv_raw)
lib.gpu_free(d_conv)
lib.gpu_free(d_pool)
lib.gpu_free(d_fc_in)
lib.gpu_free(d_fc_out)
lib.gpu_free(d_pool_grad)
lib.gpu_free(d_conv_grad)
lib.gpu_free(d_conv_raw_grad)
lib.gpu_free(d_conv_grad_w)
print("\nDone!")
