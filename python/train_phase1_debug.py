#!/usr/bin/env python3
"""Phase 1: mini-AlexNet with 2 Conv layers - DEBUG VERSION"""
import ctypes
import numpy as np
import time
from ctypes import c_void_p, c_float, c_int
import pickle
import os
import sys

so = os.path.join(os.path.dirname(__file__), "..", "cpp", "libminimal_cuda_cnn.so")
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
lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]
lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.reorganize_backward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]

BATCH = 64
LR = 0.01
LEAKY_ALPHA = 0.1
EPOCHS = 10

C1_IN, C1_OUT = 3, 32
H, W = 32, 32
KH1, KW1 = 3, 3
outH1, outW1 = H - KH1 + 1, W - KW1 + 1
poolH1, poolW1 = outH1 // 2, outW1 // 2

C2_IN, C2_OUT = 32, 64
KH2, KW2 = 3, 3
outH2, outW2 = poolH1 - KH2 + 1, poolW1 - KW2 + 1
poolH2, poolW2 = outH2 // 2, outW2 // 2

FC_IN = C2_OUT * poolH2 * poolW2

print(f"=== Phase 1 Debug ===")
print(f"Conv1: {C1_IN}→{C1_OUT}, out={outH1}×{outW1}, pool={poolH1}×{poolW1}")
print(f"Conv2: {C2_IN}→{C2_OUT}, out={outH2}×{outW2}, pool={poolH2}×{poolW2}")
print(f"FC_IN: {FC_IN}")

np.random.seed(42)
std1 = np.sqrt(2.0 / (C1_IN * KH1 * KW1))
std2 = np.sqrt(2.0 / (C2_IN * KH2 * KW2))
w_conv1 = np.random.randn(C1_OUT * C1_IN * KH1 * KW1).astype(np.float32) * std1 * 0.1
w_conv2 = np.random.randn(C2_OUT * C2_IN * KH2 * KW2).astype(np.float32) * std2 * 0.1
fc_w = np.random.randn(10 * FC_IN).astype(np.float32) * np.sqrt(2.0 / FC_IN) * 0.1
fc_b = np.zeros(10, dtype=np.float32)

print("Allocating GPU weights...")
d_w_conv1 = lib.gpu_malloc(C1_OUT * C1_IN * KH1 * KW1 * 4)
d_w_conv2 = lib.gpu_malloc(C2_OUT * C2_IN * KH2 * KW2 * 4)
d_fc_w = lib.gpu_malloc(10 * FC_IN * 4)
d_fc_b = lib.gpu_malloc(10 * 4)
lib.gpu_memcpy_h2d(d_w_conv1, w_conv1.ctypes.data, C1_OUT * C1_IN * KH1 * KW1 * 4)
lib.gpu_memcpy_h2d(d_w_conv2, w_conv2.ctypes.data, C2_OUT * C2_IN * KH2 * KW2 * 4)
lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10 * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 10 * 4)
print("GPU weights OK")

train_x, train_y = [], []
data_root = os.path.join(os.path.dirname(__file__), "..", "data", "cifar-10-batches-py")
for i in range(1, 2):  # Just batch 1 for debugging
    with open(os.path.join(data_root, f"data_batch_{i}"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        imgs = batch[b"data"].astype(np.float32) / 255.0
        imgs = imgs.reshape(-1, 3, 32, 32)
        train_x.append(imgs)
        train_y.extend(batch[b"labels"])
x_all = np.concatenate(train_x)[:BATCH]
y_all = np.array(train_y)[:BATCH]
print(f"Data: {x_all.shape}")

NBATCHES = 1
x = x_all
y = y_all

print(f"\n=== Batch 0 ===")
print(f"x shape: {x.shape}, y: {y}")

d_x = lib.gpu_malloc(BATCH * C1_IN * H * W * 4)
lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C1_IN * H * W * 4)
print("d_x allocated")

col1_size = C1_IN * KH1 * KW1 * BATCH * outH1 * outW1
d_col1 = lib.gpu_malloc(col1_size * 4)
d_conv1_raw = lib.gpu_malloc(C1_OUT * BATCH * outH1 * outW1 * 4)
lib.im2col_forward(d_x, d_col1, BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1)
lib.gemm_forward(d_w_conv1, d_col1, d_conv1_raw, C1_OUT, BATCH * outH1 * outW1, C1_IN * KH1 * KW1)
lib.leaky_relu_forward(d_conv1_raw, c_float(LEAKY_ALPHA), C1_OUT * BATCH * outH1 * outW1)
print("Conv1 + ReLU forward OK")

d_conv1 = lib.gpu_malloc(C1_OUT * BATCH * outH1 * outW1 * 4)
lib.reorganize_forward(d_conv1_raw, d_conv1, BATCH, C1_OUT, outH1, outW1)

d_pool1 = lib.gpu_malloc(C1_OUT * BATCH * poolH1 * poolW1 * 4)
d_max_idx1 = lib.gpu_malloc(C1_OUT * BATCH * poolH1 * poolW1 * 4)
lib.maxpool_forward_store(d_pool1, d_conv1, d_max_idx1, BATCH, C1_OUT, outH1, outW1)
print("Pool1 forward OK")

col2_size = C2_IN * KH2 * KW2 * BATCH * outH2 * outW2
d_col2 = lib.gpu_malloc(col2_size * 4)
d_conv2_raw = lib.gpu_malloc(C2_OUT * BATCH * outH2 * outW2 * 4)
lib.im2col_forward(d_pool1, d_col2, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)
lib.gemm_forward(d_w_conv2, d_col2, d_conv2_raw, C2_OUT, BATCH * outH2 * outW2, C2_IN * KH2 * KW2)
lib.leaky_relu_forward(d_conv2_raw, c_float(LEAKY_ALPHA), C2_OUT * BATCH * outH2 * outW2)
print("Conv2 + ReLU forward OK")

d_conv2 = lib.gpu_malloc(C2_OUT * BATCH * outH2 * outW2 * 4)
lib.reorganize_forward(d_conv2_raw, d_conv2, BATCH, C2_OUT, outH2, outW2)

d_pool2 = lib.gpu_malloc(C2_OUT * BATCH * poolH2 * poolW2 * 4)
d_max_idx2 = lib.gpu_malloc(C2_OUT * BATCH * poolH2 * poolW2 * 4)
lib.maxpool_forward_store(d_pool2, d_conv2, d_max_idx2, BATCH, C2_OUT, outH2, outW2)
print("Pool2 forward OK")

h_pool2 = np.zeros((BATCH, FC_IN), dtype=np.float32)
lib.gpu_memcpy_d2h(h_pool2.ctypes.data, d_pool2, BATCH * FC_IN * 4)
d_fc_in = lib.gpu_malloc(BATCH * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_in, h_pool2.ctypes.data, BATCH * FC_IN * 4)

d_fc_out = lib.gpu_malloc(BATCH * 10 * 4)
lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
print("FC forward OK")

h_out = np.zeros((BATCH, 10), dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH * 10 * 4)

h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
exp_out = np.exp(h_out_shifted)
probs = exp_out / exp_out.sum(axis=1, keepdims=True)
loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
print(f"Loss: {loss:.4f}")

pred = np.argmax(probs, axis=1)
correct = np.sum(pred == y)
print(f"Pred: {pred}, True: {y}, Acc: {correct}/{BATCH}")

print("\n=== BACKWARD ===")
labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
labels_onehot[np.arange(BATCH), y] = 1.0
d_loss = probs - labels_onehot
print("d_loss computed OK")

grad_fc_w = d_loss.T @ h_pool2
grad_fc_w_clipped = np.clip(grad_fc_w, -1.0, 1.0).flatten().astype(np.float32)
d_fc_grad_w = lib.gpu_malloc(10 * FC_IN * 4)
lib.gpu_memcpy_h2d(d_fc_grad_w, grad_fc_w_clipped.ctypes.data, 10 * FC_IN * 4)
lib.apply_sgd_update(d_fc_w, d_fc_grad_w, c_float(LR), 10 * FC_IN)
print("FC backward OK")

fc_w_reshaped = fc_w.reshape(10, FC_IN)
grad_pool2 = d_loss @ fc_w_reshaped
grad_pool2_clipped = np.clip(grad_pool2, -1.0, 1.0).astype(np.float32)
print(f"grad_pool2 shape: {grad_pool2.shape}, size: {grad_pool2_clipped.nbytes}")

d_pool2_grad = lib.gpu_malloc(C2_OUT * BATCH * poolH2 * poolW2 * 4)
print(f"d_pool2_grad allocated: {d_pool2_grad}")
lib.gpu_memcpy_h2d(d_pool2_grad, grad_pool2_clipped.flatten().ctypes.data, C2_OUT * BATCH * poolH2 * poolW2 * 4)
print("grad_pool2 copied to GPU")

d_conv2_grad = lib.gpu_malloc(C2_OUT * BATCH * outH2 * outW2 * 4)
print(f"d_conv2_grad allocated: {d_conv2_grad}")
lib.gpu_memset(d_conv2_grad, 0, C2_OUT * BATCH * outH2 * outW2 * 4)
print("d_conv2_grad zeroed")

print(f"\nCalling maxpool_backward_use_idx with:")
print(f"  d_pool2_grad: {d_pool2_grad}")
print(f"  d_max_idx2: {d_max_idx2}")
print(f"  d_conv2_grad: {d_conv2_grad}")
print(f"  BATCH={BATCH}, C2_OUT={C2_OUT}, outH2={outH2}, outW2={outW2}")
sys.stdout.flush()

lib.maxpool_backward_use_idx(d_pool2_grad, d_max_idx2, d_conv2_grad, BATCH, C2_OUT, outH2, outW2)
print("Pool2 backward OK")
sys.stdout.flush()

d_w_conv2_grad = lib.gpu_malloc(C2_OUT * C2_IN * KH2 * KW2 * 4)
lib.gpu_memset(d_w_conv2_grad, 0, C2_OUT * C2_IN * KH2 * KW2 * 4)
print(f"\nCalling conv_backward with:")
print(f"  grad_out (d_conv2_grad): {d_conv2_grad}")
print(f"  input (d_pool1): {d_pool1}")
print(f"  weights (d_w_conv2): {d_w_conv2}")
print(f"  grad_weights (d_w_conv2_grad): {d_w_conv2_grad}")
print(f"  N={BATCH}, C={C2_IN}, H={poolH1}, W={poolW1}, KH={KH2}, KW={KW2}, outH={outH2}, outW={outW2}, OC={C2_OUT}")
sys.stdout.flush()

lib.conv_backward(d_conv2_grad, d_pool1, d_w_conv2, d_w_conv2_grad,
                  BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2, C2_OUT)
print("Conv2 backward OK")
sys.stdout.flush()

lib.apply_sgd_update(d_w_conv2, d_w_conv2_grad, c_float(LR), C2_OUT * C2_IN * KH2 * KW2)
print("Conv2 weight update OK")

print("\n=== FIRST BATCH COMPLETE ===")
lib.gpu_free(d_x)
lib.gpu_free(d_col1)
lib.gpu_free(d_conv1_raw)
lib.gpu_free(d_conv1)
lib.gpu_free(d_pool1)
lib.gpu_free(d_max_idx1)
lib.gpu_free(d_col2)
lib.gpu_free(d_conv2_raw)
lib.gpu_free(d_conv2)
lib.gpu_free(d_pool2)
lib.gpu_free(d_max_idx2)
lib.gpu_free(d_fc_in)
lib.gpu_free(d_fc_out)
lib.gpu_free(d_fc_grad_w)
lib.gpu_free(d_pool2_grad)
lib.gpu_free(d_conv2_grad)
lib.gpu_free(d_w_conv2_grad)
lib.gpu_free(d_w_conv1)
lib.gpu_free(d_w_conv2)
lib.gpu_free(d_fc_w)
lib.gpu_free(d_fc_b)
print("All freed")
