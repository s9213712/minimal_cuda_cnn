#!/usr/bin/env python3
"""Debug Conv1 backward - isolate which step causes damage"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_float, c_int
import os, pickle, time

workspace = "/mnt/c/Users/user/.openclaw/workspace"
so = os.path.join(workspace, "NN/minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so")
lib = ctypes.CDLL(so)

lib.gpu_malloc.argtypes = [ctypes.c_size_t]; lib.gpu_malloc.restype = c_void_p
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
lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.im2col_backward.argtypes = [c_void_p]*10
lib.gemm_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]

def g2h(ptr, size):
    h = np.zeros(size, dtype=np.float32)
    lib.gpu_memcpy_d2h(h.ctypes.data, ptr, size * 4)
    return h

BATCH = 64
LR = 0.005

C1_IN, C1_OUT = 3, 32; H, W = 32, 32; KH1, KW1 = 3, 3
outH1, outW1 = H - KH1 + 1, W - KW1 + 1
poolH1, poolW1 = outH1 // 2, outW1 // 2

C2_IN, C2_OUT = 32, 64; KH2, KW2 = 3, 3
outH2, outW2 = poolH1 - KH2 + 1, poolW1 - KW2 + 1
poolH2, poolW2 = outH2 // 2, outW2 // 2

FC_IN = C2_OUT * poolH2 * poolW2

# Load CIFAR
data_root = os.path.join(workspace, "NN/minimal_cuda_cnn/data/cifar-10-batches-py")
with open(os.path.join(data_root, "data_batch_1"), "rb") as f:
    b = pickle.load(f, encoding="bytes")
    x_all = b[b"data"].astype(np.float32)/255.0
    x_all = x_all.reshape(-1, 3, 32, 32)
    y_all = np.array(b[b"labels"])

# Compare 4 scenarios on same batch
def run_epoch(scenario, epochs=2):
    np.random.seed(42)
    std = 0.05
    w_conv1 = np.random.randn(C1_OUT*C1_IN*KH1*KW1).astype(np.float32) * std
    w_conv2 = np.random.randn(C2_OUT*C2_IN*KH2*KW2).astype(np.float32) * std
    fc_w = np.random.randn(10*FC_IN).astype(np.float32) * std
    fc_b = np.zeros(10, dtype=np.float32)

    d_wc1 = lib.gpu_malloc(C1_OUT*C1_IN*KH1*KW1*4)
    d_wc2 = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
    d_fc_w = lib.gpu_malloc(10*FC_IN*4)
    d_fc_b = lib.gpu_malloc(10*4)
    lib.gpu_memcpy_h2d(d_wc1, w_conv1.ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
    lib.gpu_memcpy_h2d(d_wc2, w_conv2.ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)
    lib.gpu_memcpy_h2d(d_fc_w, fc_w.ctypes.data, 10*FC_IN*4)
    lib.gpu_memcpy_h2d(d_fc_b, fc_b.ctypes.data, 10*4)

    correct = 0; NBATCHES = x_all.shape[0] // BATCH
    indices = np.random.permutation(x_all.shape[0])

    for epoch in range(epochs):
        for batch_idx in range(NBATCHES):
            idx_s = batch_idx * BATCH; idx_e = idx_s + BATCH
            x = x_all[indices[idx_s:idx_e]]; y = y_all[indices[idx_s:idx_e]]

            # FORWARD (shared)
            d_x = lib.gpu_malloc(BATCH*C1_IN*H*W*4)
            lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH*C1_IN*H*W*4)

            d_col1 = lib.gpu_malloc(C1_IN*KH1*KW1*BATCH*outH1*outW1*4)
            d_cr1 = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
            lib.im2col_forward(d_x, d_col1, BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1)
            lib.gemm_forward(d_wc1, d_col1, d_cr1, C1_OUT, BATCH*outH1*outW1, C1_IN*KH1*KW1)
            lib.leaky_relu_forward(d_cr1, c_float(0.1), C1_OUT*BATCH*outH1*outW1)
            d_c1 = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
            lib.reorganize_forward(d_cr1, d_c1, BATCH, C1_OUT, outH1, outW1)
            d_p1 = lib.gpu_malloc(C1_OUT*BATCH*poolH1*poolW1*4)
            d_m1 = lib.gpu_malloc(C1_OUT*BATCH*poolH1*poolW1*4)
            lib.maxpool_forward_store(d_p1, d_c1, d_m1, BATCH, C1_OUT, outH1, outW1)

            d_col2 = lib.gpu_malloc(C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
            d_cr2 = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
            lib.im2col_forward(d_p1, d_col2, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)
            lib.gemm_forward(d_wc2, d_col2, d_cr2, C2_OUT, BATCH*outH2*outW2, C2_IN*KH2*KW2)
            lib.leaky_relu_forward(d_cr2, c_float(0.1), C2_OUT*BATCH*outH2*outW2)
            d_c2 = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
            lib.reorganize_forward(d_cr2, d_c2, BATCH, C2_OUT, outH2, outW2)
            d_p2 = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
            d_m2 = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
            lib.maxpool_forward_store(d_p2, d_c2, d_m2, BATCH, C2_OUT, outH2, outW2)

            h_p2 = np.zeros((BATCH, FC_IN), dtype=np.float32)
            lib.gpu_memcpy_d2h(h_p2.ctypes.data, d_p2, BATCH*FC_IN*4)
            d_fc_in = lib.gpu_malloc(BATCH*FC_IN*4)
            lib.gpu_memcpy_h2d(d_fc_in, h_p2.ctypes.data, BATCH*FC_IN*4)
            d_fc_out = lib.gpu_malloc(BATCH*10*4)
            lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
            h_out = np.zeros((BATCH, 10), dtype=np.float32)
            lib.gpu_memcpy_d2h(h_out.ctypes.data, d_fc_out, BATCH*10*4)

            h_out_s = h_out - h_out.max(axis=1, keepdims=True)
            probs = np.exp(h_out_s) / np.exp(h_out_s).sum(axis=1, keepdims=True)
            correct += np.sum(np.argmax(probs, axis=1) == y)

            # BACKWARD
            labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
            labels_onehot[np.arange(BATCH), y] = 1.0
            d_loss = probs - labels_onehot

            # FC
            grad_fc_w = d_loss.T @ h_p2
            d_fc_gw = lib.gpu_malloc(10*FC_IN*4)
            lib.gpu_memcpy_h2d(d_fc_gw, np.clip(grad_fc_w, -1.0, 1.0).flatten().ctypes.data, 10*FC_IN*4)
            lib.apply_sgd_update(d_fc_w, d_fc_gw, c_float(LR), 10*FC_IN)
            h_fc_w = np.clip(g2h(d_fc_w, 10*FC_IN), -1.0, 1.0)
            lib.gpu_memcpy_h2d(d_fc_w, h_fc_w.ctypes.data, 10*FC_IN*4)

            # Conv2 backward (ALWAYS)
            grad_pool2 = d_loss @ fc_w.reshape(10, FC_IN)
            d_p2g = lib.gpu_malloc(C2_OUT*BATCH*poolH2*poolW2*4)
            lib.gpu_memcpy_h2d(d_p2g, np.clip(grad_pool2, -1.0, 1.0).flatten().ctypes.data, C2_OUT*BATCH*poolH2*poolW2*4)
            d_c2gr = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
            lib.gpu_memset(d_c2gr, 0, C2_OUT*BATCH*outH2*outW2*4)
            lib.maxpool_backward_use_idx(d_p2g, d_m2, d_c2gr, BATCH, C2_OUT, outH2, outW2)
            d_c2g = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
            lib.gpu_memset(d_c2g, 0, C2_OUT*BATCH*outH2*outW2*4)
            lib.reorganize_backward(d_c2gr, d_c2g, BATCH, C2_OUT, outH2, outW2)
            lib.leaky_relu_backward(d_cr2, d_c2g, c_float(0.1), C2_OUT*BATCH*outH2*outW2)
            d_wc2g = lib.gpu_malloc(C2_OUT*C2_IN*KH2*KW2*4)
            lib.gpu_memset(d_wc2g, 0, C2_OUT*C2_IN*KH2*KW2*4)
            lib.conv_backward(d_c2g, d_p1, d_wc2, d_wc2g, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2, C2_OUT)
            lib.apply_sgd_update(d_wc2, d_wc2g, c_float(LR), C2_OUT*C2_IN*KH2*KW2)
            h_wc2 = np.clip(g2h(d_wc2, C2_OUT*C2_IN*KH2*KW2), -2.0, 2.0)
            lib.gpu_memcpy_h2d(d_wc2, h_wc2.ctypes.data, C2_OUT*C2_IN*KH2*KW2*4)

            # ---- SCENARIO-SPECIFIC ----
            if scenario == "conv2_only":
                pass  # skip Conv1 backward

            elif scenario == "conv1_no_sgd":
                # Compute Conv1 gradient but don't apply SGD
                d_c1gr = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
                lib.gpu_memset(d_c1gr, 0, C1_OUT*BATCH*outH1*outW1*4)
                # gemm_backward -> im2col_backward -> pool1_backward
                d_col2g = lib.gpu_malloc(C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
                lib.gpu_memset(d_col2g, 0, C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
                d_cr2g_unused = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
                lib.gpu_memset(d_cr2g_unused, 0, C2_OUT*BATCH*outH2*outW2*4)
                lib.gemm_backward(d_c2g, d_wc2, d_col2, d_cr2g_unused, d_col2g, C2_OUT, BATCH*outH2*outW2, C2_IN*KH2*KW2)
                d_p1g = lib.gpu_malloc(C2_IN*BATCH*poolH1*poolW1*4)
                lib.gpu_memset(d_p1g, 0, C2_IN*BATCH*poolH1*poolW1*4)
                lib.im2col_backward(d_col2g, d_p1g, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)
                lib.maxpool_backward_use_idx(d_p1g, d_m1, d_c1gr, BATCH, C1_OUT, outH1, outW1)
                # leak + reorganize
                d_c1g = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
                lib.gpu_memset(d_c1g, 0, C1_OUT*BATCH*outH1*outW1*4)
                lib.reorganize_backward(d_c1gr, d_c1g, BATCH, C1_OUT, outH1, outW1)
                lib.leaky_relu_backward(d_cr1, d_c1g, c_float(0.1), C1_OUT*BATCH*outH1*outW1)
                # don't update conv1 weights (just compute gradient)
                lib.gpu_free(d_c1gr); lib.gpu_free(d_col2g); lib.gpu_free(d_cr2g_unused); lib.gpu_free(d_p1g); lib.gpu_free(d_c1g)

            elif scenario == "conv1_sgd":
                # Full Conv1 backward + SGD
                d_c1gr = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
                lib.gpu_memset(d_c1gr, 0, C1_OUT*BATCH*outH1*outW1*4)
                d_col2g = lib.gpu_malloc(C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
                lib.gpu_memset(d_col2g, 0, C2_IN*KH2*KW2*BATCH*outH2*outW2*4)
                d_cr2g_unused = lib.gpu_malloc(C2_OUT*BATCH*outH2*outW2*4)
                lib.gpu_memset(d_cr2g_unused, 0, C2_OUT*BATCH*outH2*outW2*4)
                lib.gemm_backward(d_c2g, d_wc2, d_col2, d_cr2g_unused, d_col2g, C2_OUT, BATCH*outH2*outW2, C2_IN*KH2*KW2)
                d_p1g = lib.gpu_malloc(C2_IN*BATCH*poolH1*poolW1*4)
                lib.gpu_memset(d_p1g, 0, C2_IN*BATCH*poolH1*poolW1*4)
                lib.im2col_backward(d_col2g, d_p1g, BATCH, C2_IN, poolH1, poolW1, KH2, KW2, outH2, outW2)
                lib.maxpool_backward_use_idx(d_p1g, d_m1, d_c1gr, BATCH, C1_OUT, outH1, outW1)
                d_c1g = lib.gpu_malloc(C1_OUT*BATCH*outH1*outW1*4)
                lib.gpu_memset(d_c1g, 0, C1_OUT*BATCH*outH1*outW1*4)
                lib.reorganize_backward(d_c1gr, d_c1g, BATCH, C1_OUT, outH1, outW1)
                lib.leaky_relu_backward(d_cr1, d_c1g, c_float(0.1), C1_OUT*BATCH*outH1*outW1)
                d_wc1g = lib.gpu_malloc(C1_OUT*C1_IN*KH1*KW1*4)
                lib.gpu_memset(d_wc1g, 0, C1_OUT*C1_IN*KH1*KW1*4)
                lib.conv_backward(d_c1g, d_x, d_wc1, d_wc1g, BATCH, C1_IN, H, W, KH1, KW1, outH1, outW1, C1_OUT)
                lib.apply_sgd_update(d_wc1, d_wc1g, c_float(LR), C1_OUT*C1_IN*KH1*KW1)
                h_wc1 = np.clip(g2h(d_wc1, C1_OUT*C1_IN*KH1*KW1), -2.0, 2.0)
                lib.gpu_memcpy_h2d(d_wc1, h_wc1.ctypes.data, C1_OUT*C1_IN*KH1*KW1*4)
                lib.gpu_free(d_c1gr); lib.gpu_free(d_col2g); lib.gpu_free(d_cr2g_unused)
                lib.gpu_free(d_p1g); lib.gpu_free(d_c1g); lib.gpu_free(d_wc1g)

            # FREE (shared)
            lib.gpu_free(d_x); lib.gpu_free(d_col1); lib.gpu_free(d_cr1); lib.gpu_free(d_c1)
            lib.gpu_free(d_p1); lib.gpu_free(d_m1); lib.gpu_free(d_col2); lib.gpu_free(d_cr2)
            lib.gpu_free(d_c2); lib.gpu_free(d_p2); lib.gpu_free(d_m2); lib.gpu_free(d_fc_in)
            lib.gpu_free(d_fc_out); lib.gpu_free(d_fc_gw); lib.gpu_free(d_p2g)
            lib.gpu_free(d_c2gr); lib.gpu_free(d_c2g); lib.gpu_free(d_wc2g)

    acc = correct / NBATCHES / BATCH * 100
    lib.gpu_free(d_wc1); lib.gpu_free(d_wc2); lib.gpu_free(d_fc_w); lib.gpu_free(d_fc_b)
    return acc

print("Running comparison (2 epochs each)...")
scenarios = ["conv2_only", "conv1_no_sgd", "conv1_sgd"]
for s in scenarios:
    t0 = time.time()
    acc = run_epoch(s, epochs=2)
    print(f"{s}: {acc:.2f}%, time={time.time()-t0:.0f}s")