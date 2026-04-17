#!/usr/bin/env python3
"""Training with reorganize fix"""
import ctypes
import numpy as np
import os
import time
from ctypes import c_void_p, c_float, c_int
import pickle

class Bridge:
    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)
        self.lib.gpu_malloc.argtypes = [ctypes.c_size_t]
        self.lib.gpu_malloc.restype = c_void_p
        self.lib.gpu_free.argtypes = [c_void_p]
        self.lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        self.lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        self.lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
        self.lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
        self.lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        self.lib.apply_relu.argtypes = [c_void_p, c_int]
        self.lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        self.lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        self.lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]
        self.lib.reorganize_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]

    def malloc(self, size): return self.lib.gpu_malloc(size)
    def free(self, ptr): self.lib.gpu_free(ptr)
    def h2d(self, dst, src, size): self.lib.gpu_memcpy_h2d(dst, src, size)
    def d2h(self, dst, src, size): self.lib.gpu_memcpy_d2h(dst, src, size)
    def memset(self, ptr, val, size): self.lib.gpu_memset(ptr, val, size)


def load_cifar10(root):
    train_x, train_y = [], []
    for i in range(1, 6):
        with open(os.path.join(root, f"data_batch_{i}"), "rb") as f:
            batch = pickle.load(f, encoding="bytes")
            imgs = batch[b"data"].astype(np.float32) / 255.0
            imgs = imgs.reshape(-1, 3, 32, 32)
            train_x.append(imgs)
            train_y.extend(batch[b"labels"])
    return np.concatenate(train_x), np.array(train_y)


def train():
    so_path = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
    data_root = "minimal_cuda_cnn/data/cifar-10-batches-py"
    
    print("=== Training with Reorganize Fix ===")
    
    print("Loading data...")
    train_x, train_y = load_cifar10(data_root)
    print(f"Train: {train_x.shape}")
    
    bridge = Bridge(so_path)
    
    BATCH, EPOCHS, LR = 64, 5, 0.1
    print(f"Config: batch={BATCH}, epochs={EPOCHS}, lr={LR}")
    
    N, C, H, W = BATCH, 3, 32, 32
    KH, KW, OC = 3, 3, 32
    outH, outW = H - KH + 1, W - KW + 1
    FC_IN = OC * 15 * 15
    
    w_conv = np.random.randn(OC * C * KH * KW).astype(np.float32) * np.sqrt(2.0 / (C * KH * KW + OC * KH * KW))
    fc_w = np.random.randn(10 * FC_IN).astype(np.float32) * np.sqrt(2.0 / (FC_IN + 10))
    fc_b = np.zeros(10, dtype=np.float32)
    
    d_w_conv = bridge.malloc(OC * C * KH * KW * 4)
    d_fc_w = bridge.malloc(10 * FC_IN * 4)
    d_fc_b = bridge.malloc(10 * 4)
    bridge.h2d(d_w_conv, w_conv.ctypes.data, OC * C * KH * KW * 4)
    bridge.h2d(d_fc_w, fc_w.ctypes.data, 10 * FC_IN * 4)
    bridge.h2d(d_fc_b, fc_b.ctypes.data, 10 * 4)
    
    N_TRAIN = train_x.shape[0]
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        t0 = time.time()
        total_loss = 0.0
        correct = 0
        
        indices = np.random.permutation(N_TRAIN)
        for batch_idx in range(N_TRAIN // BATCH):
            batch_start = batch_idx * BATCH
            x_batch = train_x[indices[batch_start:batch_start + BATCH]]
            y_batch = train_y[indices[batch_start:batch_start + BATCH]]
            
            d_x = bridge.malloc(BATCH * C * H * W * 4)
            bridge.h2d(d_x, x_batch.ctypes.data, BATCH * C * H * W * 4)
            
            # Forward Conv
            col_size = C * KH * KW * BATCH * outH * outW
            d_col = bridge.malloc(col_size * 4)
            d_conv_raw = bridge.malloc(OC * BATCH * outH * outW * 4)
            bridge.lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)
            bridge.lib.gemm_forward(d_w_conv, d_col, d_conv_raw, OC, BATCH * outH * outW, C * KH * KW)
            bridge.lib.apply_relu(d_conv_raw, OC * BATCH * outH * outW)
            
            # Reorganize: (OC, N) -> (N, OC, H, W)
            d_conv = bridge.malloc(OC * BATCH * outH * outW * 4)
            bridge.lib.reorganize_forward(d_conv_raw, d_conv, BATCH, OC, outH, outW)
            
            # Pool: (N, OC, 30, 30) -> (N, OC, 15, 15)
            d_pool = bridge.malloc(OC * BATCH * 15 * 15 * 4)
            bridge.lib.apply_maxpool(d_conv, d_pool, BATCH, OC, outH, outW)
            
            # Flatten
            h_pool = np.zeros((BATCH, FC_IN), dtype=np.float32)
            bridge.d2h(h_pool.ctypes.data, d_pool, BATCH * FC_IN * 4)
            d_fc_in = bridge.malloc(BATCH * FC_IN * 4)
            bridge.h2d(d_fc_in, h_pool.ctypes.data, BATCH * FC_IN * 4)
            
            # FC
            d_fc_out = bridge.malloc(BATCH * 10 * 4)
            bridge.lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, BATCH, FC_IN, 10)
            
            # Loss
            h_out = np.zeros((BATCH, 10), dtype=np.float32)
            bridge.d2h(h_out.ctypes.data, d_fc_out, BATCH * 10 * 4)
            h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
            exp_out = np.exp(h_out_shifted)
            probs = exp_out / exp_out.sum(axis=1, keepdims=True)
            loss = -np.mean(np.log(probs[np.arange(BATCH), y_batch] + 1e-10))
            total_loss += loss
            
            # Backward - FC
            labels_onehot = np.zeros((BATCH, 10), dtype=np.float32)
            labels_onehot[np.arange(BATCH), y_batch] = 1.0
            d_loss = probs - labels_onehot
            
            grad_fc_w = d_loss.T @ h_pool
            grad_fc_w = np.clip(grad_fc_w, -1.0, 1.0).flatten().astype(np.float32)
            
            d_fc_grad_w = bridge.malloc(10 * FC_IN * 4)
            bridge.h2d(d_fc_grad_w, grad_fc_w.ctypes.data, 10 * FC_IN * 4)
            bridge.lib.apply_sgd_update(d_fc_w, d_fc_grad_w, LR, 10 * FC_IN)
            
            # Accuracy
            pred = np.argmax(probs, axis=1)
            correct += np.sum(pred == y_batch)
            
            # Cleanup
            bridge.free(d_x)
            bridge.free(d_col)
            bridge.free(d_conv_raw)
            bridge.free(d_conv)
            bridge.free(d_pool)
            bridge.free(d_fc_in)
            bridge.free(d_fc_out)
            bridge.free(d_fc_grad_w)
            
            if (batch_idx + 1) % 200 == 0:
                print(f"  Batch {batch_idx+1}: Loss={loss:.4f}, Acc={correct/(batch_idx+1)/BATCH*100:.1f}%")
        
        acc = correct / N_TRAIN * 100
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss={total_loss/(N_TRAIN/BATCH):.4f}, Acc={acc:.2f}%, Time={time.time()-t0:.1f}s")
    
    bridge.free(d_w_conv)
    bridge.free(d_fc_w)
    bridge.free(d_fc_b)
    print("Done!")


if __name__ == "__main__":
    train()
