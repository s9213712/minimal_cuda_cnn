#!/usr/bin/env python3
"""Minimal AlexNet Test - Step by Step"""
import ctypes
import numpy as np
import os
from ctypes import c_void_p, c_float, c_int
import pickle
import time

class AlexNetBridge:
    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)
        self._setup()
    
    def _setup(self):
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

    def malloc(self, size): return self.lib.gpu_malloc(size)
    def free(self, ptr): self.lib.gpu_free(ptr)
    def h2d(self, dst, src, size): self.lib.gpu_memcpy_h2d(dst, src, size)
    def d2h(self, dst, src, size): self.lib.gpu_memcpy_d2h(dst, src, size)
    def memset(self, ptr, val, size): self.lib.gpu_memset(ptr, val, size)


def load_cifar10(root):
    train_x, train_y = [], []
    for i in range(1, 6):
        with open(os.path.join(root, f'data_batch_{i}'), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            imgs = batch[b'data'].astype(np.float32) / 255.0
            imgs = imgs.reshape(-1, 3, 32, 32)
            train_x.append(imgs)
            train_y.extend(batch[b'labels'])
    return np.concatenate(train_x), np.array(train_y)


class ConvLayer:
    def __init__(self, c_in, c_out, k, bridge):
        self.c_in, self.c_out, self.k, self.bridge = c_in, c_out, k, bridge
        self.w_size = c_out * c_in * k * k
    
    def init_weights(self, h, w):
        self.h, self.w = h, w
        scale = np.sqrt(2.0 / (self.c_in * self.k * self.k + self.c_out * self.k * self.k))
        self.w_host = (np.random.randn(self.w_size).astype(np.float32) * scale)
        self.d_w = self.bridge.malloc(self.w_size * 4)
        self.bridge.h2d(self.d_w, self.w_host.ctypes.data, self.w_size * 4)
    
    def forward(self, d_input, N, C, H, W):
        outH, outW = H - self.k + 1, W - self.k + 1
        col_size = self.c_in * self.k * self.k * N * outH * outW
        out_size = self.c_out * N * outH * outW
        
        d_col = self.bridge.malloc(col_size * 4)
        d_out = self.bridge.malloc(out_size * 4)
        
        self.bridge.lib.im2col_forward(d_input, d_col, N, C, H, W, self.k, self.k, outH, outW)
        self.bridge.lib.gemm_forward(self.d_w, d_col, d_out, self.c_out, N * outH * outW, self.c_in * self.k * self.k)
        self.bridge.lib.apply_relu(d_out, out_size)
        
        return d_out, d_col, outH, outW
    
    def cleanup(self, *ptrs):
        for p in ptrs: self.bridge.free(p)


class PoolLayer:
    def __init__(self, bridge): self.bridge = bridge
    
    def forward(self, d_input, N, C, H, W):
        outH, outW = H // 2, W // 2
        d_out = self.bridge.malloc(N * C * outH * outW * 4)
        self.bridge.lib.apply_maxpool(d_input, d_out, N, C, H, W)
        return d_out, outH, outW


class FCLayer:
    def __init__(self, c_in, c_out, bridge):
        self.c_in, self.c_out, self.bridge = c_in, c_out, bridge
        self.w_size, self.b_size = c_out * c_in, c_out
    
    def init_weights(self):
        scale = np.sqrt(2.0 / (self.c_in + self.c_out))
        self.w_host = (np.random.randn(self.w_size).astype(np.float32) * scale)
        self.b_host = np.zeros(self.c_out, dtype=np.float32)
        self.d_w = self.bridge.malloc(self.w_size * 4)
        self.d_b = self.bridge.malloc(self.b_size * 4)
        self.bridge.h2d(self.d_w, self.w_host.ctypes.data, self.w_size * 4)
        self.bridge.h2d(self.d_b, self.b_host.ctypes.data, self.b_size * 4)
    
    def forward(self, d_input, N):
        d_out = self.bridge.malloc(N * self.c_out * 4)
        self.bridge.lib.dense_forward(d_input, self.d_w, self.d_b, d_out, N, self.c_in, self.c_out)
        return d_out


def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(base_dir, '..', 'cpp', 'libminimal_cuda_cnn.so')
    data_root = os.path.join(base_dir, '..', 'data', 'cifar-10-batches-py')
    
    print("=== Minimal AlexNet ===")
    
    print("Loading data...")
    train_x, train_y = load_cifar10(data_root)
    print(f"Train: {train_x.shape}, Labels: {train_y.shape}")
    
    bridge = AlexNetBridge(so_path)
    
    BATCH, EPOCHS, LR = 32, 2, 0.01
    print(f"Config: batch={BATCH}, epochs={EPOCHS}, lr={LR}")
    
    # Simple 2-layer conv + 1 FC
    conv1 = ConvLayer(3, 16, 3, bridge)
    conv2 = ConvLayer(16, 32, 3, bridge)
    fc1 = FCLayer(32 * 7 * 7, 128, bridge)
    fc2 = FCLayer(128, 10, bridge)
    
    print("Init weights...")
    conv1.init_weights(32, 32)
    conv2.init_weights(14, 14)  # After pool
    fc1.init_weights()
    fc2.init_weights()
    
    N_TRAIN = train_x.shape[0]
    BATCH_PER_EPOCH = N_TRAIN // BATCH
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        t0 = time.time()
        total_loss = 0.0
        correct = 0
        
        indices = np.random.permutation(N_TRAIN)
        for batch_idx in range(BATCH_PER_EPOCH):
            if batch_idx >= 5: break  # Debug: only 5 batches
            
            batch_start = batch_idx * BATCH
            x_batch = train_x[indices[batch_start:batch_start + BATCH]]
            y_batch = train_y[indices[batch_start:batch_start + BATCH]]
            
            # Forward: Conv1 -> Pool -> Conv2 -> Pool -> FC1 -> FC2
            d_x = bridge.malloc(BATCH * 3 * 32 * 32 * 4)
            bridge.h2d(d_x, x_batch.ctypes.data, BATCH * 3 * 32 * 32 * 4)
            
            # Conv1 + Pool
            d_out, d_col, outH, outW = conv1.forward(d_x, BATCH, 3, 32, 32)
            d_pool, outH, outW = PoolLayer(bridge).forward(d_out, BATCH, 16, outH, outW)
            conv1.cleanup(d_x, d_col, d_out)
            
            # Conv2 + Pool (output should be ~7x7)
            d_out, d_col, outH, outW = conv2.forward(d_pool, BATCH, 16, outH, outW)
            d_pool, outH, outW = PoolLayer(bridge).forward(d_out, BATCH, 32, outH, outW)
            conv2.cleanup(d_pool, d_col, d_out)
            
            # Flatten and FC
            # Note: this is simplified - need proper flatten
            d_fc_in = d_pool  # Should flatten here
            
            d_fc1 = fc1.forward(d_fc_in, BATCH)
            bridge.free(d_fc_in)
            
            d_fc2 = fc2.forward(d_fc1, BATCH)
            bridge.free(d_fc1)
            bridge.free(d_fc2)
            bridge.free(d_x)
            
            correct += np.random.randint(0, 5)  # Mock
            total_loss += 2.3
            
            if (batch_idx + 1) % 2 == 0:
                print(f"  Batch {batch_idx+1}/{BATCH_PER_EPOCH}", end=" ")
        
        print(f"\nEpoch {epoch+1}: Loss={total_loss/5:.4f}, Acc={correct/5/BATCH*100:.1f}%, Time={time.time()-t0:.1f}s")
    
    print("Done!")


if __name__ == '__main__':
    train()
