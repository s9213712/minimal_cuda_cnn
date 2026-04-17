#!/usr/bin/env python3
"""Simplified AlexNet Training Test"""
import ctypes
import numpy as np
import os
from ctypes import c_void_p, c_float, c_int
import pickle

class SimpleBridge:
    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)
        # Memory
        self.lib.gpu_malloc.argtypes = [ctypes.c_size_t]
        self.lib.gpu_malloc.restype = c_void_p
        self.lib.gpu_free.argtypes = [c_void_p]
        self.lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        self.lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        self.lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
        
        # Conv
        self.lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
        self.lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        self.lib.apply_relu.argtypes = [c_void_p, c_int]
        self.lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        
        # Dense
        self.lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        
        # Update
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
            # Reshape to NCHW
            imgs = imgs.reshape(-1, 3, 32, 32)
            train_x.append(imgs)
            train_y.extend(batch[b'labels'])
    
    with open(os.path.join(root, 'test_batch'), 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        test_x = batch[b'data'].astype(np.float32) / 255.0
        test_x = test_x.reshape(-1, 3, 32, 32)
        test_y = batch[b'labels']
    
    return (np.concatenate(train_x), np.array(train_y)), (test_x, np.array(test_y))


def test_simple():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(base_dir, '..', 'cpp', 'libminimal_cuda_cnn.so')
    data_root = os.path.join(base_dir, '..', 'data', 'cifar-10-batches-py')
    
    print("=== Simple Test ===")
    print(f"SO: {so_path}")
    print(f"Data: {data_root}")
    
    # Load CIFAR-10
    (train_x, train_y), _ = load_cifar10(data_root)
    print(f"\nTrain: {train_x.shape}, Labels: {train_y.shape}")
    print(f"Labels range: {train_y.min()} - {train_y.max()}")
    
    # Simple 1-layer conv test
    bridge = SimpleBridge(so_path)
    
    N, C, H, W = 4, 3, 32, 32  # Small batch
    KH, KW = 3, 3
    OUT_C = 8  # Small output
    
    x = train_x[:N].reshape(N, C, H, W)
    
    # Alloc
    x_size = N * C * H * W * 4
    w_size = OUT_C * C * KH * KW * 4
    outH, outW = H - KH + 1, W - KW + 1
    col_size = C * KH * KW * N * outH * outW * 4
    out_size = OUT_C * N * outH * outW * 4
    
    d_x = bridge.malloc(x_size)
    d_w = bridge.malloc(w_size)
    d_col = bridge.malloc(col_size)
    d_out = bridge.malloc(out_size)
    
    bridge.h2d(d_x, x.ctypes.data, x_size)
    
    # Xavier init weights
    scale = np.sqrt(2.0 / (C * KH * KW + OUT_C * KH * KW))
    w = (np.random.randn(OUT_C * C * KH * KW) * scale).astype(np.float32)
    bridge.h2d(d_w, w.ctypes.data, w_size)
    
    print(f"\nRunning conv forward...")
    print(f"Input: {N}x{C}x{H}x{W}")
    print(f"Weights: {OUT_C}x{C}x{KH}x{KW}")
    print(f"Output: {OUT_C}x{outH}x{outW}")
    
    # Forward
    bridge.lib.im2col_forward(d_x, d_col, N, C, H, W, KH, KW, outH, outW)
    bridge.lib.gemm_forward(d_w, d_col, d_out, OUT_C, N * outH * outW, C * KH * KW)
    bridge.lib.apply_relu(d_out, OUT_C * N * outH * outW)
    bridge.lib.apply_maxpool(d_out, d_out, N, OUT_C, outH, outW)
    
    print("Forward OK!")
    
    # Get output
    h_out = np.zeros(out_size, dtype=np.float32)
    bridge.d2h(h_out.ctypes.data, d_out, out_size)
    print(f"Output sum: {h_out.sum():.4f}, max: {h_out.max():.4f}")
    
    # Cleanup
    bridge.free(d_x)
    bridge.free(d_w)
    bridge.free(d_col)
    bridge.free(d_out)
    
    print("\nSimple test PASSED!")


if __name__ == '__main__':
    test_simple()
