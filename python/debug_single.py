#!/usr/bin/env python3
"""Debug single batch"""
import ctypes
import numpy as np
import os
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

    def malloc(self, size): return self.lib.gpu_malloc(size)
    def free(self, ptr): self.lib.gpu_free(ptr)
    def h2d(self, dst, src, size): self.lib.gpu_memcpy_h2d(dst, src, size)
    def d2h(self, dst, src, size): self.lib.gpu_memcpy_d2h(dst, src, size)
    def memset(self, ptr, val, size): self.lib.gpu_memset(ptr, val, size)


def load_cifar10_batch(root, batch_num):
    with open(os.path.join(root, f'data_batch_{batch_num}'), 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        imgs = batch[b'data'].astype(np.float32) / 255.0
        imgs = imgs.reshape(-1, 3, 32, 32)
        return imgs, np.array(batch[b'labels'])


def test():
    so_path = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
    data_root = "minimal_cuda_cnn/data/cifar-10-batches-py"
    
    print("=== Single Batch Test ===")
    
    bridge = Bridge(so_path)
    print("Bridge loaded")
    
    # Load single batch
    train_x, train_y = load_cifar10_batch(data_root, 1)
    print(f"Data loaded: {train_x.shape}")
    
    BATCH = 8  # Small batch
    x = train_x[:BATCH]
    y = train_y[:BATCH]
    
    N, C, H, W = BATCH, 3, 32, 32
    KH, KW, OC = 3, 3, 16  # Small conv
    outH = H - KH + 1
    outW = W - KW + 1
    
    # Init weights
    w = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.05
    print(f"Weights: {w.shape}")
    
    d_x = bridge.malloc(BATCH * C * H * W * 4)
    d_w = bridge.malloc(OC * C * KH * KW * 4)
    bridge.h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
    bridge.h2d(d_w, w.ctypes.data, OC * C * KH * KW * 4)
    print("Copied to GPU")
    
    # Conv forward
    col_size = C * KH * KW * BATCH * outH * outW
    d_col = bridge.malloc(col_size * 4)
    d_out = bridge.malloc(OC * BATCH * outH * outW * 4)
    
    print("Running im2col_forward...")
    bridge.lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)
    print("im2col done")
    
    print("Running gemm_forward...")
    bridge.lib.gemm_forward(d_w, d_col, d_out, OC, BATCH * outH * outW, C * KH * KW)
    print("gemm done")
    
    print("Running apply_relu...")
    bridge.lib.apply_relu(d_out, OC * BATCH * outH * outW)
    print("relu done")
    
    # Pool
    d_pool = bridge.malloc(OC * BATCH * 15 * 15 * 4)
    print("Running apply_maxpool...")
    bridge.lib.apply_maxpool(d_out, d_pool, BATCH, OC, outH, outW)
    print("maxpool done")
    
    # Copy result back
    result = np.zeros((OC * BATCH * 15 * 15,), dtype=np.float32)
    bridge.d2h(result.ctypes.data, d_pool, OC * BATCH * 15 * 15 * 4)
    print(f"Result: sum={result.sum():.2f}, max={result.max():.2f}")
    
    # Cleanup
    bridge.free(d_x)
    bridge.free(d_w)
    bridge.free(d_col)
    bridge.free(d_out)
    bridge.free(d_pool)
    
    print("SUCCESS!")


if __name__ == '__main__':
    test()
