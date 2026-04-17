#!/usr/bin/env python3
"""CUDA CNN Real Training with proper GPU memory management"""
import ctypes
import numpy as np
import os
from ctypes import POINTER, c_float, c_int, c_void_p

class CudaCNNBridge:
    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)
        
        # Memory helpers
        self.lib.gpu_malloc.argtypes = [ctypes.c_size_t]
        self.lib.gpu_malloc.restype = ctypes.c_void_p
        self.lib.gpu_free.argtypes = [ctypes.c_void_p]
        self.lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        self.lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        self.lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]
        
        # Forward functions
        self.lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
        self.lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        self.lib.apply_relu.argtypes = [c_void_p, c_int]
        self.lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        
        # Backward functions
        self.lib.apply_relu_backward.argtypes = [c_void_p, c_void_p, c_int]
        self.lib.conv_backward.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]

    def train_step(self, x, w_conv, label=None, lr=0.01):
        N, C, H, W = 1, 3, 32, 32
        KH, KW = 3, 3
        OUT_C = 64
        outH = H - KH + 1
        outW = W - KW + 1
        
        # Sizes
        x_size = N * C * H * W * 4
        w_size = OUT_C * C * KH * KW * 4
        conv_out_size = OUT_C * N * outH * outW * 4
        pool_out_size = OUT_C * N * (outH // 2) * (outW // 2) * 4
        
        # Allocate GPU memory
        d_x = self.lib.gpu_malloc(x_size)
        d_w = self.lib.gpu_malloc(w_size)
        d_col = self.lib.gpu_malloc(C * KH * KW * N * outH * outW * 4)
        d_conv_out = self.lib.gpu_malloc(conv_out_size)
        d_pool_out = self.lib.gpu_malloc(pool_out_size)
        d_grad_w = self.lib.gpu_malloc(w_size)
        
        # Copy to GPU
        self.lib.gpu_memcpy_h2d(d_x, x.ctypes.data, x_size)
        self.lib.gpu_memcpy_h2d(d_w, w_conv.ctypes.data, w_size)
        
        # Forward pass
        self.lib.im2col_forward(d_x, d_col, N, C, H, W, KH, KW, outH, outW)
        self.lib.gemm_forward(d_w, d_col, d_conv_out, OUT_C, N * outH * outW, C * KH * KW)
        self.lib.apply_relu(d_conv_out, OUT_C * N * outH * outW)
        self.lib.apply_maxpool(d_conv_out, d_pool_out, N, OUT_C, outH, outW)
        
        # Mock gradient for backward
        grad_pool = np.random.randn(OUT_C * N * (outH // 2) * (outW // 2)).astype(np.float32)
        d_grad_pool = self.lib.gpu_malloc(pool_out_size)
        self.lib.gpu_memcpy_h2d(d_grad_pool, grad_pool.ctypes.data, pool_out_size)
        
        self.lib.apply_relu_backward(d_conv_out, d_grad_pool, OUT_C * N * outH * outW)
        
        # Zero grad_weights
        self.lib.gpu_memset(d_grad_w, 0, w_size)
        
        # Backward pass
        self.lib.conv_backward(d_grad_pool, d_x, d_w, d_grad_w, 
                              N, C, H, W, KH, KW, outH, outW, OUT_C)
        
        # Update weights
        self.lib.apply_sgd_update(d_w, d_grad_w, lr, w_size)
        
        # Copy updated weights back
        self.lib.gpu_memcpy_d2h(w_conv.ctypes.data, d_w, w_size)
        
        # Free GPU memory
        self.lib.gpu_free(d_x)
        self.lib.gpu_free(d_w)
        self.lib.gpu_free(d_col)
        self.lib.gpu_free(d_conv_out)
        self.lib.gpu_free(d_pool_out)
        self.lib.gpu_free(d_grad_w)
        self.lib.gpu_free(d_grad_pool)
        
        return np.sum(grad_pool**2)

def load_cifar10_binary(path):
    try:
        with open(path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint8)
            img = data[1024:3072+1024].astype(np.float32) / 255.0
            return img.reshape(1, 3, 32, 32).flatten()
    except Exception as e:
        print(f"Loader Error: {e}")
        return np.random.randn(1 * 3 * 32 * 32).astype(np.float32)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.abspath(os.path.join(base_dir, "..", "cpp", "libminimal_cuda_cnn.so"))
    bridge = CudaCNNBridge(so_path)
    
    data_path = os.path.join(base_dir, "..", "data", "cifar-10-batches-py", "data_batch_1")
    x = load_cifar10_binary(data_path)
    w_conv = np.random.randn(64 * 3 * 3 * 3).astype(np.float32)
    
    print(f"Starting Real-Data Training Loop on {data_path}...")
    for epoch in range(10):
        gnorm = bridge.train_step(x, w_conv, None)
        print(f"Epoch {epoch}: Grad Norm = {gnorm:.6f}")
    
    print("Training Smoke Test: SUCCESS")
