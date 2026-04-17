#!/usr/bin/env python3
"""Compare CUDA conv_backward with PyTorch conv2d backward"""
import ctypes
import numpy as np
import torch
import torch.nn as nn
import os

class ConvBackwardVerifier:
    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)
        
        # Memory helpers
        self.lib.gpu_malloc.argtypes = [ctypes.c_size_t]
        self.lib.gpu_malloc.restype = ctypes.c_void_p
        self.lib.gpu_free.argtypes = [ctypes.c_void_p]
        self.lib.gpu_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.gpu_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.gpu_memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        
        # Conv backward
        self.lib.conv_backward.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        
        # Constants
        self.H2D = 2
        self.D2H = 1

    def cuda_conv_backward(self, input_data, weights, grad_out, N, C, H, W, KH, KW, OUT_C):
        outH = H - KH + 1
        outW = W - KW + 1
        
        input_size = N * C * H * W * 4
        weight_size = OUT_C * C * KH * KW * 4
        grad_out_size = OUT_C * N * outH * outW * 4
        
        # Allocate GPU memory
        d_input = self.lib.gpu_malloc(input_size)
        d_weights = self.lib.gpu_malloc(weight_size)
        d_grad_out = self.lib.gpu_malloc(grad_out_size)
        d_grad_weights = self.lib.gpu_malloc(weight_size)
        
        # Copy to GPU
        self.lib.gpu_memcpy_h2d(d_input, input_data.ctypes.data, input_size)
        self.lib.gpu_memcpy_h2d(d_weights, weights.ctypes.data, weight_size)
        self.lib.gpu_memcpy_h2d(d_grad_out, grad_out.ctypes.data, grad_out_size)
        self.lib.gpu_memset(d_grad_weights, 0, weight_size)
        
        # Call conv_backward
        self.lib.conv_backward(d_grad_out, d_input, d_weights, d_grad_weights, 
                              N, C, H, W, KH, KW, outH, outW, OUT_C)
        
        # Copy result back
        grad_weights = np.zeros(OUT_C * C * KH * KW, dtype=np.float32)
        self.lib.gpu_memcpy_d2h(grad_weights.ctypes.data, d_grad_weights, weight_size)
        
        # Free GPU memory
        self.lib.gpu_free(d_input)
        self.lib.gpu_free(d_weights)
        self.lib.gpu_free(d_grad_out)
        self.lib.gpu_free(d_grad_weights)
        
        return grad_weights

def test_conv_backward():
    so_path = "/mnt/c/Users/user/.openclaw/workspace/NN/minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
    
    # Test parameters
    N, C, H, W = 1, 3, 8, 8
    KH, KW = 3, 3
    OUT_C = 16
    outH = H - KH + 1
    outW = W - KW + 1
    
    np.random.seed(42)
    input_data = np.random.randn(N * C * H * W).astype(np.float32)
    weights = np.random.randn(OUT_C * C * KH * KW).astype(np.float32)
    grad_out = np.random.randn(OUT_C * N * outH * outW).astype(np.float32)
    
    # PyTorch reference
    conv = nn.Conv2d(C, OUT_C, KH, bias=None)
    conv.weight.data = torch.from_numpy(weights.reshape(OUT_C, C, KH, KW))
    
    input_t = torch.from_numpy(input_data.reshape(N, C, H, W)).clone()
    input_t.requires_grad = True
    
    out = conv(input_t)
    out.backward(torch.from_numpy(grad_out.reshape(N, OUT_C, outH, outW)))
    
    pt_grad_w = conv.weight.grad.data.numpy().flatten()
    
    # CUDA result
    verifier = ConvBackwardVerifier(so_path)
    cuda_grad_w = verifier.cuda_conv_backward(input_data, weights, grad_out, N, C, H, W, KH, KW, OUT_C)
    
    print("=== Conv Backward Comparison ===")
    print(f"Shape: Input={input_data.shape}, Weights={weights.shape}, GradOut={grad_out.shape}")
    print(f"PyTorch grad_W[0:10]: {pt_grad_w[:10]}")
    print(f"CUDA    grad_W[0:10]: {cuda_grad_w[:10]}")
    
    diff = np.abs(pt_grad_w - cuda_grad_w)
    print(f"\nDifference stats:")
    print(f"  Max: {diff.max():.6f}")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Relative: {diff.mean() / np.abs(pt_grad_w).mean():.6f}")
    
    # Check layout
    print("\n=== Layout Check ===")
    print(f"PyTorch stride: {conv.weight.grad.stride()}")
    
    if diff.mean() < 1e-3:
        print("\n✓ Gradients MATCH!")
    else:
        print("\n✗ Gradients MISMATCH!")
        print("Possible issues:")
        print("  1. Weight memory layout (OCxCxKHxKW vs flattened)")
        print("  2. Convolution index calculation")

if __name__ == "__main__":
    test_conv_backward()
