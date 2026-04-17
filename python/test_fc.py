#!/usr/bin/env python3
"""Test FC + Loss + Backward"""
import ctypes
import numpy as np
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
        self.lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        self.lib.dense_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        self.lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]
        self.lib.softmax_cross_entropy.argtypes = [c_void_p, c_void_p, c_int, c_int]
        self.lib.softmax_cross_entropy.restype = c_float

    def malloc(self, size): return self.lib.gpu_malloc(size)
    def free(self, ptr): self.lib.gpu_free(ptr)
    def h2d(self, dst, src, size): self.lib.gpu_memcpy_h2d(dst, src, size)
    def d2h(self, dst, src, size): self.lib.gpu_memcpy_d2h(dst, src, size)
    def memset(self, ptr, val, size): self.lib.gpu_memset(ptr, val, size)


def test():
    so_path = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
    print("=== FC + Softmax Test ===")
    
    bridge = Bridge(so_path)
    
    BATCH, IN_F, OUT_F = 8, 128, 10
    LR = 0.1
    
    # Random data
    x = np.random.randn(BATCH, IN_F).astype(np.float32)
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7])  # Labels
    
    # Init weights
    w = np.random.randn(OUT_F, IN_F).astype(np.float32) * 0.05
    b = np.zeros(OUT_F, dtype=np.float32)
    
    d_x = bridge.malloc(BATCH * IN_F * 4)
    d_w = bridge.malloc(OUT_F * IN_F * 4)
    d_b = bridge.malloc(OUT_F * 4)
    d_out = bridge.malloc(BATCH * OUT_F * 4)
    
    bridge.h2d(d_x, x.ctypes.data, BATCH * IN_F * 4)
    bridge.h2d(d_w, w.ctypes.data, OUT_F * IN_F * 4)
    bridge.h2d(d_b, b.ctypes.data, OUT_F * 4)
    
    print("Forward...")
    bridge.lib.dense_forward(d_x, d_w, d_b, d_out, BATCH, IN_F, OUT_F)
    
    # Get output
    h_out = np.zeros((BATCH, OUT_F), dtype=np.float32)
    bridge.d2h(h_out.ctypes.data, d_out, BATCH * OUT_F * 4)
    
    # Softmax
    h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
    probs = np.exp(h_out_shifted) / np.exp(h_out_shifted).sum(axis=1, keepdims=True)
    
    # Loss
    loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
    print(f"Loss: {loss:.4f}")
    
    # Accuracy
    pred = np.argmax(probs, axis=1)
    acc = np.mean(pred == y) * 100
    print(f"Initial Acc: {acc:.1f}%")
    
    # Backward
    print("Backward...")
    d_labels = bridge.malloc(BATCH * OUT_F * 4)
    labels_onehot = np.zeros((BATCH, OUT_F), dtype=np.float32)
    labels_onehot[np.arange(BATCH), y] = 1.0
    bridge.h2d(d_labels, labels_onehot.ctypes.data, BATCH * OUT_F * 4)
    
    d_loss = bridge.malloc(BATCH * OUT_F * 4)
    bridge.h2d(d_loss, (probs - labels_onehot).ctypes.data, BATCH * OUT_F * 4)
    
    d_grad_w = bridge.malloc(OUT_F * IN_F * 4)
    bridge.lib.dense_backward(d_loss, d_x, d_w, d_grad_w, BATCH, IN_F, OUT_F)
    
    # Update
    print("Update...")
    bridge.lib.apply_sgd_update(d_w, d_grad_w, LR, OUT_F * IN_F)
    
    # Test again
    bridge.lib.dense_forward(d_x, d_w, d_b, d_out, BATCH, IN_F, OUT_F)
    bridge.d2h(h_out.ctypes.data, d_out, BATCH * OUT_F * 4)
    
    h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
    probs = np.exp(h_out_shifted) / np.exp(h_out_shifted).sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(BATCH), y] + 1e-10))
    pred = np.argmax(probs, axis=1)
    acc = np.mean(pred == y) * 100
    print(f"After Update Acc: {acc:.1f}%")
    
    bridge.free(d_x)
    bridge.free(d_w)
    bridge.free(d_b)
    bridge.free(d_out)
    bridge.free(d_labels)
    bridge.free(d_loss)
    bridge.free(d_grad_w)
    
    print("SUCCESS!")


if __name__ == '__main__':
    test()
