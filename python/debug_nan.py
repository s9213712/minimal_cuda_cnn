#!/usr/bin/env python3
"""Debug NaN in loss"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_int
import pickle

class Bridge:
    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)
        self.lib.gpu_malloc.argtypes = [ctypes.c_size_t]
        self.lib.gpu_malloc.restype = c_void_p
        self.lib.gpu_free.argtypes = [c_void_p]
        self.lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        self.lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        self.lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]

    def malloc(self, size): return self.lib.gpu_malloc(size)
    def free(self, ptr): self.lib.gpu_free(ptr)
    def h2d(self, dst, src, size): self.lib.gpu_memcpy_h2d(dst, src, size)
    def d2h(self, dst, src, size): self.lib.gpu_memcpy_d2h(dst, src, size)


def test():
    so_path = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
    data_root = "minimal_cuda_cnn/data/cifar-10-batches-py"
    
    bridge = Bridge(so_path)
    
    # Load a batch
    with open(f"{data_root}/data_batch_1", 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        x = batch[b'data'][:8].astype(np.float32) / 255.0
        y = np.array(batch[b'labels'][:8])
    
    x = x.reshape(-1, 3, 32, 32)
    FC_IN = 32 * 15 * 15
    
    # Init FC weights
    fc_w = np.random.randn(10 * FC_IN).astype(np.float32) * 0.01
    fc_b = np.zeros(10, dtype=np.float32)
    
    # Flatten input
    x_flat = x.reshape(8, -1)
    
    d_fc_in = bridge.malloc(8 * FC_IN * 4)
    d_fc_w = bridge.malloc(10 * FC_IN * 4)
    d_fc_b = bridge.malloc(10 * 4)
    d_fc_out = bridge.malloc(8 * 10 * 4)
    
    bridge.h2d(d_fc_in, x_flat.ctypes.data, 8 * FC_IN * 4)
    bridge.h2d(d_fc_w, fc_w.ctypes.data, 10 * FC_IN * 4)
    bridge.h2d(d_fc_b, fc_b.ctypes.data, 10 * 4)
    
    bridge.lib.dense_forward(d_fc_in, d_fc_w, d_fc_b, d_fc_out, 8, FC_IN, 10)
    
    h_out = np.zeros((8, 10), dtype=np.float32)
    bridge.d2h(h_out.ctypes.data, d_fc_out, 8 * 10 * 4)
    
    print("h_out stats:")
    print(f"  min: {h_out.min():.4f}, max: {h_out.max():.4f}")
    print(f"  any NaN: {np.isnan(h_out).any()}")
    print(f"  any Inf: {np.isinf(h_out).any()}")
    
    # Softmax
    h_out_shifted = h_out - h_out.max(axis=1, keepdims=True)
    print(f"h_out_shifted min: {h_out_shifted.min():.4f}")
    
    exp_out = np.exp(h_out_shifted)
    print(f"exp_out min: {exp_out.min():.4f}, max: {exp_out.max():.4f}")
    
    probs = exp_out / exp_out.sum(axis=1, keepdims=True)
    print(f"probs min: {probs.min():.4f}, max: {probs.max():.4f}")
    print(f"probs any NaN: {np.isnan(probs).any()}")
    
    # Loss
    loss = -np.mean(np.log(probs[np.arange(8), y] + 1e-10))
    print(f"Loss: {loss:.4f}")
    
    bridge.free(d_fc_in)
    bridge.free(d_fc_w)
    bridge.free(d_fc_b)
    bridge.free(d_fc_out)
    print("Done!")


if __name__ == '__main__':
    test()
