#!/usr/bin/env python3
"""Debug dense_forward"""
import ctypes
import numpy as np
from ctypes import c_void_p, c_int

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
    
    bridge = Bridge(so_path)
    
    # Simple test: input=1, weights small
    BATCH, IN_F, OUT_F = 4, 128, 10
    
    x = np.ones((BATCH, IN_F), dtype=np.float32) * 0.1
    w = np.ones((OUT_F, IN_F), dtype=np.float32) * 0.01
    b = np.zeros(OUT_F, dtype=np.float32)
    
    print(f"Input: min={x.min():.4f}, max={x.max():.4f}")
    print(f"Weights: min={w.min():.4f}, max={w.max():.4f}")
    
    d_x = bridge.malloc(BATCH * IN_F * 4)
    d_w = bridge.malloc(OUT_F * IN_F * 4)
    d_b = bridge.malloc(OUT_F * 4)
    d_out = bridge.malloc(BATCH * OUT_F * 4)
    
    bridge.h2d(d_x, x.ctypes.data, BATCH * IN_F * 4)
    bridge.h2d(d_w, w.ctypes.data, OUT_F * IN_F * 4)
    bridge.h2d(d_b, b.ctypes.data, OUT_F * 4)
    
    bridge.lib.dense_forward(d_x, d_w, d_b, d_out, BATCH, IN_F, OUT_F)
    
    h_out = np.zeros((BATCH, OUT_F), dtype=np.float32)
    bridge.d2h(h_out.ctypes.data, d_out, BATCH * OUT_F * 4)
    
    print(f"Output: min={h_out.min():.4f}, max={h_out.max():.4f}")
    print(f"Output any NaN: {np.isnan(h_out).any()}")
    
    # Test with random data
    x2 = np.random.randn(BATCH, IN_F).astype(np.float32) * 0.01
    w2 = np.random.randn(OUT_F, IN_F).astype(np.float32) * 0.01
    
    bridge.h2d(d_x, x2.ctypes.data, BATCH * IN_F * 4)
    bridge.h2d(d_w, w2.ctypes.data, OUT_F * IN_F * 4)
    
    bridge.lib.dense_forward(d_x, d_w, d_b, d_out, BATCH, IN_F, OUT_F)
    
    bridge.d2h(h_out.ctypes.data, d_out, BATCH * OUT_F * 4)
    
    print(f"Output2: min={h_out.min():.4f}, max={h_out.max():.4f}")
    print(f"Output2 any NaN: {np.isnan(h_out).any()}")
    
    bridge.free(d_x)
    bridge.free(d_w)
    bridge.free(d_b)
    bridge.free(d_out)
    
    print("Done!")


if __name__ == '__main__':
    test()
