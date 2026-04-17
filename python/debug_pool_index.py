#!/usr/bin/env python3
"""Debug maxpool index calculation"""
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
        self.lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]

    def malloc(self, size): return self.lib.gpu_malloc(size)
    def free(self, ptr): self.lib.gpu_free(ptr)
    def h2d(self, dst, src, size): self.lib.gpu_memcpy_h2d(dst, src, size)
    def d2h(self, dst, src, size): self.lib.gpu_memcpy_d2h(dst, src, size)


def test():
    so_path = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
    
    bridge = Bridge(so_path)
    
    # Create simple test input: (1, 1, 4, 4)
    # Expected output: (1, 1, 2, 2)
    # Layout: ((n*c + c) * h + h) * w + w
    input_data = np.array([
        [[[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 16]]]
    ], dtype=np.float32)
    
    print("Input (NCHW):")
    print(input_data[0, 0])
    print(f"Flat: {input_data.flatten()}")
    
    # Check what index maps to what
    N, C, H, W = 1, 1, 4, 4
    print("\nIndex mapping (expected):")
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    flat_idx = ((n * C + c) * H + h) * W + w
                    print(f"  N={n}, C={c}, H={h}, W={w} -> flat_idx={flat_idx}, val={input_data.flatten()[flat_idx]}")
    
    d_input = bridge.malloc(N * C * H * W * 4)
    bridge.h2d(d_input, input_data.ctypes.data, N * C * H * W * 4)
    
    d_output = bridge.malloc(N * C * 2 * 2 * 4)
    bridge.lib.apply_maxpool(d_input, d_output, N, C, H, W)
    
    h_output = np.zeros((1, 1, 2, 2), dtype=np.float32)
    bridge.d2h(h_output.ctypes.data, d_output, N * C * 2 * 2 * 4)
    print("\nOutput:")
    print(h_output[0, 0])
    
    bridge.free(d_input)
    bridge.free(d_output)
    print("Done!")


if __name__ == '__main__':
    test()
