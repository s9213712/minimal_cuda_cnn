#!/usr/bin/env python3
"""Debug GEMM with OC=32"""
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
        self.lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
        self.lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]

    def malloc(self, size): return self.lib.gpu_malloc(size)
    def free(self, ptr): self.lib.gpu_free(ptr)
    def h2d(self, dst, src, size): self.lib.gpu_memcpy_h2d(dst, src, size)
    def d2h(self, dst, src, size): self.lib.gpu_memcpy_d2h(dst, src, size)


def test():
    so_path = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
    data_root = "minimal_cuda_cnn/data/cifar-10-batches-py"
    
    bridge = Bridge(so_path)
    
    with open(f"{data_root}/data_batch_1", 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        x = batch[b'data'][:8].astype(np.float32) / 255.0
    x = x.reshape(-1, 3, 32, 32)
    
    BATCH, C, H, W = 8, 3, 32, 32
    KH, KW, OC = 3, 3, 32
    outH, outW = H - KH + 1, W - KW + 1
    
    print(f"Config: BATCH={BATCH}, C={C}, H={H}, W={W}, OC={OC}, outH={outH}, outW={outW}")
    
    # Check im2col output
    np.random.seed(42)
    w = np.random.randn(OC * C * KH * KW).astype(np.float32)
    
    print(f"Weights: shape=({OC*C*KH*KW}), min={w.min():.4f}, max={w.max():.4f}")
    
    d_x = bridge.malloc(BATCH * C * H * W * 4)
    d_w = bridge.malloc(OC * C * KH * KW * 4)
    bridge.h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)
    bridge.h2d(d_w, w.ctypes.data, OC * C * KH * KW * 4)
    
    col_size = C * KH * KW * BATCH * outH * outW
    d_col = bridge.malloc(col_size * 4)
    bridge.lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)
    
    h_col = np.zeros((C * KH * KW, BATCH * outH * outW), dtype=np.float32)
    bridge.d2h(h_col.ctypes.data, d_col, col_size * 4)
    print(f"im2col output: shape={h_col.shape}, min={h_col.min():.4f}, max={h_col.max():.4f}")
    
    # GEMM: d_w @ d_col -> d_out
    # d_w: (OC, C*KH*KW) = (32, 27)
    # d_col: (C*KH*KW, BATCH*outH*outW) = (27, 8*30*30) = (27, 7200)
    # d_out: (OC, BATCH*outH*outW) = (32, 7200)
    M, K, N = OC, C * KH * KW, BATCH * outH * outW
    print(f"\nGEMM: M={M} (OC), K={K} (C*KH*KW), N={N} (BATCH*outH*outW)")
    
    d_out = bridge.malloc(M * N * 4)
    bridge.lib.gemm_forward(d_w, d_col, d_out, M, N, K)
    
    h_out = np.zeros((M, N), dtype=np.float32)
    bridge.d2h(h_out.ctypes.data, d_out, M * N * 4)
    print(f"GEMM output: shape={h_out.shape}, min={h_out.min():.4f}, max={h_out.max():.4f}")
    print(f"GEMM output[0,0:10]: {h_out[0,0:10]}")
    
    bridge.free(d_x)
    bridge.free(d_w)
    bridge.free(d_col)
    bridge.free(d_out)
    print("Done!")


if __name__ == '__main__':
    test()
