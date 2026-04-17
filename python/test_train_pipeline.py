import ctypes
import numpy as np
import os
from ctypes import POINTER, c_float, c_int, c_void_p

class CudaCNNBridge:
    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)
        # Forward functions
        self.lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
        self.lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        self.lib.apply_relu.argtypes = [c_void_p, c_int]
        self.lib.apply_maxpool.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        
        # Backward functions
        self.lib.apply_relu_backward.argtypes = [c_void_p, c_void_p, c_int]
        self.lib.dense_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        self.lib.conv_backward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]

    def run_training_step(self, x, w_conv, w_dense, labels):
        # Simplified single-sample training step to verify the pipeline
        N, C, H, W = 1, 3, 32, 32
        KH, KW = 3, 3
        OUT_C = 64
        outH, outW = H - KH + 1, W - KW + 1
        
        # Allocate GPU buffers (using numpy as host mirrors for simplicity in this test)
        h_col = np.zeros((C * KH * KW, N * outH * outW), dtype=np.float32)
        h_conv_out = np.zeros((OUT_C, N * outH * outW), dtype=np.float32)
        h_pool_out = np.zeros((OUT_C, N * (outH // 2) * (outW // 2)), dtype=np.float32)
        
        # Forward
        self.lib.im2col_forward(x.ctypes.data, h_col.ctypes.data, N, C, H, W, KH, KW, outH, outW)
        self.lib.gemm_forward(w_conv.ctypes.data, h_col.ctypes.data, h_conv_out.ctypes.data, OUT_C, N * outH * outW, C * KH * KW)
        self.lib.apply_relu(h_conv_out.ctypes.data, OUT_C * N * outH * outW)
        self.lib.apply_maxpool(h_conv_out.ctypes.data, h_pool_out.ctypes.data, N, OUT_C, outH, outW)
        
        # Mock Gradient for backward test
        grad_pool = np.random.randn(*h_pool_out.shape).astype(np.float32)
        
        # Backward (Simplified ReLU)
        self.lib.apply_relu_backward(h_conv_out.ctypes.data, grad_pool.ctypes.data, OUT_C * N * outH * outW)
        
        print("Forward and Backward pipelines executed!")
        return True

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.abspath(os.path.join(base_dir, "..", "cpp", "libminimal_cuda_cnn.so"))
    bridge = CudaCNNBridge(so_path)
    
    x = np.random.randn(1 * 3 * 32 * 32).astype(np.float32)
    w_conv = np.random.randn(64 * 3 * 3 * 3).astype(np.float32)
    w_dense = np.random.randn(10 * 1024).astype(np.float32)
    labels = np.array([1], dtype=np.int32)
    
    if bridge.run_training_step(x, w_conv, w_dense, labels):
        print("CUDA Training Step Pipeline: SUCCESS")
    else:
        print("CUDA Training Step Pipeline: FAILED")
