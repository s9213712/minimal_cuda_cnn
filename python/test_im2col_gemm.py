import ctypes
import numpy as np
from ctypes import c_void_p, c_int
import pickle

so = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
lib = ctypes.CDLL(so)
lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_malloc.restype = c_void_p
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.im2col_forward.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]

# Load CIFAR image
with open("minimal_cuda_cnn/data/cifar-10-batches-py/data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"][:1].astype(np.float32) / 255.0
x = x.reshape(1, 3, 32, 32)

BATCH, C, H, W = 1, 3, 32, 32
KH, KW, OC = 3, 3, 32
outH, outW = H - KH + 1, W - KW + 1

# im2col
d_x = lib.gpu_malloc(BATCH * C * H * W * 4)
lib.gpu_memcpy_h2d(d_x, x.ctypes.data, BATCH * C * H * W * 4)

col_size = C * KH * KW * BATCH * outH * outW
d_col = lib.gpu_malloc(col_size * 4)
lib.im2col_forward(d_x, d_col, BATCH, C, H, W, KH, KW, outH, outW)

h_col = np.zeros((C * KH * KW, BATCH * outH * outW), dtype=np.float32)
lib.gpu_memcpy_d2h(h_col.ctypes.data, d_col, col_size * 4)

print(f"im2col output: shape={h_col.shape}")
print(f"col[0,:10]: {h_col[0,:10]}")  # First channel patch at first position
print(f"col[0,0]: {h_col[0,0]}, col[0,1]: {h_col[0,1]}")

# Random weights
np.random.seed(42)
w = np.random.randn(OC * C * KH * KW).astype(np.float32) * 0.05
print(f"w: min={w.min():.4f}, max={w.max():.4f}")

d_w = lib.gpu_malloc(OC * C * KH * KW * 4)
lib.gpu_memcpy_h2d(d_w, w.ctypes.data, OC * C * KH * KW * 4)

# GEMM: (OC, K) @ (K, N) = (OC, N)
M, K, N = OC, C * KH * KW, BATCH * outH * outW
print(f"\nGEMM: M={M}, K={K}, N={N}")

d_out = lib.gpu_malloc(M * N * 4)
lib.gemm_forward(d_w, d_col, d_out, M, N, K)

h_out = np.zeros((M, N), dtype=np.float32)
lib.gpu_memcpy_d2h(h_out.ctypes.data, d_out, M * N * 4)

print(f"gemm output: min={h_out.min():.6f}, max={h_out.max():.6f}")
print(f"h_out[0,:10]: {h_out[0,:10]}")
print(f"h_out[1,:10]: {h_out[1,:10]}")

lib.gpu_free(d_x)
lib.gpu_free(d_col)
lib.gpu_free(d_w)
lib.gpu_free(d_out)
