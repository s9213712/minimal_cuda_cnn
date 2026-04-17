import ctypes
import numpy as np
from ctypes import c_void_p, c_int

so = "minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so"
lib = ctypes.CDLL(so)
lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_malloc.restype = c_void_p
lib.gpu_free.argtypes = [c_void_p]
lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]

# Test: A(2,3) @ B(3,4) = C(2,4)
A = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
B = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=np.float32)
C = np.zeros((2,4), dtype=np.float32)

dA = lib.gpu_malloc(2*3*4)
dB = lib.gpu_malloc(3*4*4)
dC = lib.gpu_malloc(2*4*4)
lib.gpu_memcpy_h2d(dA, A.ctypes.data, 2*3*4)
lib.gpu_memcpy_h2d(dB, B.ctypes.data, 3*4*4)
lib.gemm_forward(dA, dB, dC, 2, 4, 3)
lib.gpu_memcpy_d2h(C.ctypes.data, dC, 2*4*4)
print("A:", A.tolist())
print("C:", C.tolist())
print("Expected:", A.tolist())
lib.gpu_free(dA)
lib.gpu_free(dB)
lib.gpu_free(dC)
