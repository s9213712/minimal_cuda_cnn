#!/usr/bin/env python3
import ctypes, os
so = '/mnt/c/Users/user/.openclaw/workspace/NN/minimal_cuda_cnn/cpp/libminimal_cuda_cnn.so'
print(f'File exists: {os.path.exists(so)}')
lib = ctypes.CDLL(so)
print('Library loaded OK')
