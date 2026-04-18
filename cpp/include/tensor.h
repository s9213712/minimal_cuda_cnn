#pragma once
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// 4D Tensor 結構 (Batch, Channel, Height, Width)
struct CudaTensor {
    float* data;
    int n, c, h, w;
    size_t total_size;

    CudaTensor(int n, int c, int h, int w) : n(n), c(c), h(h), w(w) {
        total_size = (size_t)n * c * h * w * sizeof(float);
        CUDA_CHECK(cudaMalloc(&data, total_size));
    }

    ~CudaTensor() {
        CUDA_CHECK(cudaFree(data));
    }

    void copy_from_host(const float* host_ptr) {
        CUDA_CHECK(cudaMemcpy(data, host_ptr, total_size, cudaMemcpyHostToDevice));
    }

    void copy_to_host(float* host_ptr) {
        CUDA_CHECK(cudaMemcpy(host_ptr, data, total_size, cudaMemcpyDeviceToHost));
    }
};

// GPU 狀態監測函數 (由 C++ 呼叫)
extern "C" void check_gpu_status();
