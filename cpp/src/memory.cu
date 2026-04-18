#include <cuda_runtime.h>
#include <cstdlib>
#include "cuda_check.h"

extern "C" {
    void* gpu_malloc(size_t size) {
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        return ptr;
    }

    void gpu_free(void* ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }

    void gpu_memcpy_h2d(void* dst, const void* src, size_t size) {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    }

    void gpu_memcpy_d2h(void* dst, const void* src, size_t size) {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }

    void gpu_memset(void* dst, int value, size_t size) {
        CUDA_CHECK(cudaMemset(dst, value, size));
    }
}
