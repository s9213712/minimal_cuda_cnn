#include <cuda_runtime.h>
#include <cstdlib>

extern "C" {
    void* gpu_malloc(size_t size) {
        void* ptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }

    void gpu_free(void* ptr) {
        cudaFree(ptr);
    }

    void gpu_memcpy_h2d(void* dst, const void* src, size_t size) {
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }

    void gpu_memcpy_d2h(void* dst, const void* src, size_t size) {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }

    void gpu_memset(void* dst, int value, size_t size) {
        cudaMemset(dst, value, size);
    }
}
