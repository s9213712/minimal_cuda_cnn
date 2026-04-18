#include "cuda_check.h"

__global__ void sgd_update_kernel(float* weights, const float* grad, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad[idx];
    }
}

__global__ void momentum_update_kernel(
    float* weights,
    const float* grad,
    float* velocity,
    float lr,
    float momentum,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        velocity[idx] = momentum * velocity[idx] - lr * grad[idx];
        weights[idx] += velocity[idx];
    }
}

extern "C" {
    void apply_sgd_update(float* d_weights, float* d_grad, float lr, int size) {
        int tpb = 256;
        int bpg = (size + tpb - 1) / tpb;
        sgd_update_kernel<<<bpg, tpb>>>(d_weights, d_grad, lr, size);
        CUDA_KERNEL_CHECK();
    }

    void apply_momentum_update(
        float* d_weights,
        float* d_grad,
        float* d_velocity,
        float lr,
        float momentum,
        int size
    ) {
        int tpb = 256;
        int bpg = (size + tpb - 1) / tpb;
        momentum_update_kernel<<<bpg, tpb>>>(d_weights, d_grad, d_velocity, lr, momentum, size);
        CUDA_KERNEL_CHECK();
    }
}
