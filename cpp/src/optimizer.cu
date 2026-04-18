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

__global__ void conv_update_fused_kernel(
    float* weights,
    float* grad,
    float* velocity,
    float lr,
    float momentum,
    float weight_decay,
    float clip_val,
    float normalizer,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / normalizer + weight_decay * weights[idx];
        g = fmaxf(-clip_val, fminf(clip_val, g));
        velocity[idx] = momentum * velocity[idx] - lr * g;
        weights[idx] += velocity[idx];
    }
}

__global__ void clip_inplace_kernel(float* values, float clip_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        values[idx] = fmaxf(-clip_val, fminf(clip_val, values[idx]));
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

    void conv_update_fused(
        float* d_weights,
        float* d_grad,
        float* d_velocity,
        float lr,
        float momentum,
        float weight_decay,
        float clip_val,
        float normalizer,
        int size
    ) {
        int tpb = 256;
        int bpg = (size + tpb - 1) / tpb;
        conv_update_fused_kernel<<<bpg, tpb>>>(
            d_weights, d_grad, d_velocity,
            lr, momentum, weight_decay, clip_val, normalizer, size
        );
        CUDA_KERNEL_CHECK();
    }

    void clip_inplace(float* d_values, float clip_val, int size) {
        int tpb = 256;
        int bpg = (size + tpb - 1) / tpb;
        clip_inplace_kernel<<<bpg, tpb>>>(d_values, clip_val, size);
        CUDA_KERNEL_CHECK();
    }
}
