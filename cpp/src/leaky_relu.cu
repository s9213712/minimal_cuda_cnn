#include "cuda_check.h"

// Leaky ReLU forward kernel (CNHW layout: C,N,H,W flatten)
__global__ void leaky_relu_forward_kernel(float* data, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (data[idx] > 0) ? data[idx] : alpha * data[idx];
    }
}

// Leaky ReLU backward kernel (CNHW layout)
__global__ void leaky_relu_backward_kernel(float* data, float* grad, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = (data[idx] > 0) ? grad[idx] : alpha * grad[idx];
    }
}

// Leaky ReLU forward kernel (NCHW layout: N,C,H,W flatten) - for use after reorganize
__global__ void leaky_relu_forward_nchw_kernel(float* data, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (data[idx] > 0) ? data[idx] : alpha * data[idx];
    }
}

// Leaky ReLU backward kernel (NCHW layout) - data and grad both in NCHW format
// data: forward output (N,C,H,W), grad: gradient (N,C,H,W)
__global__ void leaky_relu_backward_nchw_kernel(float* data, float* grad, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = (data[idx] > 0) ? grad[idx] : alpha * grad[idx];
    }
}

extern "C" void leaky_relu_forward(float* data, float alpha, int size) {
    int tpb = 256;
    leaky_relu_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(data, alpha, size);
    CUDA_KERNEL_CHECK();
}

extern "C" void leaky_relu_backward(float* data, float* grad, float alpha, int size) {
    int tpb = 256;
    leaky_relu_backward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(data, grad, alpha, size);
    CUDA_KERNEL_CHECK();
}

// NCHW layout versions (data and grad both in N,C,H,W flatten format)
extern "C" void leaky_relu_forward_nchw(float* data, float alpha, int size) {
    int tpb = 256;
    leaky_relu_forward_nchw_kernel<<<(size + tpb - 1) / tpb, tpb>>>(data, alpha, size);
    CUDA_KERNEL_CHECK();
}

extern "C" void leaky_relu_backward_nchw(float* data, float* grad, float alpha, int size) {
    int tpb = 256;
    leaky_relu_backward_nchw_kernel<<<(size + tpb - 1) / tpb, tpb>>>(data, grad, alpha, size);
    CUDA_KERNEL_CHECK();
}
