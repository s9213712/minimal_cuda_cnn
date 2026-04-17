#pragma once
#include "tensor.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>

// 層類基類
class Layer {
public:
    virtual ~Layer() {}
    virtual CudaTensor* forward(CudaTensor* input) = 0;
};

// 卷積層
class ConvLayer : public Layer {
    int in_c, out_c, kh, kw;
    float* d_weights;
    float* d_grad_weights;
public:
    ConvLayer(int in_c, int out_c, int kh, int kw) : in_c(in_c), out_c(out_c), kh(kh), kw(kw) {
        size_t size = (size_t)out_c * in_c * kh * kw * sizeof(float);
        cudaMalloc(&d_weights, size);
        cudaMalloc(&d_grad_weights, size);
        cudaMemset(d_grad_weights, 0, size);
    }
    ~ConvLayer() { 
        cudaFree(d_weights); 
        cudaFree(d_grad_weights);
    }
    
    void set_weights(const float* weights) {
        size_t size = (size_t)out_c * in_c * kh * kw * sizeof(float);
        cudaMemcpy(d_weights, weights, size, cudaMemcpyHostToDevice);
    }
    
    float* get_weights() { return d_weights; }
    float* get_grad_weights() { return d_grad_weights; }
    void clear_grads() {
        size_t size = (size_t)out_c * in_c * kh * kw * sizeof(float);
        cudaMemset(d_grad_weights, 0, size);
    }
    
    CudaTensor* forward(CudaTensor* input) override;
};

// 激活層 (ReLU)
class ReLULayer : public Layer {
public:
    CudaTensor* forward(CudaTensor* input) override;
};

// 池化層 (MaxPool)
class MaxPoolLayer : public Layer {
public:
    CudaTensor* forward(CudaTensor* input) override;
};
