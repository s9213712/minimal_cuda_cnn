#pragma once
#include "cuda_check.h"
#include "network.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>

class DenseLayer : public Layer {
    int in_features, out_features;
    float* d_weights;
    float* d_bias;
public:
    DenseLayer(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
        CUDA_CHECK(cudaMalloc(&d_weights, in_f * out_f * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias, out_f * sizeof(float)));
    }
    ~DenseLayer() {
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_bias));
    }
    CudaTensor* forward(CudaTensor* input) override;
};
