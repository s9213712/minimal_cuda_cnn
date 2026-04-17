#pragma once
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
        cudaMalloc(&d_weights, in_f * out_f * sizeof(float));
        cudaMalloc(&d_bias, out_f * sizeof(float));
    }
    ~DenseLayer() {
        cudaFree(d_weights);
        cudaFree(d_bias);
    }
    CudaTensor* forward(CudaTensor* input) override;
};
