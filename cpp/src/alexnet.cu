#include "network.h"
#include "tensor.h"
#include "cuda_check.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

extern "C" {
    void im2col_forward(float* d_input, float* d_output, int N, int C, int H, int W, int KH, int KW, int outH, int outW);
    void gemm_forward(float* d_A, float* d_B, float* d_C, int M, int N, int K);
    void apply_relu(float* d_data, int size);
    void apply_maxpool(float* d_input, float* d_output, int n, int c, int h, int w);
}

class AlexNet {
    // This is a simplified version for CIFAR-10
    struct LayerParams {
        int in_c, out_c, kh, kw;
    };
    std::vector<LayerParams> conv_params;
    std::vector<float*> weights;

public:
    AlexNet() {
        // Simplified AlexNet-like params for 32x32
        conv_params = {
            {3, 64, 3, 3},   // L1
            {64, 192, 3, 3}, // L2
            {192, 384, 3, 3},// L3
            {384, 256, 3, 3},// L4
            {256, 256, 3, 3} // L5
        };
        for (auto& p : conv_params) {
            float* w;
            CUDA_CHECK(cudaMalloc(&w, p.out_c * p.in_c * p.kh * p.kw * sizeof(float)));
            weights.push_back(w);
        }
    }

    ~AlexNet() {
        for (auto w : weights) CUDA_CHECK(cudaFree(w));
    }

    float* forward(float* d_input, int N, int C, int H, int W) {
        float* current_data = d_input;
        int curr_c = C, curr_h = H, curr_w = W;

        for (size_t i = 0; i < conv_params.size(); ++i) {
            auto& p = conv_params[i];
            int outH = curr_h - p.kh + 1; // Simplified: no padding for now
            int outW = curr_w - p.kw + 1;
            
            float* d_col;
            CUDA_CHECK(cudaMalloc(&d_col, p.in_c * p.kh * p.kw * N * outH * outW * sizeof(float)));
            float* d_conv_out;
            CUDA_CHECK(cudaMalloc(&d_conv_out, p.out_c * N * outH * outW * sizeof(float)));

            im2col_forward(current_data, d_col, N, curr_c, curr_h, curr_w, p.kh, p.kw, outH, outW);
            gemm_forward(weights[i], d_col, d_conv_out, p.out_c, N * outH * outW, p.in_c * p.kh * p.kw);
            apply_relu(d_conv_out, p.out_c * N * outH * outW);
            
            // MaxPool every few layers
            if (i == 0 || i == 1 || i == 3 || i == 4) {
                float* d_pool_out;
                CUDA_CHECK(cudaMalloc(&d_pool_out, p.out_c * N * (outH/2) * (outW/2) * sizeof(float)));
                apply_maxpool(d_conv_out, d_pool_out, N, p.out_c, outH, outW);
                
                if (current_data != d_input) CUDA_CHECK(cudaFree(current_data));
                CUDA_CHECK(cudaFree(d_conv_out));
                current_data = d_pool_out;
                curr_h /= 2; curr_w /= 2;
            } else {
                if (current_data != d_input) CUDA_CHECK(cudaFree(current_data));
                current_data = d_conv_out;
            }
            
            curr_c = p.out_c;
            curr_h = (i == 0 || i == 1 || i == 3 || i == 4) ? curr_h : outH;
            curr_w = (i == 0 || i == 1 || i == 3 || i == 4) ? curr_w : outW;
            CUDA_CHECK(cudaFree(d_col));
        }
        return current_data;
    }
};
