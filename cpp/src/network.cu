#include "network.h"
#include "tensor.h"
#include "cuda_check.h"
#include <iostream>
#include <cuda_runtime.h>

extern "C" {
    void apply_relu(float* d_data, int size);
    void apply_maxpool(float* d_input, float* d_output, int n, int c, int h, int w);
    void im2col_forward(float* d_input, float* d_output, int N, int C, int H, int W, int KH, int KW, int outH, int outW);
    void gemm_forward(float* d_A, float* d_B, float* d_C, int M, int N, int K);
}

CudaTensor* ConvLayer::forward(CudaTensor* input) {
    int outH = input->h - kh + 1;
    int outW = input->w - kw + 1;
    CudaTensor* output = new CudaTensor(input->n, out_c, outH, outW);

    float* d_col;
    int col_rows = input->c * kh * kw;
    int col_cols = input->n * outH * outW;
    CUDA_CHECK(cudaMalloc(&d_col, col_rows * col_cols * sizeof(float)));

    // 1. Image -> Column (GPU)
    im2col_forward(input->data, d_col, input->n, input->c, input->h, input->w, kh, kw, outH, outW);

    // 2. GEMM: Output = Weights * Col (GPU)
    gemm_forward(d_weights, d_col, output->data, out_c, col_cols, col_rows);

    CUDA_CHECK(cudaFree(d_col));
    return output;
}

CudaTensor* ReLULayer::forward(CudaTensor* input) {
    CudaTensor* output = new CudaTensor(input->n, input->c, input->h, input->w);
    float* d_input = input->data;
    float* d_output = output->data;
    int size = input->n * input->c * input->h * input->w;

    // We need a kernel that does: output[i] = fmaxf(0, input[i])
    // Since apply_relu is in-place, let's implement a copy-relu or just use a lambda/small kernel
    // For now, let's just perform a cudaMemcpy then apply_relu for correctness
    CUDA_CHECK(cudaMemcpy(d_output, d_input, size * sizeof(float), cudaMemcpyDeviceToDevice));
    apply_relu(d_output, size);
    
    return output;
}

CudaTensor* MaxPoolLayer::forward(CudaTensor* input) {
    CudaTensor* output = new CudaTensor(input->n, input->c, input->h / 2, input->w / 2);
    apply_maxpool(input->data, output->data, input->n, input->c, input->h, input->w);
    return output;
}
