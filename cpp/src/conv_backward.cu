__global__ void conv_backward_kernel(float* grad_out, float* input, float* weights, float* grad_weights, 
                                       int N, int C, int H, int W, int KH, int KW, int outH, int outW, int OUT_C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = OUT_C * C * KH * KW * N * outH * outW;
    if (idx >= total_elements) return;

    int row = idx / (N * outH * outW); // OUT_C * C * KH * KW
    int col = idx % (N * outH * outW); // N * outH * outW

    int oc = row / (C * KH * KW);
    int c = (row / (KH * KW)) % C;
    int kh = (row / KW) % KH;
    int kw = row % KW;

    int n = col / (outH * outW);
    int ow = col % outW;
    int oh = (col / outW) % outH;

    int h_in = oh + kh;
    int w_in = ow + kw;

    // Weight gradient = Input * Gradient_Out
    atomicAdd(&grad_weights[oc * (C * KH * KW) + (c * KH * KW + kh * KW + kw)], 
              input[((n * C + c) * H + h_in) * W + w_in] * grad_out[oc * (N * outH * outW) + col]);
}

// Input gradient: d_input[n,c,h,w] += sum over oc,kh,kw of: d_conv2_grad[oc,n,oh,ow] * weights[oc,c,kh,kw]
// where oh = h - kh, ow = w - kw (only if oh,ow are valid)
__global__ void conv_backward_input_kernel(float* grad_out, float* weights, float* grad_input,
                                            int N, int C, int H, int W, int KH, int KW, int outH, int outW, int OUT_C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;  // one thread per input element
    if (idx >= total_elements) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    float sum = 0.0f;
    for (int oc = 0; oc < OUT_C; oc++) {
        for (int kh = 0; kh < KH; kh++) {
            int oh = h - kh;
            if (oh < 0 || oh >= outH) continue;
            for (int kw = 0; kw < KW; kw++) {
                int ow = w - kw;
                if (ow < 0 || ow >= outW) continue;
                // grad_out[oc, n, oh, ow]
                // weights[oc, c, kh, kw]
                int g_out_idx = oc * (N * outH * outW) + n * (outH * outW) + oh * outW + ow;
                int w_idx = oc * (C * KH * KW) + c * (KH * KW) + kh * KW + kw;
                sum += grad_out[g_out_idx] * weights[w_idx];
            }
        }
    }
    grad_input[idx] = sum;
}

extern "C" {
    void conv_backward(float* grad_out, float* input, float* weights, float* grad_weights, float* grad_input,
                       int N, int C, int H, int W, int KH, int KW, int outH, int outW, int OUT_C) {
        // Weight gradient
        int total_w = OUT_C * C * KH * KW * N * outH * outW;
        int tpb = 256;
        conv_backward_kernel<<<(total_w + tpb - 1) / tpb, tpb>>>(grad_out, input, weights, grad_weights, N, C, H, W, KH, KW, outH, outW, OUT_C);
        
        // Input gradient
        int total_in = N * C * H * W;
        conv_backward_input_kernel<<<(total_in + tpb - 1) / tpb, tpb>>>(grad_out, weights, grad_input, N, C, H, W, KH, KW, outH, outW, OUT_C);
        
        cudaDeviceSynchronize();
    }
}
