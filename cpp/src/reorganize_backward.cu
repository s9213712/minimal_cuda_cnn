// Reverse reorganize: (N, OC, H, W) NCHW -> (OC, N, H, W) CNHW
// Correct mathematical inverse: grad_input[c, n, h, w] = grad_output[n, c, h, w]
// Each output element maps to exactly one input element (no accumulation needed)
__global__ void reorganize_backward_kernel(const float* grad_output, float* grad_input, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    
    // Interpret idx as NCHW coordinates
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);
    
    // grad_output[n, c, h, w] -> grad_input[c, n, h, w]
    // grad_input linear = c*N*H*W + n*H*W + h*W + w
    int in_idx = c * (N * H * W) + n * H * W + h * W + w;
    grad_input[in_idx] = grad_output[idx];
}

extern "C" void reorganize_backward(float* d_grad_output, float* d_grad_input, int N, int C, int H, int W) {
    int total = N * C * H * W;
    int tpb = 256;
    reorganize_backward_kernel<<<(total + tpb - 1) / tpb, tpb>>>(d_grad_output, d_grad_input, N, C, H, W);
    cudaDeviceSynchronize();
}
