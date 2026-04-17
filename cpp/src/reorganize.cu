// Reorganize kernel: (OC, N_spatial) -> (N, OC, H, W)
__global__ void reorganize_kernel(const float* input, float* output, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);
    
    // Output idx = ((n * C + c) * H + h) * W + w
    // Input is (C, N*H*W), so channel c, spatial index = n*H*W + h*W + w
    int in_idx = c * (N * H * W) + n * H * W + h * W + w;
    
    output[idx] = input[in_idx];
}

// Host wrapper
extern "C" void reorganize_forward(const float* input, float* output, int N, int C, int H, int W) {
    dim3 block(256);
    dim3 grid((N * C * H * W + 255) / 256);
    reorganize_kernel<<<grid, block>>>(input, output, N, C, H, W);
}
