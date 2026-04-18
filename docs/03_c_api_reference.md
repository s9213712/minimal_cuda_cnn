# C API 參考

`libminimal_cuda_cnn.so` 主要透過 `extern "C"` 匯出函式，適合 Python `ctypes`、C/C++ 或其他 FFI 呼叫。

## GPU 記憶體

```c
void* gpu_malloc(size_t size);
void gpu_free(void* ptr);
void gpu_memcpy_h2d(void* dst, const void* src, size_t size);
void gpu_memcpy_d2h(void* dst, const void* src, size_t size);
void gpu_memset(void* dst, int value, size_t size);
```

`size` 是 bytes，不是元素數。`float32` buffer 大小通常是 `num_elements * 4`。

## Forward API

```c
void im2col_forward(float* input, float* col,
                    int N, int C, int H, int W,
                    int KH, int KW, int outH, int outW);

void gemm_forward(float* A, float* B, float* C,
                  int M, int N, int K);

void dense_forward(float* input, float* weights, float* bias, float* output,
                   int N, int in_f, int out_f);

void apply_relu(float* data, int size);
void leaky_relu_forward(float* data, float alpha, int size);

void maxpool_forward_store(float* output, float* input, int* max_idx,
                           int N, int C, int H, int W);

void softmax_forward(float* input, float* output, int N, int features);
```

典型 convolution forward：

```text
NCHW input
-> im2col_forward
-> gemm_forward(weights, col, conv_raw)
-> activation
```

`gemm_forward` 的 conv output 常用作 CNHW：`(OUT_C, N, outH, outW)`。

## Backward API

```c
void dense_backward_full(float* d_out, float* input, float* weights,
                         float* d_input, float* d_weights, float* d_bias,
                         int N, int in_f, int out_f);

void conv_backward(float* grad_out, float* input, float* weights,
                   float* grad_weights, float* grad_input,
                   int N, int C, int H, int W,
                   int KH, int KW, int outH, int outW, int OUT_C);

void leaky_relu_backward(float* data, float* grad, float alpha, int size);

void maxpool_backward_use_idx(float* grad_out, int* max_idx, float* grad_input,
                              int N, int C, int H, int W);

void softmax_backward(float* grad_out, float* probs, int N, int features);
```

`conv_backward` 會計算：

```text
grad_weights: dL/dW
grad_input:   dL/dInput
```

目前訓練程式會對部分 gradient 做 clipping 或 batch 平均，這些縮放通常在 Python 端完成。

## Optimizer

```c
void apply_sgd_update(float* weights, float* grad, float lr, int size);
```

行為：

```text
weights[i] -= lr * grad[i]
```

## Layout 轉換

```c
void nchw_to_cnhw(float* input, float* output, int N, int C, int H, int W);
void cnhw_to_nchw(float* input, float* output, int N, int C, int H, int W);
```

使用重點：

```text
im2col_forward input: NCHW
conv raw output: CNHW
maxpool_forward_store input/output: CNHW
dense_forward input: row-major (N, features)
```

## Softmax loss

```c
float softmax_cross_entropy(float* input, float* labels, float* probs, float* loss,
                            int N, int features);
```

`labels` 是 one-hot label。此函式會計算 softmax 與 loss，不會修改 label buffer。若 `loss` 非 null，會把 scalar loss 寫到 device pointer。

