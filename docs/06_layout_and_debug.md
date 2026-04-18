# Layout 與 Debug 指南

本文整理使用 `.so` 時最容易出錯的 layout、記憶體大小與 CUDA debug 流程。

## Layout 規則

這個 `.so` 同時存在 NCHW 與 CNHW。

| 資料 | 常用 layout |
|---|---|
| 原始影像 input | NCHW，`(N, C, H, W)` |
| `im2col_forward` input | NCHW |
| `gemm_forward` conv output | CNHW，實際 shape 可視為 `(OUT_C, N, outH, outW)` |
| `maxpool_forward_store` input/output | CNHW |
| `dense_forward` input | 一般 row-major `(N, features)`，可由 CNHW pool output 轉 NCHW 後 flatten |
| `conv_backward` input | input 使用 NCHW；`grad_out` 在目前訓練流程可直接使用 conv raw 的 CNHW buffer |
| `conv_backward_precol` col | 來自同層 forward `im2col_forward` 的 row-major `(C*KH*KW, N*outH*outW)` buffer |

建議習慣：

```text
影像或下一層 conv 前：NCHW
conv raw / activation / maxpool：CNHW
進 dense 前：CNHW -> NCHW -> flatten
```

## 常見錯誤

### buffer 大小用錯

CUDA allocation 使用 bytes：

```python
ptr = lib.gpu_malloc(num_float32 * 4)
```

NumPy upload 可以用：

```python
ptr = lib.gpu_malloc(arr.nbytes)
```

### `ctypes.argtypes` 不一致

如果 `argtypes` 少一個參數或型別錯誤，很容易 crash 或得到錯誤結果。先對照 [03_c_api_reference.md](03_c_api_reference.md)。

### CNHW/NCHW 沒轉

最典型錯誤：

```text
conv raw 是 CNHW
但下一層 im2col_forward 需要 NCHW
```

這時要先做：

```python
lib.cnhw_to_nchw(d_conv_raw, d_conv_nchw, N, C, H, W)
```

### double free

每個 GPU pointer 只呼叫一次 `gpu_free`。若把同一 pointer 放進多個 cache list，會在 batch 結束時重複釋放。

## CUDA error check

專案內已使用：

```cpp
CUDA_CHECK(cudaMalloc(...));
CUDA_KERNEL_CHECK();
```

若 CUDA kernel launch 或 runtime API 出錯，程式會印出類似：

```text
CUDA error at src/memory.cu:8: cudaMalloc(&ptr, size) failed: ...
```

## 最小驗證清單

每次改 `.cu` 後建議跑：

```bash
make -C cpp
python3 -u /tmp/so_function_check.py
cuda-memcheck python3 -u /tmp/so_function_check.py
```

如果改到 optimizer，確認 `.so` 有匯出 Momentum update：

```bash
nm -D cpp/libminimal_cuda_cnn.so | grep -E 'apply_momentum_update|conv_update_fused|clip_inplace'
```

如果改到 convolution backward，確認 precol API 有匯出：

```bash
nm -D cpp/libminimal_cuda_cnn.so | grep conv_backward_precol
```

若要確認 CIFAR-10 訓練腳本能啟動：

```bash
timeout 70s python3 -u python/train_split.py
```

完整 CIFAR-10 訓練：

```bash
python3 -u python/train_split.py
```

## 常見環境問題

### `CUDA driver version is insufficient for CUDA runtime version`

通常發生在 sandbox、WSL GPU 權限或 driver/runtime 不匹配。先確認：

```bash
nvidia-smi
```

再用正常 shell 執行 Python 腳本。如果在受限環境內跑，可能需要允許 GPU 存取。

### `compute-sanitizer` 不能跑

部分 WSL/WDDM 環境會出現 debugger interface 不支援。這時可改用：

```bash
cuda-memcheck python3 -u your_script.py
```

### 訓練準確率不升

先從 MNIST 小模型確認：

1. 只跑 FC baseline，確認 loss 會下降。
2. 加 Conv + ReLU，不加 Pool。
3. 再加 Pool。
4. 每一步都檢查 gradient scale。`conv_backward`/`conv_backward_precol` 的 weight gradient 會累加傳入的 `grad_out`，目前 CIFAR trainer 會先在 logits gradient 做 batch mean，再交給後續 backward。
5. 使用 Momentum SGD 時，velocity buffer 不能每個 batch 重設；它必須從訓練開始保留到訓練結束。
