# 專案檔案說明

本文整理 `cpp/include` 與 `cpp/src` 中各檔案的責任。預設訓練流程主要透過 `extern "C"` 匯出的 C API 呼叫 `.so`，C++ layer 類別則偏向 C++ 端直接使用。

## 目錄結構

```text
minimal_cuda_cnn/
├── cpp/
│   ├── Makefile
│   ├── include/
│   │   ├── cuda_check.h
│   │   ├── dataloader.h
│   │   ├── dense_layer.h
│   │   ├── network.h
│   │   └── tensor.h
│   ├── src/
│   │   ├── core.cu
│   │   ├── memory.cu
│   │   ├── loss_layer.cu
│   │   ├── conv_backward.cu
│   │   ├── dense_layer.cu
│   │   ├── optimizer.cu
│   │   ├── layout_convert.cu
│   │   ├── maxpool_store.cu
│   │   ├── maxpool_backward_use_idx.cu
│   │   └── ...
│   └── libminimal_cuda_cnn.so
├── python/
│   ├── train_split.py
│   ├── train_split_torch_baseline.py
│   ├── train_config.py
│   ├── cifar10_data.py
│   ├── cuda_backend.py
│   ├── model_forward.py
│   ├── model_init.py
│   └── model_weights.py
└── docs/
```

## include

| 檔案 | 作用 |
|---|---|
| `cuda_check.h` | CUDA 錯誤檢查工具。`CUDA_CHECK(expr)` 檢查 runtime API 回傳值；`CUDA_KERNEL_CHECK()` 檢查 kernel launch error 並同步 GPU。 |
| `tensor.h` | `CudaTensor` C++ RAII 包裝，管理 GPU tensor 記憶體，提供 host/device copy。 |
| `network.h` | C++ layer 介面與 `ConvLayer`、`ReLULayer`、`MaxPoolLayer` 宣告。 |
| `dense_layer.h` | C++ `DenseLayer` 宣告。 |
| `dataloader.h` | 簡單 C++ dataloader 骨架，目前不是 Python 訓練主流程必需。 |

## src

| 檔案 | 作用 |
|---|---|
| `memory.cu` | 匯出 `gpu_malloc`、`gpu_free`、`gpu_memcpy_h2d`、`gpu_memcpy_d2h`、`gpu_memset`，供 Python/C++ 管理 GPU 記憶體。 |
| `core.cu` | 基礎 forward kernel：`im2col_forward`、`gemm_forward`、`apply_relu`、`apply_maxpool`。 |
| `backward.cu` | ReLU backward 與不保存 index 的 NCHW maxpool backward。 |
| `conv_backward.cu` | 卷積層 backward：計算 conv weight gradient 與 input gradient。 |
| `dense_layer.cu` | 全連接層 forward/backward：`dense_forward`、`dense_backward_full`。 |
| `loss_layer.cu` | `softmax_forward`、`softmax_cross_entropy`、`softmax_backward`，另含 `im2col_backward`、`gemm_backward`。 |
| `optimizer.cu` | Optimizer kernel。`apply_sgd_update` 執行純 SGD；`apply_momentum_update` 執行 Momentum SGD，更新公式為 `velocity = momentum * velocity - lr * grad`、`weights += velocity`。 |
| `layout_convert.cu` | `nchw_to_cnhw` 與 `cnhw_to_nchw`。部分 kernel 輸出以 CNHW 儲存，訓練時常需要轉換。 |
| `reorganize.cu` / `reorganize_backward.cu` | 舊版 layout 重排 API。新程式建議優先用 `layout_convert.cu` 的明確 NCHW/CNHW 函式。 |
| `maxpool_store.cu` | 帶 max index 的 maxpool forward：`maxpool_forward_store`。 |
| `maxpool_backward_use_idx.cu` | 搭配 `maxpool_forward_store` 做 maxpool backward。 |
| `maxpool_backward_nchw.cu` | NCHW maxpool backward 版本。 |
| `leaky_relu.cu` | LeakyReLU forward/backward，含 CNHW 與 NCHW 命名版本。 |
| `layer_norm.cu` | LayerNorm forward/backward。 |
| `batch_norm.cu` | BatchNorm forward/backward。注意：目前預設 Makefile 沒有把它編進 `.so`，若要使用需加入 `cpp/Makefile` 的編譯清單。 |
| `network.cu` | C++ layer 類別的 forward 實作，供 C++ 端使用。 |
| `alexnet.cu`、`resnet_backward.cu` | 實驗性網路/ResNet backward 程式碼，不是目前 CIFAR/MNIST 主流程必需，也未在預設 Makefile 編進 `.so`。 |
| `gpu_monitor.cu` | `check_gpu_status()`，呼叫 `nvidia-smi` 印出 GPU 使用率。 |

## python

| 檔案 | 作用 |
|---|---|
| `train_split.py` | 手寫 CUDA CIFAR-10 trainer。主要保留訓練 loop、backward pass、scheduler、early stopping。 |
| `train_split_torch_baseline.py` | PyTorch baseline。使用同一個資料切分、模型架構、初始權重、Momentum SGD 條件，方便和 CUDA trainer 比較。 |
| `train_config.py` | 共享訓練參數與模型 shape。包含 `BATCH`、`EPOCHS`、`LR_*`、`MOMENTUM`、`WEIGHT_DECAY`、gradient clipping、train/val split 與 conv shape。 |
| `cifar10_data.py` | CIFAR-10 batch 讀取、train/val/test split、CIFAR mean/std normalization。 |
| `cuda_backend.py` | Python `ctypes` binding、`.so` 載入、GPU memory helper、conv/maxpool helper、Momentum conv update helper。 |
| `model_init.py` | CPU 端共享 initial weights。CUDA trainer 和 PyTorch baseline 都從這裡取得同一組初始權重。 |
| `model_weights.py` | CUDA 權重上傳、velocity buffer 初始化、checkpoint save/load、GPU pointer 釋放。 |
| `model_forward.py` | CUDA inference forward pass 與 validation/test evaluation。 |
