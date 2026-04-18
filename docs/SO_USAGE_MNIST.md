# libminimal_cuda_cnn.so 使用教學索引

這組文件說明如何編譯與使用 `cpp/libminimal_cuda_cnn.so`，並用 MNIST 示範如何從 Python 或 C++ 呼叫 CUDA C API 做訓練與驗證。

建議閱讀順序：

1. [01_project_files.md](01_project_files.md)：`cpp/include` 與 `cpp/src` 各 `.h/.cu` 檔案用途。
2. [02_build_shared_library.md](02_build_shared_library.md)：如何編譯 `.so`、調整 GPU 架構、檢查匯出符號。
3. [03_c_api_reference.md](03_c_api_reference.md)：主要 C API、函式 prototype、forward/backward/update 介面。
4. [04_python_ctypes_mnist.md](04_python_ctypes_mnist.md)：Python `ctypes` 載入 `.so`，以及 MNIST CNN 訓練範例。
5. [05_cpp_linking.md](05_cpp_linking.md)：C++ 如何連結 `.so`，以及最小 inference 範例。
6. [06_layout_and_debug.md](06_layout_and_debug.md)：NCHW/CNHW layout 規則、常見錯誤、`cuda-memcheck` 驗證流程。

目前專案中的完整 CIFAR-10 訓練腳本位於 [train_split.py](/home/s92137/NN/minimal_cuda_cnn/python/train_split.py)，也使用同一個 `.so`。PyTorch 對照 baseline 位於 [train_split_torch_baseline.py](/home/s92137/NN/minimal_cuda_cnn/python/train_split_torch_baseline.py)，兩者共用 `train_config.py`、`cifar10_data.py` 與 `model_init.py` 來維持相同資料切分、初始權重與 Momentum SGD 條件。
