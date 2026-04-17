#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_gpu_status() {
    // 呼叫 nvidia-smi 獲取即時狀態
    system("nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits");
}
