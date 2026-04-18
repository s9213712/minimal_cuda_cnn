"""Training and model configuration for train_split.py."""

import os


BATCH = 64
EPOCHS = 50
EVAL_MAX_BATCHES = 50

N_TRAIN = 8000
N_VAL = 2000
TRAIN_BATCH_IDS = (1,)
DATASET_SEED = 42
INIT_SEED = 42

LR_CONV1 = 0.002
LR_CONV = 0.002
LR_FC = 0.002
LR_PLATEAU_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-5

MOMENTUM = 0.9
LEAKY_ALPHA = 0.1
WEIGHT_DECAY = 5e-4
GRAD_CLIP_CONV = 1.0
GRAD_CLIP_FC = 1.0
GRAD_CLIP_BIAS = 5.0
GRAD_POOL_CLIP = 1.0

EARLY_STOP_PATIENCE = 8
MIN_DELTA = 0.05
BEST_MODEL_FILENAME = "best_model_split.npz"

GRAD_DEBUG = os.environ.get("GRAD_DEBUG") == "1"
GRAD_DEBUG_BATCHES = int(os.environ.get("GRAD_DEBUG_BATCHES", "1"))

C1_IN, C1_OUT = 3, 32
C2_IN, C2_OUT = 32, 32
C3_IN, C3_OUT = 32, 64
C4_IN, C4_OUT = 64, 64

H, W = 32, 32
KH, KW = 3, 3

H1, W1 = H - KH + 1, W - KW + 1          # 30x30
H2, W2 = H1 - KH + 1, W1 - KW + 1        # 28x28
P1H, P1W = H2 // 2, W2 // 2              # 14x14
H3, W3 = P1H - KH + 1, P1W - KW + 1      # 12x12
H4, W4 = H3 - KH + 1, W3 - KW + 1        # 10x10
P2H, P2W = H4 // 2, W4 // 2              # 5x5
FC_IN = C4_OUT * P2H * P2W
