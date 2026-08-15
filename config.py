import torch

# 模型
MODEL_NAME = "basic_cnn"

# 实验
EXPERIMENT_TAG = "exp04"
EXPERIMENT_NAME = f"{MODEL_NAME}_{EXPERIMENT_TAG}"
OUTPUT_ROOT = "./outputs"
OUTPUT_DIR = f"{OUTPUT_ROOT}/{EXPERIMENT_NAME}"

# 数据相关
BATCH_SIZE = 64
DATA_DIR = "./data"
VAL_SIZE = 5000

# 训练相关
LR = 1e-3
EPOCHS = 10

# 学习率调度策略
USE_SCHEDULER = True
STEP_SIZE = 1
GAMMA = 0.98


# 设备
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# TensorBoard
LOG_DIR = f"{OUTPUT_DIR}/tensorboard"


# 模型保存
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best.pth"

# 随机种子
SEED = 42

