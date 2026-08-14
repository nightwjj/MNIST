import torch


# 数据相关
BATCH_SIZE = 64
DATA_DIR = "./data"


# 训练相关
LR = 1e-3
EPOCHS = 10


# 设备
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# TensorBoard
LOG_DIR = "./logs_train/exp1"


# 模型保存
CHECKPOINT_DIR = "./checkpoints"

BEST_MODEL_PATH = "./checkpoints/best_exp3.pth"

LAST_MODEL_PATH = "./checkpoints/last.pth"

# 是否断点续训
RESUME = False