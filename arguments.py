import argparse

import config


MODEL_CHOICES = (
    "basic_cnn",
    "m3_cnn",
    "resnet",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练 MNIST 手写数字分类模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型参数
    model_group = parser.add_argument_group("模型参数")

    model_group.add_argument(
        "--model",
        type=str,
        choices=MODEL_CHOICES,
        default="basic_cnn",
        help="选择要训练的模型",
    )

    # 数据参数
    data_group = parser.add_argument_group("数据参数")

    data_group.add_argument(
        "--data-dir",
        type=str,
        default=config.DATA_DIR,
        help="MNIST 数据集保存路径",
    )

    data_group.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="每个 Batch 的样本数量",
    )

    data_group.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader 使用的子进程数量",
    )

    # 训练参数
    train_group = parser.add_argument_group("训练参数")

    train_group.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="训练轮数",
    )

    train_group.add_argument(
        "--lr",
        type=float,
        default=config.LR,
        help="初始学习率",
    )

    train_group.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="随机种子",
    )

    train_group.add_argument(
        "--device",
        type=str,
        choices=("cpu", "cuda"),
        default=str(config.DEVICE),
        help="训练设备",
    )

    # 学习率调度器
    scheduler_group = parser.add_argument_group("学习率调度器")

    scheduler_group.add_argument(
        "--scheduler",
        type=str,
        choices=("none", "step"),
        default="step" if config.USE_SCHEDULER else "none",
        help="学习率调度策略",
    )

    scheduler_group.add_argument(
        "--step-size",
        type=int,
        default=config.STEP_SIZE,
        help="StepLR 每隔多少轮调整一次学习率",
    )

    scheduler_group.add_argument(
        "--gamma",
        type=float,
        default=config.GAMMA,
        help="StepLR 学习率衰减系数",
    )

    # 实验记录
    output_group = parser.add_argument_group("实验记录")

    output_group.add_argument(
        "--log-dir",
        type=str,
        default=config.LOG_DIR,
        help="TensorBoard 日志目录",
    )

    output_group.add_argument(
        "--best-model-path",
        type=str,
        default=config.BEST_MODEL_PATH,
        help="最佳模型保存路径",
    )

    output_group.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="需要继续训练的 Checkpoint 路径",
    )

    return parser.parse_args()