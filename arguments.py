import argparse

import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练 MNIST 手写数字分类模型"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help=f"训练轮数，默认值为 {config.EPOCHS}"
    )

    return parser.parse_args()