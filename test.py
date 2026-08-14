from torch import nn

import config

from dataset import get_dataloader
from models.basic_cnn import Basic_cnn
from train import evaluate
from utils.checkpoint import load_checkpoint

def main():

    # 获取数据
    _, test_loader = get_dataloader()

    # 创建模型
    model = Basic_cnn().to(config.DEVICE)

    epoch, accuracy = load_checkpoint(
        model=model,
        optimizer=None, 
        path=config.BEST_MODEL_PATH,
        device=config.DEVICE
    )

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 测试
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        loss_fn,
        config.DEVICE
    )

    print(f"加载的模型来自第 {epoch} 轮")
    print(f"保存时准确率：{accuracy:.4f}")
    print(f"测试集 loss：{test_loss:.4f}")
    print(f"测试集准确率：{test_accuracy:.4f}")


if __name__ == "__main__":
    main()
