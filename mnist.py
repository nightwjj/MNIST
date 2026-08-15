import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn


import config


from models.basic_cnn import Basic_cnn
from models.m3_cnn import M3CNN
from models.resnet import SmallResNet
from dataset import get_dataloader
from train import train_one_epoch, evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.seed import set_seed

from arguments import parse_args

def main():
    args = parse_args()

    set_seed(config.SEED)

    train_loader, val_loader, test_loader = get_dataloader()


    len_train = len(train_loader.dataset)
    len_val = len(val_loader.dataset)
    len_test = len(test_loader.dataset)


    print(
        f"训练集长度：{len_train}, "
        f"验证集长度：{len_val}, "
        f"测试集长度：{len_test}"
    )


    # 创建模型
    # model = Basic_cnn().to(config.DEVICE)
    model = M3CNN().to(config.DEVICE)
    # model = SmallResNet().to(config.DEVICE)


    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # CrossEntropyLoss 本身没有必须移动到 GPU 的可训练参数
    # if torch.cuda.is_available():
    #     loss_fn = loss_fn.cuda()


    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = None
    if config.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=config.STEP_SIZE, gamma=config.GAMMA
        )


    # 添加tensorboard
    writer = SummaryWriter(config.LOG_DIR)

    # 保存准确率最高模型
    best_accuracy = 0.0


    for i in range(args.epochs):
        print(f"--------第{i+1}轮训练开始--------")

        # 训练开始
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optim,
            device=config.DEVICE
            )

        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=config.DEVICE,
            )

        print(f"训练集loss：{train_loss:.4f}，训练集正确率：{train_accuracy:.4f}")
        print(f"验证集loss：{val_loss:.4f}，验证集正确率：{val_accuracy:.4f}")

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        writer.add_scalar("Loss/train", train_loss, i)
        writer.add_scalar("Accuracy/train", train_accuracy, i)
        writer.add_scalar("Loss/val", val_loss, i)
        writer.add_scalar("Accuracy/val", val_accuracy, i)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            save_checkpoint(
                model=model,
                optimizer=optim,
                epoch=i + 1,
                accuracy=val_accuracy,
                path=config.BEST_MODEL_PATH,
            )

            print(f"保存最佳模型，"f"验证集 accuracy：{best_accuracy:.4f}")

    print(f"最高验证集准确率：{best_accuracy:.4f}")

    print("---------加载最佳模型----------")

    best_epoch, best_val_accuracy = load_checkpoint(
        model=model,
        optimizer=None,
        path=config.BEST_MODEL_PATH, device=config.DEVICE
    )

    test_loss, test_accuracy = evaluate(
                model=model, data_loader=test_loader,
                loss_fn=loss_fn, device=config.DEVICE,
                )

    print(f"最佳模型来自第 {best_epoch} 轮")
    print(f"最佳验证集准确率：{best_val_accuracy:.4f}")
    print(f"最终测试集 loss：{test_loss:.4f}")
    print(f"最终测试集准确率：{test_accuracy:.4f}")

    writer.close()

if __name__ == "__main__":
    main()
