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

def main():

    train_loader, test_loader = get_dataloader()


    len_train = len(train_loader.dataset)
    len_test = len(test_loader.dataset)


    print(
        f"训练集长度：{len_train}, "
        f"测试集长度：{len_test}"
    )


    # 创建模型
    # model = Basic_cnn().to(config.DEVICE)
    # model = M3CNN().to(config.DEVICE)
    model = SmallResNet().to(config.DEVICE)


    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # CrossEntropyLoss 本身没有必须移动到 GPU 的可训练参数
    # if torch.cuda.is_available():
    #     loss_fn = loss_fn.cuda()


    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=config.STEP_SIZE, gamma=config.GAMMA
    )


    # 添加tensorboard
    writer = SummaryWriter(config.DATA_DIR)

    # 保存准确率最高模型
    best_accuracy = 0.0


    for i in range(config.EPOCHS):
        print(f"--------第{i+1}轮训练开始--------")

        # 训练开始
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optim,
            device=config.DEVICE
            )

        test_loss, test_accuracy = evaluate(
                model=model,
                test_loader=test_loader,
                loss_fn=loss_fn,
                device=config.DEVICE
            )

        print(f"训练集loss：{train_loss:.4f}，训练集正确率：{train_accuracy:.4f}")
        print(f"测试集loss：{test_loss:.4f}，测试集正确率：{test_accuracy:.4f}")

        # 更新学习率
        if config.USE_SCHEDULER:
            scheduler.step()

        writer.add_scalar("Loss/train", train_loss, i)
        writer.add_scalar("Accuracy/train", train_accuracy, i)
        writer.add_scalar("Loss/test", test_loss, i)
        writer.add_scalar("Accuracy/test", test_accuracy, i)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_checkpoint(model=model,
                            optimizer=optim,
                            epoch=i+1,
                            accuracy=test_accuracy,
                            path=config.BEST_MODEL_PATH
                            )

            print(f"保存最佳模型，"f"accuracy：{best_accuracy:.4f}")

    print(f"本轮最高准确率：{best_accuracy:.4f}")

    writer.close()

if __name__ == "__main__":
    main()
