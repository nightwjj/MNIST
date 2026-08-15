import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn


import config

from models.build import build_model
from dataset import get_dataloader
from train import train_one_epoch, evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.seed import set_seed
from utils.experiment import (save_experiment_config,save_experiment_results)

from arguments import parse_args

def main():
    args = parse_args()

    experiment_config = {
                        "experiment_name": config.EXPERIMENT_NAME,
                        "model_name": config.MODEL_NAME,
                        "batch_size": config.BATCH_SIZE,
                        "val_size": config.VAL_SIZE,
                        "learning_rate": config.LR,
                        "epochs": args.epochs,
                        "optimizer": "Adam",
                        "use_scheduler": config.USE_SCHEDULER,
                        "step_size": (
                            config.STEP_SIZE
                            if config.USE_SCHEDULER
                            else None
                        ),
                        "gamma": (
                            config.GAMMA
                            if config.USE_SCHEDULER
                            else None
                        ),
                        "seed": config.SEED,
                        "device": str(config.DEVICE),
                        }
    config_path = save_experiment_config(
        experiment_config=experiment_config,
        output_dir=config.OUTPUT_DIR,
    )

    print(f"实验配置已保存：{config_path}")

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
    model = build_model(config.MODEL_NAME)
    model = model.to(config.DEVICE)

    print(f"当前模型：{config.MODEL_NAME}")
    print(f"实验名称：{config.EXPERIMENT_NAME}")
    print(f"实验目录：{config.OUTPUT_DIR}")

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

    experiment_results = {
        "experiment_name": config.EXPERIMENT_NAME,
        "model_name": config.MODEL_NAME,
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_accuracy),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }

    results_path = save_experiment_results(
        experiment_results=experiment_results,
        output_dir=config.OUTPUT_DIR,
    )

    print(f"实验结果已保存：{results_path}")

    writer.close()

if __name__ == "__main__":
    main()
