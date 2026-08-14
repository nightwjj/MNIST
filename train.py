import torch

def train_one_epoch(model, train_loader, 
                    loss_fn, optimizer, device):
    model.train()

    total_loss = 0
    total_accuracy = 0

    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 前向传播
        outputs = model(imgs)

        # 计算损失
        loss = loss_fn(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 更新
        optimizer.step()

        # 累计loss
        total_loss += loss.item()

        total_accuracy += (outputs.argmax(1) == targets).sum()

    # 一轮训练平均loss
    avg_loss = total_loss / len(train_loader)

    # 一轮训练准确率
    accuracy = total_accuracy / len(train_loader.dataset)

    return avg_loss, accuracy

def evaluate(model, test_loader, 
            loss_fn, device):

    model.eval()

    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)

            loss = loss_fn(outputs, targets)

            total_loss += loss.item()

            total_accuracy += (outputs.argmax(1) == targets).sum() 

    avg_loss = total_loss / len(test_loader)

    accuracy = total_accuracy / len(test_loader.dataset)

    return avg_loss, accuracy

