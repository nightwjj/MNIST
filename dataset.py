import torchvision
import torch
from torch.utils.data import DataLoader, Subset, random_split
import config

batch_size = config.BATCH_SIZE
root=config.DATA_DIR

# mean = (0.1307, )
# std = (0.3081, )
mean = (0.5, )
std = (0.5, )


def get_dataloader():
    train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),

            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    train_full_data = torchvision.datasets.MNIST(
    root=root,
    train=True,
    download=True,
    transform=train_transform,
    )

    val_full_data = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    train_size = len(train_full_data) - config.VAL_SIZE
    val_size = config.VAL_SIZE

    generator = torch.Generator()
    generator.manual_seed(config.SEED)

    train_subset, val_subset = random_split(
        train_full_data,
        lengths=[train_size, val_size],
        generator=generator,
    )

    train_data = train_subset

    val_data = Subset(
        val_full_data,
        val_subset.indices,
    )
    
    test_data = torchvision.datasets.MNIST(
        root=root, 
        train=False, 
        download=True, 
        transform=transform
    )

    # # 训练集与测试集长度
    # len_train = len(train_data)
    # len_test = len(test_data)
    # print(f"训练集长度：{len_train}, 测试集长度：{len_test}")

    # DataLoader加载数据集
    train_loader = DataLoader(
        train_data, 
        batch_size = batch_size, 
        shuffle = True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
    )   

    test_loader = DataLoader(
        test_data, 
        batch_size = batch_size, 
        shuffle = False
    )

    return train_loader, val_loader, test_loader
  