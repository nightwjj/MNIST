import torchvision
from torch.utils.data import DataLoader
import config

batch_size = config.BATCH_SIZE
root=config.DATA_DIR

def get_dataloader():
    train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(degree=10, translate=(0.1, 0.1)),

            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
    ])

    train_data = torchvision.datasets.MNIST(
        root=root, 
        train=True, 
        download=True, 
        transform=train_transform
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

    test_loader = DataLoader(
        test_data, 
        batch_size = batch_size, 
        shuffle = False
    )

    return train_loader, test_loader

    