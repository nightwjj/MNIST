from turtle import forward

from numpy import identity
import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, 
                               padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        # 原地替换，不保留原来张量
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                                       kernel_size=3, stride=1, 
                                       padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()

        if (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,out_channels,
                    kernel_size=1,stride=stride,bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity

        out = self.relu(out)

        return out

class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,
            stride=1, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(inplace=True)

        self.layer1 = BasicBlock(32, 32, stride=1)

        self.layer2 = BasicBlock(32, 64, stride=2)

        self.layer3 = BasicBlock(64, 128, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.avgpool(x)

        x = torch.flatten(x,1)

        x = self.fc(x)

        return x


        

