import torch
import torch.nn as nn


class M3CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 48, kernel_size=3, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            nn.Conv2d(48, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 80, kernel_size=3, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(),

            nn.Conv2d(80, 96, kernel_size=3, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.Conv2d(96, 112, kernel_size=3, bias=False),
            nn.BatchNorm2d(112),
            nn.ReLU(),

            nn.Conv2d(112, 128, kernel_size=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 144, kernel_size=3, bias=False),
            nn.BatchNorm2d(144),
            nn.ReLU(),

            nn.Conv2d(144, 160, kernel_size=3, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            nn.Conv2d(160, 176, kernel_size=3, bias=False),
            nn.BatchNorm2d(176),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(176 * 8 * 8, 10, bias=False),
            nn.BatchNorm1d(10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x