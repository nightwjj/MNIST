from torch import nn

class Basic_cnn(nn.Module):
    def __init__(self):
        super(Basic_cnn, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 

            nn.MaxPool2d(2), 

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x