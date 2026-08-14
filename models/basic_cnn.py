from torch import nn

class Basic_cnn(nn.Module):
    def __init__(self):
        super(Basic_cnn, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2), 

            nn.Flatten(),
            nn.Linear(288, 64),
            
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x