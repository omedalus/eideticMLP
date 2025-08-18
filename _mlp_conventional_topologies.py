import torch.nn as nn


# Standard MLP for MNIST (input 784, two hidden layers, output 10)
class MLP_2HLStandard(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.ReLU(),
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
        )

    def forward(self, x):
        return self.net(x)


# MLP with skip connections
class MLP_2HLSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_skip = nn.Linear(
            784, 32
        )  # skip connection from input to 2nd hidden layer
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1) + self.fc_skip(x))
        out = self.fc3(h2)
        return out
