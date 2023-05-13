import torch
import torch.nn as nn


class SimpleCancerCNN(nn.Module):
    def __init__(self):
        super(SimpleCancerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 128 * 128, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 128 * 128)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x