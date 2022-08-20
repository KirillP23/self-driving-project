import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(288, 1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        output = self.fc1(x)
        return output