import torch
from torch import nn


class ResNet_AMC(torch.nn.Module):
    def __init__(self):
        super(ResNet_AMC, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, (1, 3), padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, (2, 3), padding='same')
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 80, (1, 3), padding='same')
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(80, 80, (1, 3), padding='same')
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(163840, 128)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(128, 24)
        
    def forward(self, x):
        input = x.unsqueeze(1)
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.conv2(x)
        x = input + x
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
