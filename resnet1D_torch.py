import numpy as np
import torch
from torch import nn

class BottleneckResidual(nn.Module):
    def __init__(self,in_channel,out_channels,stride):
        super(BottleneckResidual, self).__init__()
        out_channel1,out_channel2,out_channel3 = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel,out_channel1,1,stride=stride),
            nn.BatchNorm1d(out_channel1),
            nn.ReLU()
            )
        self.conv2 =nn.Sequential(
            nn.Conv1d(out_channel1,out_channel2,3,padding='same'),
            nn.BatchNorm1d(out_channel2),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv1d(out_channel2,out_channel3,1),
            nn.BatchNorm1d(out_channel3),
            )
        self.is_projection = False
        if not in_channel == out_channel3: # match dimension
            self.is_projection = True
            self.projection = nn.Sequential(
                nn.Conv1d(in_channel,out_channel3,1,stride=stride),
                nn.BatchNorm1d(out_channel3),
                )

    def forward(self,x):
        if self.is_projection:
            shortcut = self.projection(x)
        else:
            shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += shortcut
        return nn.ReLU()(x)

class ResNet50(nn.Module):
    def __init__(self,in_channel,nclasses):
        super(ResNet50, self).__init__()
        self.entry = nn.Sequential(
            nn.Conv1d(in_channel,64,7,2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3,2)
        )
        self.backbone = nn.ModuleList()
        # conv2
        self.backbone.append(BottleneckResidual(64,[64,64,256],1))
        for i in np.arange(2):
            self.backbone.append(BottleneckResidual(256,[64,64,256],1))
        # conv3
        self.backbone.append(BottleneckResidual(256,[128,128,512],2))
        for i in np.arange(3):
            self.backbone.append(BottleneckResidual(512,[128,128,512],1))
        # conv4
        self.backbone.append(BottleneckResidual(512,[256,256,1024],2))
        for i in np.arange(5):
            self.backbone.append(BottleneckResidual(1024,[256,256,1024],1))
        # conv5
        self.backbone.append(BottleneckResidual(1024,[512,512,2048],2))
        for i in np.arange(2):
            self.backbone.append(BottleneckResidual(2048,[512,512,2048],1))     
        self.exit = nn.Linear(2048,nclasses)

    def forward(self,x):
        x = self.entry(x)
        for i,e in enumerate(self.backbone):
            x = e(x)
        # global avg pooling
        x = torch.mean(x,dim=2)
        x = self.exit(x)
        return nn.Softmax(dim=1)(x)