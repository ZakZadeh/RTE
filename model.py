import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as func
from torchvision.models import resnet50, ResNet50_Weights

""" -------------------------------------------------------------------------"""
""" Predictor """   
class Predictor(nn.Module):
    def __init__(self, params):
        super(Predictor, self).__init__()
        self.nf = params.nFeature
        self.nClass = params.nClass
        
        self.fc1 = nn.Sequential(
            nn.Linear(self.nf , self.nf),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(self.nf , self.nClass),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

""" -------------------------------------------------------------------------"""
""" CNN4 """
class CNN4(nn.Module):
    def __init__(self, params):
        super(CNN4, self).__init__()
        self.nf = 64
        self.nC = params.nChannel
        self.nClass = params.nClass
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.nC, self.nf, 4, 4, 1),
            nn.BatchNorm2d(self.nf),
            nn.ReLU(True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.nf, self.nf * 2, 4, 4, 1),
            nn.BatchNorm2d(self.nf * 2),
            nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.nf * 2, self.nf * 4, 4, 4, 1),
            nn.BatchNorm2d(self.nf * 4),
            nn.ReLU(True),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.nf * 4, self.nf * 8, 4, 4, 1),
            nn.BatchNorm2d(self.nf * 8),
            nn.ReLU(True),
        )

        # self.tran1 = nn.Sequential(
        #    nn.TransformerEncoderLayer(self.nf, 16),
        # )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.squeeze()
        # x = x.permute(2,0,1)
        # x = self.tran1(x)
        # x = x.permute(1,2,0)
        return x

""" -------------------------------------------------------------------------"""
""" ResNet50 """
class ResNet50(nn.Module):
    def __init__(self, params):
        super(ResNet50, self).__init__()
        self.nClass = params.nClass
        self.weights = ResNet50_Weights.DEFAULT
        self.resnet = nn.Sequential(
            resnet50(weights=self.weights),
        )

    def forward(self, x):
        x = self.resnet(x)
        return x