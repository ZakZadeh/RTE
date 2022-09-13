import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as func
from torchvision.models import resnet50, ResNet50_Weights

""" -------------------------------------------------------------------------"""
""" Image Encoder """   
""" -------------------------------------------------------------------------"""

""" ResNet50 """
class ResNet50(nn.Module):
    def __init__(self, params):
        super(ResNet50, self).__init__()
        self.weights = ResNet50_Weights.DEFAULT
        self.resnet = nn.Sequential(
            resnet50(weights=self.weights),
        )

    def forward(self, x):
        x = self.resnet(x)
        return x

""" -------------------------------------------------------------------------"""
""" Laser Encoder """   
""" -------------------------------------------------------------------------"""
    
""" -------------------------------------------------------------------------"""
""" CNN1D """
class CNN1D(nn.Module):
    def __init__(self, params):
        super(CNN1D, self).__init__()
        self.nf = params.nFeature
        self.nC = 1
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.nC, self.nf // 32, 4, 4, 0),
            nn.BatchNorm1d(self.nf // 32),
            nn.ReLU(True),
            nn.MaxPool1d(4),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.nf // 32, self.nf // 16, 4, 4, 0),
            nn.BatchNorm1d(self.nf // 16),
            nn.ReLU(True),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.nf // 16, self.nf // 8, 4, 4, 0),
            nn.BatchNorm1d(self.nf // 8),
            nn.ReLU(True),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(self.nf // 8, self.nf // 4, 4, 4, 0),
            nn.BatchNorm1d(self.nf // 4),
            nn.ReLU(True),
        )
        
        # self.tran1 = nn.Sequential(
        #    nn.TransformerEncoderLayer(self.nf, 16),
        # )
            
    def forward(self, x):
        x = x.unsqueeze(1)
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
""" Projector """   
""" -------------------------------------------------------------------------"""

""" -------------------------------------------------------------------------"""
""" CatFusion """   
class CatFusion(nn.Module):
    def __init__(self, params):
        super(CatFusion, self).__init__()
        self.nf = params.nFeature
        
        self.fc1 = nn.Sequential(
            nn.Linear((self.nf * 5) // 4, self.nf * 4),
            nn.BatchNorm1d(self.nf * 4),
            nn.ReLU(True),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(self.nf * 4, self.nf),
            nn.BatchNorm1d(self.nf),
            nn.ReLU(True),
        )
        
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.fc1(x)
        x = self.fc3(x)
        return x

""" -------------------------------------------------------------------------"""
""" AttenFusion """   
class AttenFusion(nn.Module):
    def __init__(self, params):
        super(AttenFusion, self).__init__()
        self.nf = params.nFeature
        
        self.fc1 = nn.Sequential(
            nn.Linear((self.nf * 5) // 4, self.nf * 4),
            nn.BatchNorm1d(self.nf * 4),
            nn.ReLU(True),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(self.nf * 4, self.nf),
            nn.BatchNorm1d(self.nf),
            nn.ReLU(True),
        )
        
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.fc1(x)
        x = self.fc3(x)
        return x

""" -------------------------------------------------------------------------"""
""" Predictor """   
""" -------------------------------------------------------------------------"""

""" Lin """   
class Lin(nn.Module):
    def __init__(self, params):
        super(Lin, self).__init__()
        self.nf = params.nFeature
        self.nClass = params.nClass
        
        self.fc = nn.Sequential(
            nn.Linear(self.nf, self.nClass),
        )

    def forward(self, x):
        x = self.fc(x)
        return x