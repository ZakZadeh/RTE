import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as func
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import vit_h_14, ViT_H_14_Weights

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
    
""" EffNet """
class EffNet(nn.Module):
    def __init__(self, params):
        super(EffNet, self).__init__()
        self.weights = EfficientNet_V2_M_Weights.DEFAULT
        self.effnet = nn.Sequential(
            efficientnet_v2_m(weights=self.weights),
        )

    def forward(self, x):
        x = self.effnet(x)
        return x

""" ViT """
class ViT(nn.Module):
    def __init__(self, params):
        super(ViT, self).__init__()
        self.weights = ViT_B_16_Weights.DEFAULT
        self.vit = nn.Sequential(
            vit_b_16(weights=self.weights),
        )

    def forward(self, x):
        x = self.vit(x)
        return x
    
""" ViTBig """
class ViTBig(nn.Module):
    def __init__(self, params):
        super(ViTBig, self).__init__()
        self.weights = ViT_H_14_Weights.DEFAULT
        self.vitBig = nn.Sequential(
            vision_transformer(weights=self.weights),
        )

    def forward(self, x):
        x = self.vitBig(x)
        return x
    
""" -------------------------------------------------------------------------"""
""" Projector """   
""" -------------------------------------------------------------------------"""

""" -------------------------------------------------------------------------"""
""" Identity """   
class Identity(nn.Module):
    def __init__(self, params):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

""" -------------------------------------------------------------------------"""
""" MLP2 """   
class MLP2(nn.Module):
    def __init__(self, params):
        super(MLP2, self).__init__()
        self.nf = params.nFeature
        
        self.fc1 = nn.Sequential(
            nn.Linear(self.nf, self.nf),
            nn.BatchNorm1d(self.nf),
            nn.ReLU(True),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(self.nf, self.nf),
            nn.BatchNorm1d(self.nf),
            nn.ReLU(True),
        )
                        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

""" -------------------------------------------------------------------------"""
""" CatFusion """   
class CatFusion(nn.Module):
    def __init__(self, params):
        super(CatFusion, self).__init__()
        self.nf = params.nFeature
        self.nL = params.laserDim
        
        self.fcI = nn.Sequential(
            nn.Linear(self.nf, self.nf // 2),
            nn.BatchNorm1d(self.nf // 2),
            nn.ReLU(True),
        )
        
        self.fcL = nn.Sequential(
            nn.Linear(self.nL, self.nf // 2),
            nn.BatchNorm1d(self.nf // 2),
            nn.ReLU(True),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.nf, self.nf),
            nn.BatchNorm1d(self.nf),
            nn.ReLU(True),
        )
        
    def forward(self, xI, xL):
        xI = self.fcI(xI)
        xL = self.fcL(xI)
        # print(xI.size(), xL.size())
        x = torch.cat((xI, xL), 1)
        x = self.fc(x)
        return x

""" -------------------------------------------------------------------------"""
""" AttenFusion """   
class AttenFusion(nn.Module):
    def __init__(self, params):
        super(AttenFusion, self).__init__()
        self.nf = params.nFeature
        self.nL = params.laserDim
        
        self.fcI = nn.Sequential(
            nn.Linear(self.nf, self.nf),
            nn.BatchNorm1d(self.nf),
            nn.ReLU(True),
        )
        
        self.fcL = nn.Sequential(
            nn.Linear(self.nL, self.nf),
            nn.BatchNorm1d(self.nf),
            nn.ReLU(True),
        )
        
        self.mha = nn.MultiheadAttention(self.nf, 1)
        
    def forward(self, x1, x2):
        xI = self.fcI(xI)
        xL = self.fcL(xI)
        # print(xI.size(), xL.size())
        x, attnWeights = self.mha(xL, xI, xI)
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