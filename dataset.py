import os
import joblib
import numpy as np
import torch
import utils
import glob
import pickle
from PIL import Image
import torch.nn.functional as func
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd

class Rosbag(Dataset):
    def __init__(self, splitMode, params):
        self.splitMode = splitMode
        self.useLaser = params.useLaser:
        self.trainSet = params.trainSet
        self.testSet = params.testSet
        self.imageDim = params.imageDim
        self.nClass = params.nClass
        self.path = params.path
        dirsList = []
        if self.splitMode == "train":
            for dirPath in self.trainSet:
                dirsList.append(self.path + "/labels/" + dirPath + "/labels.csv")
        else:
            for dirPath in self.testSet:
                dirsList.append(self.path + "/labels/" + dirPath + "/labels.csv")
        self.data = []
        for dirPath in dirsList:
            dataDF = pd.read_csv(dirPath, index_col = 0)
            img = dataDF['img'].tolist()
            joints = dataDF['joints'].tolist()
            laser = dataDF['front_laser'].tolist()
            label = dataDF['label'].tolist()
            a = [img, joints, laser, label]
            dataList = np.transpose(np.vstack((img, joints, laser, label)))
            self.data.extend(dataList)
        if self.splitMode == "train":
            params.weightedLoss = self.getClassRatio()
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dataPaths = self.data[idx]
        imgPath = dataPaths[0]
        jointsPath = dataPaths[1]
        fLaserPath = dataPaths[2]
        label = int(dataPaths[3])

        transform = transforms.Compose([transforms.Resize(self.imageDim),
            transforms.CenterCrop(self.imageDim),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        img = transform(Image.open(imgPath))

        # with (open(jointsPath, "rb")) as f:
        #    joints = pickle.load(f)
        #    velocity = joints["velocity"]
        #    label = self.getLabel(velocity)

        if self.useLaser:
            with (open(fLaserPath, "rb")) as f:
                fLaserDic = pickle.load(f)
                fLaser = np.array(fLaserDic["ranges"], dtype = np.float32)
        else:
            fLaser = None

        return img, label, fLaser
    
    def getClassRatio(self):
        counts = np.zeros(self.nClass)
        for sample in self.data:
            label = int(sample[3])
            counts[label] += 1
        for i in range(len(counts)):
            counts[i] = 1/(counts[i]+1e-6)
        weightedLoss = counts / np.amin(counts)
        return weightedLoss        