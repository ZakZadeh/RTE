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

class Rosbag(Dataset):
    def __init__(self, splitMode, params):
        self.splitMode = splitMode
        self.trainSet = params.trainSet
        self.testSet = params.testSet
        self.imageDim = params.imageDim
        self.nClass = params.nClass
        self.path = params.path
        dirsList = []
        if self.splitMode == "train":
            for dirPath in self.trainSet:
                dirsList.append(self.path + "/" + dirPath)
        else:
            for dirPath in self.testSet:
                dirsList.append(self.path + "/" + dirPath)
        self.data = []
        for dirPath in dirsList:
            imgDir = dirPath + "/img/"
            jntDir = dirPath + "/joints/"       
            fLsrDir = dirPath + "/front_laser/"       
            imgPathList = sorted(glob.glob(imgDir + "*"))
            jntPathList = sorted(glob.glob(jntDir + "*"))
            fLsrPathList = sorted(glob.glob(fLsrDir + "*"))
            oneData = [{} for _ in imgPathList]
            for idx, imgPath in enumerate(imgPathList):
                jointPath = jntPathList[idx]
                fLaserPath = fLsrPathList[idx]
                oneData[idx] = {"img": imgPath, "joints": jointPath, "front_laser": fLaserPath}
            self.data.extend(oneData)
        if self.splitMode == "train":
            params.weightedLoss = self.getClassRatio()
        # self.filterData()
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dataPaths = self.data[idx]
        imgPath = dataPaths["img"]
        jointsPath = dataPaths["joints"]
        fLaserPath = dataPaths["front_laser"]
        transform = transforms.Compose([transforms.Resize(self.imageDim),
            transforms.CenterCrop(self.imageDim),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        img = transform(Image.open(imgPath))
        
        try:
            with (open(jointsPath, "rb")) as f:
                joints = pickle.load(f)
                velocity = joints["velocity"]
                label = self.getLabel(velocity)
        except OSError:
            print("Joint file Not Found" + jointsPath)
            label = 0
        
        try:
            with (open(fLaserPath, "rb")) as f:
                fLaserDic = pickle.load(f)
                fLaser = np.array(fLaserDic["ranges"], dtype = np.float32)
        except OSError:
            print("Laser file Not Found" + jointsPath)
            label = None
        return img, label, fLaser
    
    def getLabel(self, velocity):
        meanFrontVel = (velocity[2] + velocity[3]) / 2
        if meanFrontVel >= 1.25:
            label = 1
        else:
            label = 0
        return label
        
    def getClassRatio(self):
        counts = np.zeros(self.nClass)
        for sample in self.data:
            jointsPath = sample["joints"]
            try:
                with (open(jointsPath, "rb")) as f:
                    joints = pickle.load(f)
                    velocity = joints["velocity"]
                    label = self.getLabel(velocity)
                    counts[label] += 1
            except OSError:
                continue
        for i in range(len(counts)):
            counts[i] = 1/(counts[i]+1e-6)
        weightedLoss = counts / np.amin(counts)
        return weightedLoss        
            
            
    def filterData(self):
        newData = []
        for idx, sample in enumerate(self.data):
            jointsPath = sample["joints"]
            with (open(jointsPath, "rb")) as f:
                joints = pickle.load(f)
                velocity = joints["velocity"]
                meanFrontVel = (velocity[2] + velocity[3]) / 2
                if meanFrontVel >= 0:
                    newData.append(sample)
        self.data = newData