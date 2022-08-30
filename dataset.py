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
        self.path = params.path
        if self.splitMode == "train":
            self.dir = self.path + "/" + self.trainSet
        else:
            self.dir = self.path + "/" + self.testSet
        self.imgDir = self.dir + "/img/"
        self.jntDir = self.dir + "/joints/"       
        self.fLsrDir = self.dir + "/front_laser/"       
        imgPathList = sorted(glob.glob(self.imgDir + "*"))
        jntPathList = sorted(glob.glob(self.jntDir + "*"))
        fLsrPathList = sorted(glob.glob(self.fLsrDir + "*"))
        self.data = [{} for _ in imgPathList]
        
        for idx, imgPath in enumerate(imgPathList):
            jointPath = jntPathList[idx]
            fLaserPath = fLsrPathList[idx]
            self.data[idx] = {"img": imgPath, "joints": jointPath, "front_laser": fLaserPath}
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dataPaths = self.data[idx]
        imgPath = dataPaths["img"]
        jointsPath = dataPaths["joints"]
        fLaserPath = dataPaths["front_laser"]
        transform = transforms.Compose([transforms.Resize(self.imageDim),
            transforms.RandomCrop(self.imageDim),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        img = transform(Image.open(imgPath))
        
        try:
            with (open(jointsPath, "rb")) as f:
                joints = pickle.load(f)
                velocity = joints["velocity"]
                meanFrontVel = (velocity[2] + velocity[3]) / 2
                if meanFrontVel >= 1:
                    label = 1
                else:
                    label = 0
        except OSError:
            print("File Not Found" + jointsPath)
            label = 0
        return img, label