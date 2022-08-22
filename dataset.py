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
        self.imageSize = params.imageSize
        self.path = params.path + params.dataset
        self.imgDir = self.path + "/img/"
        self.jntDir = self.path + "/joints/"
        imgPathList = sorted(glob.glob(self.imgDir + "*"))
        jntPathList = sorted(glob.glob(self.jntDir + "*"))
        self.data = [{} for _ in imgPathList]
        
        for idx, imgPath in enumerate(imgPathList):
            # timeStamp = imgPath.split("/")[-1]
            # timeStamp = timeStamp.split(".")[0]
            # jointPath = self.jointDir + timeStamp + ".pkl"
            jointPath = jntPathList[idx]
            self.data[idx] = {"img": imgPath, "joints": jointPath}
            
        splitIdx = int(.8 * len(self.data))
        if self.splitMode == "train":
            self.data = self.data[:splitIdx]
        else:
            self.data = self.data[splitIdx:]
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dataPaths = self.data[idx]
        imgPath = dataPaths["img"]
        jointsPath = dataPaths["joints"]
        transform = transforms.Compose([transforms.Resize(self.imageSize),
            transforms.RandomCrop(self.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        img = transform(Image.open(imgPath))
        
        try:
            with (open(jointsPath, "rb")) as f:
                joints = pickle.load(f)
                velocity = joints["velocity"]
                if velocity[0] > velocity[1]:
                    label = 1
                else:
                    label = 0
        except OSError:
            print("File Not Found" + jointsPath)
            label = 0
        return img, label