import os
import glob
import pickle
import numpy as np 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import rosbag
import rospy
from PIL import Image
import model
import pandas as pd

def makeFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def showLoss(trnLosses, testLosses):
    plt.figure(figsize = (10,5))
    plt.title("Loss During Training & Testing")
    plt.plot(trnLosses,  label = 'Train')
    plt.plot(testLosses, label = 'Test')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def saveCkpt(netImageEncoder, netLaserEncoder, netProjector, netPredictor, epoch, params):
    resultDir = params.path + params.dataset + '/result/'
    makeFolder(resultDir)
    modelDir = resultDir + 'model/'
    makeFolder(modelDir)
    torch.save(netImageEncoder, os.path.join(modelDir, '{:05d}_image_encoder_'.format(epoch) + params.model + '.pth'))
    torch.save(netLaserEncoder, os.path.join(modelDir, '{:05d}_laser_encoder_'.format(epoch) + params.model + '.pth'))
    torch.save(netProjector, os.path.join(modelDir, '{:05d}_projector_'.format(epoch) + params.model + '.pth'))
    torch.save(netPredictor, os.path.join(modelDir, '{:05d}_predictor_'.format(epoch) + params.model + '.pth'))

def loadCkpt(params):
    modelDir = params.path + params.dataset + '/result/model/'
    epoch = params.startEpoch
    imageEncoderName = os.path.join(modelDir, '{:05d}_image_encoder_'.format(epoch) + params.model + '.pth')
    laserEncoderName = os.path.join(modelDir, '{:05d}_laser_encoder_'.format(epoch) + params.model + '.pth')
    projectorName = os.path.join(modelDir, '{:05d}_projector_'.format(epoch) + params.model + '.pth')
    predictorName = os.path.join(modelDir, '{:05d}_predictor_'.format(epoch) + params.model + '.pth')
    imageEncoder = torch.load(imageEncoderName)
    laserEncoder = torch.load(laserEncoderName)
    projector = torch.load(projectorName)
    predictor = torch.load(predictorName)
    return imageEncoder, laserEncoder, projector, predictor

def readRosbag(labelPairs):
    labelPairs = {"/data/zak/robot/rosbag/Data-08-12-22-Time-17-32-56.bag": "/data/zak/robot/extracted/heracleia/8_12_un/",
#                   "/data/zak/robot/rosbag/Data-09-09-22-Time-15-51-29.bag": "/data/zak/robot/extracted/uc/9_10/",
#                   "/data/zak/robot/rosbag/Data-09-09-22-Time-15-58-30.bag": "/data/zak/robot/extracted/uc/9_11/",
                 }

    for bagDir, outDir in labelPairs.items():
        bag = rosbag.Bag(bagDir)
        # bagInfo = bag.get_type_and_topic_info()[1]
        # topics = bagInfo.keys()
        # print(topics)
        makeFolder(outDir)
        jointDir = outDir + "joints/"
        imgDir = outDir + "img/"
        frontLaserDir = outDir + "front_laser/"
        makeFolder(jointDir)
        makeFolder(imgDir)
        makeFolder(frontLaserDir)

        for topic, msg, t in bag.read_messages():
            # print(msg.header)
            timeStamp = msg.header.stamp
            # times.append(timeStamp.to_sec())

            if topic == "joints":
                jointDict = {}
                jointDict['name'] = msg.name
                jointDict['position'] = msg.position
                jointDict['velocity'] = msg.velocity
                jointDict['effort']   = msg.effort
                name = jointDir + str(timeStamp) + ".pkl"
                with open(name, "wb") as f:
                    pickle.dump(jointDict, f)

            if topic == 'img':
                img = np.frombuffer(msg.data, dtype=np.uint8)
                img = img.reshape(msg.height, msg.width, 3)
                img = Image.fromarray(img)
                img.save(imgDir + str(timeStamp) + ".jpg")

            if topic == "front_laser":
                frontLaserDict = {}
                frontLaserDict["angle_min"] = msg.angle_min
                frontLaserDict["angle_max"] = msg.angle_max
                frontLaserDict["angle_increment"] = msg.angle_increment
                frontLaserDict["time_increment"] = msg.time_increment
                frontLaserDict["scan_time"] = msg.scan_time
                frontLaserDict["range_min"] = msg.range_min
                frontLaserDict["range_max"] = msg.range_max
                frontLaserDict["ranges"] = msg.ranges
                name = frontLaserDir + str(timeStamp) + ".pkl"
                with open(name, "wb") as f:
                    pickle.dump(frontLaserDict, f)
                    
def writeInfo(dataset, path = '/data/zak/robot/'):
    inPath = path + "extracted/" + dataset + "/"
    data = []
    datePathList = sorted(glob.glob(inPath + "*"))
    for datePath in datePathList:
        date = datePath.split("/")[-1]
        if len(date.split("_")) < 3:
            jointLabeled = True
        else:
            jointLabeled = False
        modeList = sorted(glob.glob(datePath + "/*"))
        img, jnt, lsr = [], [], []
        for modePath in modeList:
            mode = modePath.split("/")[-1]
            filePaths = sorted(glob.glob(modePath + "/*"))
            for filePath in filePaths:
                if mode == "img":
                    img.append(filePath)
                elif mode == "joints":
                    jnt.append(filePath)
                elif mode == "front_laser":
                    lsr.append(filePath)
        for i in range(len(img)):
            dataSingle = {}
            dataSingle['dataset'] = dataset
            dataSingle['date'] = date
            dataSingle['jointLabeled'] = jointLabeled
            dataSingle['img'] = img[i]
            dataSingle['joints'] = jnt[i]
            if i < len(lsr):
                dataSingle['front_laser'] = lsr[i]
            else:
                dataSingle['front_laser'] = None
            dataSingle['label'] = None
            data.append(dataSingle)
    df = pd.DataFrame.from_dict(data)
    outPath = path + "labels/" + dataset + "/info.csv"
    df.to_csv(outPath, index = True, header = True)
    
def writeLabel(datasetPath, path = '/data/zak/robot/'):
    inPath = path + "labels/" + datasetPath + "/" + "info.csv"
    data = pd.read_csv(inPath, index_col = 0)
    index = data.index.tolist()
    dataset = data['dataset'].tolist()
    date = data['date'].tolist()
    jointLabeled = data['jointLabeled'].tolist()
    img = data['img'].tolist()
    joints = data['joints'].tolist()
    laser = data['front_laser'].tolist()
    label = data['label'].tolist()
        
    filteredData = []
    for i in index:
        if jointLabeled[i]:
            try:
                with (open(joints[i], "rb")) as f:
                    velocity = pickle.load(f)["velocity"]
                    meanVel = sum(velocity) / 4
                    if min(velocity) >= 1:
                        label = 1
                    elif meanVel >= -0.1 and meanVel <= 0.1:
                        label = 0
                    else:
                        continue
            except OSError:
                contniue
            dataSingle = {}
            dataSingle['dataset'] = dataset[i]
            dataSingle['date'] = date[i]
            dataSingle['img'] = img[i]
            dataSingle['joints'] = joints[i]
            if i < len(laser):
                dataSingle['front_laser'] = laser[i]
            else:
                dataSingle['front_laser'] = None
            dataSingle['label'] = label
            filteredData.append(dataSingle)
    # print(len(filteredData))
    df = pd.DataFrame.from_dict(filteredData)
    outPath = path + "labels/" + datasetPath + "/labels.csv"
    df.to_csv(outPath, index = True, header = True)