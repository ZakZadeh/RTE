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
from sklearn.manifold import TSNE
import seaborn as sns


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
    
def showTSNE(x, y, epoch, params):
    tsne = TSNE(n_components = 2)
    tsneResults = tsne.fit_transform(x)
    tsneResultsDF = pd.DataFrame({'tsne1': tsneResults[:,0], 'tsne2': tsneResults[:,1], 'label': y})
    outDir = params.path + "/results/"
    fig, ax = plt.subplots(1)
    sns.scatterplot(x = 'tsne1', y = 'tsne2', hue = 'label', data = tsneResultsDF, ax = ax, s = 120)
    lim = (tsneResults.min() - 5, tsneResults.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.0)
    plt.savefig(os.path.join(outDir, '{:05d}.png'.format(epoch)))

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
    labelPairs = {
#                   "/blue3/Indoor_Data/Data-09-27-22-Time-12-32-44.bag": "/data/zak/robot/extracted/elb/9_27_1/",
#                   "/blue3/Indoor_Data/Data-09-27-22-Time-12-41-13.bag": "/data/zak/robot/extracted/elb/9_27_2/",
#                   "/blue3/Indoor_Data/Data-09-27-22-Time-12-49-41.bag": "/data/zak/robot/extracted/elb/9_27_3/",
                  "/blue3/Indoor_Data/Data-09-27-22-Time-12-56-25.bag": "/data/zak/robot/extracted/wh/9_27_1/",
                  "/blue3/Indoor_Data/Data-09-27-22-Time-13-00-12.bag": "/data/zak/robot/extracted/wh/9_27_2/",
                  "/blue3/Indoor_Data/Data-09-27-22-Time-13-07-48.bag": "/data/zak/robot/extracted/wh/9_27_3/",
                  "/blue3/Indoor_Data/Data-09-27-22-Time-13-11-09.bag": "/data/zak/robot/extracted/wh/9_27_4/",
                  "/blue3/Indoor_Data/Data-09-27-22-Time-13-15-19.bag": "/data/zak/robot/extracted/wh/9_27_5/",
                  "/blue3/Indoor_Data/Data-09-27-22-Time-13-16-42.bag": "/data/zak/robot/extracted/wh/9_27_6/",
                 }

    for bagDir, outDir in labelPairs.items():
        try:
            print(bagDir)
            bag = rosbag.Bag(bagDir)
            bagInfo = bag.get_type_and_topic_info()[1]
            topics = bagInfo.keys()
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
        except:
            print("Error: " + bagDir)
                    
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

def getLabelByJoint(vel):
    if min(vel) >= -.5 and max(vel) <= .5:
        return 0
    elif min(vel) >= 1:
        return 1
    return -1
    
def writeLabelByJoint(datasetPath, path = '/data/zak/robot/'):
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
                    label = getLabelByJoint(velocity)
                    if label == -1:
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
    outPath = path + "labels/" + datasetPath + "/labels_joint.csv"
    df.to_csv(outPath, index = False, header = True)