import os
import model
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
