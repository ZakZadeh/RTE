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

def saveCkpt(netEncoder, netPredictor, epoch, params):
    resultDir = params.path + params.dataset + '/result/'
    makeFolder(resultDir)
    modelDir = resultDir + 'model/'
    makeFolder(modelDir)
    torch.save(netEncoder, os.path.join(modelDir, '{:05d}_encoder_'.format(epoch) + params.model + '.pth'))
    torch.save(netPredictor, os.path.join(modelDir, '{:05d}_predictor_'.format(epoch) + params.model + '.pth'))

def loadCkpt(params):
    modelDir = params.path + params.dataset + '/result/model/'
    epoch = params.startEpoch
    encoderName = os.path.join(modelDir, '{:05d}_encoder_'.format(epoch) + params.model + '.pth')
    predictorName = os.path.join(modelDir, '{:05d}_predictor_'.format(epoch) + params.model + '.pth')
    encoder = torch.load(encoderName)
    predictor = torch.load(predictorName)
    return encoder, predictor
