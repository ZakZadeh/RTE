import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as func
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import sklearn.metrics as skm
import dataset
import model
import utils

""" -------------------------------------------------------------------------"""
""" Parameters """
parser = argparse.ArgumentParser(description = 'Robot Traversability Estimation')
parser.add_argument('--path', type=str, default='/data/zak/rosbag/labeled/', help = 'Dataset path')
parser.add_argument('--trainSet', type=str, default='heracleia', help = 'Train Set: heracleia, mocap, uc, erb')
parser.add_argument('--testSet', type=str, default='mocap', help = 'Test Set')
parser.add_argument('--encoder', type=str, default="CNN4", help = 'Encoder Model: ResNet50, CNN4')
parser.add_argument('--predictor', type=str, default="MLP4", help = 'Predictor Model: MLP4')
parser.add_argument('--nEpochs', type=int, default = 10, help = 'Num of training epochs')
parser.add_argument('--nBatch', type=int, default = 512, help = 'Batch Size')
parser.add_argument('--imageDim', type=int, default = 256, help = 'Image Dimension')
parser.add_argument('--laserDim', type=int, default = 1081, help = 'Laser Dimension')
parser.add_argument('--startEpoch', type=int, default = 0, help = 'Num of epochs to resume from')
parser.add_argument('--nGPU', type=int, default = 2, help = 'Num of GPUs available. CPU: 0')
parser.add_argument('--plotLoss', type=bool, default = False)
parser.add_argument('--nCheckpoint', type=int, default = 1, help = 'Num of epochs for checkpoints')
parser.add_argument('--saveCheckpoint', type=bool, default = False)
params = parser.parse_args()

""" Initializing """
device = torch.device("cuda:0" if (torch.cuda.is_available() and params.nGPU > 0) else "cpu")

if (params.encoder == "CNN4"):
    params.nChannel = 3
    params.nFeature = 512
    netEncoder = model.CNN4(params).to(device)
if (params.encoder == "ResNet50"):
    params.nChannel = 3
    params.nFeature = 2048
    netEncoder = model.ResNet50(params).to(device)
    nodes, _ = get_graph_node_names(netEncoder)
    netEncoder = create_feature_extractor(netEncoder, return_nodes=['resnet.0.flatten'])

if (params.predictor == "MLP4"):
    params.nClass = 2
    netPredictor = model.MLP4(params).to(device)
    
# Resume Training
if params.startEpoch != 0:
    netEncoder, netPredictor = utils.loadCkpt(params)
    netEncoder = netEncoder.to(device)
    netPredictor = netPredictor.to(device)
    
if (device.type == 'cuda') and (params.nGPU > 1):
    netEncoder = nn.DataParallel(netEncoder, list(range(params.nGPU)))
    netPredictor = nn.DataParallel(netPredictor, list(range(params.nGPU)))

if (params.encoder == "ResNet50"):
    allWeights = list(netPredictor.parameters())
else:
    allWeights = list(netEncoder.parameters()) + list(netPredictor.parameters())
optimizer = optim.Adam(allWeights, lr = 0.001, betas = (0.5, 0.999))
criterion = nn.CrossEntropyLoss()
    
""" Training """
print("Starting Training Loop...")
trainLoader = DataLoader(dataset.Rosbag('train', params), batch_size = params.nBatch, shuffle = True)
testLoader  = DataLoader(dataset.Rosbag('test', params),  batch_size = params.nBatch, shuffle = True)
trnLosses, tstLosses = [], []
accs = []

for epoch in range(params.startEpoch, params.nEpochs):
    trnLabel, trnPred = [], []
    tstLabel, tstPred = [], []

    for data, label in trainLoader:
        netEncoder.zero_grad()
        netPredictor.zero_grad()
        data, label = data.to(device), label.to(device)
        # print(data.size(), label.size())
        feature = netEncoder(data)
        if params.encoder == "ResNet50":
            feature = feature['resnet.0.flatten']
        # print(feature.size())
        output = netPredictor(feature)
        labelPred = torch.max(func.softmax(output, dim = 1), 1)[1]
        err = criterion(output, label)
        err.backward()
        optimizer.step()

        trnLosses.append(err.item())
        trnLabel.append(label)
        trnPred.append(labelPred)

    with torch.no_grad():
        for data, label in testLoader:
            data, label = data.to(device), label.to(device)
            feature = netEncoder(data)
            if params.encoder == "ResNet50":
                feature = feature['resnet.0.flatten']
            output = netPredictor(feature)
            labelPred = torch.max(func.softmax(output, dim = 1), 1)[1]
            err = criterion(output, label)

            tstLosses.append(err.item())
            tstLabel.append(label)
            tstPred.append(labelPred)

    trnLabels = torch.cat(trnLabel, dim = 0)
    trnPreds  = torch.cat(trnPred,  dim = 0)
    tstLabels = torch.cat(tstLabel, dim = 0)
    tstPreds  = torch.cat(tstPred,  dim = 0)

    trnAccuracy = skm.accuracy_score(trnLabels.cpu(), trnPreds.cpu()) * 100
    tstAccuracy = skm.accuracy_score(tstLabels.cpu(), tstPreds.cpu()) * 100
    accs.append(tstAccuracy)

    if ((epoch+1) % params.nCheckpoint == 0):
        print('[%d/%d]\tTrain Loss: %.4f\tTrain Accuracy: %.4f'
                % (epoch+1, params.nEpochs, err.item(), trnAccuracy))
        print('[%d/%d]\tTest Loss: %.4f\tTest Accuracy: %.4f'
                % (epoch+1, params.nEpochs, err.item(), tstAccuracy))
        if params.saveCheckpoint:
            utils.saveCkpt(netEncoder, netPredictor, epoch + 1, params)
if params.plotLoss:
    utils.showLoss(trnLosses, testLosses)
maxAcc = max(accs)
maxEpoch = accs.index(maxAcc)
print('[Max]\tEpoch: %d\tTest Accuracy: %.4f' % (maxEpoch+1, maxAcc))

""" Results """
if params.saveCheckpoint:
    utils.saveCkpt(netEncoder, netPredictor, params.nEpochs, params)