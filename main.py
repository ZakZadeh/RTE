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
from sklearn.metrics import confusion_matrix
import dataset
import model
import utils

""" -------------------------------------------------------------------------"""
""" Parameters """
parser = argparse.ArgumentParser(description = 'Robot Traversability Estimation')
parser.add_argument('--path', type=str, default='/data/zak/robot', help = 'Dataset path')
parser.add_argument('--trainSet', type=str, default = ['heracleia', 'mocap', 'nh', 'uc'], help ='Train Set')
parser.add_argument('--testSet', type=str, default = ['wh'], help = 'Test Set: heracleia, mocap, nh, uc, wh')
parser.add_argument('--imageEncoder', type=str, default = "ViT", help = 'ResNet50, EffNetM, ViT')
parser.add_argument('--useLaser', type=bool, default= False, help = "Uses Laser or Not")
parser.add_argument('--projector', type=str, default = "Identity", help = 'Identity, MLP2, CatFusion, AttenFusion')
parser.add_argument('--predictor', type=str, default = "Lin", help = 'Lin')
parser.add_argument('--nEpochs', type=int, default = 50, help = 'Num of training epochs')
parser.add_argument('--nBatch', type=int, default = 128, help = 'Batch Size')
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

if params.imageEncoder == "ResNet50":
    params.nChannel = 3
    params.nFeature = 2048
    netEncoder = model.ResNet50(params).to(device)
    nodes, _ = get_graph_node_names(netEncoder)
    netEncoder = create_feature_extractor(netEncoder, return_nodes=['resnet.0.flatten'])
if params.imageEncoder == "EffNet":
    params.nChannel = 3
    params.nFeature = 1280
    netEncoder = model.EffNet(params).to(device)
    nodes, _ = get_graph_node_names(netEncoder)
    netEncoder = create_feature_extractor(netEncoder, return_nodes=['effnet.0.flatten'])
if params.imageEncoder == "ViT":
    params.imageDim = 224
    params.nChannel = 3
    params.nFeature = 768
    netEncoder = model.ViT(params).to(device)
    nodes, _ = get_graph_node_names(netEncoder)
    netEncoder = create_feature_extractor(netEncoder, return_nodes=['vit.0.getitem_5'])
    
if params.projector == "Identity":
    netProjector = model.Identity(params).to(device)  
elif params.projector == "MLP2":
    netProjector = model.MLP2(params).to(device)
elif params.projector == "CatFusion":
    netProjector = model.CatFusion(params).to(device)
elif params.projector == "AttenFusion":
    netProjector = model.CatFusion(params).to(device)

if (params.predictor == "Lin"):
    params.nClass = 2
    netPredictor = model.Lin(params).to(device)

# Resume Training
if params.startEpoch != 0:
    netEncoder, netProjector, netPredictor = utils.loadCkpt(params)
    netEncoder = netEncoder.to(device)
    netPredictor = netPredictor.to(device)
    netProjector = netProjector.to(device)
    
if (device.type == 'cuda') and (params.nGPU > 1):
    netEncoder = nn.DataParallel(netEncoder, list(range(params.nGPU)))
    netPredictor = nn.DataParallel(netPredictor, list(range(params.nGPU)))
    netProjector = nn.DataParallel(netProjector, list(range(params.nGPU)))

allWeights = list(netProjector.parameters()) + list(netPredictor.parameters()) 
optimizer = optim.Adam(allWeights, lr = 0.001, betas = (0.5, 0.999))
    
""" Training """
print("Starting Training Loop...")
trainLoader = DataLoader(dataset.Rosbag('train', params), batch_size = params.nBatch, shuffle = True)
testLoader  = DataLoader(dataset.Rosbag('test', params),  batch_size = params.nBatch, shuffle = True)

weightedLoss = torch.tensor(params.weightedLoss, dtype = torch.float, device = device)
criterion = nn.CrossEntropyLoss(weightedLoss)

trnLosses, tstLosses = [], []
accs = []

for epoch in range(params.startEpoch, params.nEpochs):
    trnLabel, trnPred = [], []
    tstLabel, tstPred = [], []
    tstFeature = []

    for image, label, laser in trainLoader:
        netEncoder.zero_grad()
        netProjector.zero_grad()
        netPredictor.zero_grad()
        image, label, laser = image.to(device), label.to(device), laser.to(device)
    
        imageFeature = netEncoder(image)
        if params.imageEncoder == "ResNet50":
            imageFeature = imageFeature['resnet.0.flatten']
        elif params.imageEncoder == "EffNet":
            imageFeature = imageFeature['effnet.0.flatten']
        elif params.imageEncoder == "ViT":
            imageFeature = imageFeature['vit.0.getitem_5']
        if params.useLaser:
            feature = netProjector(imageFeature, laser)
        else:
            feature = netProjector(imageFeature)

        output = netPredictor(feature)
        labelPred = torch.max(func.softmax(output, dim = 1), 1)[1]
        err = criterion(output, label)
        err.backward()
        optimizer.step()

        trnLosses.append(err.item())
        trnLabel.append(label)
        trnPred.append(labelPred)

    with torch.no_grad():
        for image, label, laser in testLoader:
            image, label, laser = image.to(device), label.to(device), laser.to(device)
            
            imageFeature = netEncoder(image)
            if params.imageEncoder == "ResNet50":
                imageFeature = imageFeature['resnet.0.flatten']
            elif params.imageEncoder == "EffNet":
                imageFeature = imageFeature['effnet.0.flatten']
            elif params.imageEncoder == "ViT":
                imageFeature = imageFeature['vit.0.getitem_5']
            if params.useLaser:
                feature = netProjector(imageFeature, laser)
            else:
                feature = netProjector(imageFeature)
            
            output = netPredictor(feature)
            labelPred = torch.max(func.softmax(output, dim = 1), 1)[1]
            err = criterion(output, label)

            tstLosses.append(err.item())
            tstLabel.append(label)
            tstPred.append(labelPred)
            if ((epoch+1) % params.nCheckpoint == 0):
                tstFeature.append(feature)
    
    trnLabels   = torch.cat(trnLabel, dim = 0)
    trnPreds    = torch.cat(trnPred, dim = 0)
    tstLabels   = torch.cat(tstLabel, dim = 0)
    tstPreds    = torch.cat(tstPred, dim = 0)
    tstFeatures = torch.cat(tstFeature, dim = 0)

    trnAccuracy = skm.accuracy_score(trnLabels.cpu(), trnPreds.cpu()) * 100
    tstAccuracy = skm.accuracy_score(tstLabels.cpu(), tstPreds.cpu()) * 100
    confusionMatrix = confusion_matrix(tstLabels.cpu(), tstPreds.cpu())
    accs.append(tstAccuracy)

    if ((epoch+1) % params.nCheckpoint == 0):
        print('[%d/%d]\tTrain Loss: %.4f\tTrain Accuracy: %.4f'
                % (epoch+1, params.nEpochs, err.item(), trnAccuracy))
        print('[%d/%d]\tTest Loss: %.4f\tTest Accuracy: %.4f'
                % (epoch+1, params.nEpochs, err.item(), tstAccuracy))
        print(confusionMatrix)
        #utils.showTSNE(tstFeatures.cpu(), tstLabels.cpu(), epoch, params)
        if params.saveCheckpoint:
            utils.saveCkpt(netImageEncoder, netLaserEncoder, netProjector, netPredictor, epoch + 1, params)
if params.plotLoss:
    utils.showLoss(trnLosses, testLosses)
maxAcc = max(accs)
maxEpoch = accs.index(maxAcc)
print('[Max]\tEpoch: %d\tTest Accuracy: %.4f' % (maxEpoch+1, maxAcc))

""" Results """
if params.saveCheckpoint:
    utils.saveCkpt(netImageEncoder, netLaserEncoder, netProjector, netPredictor, params.nEpochs, params)