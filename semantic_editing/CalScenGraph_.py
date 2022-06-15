import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions

import MyFunc
import json
import scipy.io as io

def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

#-----------------------------------------------
#customize setting
classNum = 35
instNumEachClass = 25
realationNum = 3
AdjSize = classNum * instNumEachClass + realationNum
#-----------------------------------------------


opt = TrainOptions().parse()     #进行参数的设置和写入这些参数去opt.txt文件中
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    #打印频率，乘积和最大公约数的商
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

scGraph_dict = {}
BinaryAdjSet = torch.zeros((dataset_size, AdjSize, AdjSize))
RealAdjSet = torch.zeros((dataset_size, AdjSize, AdjSize))

testScGraph = True

for i, data in enumerate(dataset, start=epoch_iter):
    boxes, object_centers, instIDs = MyFunc.CalBoundry(data)
    detailInfo = {}
    detailInfo['boxes'] = boxes
    detailInfo['object_centers'] = object_centers
    detailInfo['instIDs'] = instIDs
    detailInfo['orderID'] = i
    print('Finsh image transfer: ', i)
    # 开始进行两个物体间距离的联系, 产生scene graph
    method = 'Distance'
    scGraph = MyFunc.CalSceneGraph(boxes, object_centers, instIDs, method)
    if testScGraph == True:
        testRes = []
        for subtuple in scGraph:
            for subsubtuple in subtuple:
                testRes.append(len(subsubtuple))
    detailInfo['scGraph'] = scGraph
    singleBracket_scGraph = MyFunc.transferScGraph2AdjMat(scGraph)
    AdjBinaryVal, AdjRealVal = MyFunc.CalTwoKindsAdjs(singleBracket_scGraph, classNum, instNumEachClass)
    BinaryAdjSet[i, :, :] = AdjBinaryVal
    RealAdjSet[i, :, :] = AdjRealVal

    assert len(data['path']) == 1
    scGraph_dict[data['path'][0]] = detailInfo
    # print(scGraph.__len__())



#save as json file
if opt.use_full_data:
    DesPath = './datasets2/cityscapes/train_scGraph/'
else:
    DesPath = './datasets/cityscapes/train_scGraph/'
# DesPath = './datasets2/cityscapes/train_scGraph/'
FileName = 'train_scGraph'+str(dataset_size)+'.json'
DesFile = DesPath + FileName


BinAdjFileName = 'train_scAdj' + 'BinaryAdjSet' + str(dataset_size) + '.npy'
BinDesAdjSetFile = DesPath + BinAdjFileName

RealAdjFileName = 'train_scAdj' + 'RealAdjSet' + str(dataset_size) + '.npy'
RealDesAdjSetFile = DesPath + RealAdjFileName

MyFunc.mkdir1(DesPath)

BinaryAdjSet = np.array(BinaryAdjSet)
RealAdjSet = np.array(RealAdjSet)
# io.savemat(DesAdjSetFile, {'BinaryAdjSet': BinaryAdjSet, 'RealAdjSet': RealAdjSet}, long_field_names=False, do_compression=True)
# io.savemat(DesAdjSetFile, {'BinaryAdjSet': BinaryAdjSet, 'RealAdjSet': RealAdjSet})

np.save(BinDesAdjSetFile, BinaryAdjSet)
np.save(RealDesAdjSetFile, RealAdjSet)
BinaryAdjSet2 = np.load(BinDesAdjSetFile)
RealAdjSet2 = np.load(RealDesAdjSetFile)

SaveFile_scGraph = open(DesFile, 'w',encoding='utf-8')
json.dump(scGraph_dict, SaveFile_scGraph, ensure_ascii=False)
SaveFile_scGraph.close()

#read saved json file
testRead = False
if testRead == True:
    ReadFile_scGraph = open(DesFile, 'r', encoding='utf-8')
    Read_scGraph = json.load(ReadFile_scGraph)
    print("Test Read Scene Graph Successfully!")









