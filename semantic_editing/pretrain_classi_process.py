import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import scipy.io as io
import json
import MyFunc
import classifier
from options.test_options import TestOptions

# if Test
opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip


# if train
# opt = TrainOptions().parse()     #进行参数的设置和写入这些参数去opt.txt文件中
# opt.use_full_data = False
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

# if train
# if opt.continue_train:
#     try:
#         start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
#     except:
#         start_epoch, epoch_iter = 1, 0
#     print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
# else:
#     start_epoch, epoch_iter = 1, 0
#
# opt.print_freq = lcm(opt.print_freq, opt.batchSize)    #打印频率，乘积和最大公约数的商
# if opt.debug:
#     opt.display_freq = 1
#     opt.print_freq = 1
#     opt.niter = 1
#     opt.niter_decay = 0
#     opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

#-------------------------------

DesPath = './datasets2/cityscapes/train_scGraph/'
# FileName = 'ShortTransformMeanShapeSet'+str(dataset_size) +'.json'
FileName = 'ShortShapeSet'+ str(dataset_size) +'.json'              #在当前实验中,并未使用转换成固定规格之后的图片
DesFile = DesPath + FileName


ReadFile_MeanShape = open(DesFile, 'r', encoding='utf-8')
MeanShape_dict2 = json.load(ReadFile_MeanShape)
print("Test Read MeanShape_dict2 Successfully!")


#Start import my own scGraph for all images one by one


# AdjFileName = 'train_scAdj' + str(dataset_size) + '.mat'
# DesAdjSetFile = DesPath + AdjFileName
# AdjSet = io.loadmat(DesAdjSetFile)

classNum = 35
instNumEachClass = 25
ExcVal = 35
opt.numEdges = 3
opt.numNodes = classNum * instNumEachClass + opt.numEdges


SuitableObjeID = eval(opt.SuitableObjeID)

#Finish import my own scGraph for all images one by one
#-------------------------------
train_feature_list = []
label_list = []

projected_label = range(0, len(SuitableObjeID))
SuitableObjeID2ProjectedLabel = {}
ProjectedLabel2SuitableObjeID = {}

for i in range(len(projected_label)):
    SuitableObjeID2ProjectedLabel[str(SuitableObjeID[i])] = projected_label[i]
    ProjectedLabel2SuitableObjeID[i] = str(SuitableObjeID[i])

# LabelList = torch.tensor(range(len(SuitableObjeID))).unsqueeze(1)
# one_hot = torch.zeros(len(SuitableObjeID), len(SuitableObjeID)).scatter_(1, LabelList, 1)


for key, sublist in MeanShape_dict2.items():
    for i in range(len(sublist)):

        # mid = MyFunc.get_edges(torch.tensor(sublist[i]['Neighbor_Sem_Trans']))
        mid = MyFunc.pt_get_edges(torch.tensor(sublist[i]['Neighbor_Sem_Trans']))
        train_feature_list.append(mid)
        # label_list.append(one_hot[:, SuitableObjeID2ProjectedLabel[key]])
        label_list.append(torch.tensor([SuitableObjeID2ProjectedLabel[key]]).long())

print('finish Transform')
input_nc = train_feature_list[0].size()[1]
pool_nd = 10

model1 = classifier.CLASSIFIER(train_feature_list, label_list, input_nc, pool_nd, len(SuitableObjeID), 0.001, 0.5, 20, 1) #14
model1.fit()


#笔记：为进行val集的设定，可以在后续的实验中，改为验证集进行测试


if opt.gpu_ids[0] == 1:
    pretrain_classifier_path2 = 'TE_pretrained_classifier_gpu1.pkl'   # 'ED_pretrained_classifier_gpu1.pkl'
elif opt.gpu_ids[0] == 0:
    pretrain_classifier_path2 = 'TE_pretrained_classifier_gpu0.pkl'   # 'ED_pretrained_classifier_gpu0.pkl'
torch.save(model1, pretrain_classifier_path2)

model2 = torch.load(pretrain_classifier_path2)
acc2_single, acc2_whole = model2.cal_acc()

# print('model1 accuracy:', acc1)
print('model2 accuracy:', acc2_single, acc2_whole)


acc2_single, acc2_whole = model1.cal_acc()

# print('model1 accuracy:', acc1)
print('model1 accuracy:', acc2_single, acc2_whole)

print('end!~')







