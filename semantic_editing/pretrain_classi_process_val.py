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
import classifier_val

# 上述描写中表明使用BCE_loss会导致不稳定的求导，这时使用BCEWithLogitsLoss()函数即可。
#This is change from JF's Mac
#This is change from JF's Ubuntu
opt = TrainOptions().parse()     #进行参数的设置和写入这些参数去opt.txt文件中
# opt.use_full_data = False
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

#-------------------------------

DesPath = './datasets/cityscapes/train_scGraph/'
# FileName = 'ShortTransformMeanShapeSet'+str(dataset_size) +'.json'
FileName = 'ShortShapeSet'+ '2975' +'.json'
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
xl = {}


projected_label = range(0, len(SuitableObjeID))
SuitableObjeID2ProjectedLabel = {}
ProjectedLabel2SuitableObjeID = {}

for i in range(len(projected_label)):
    SuitableObjeID2ProjectedLabel[str(SuitableObjeID[i])] = projected_label[i]
    ProjectedLabel2SuitableObjeID[i] = str(SuitableObjeID[i])
    xl[i] = []

# LabelList = torch.tensor(range(len(SuitableObjeID))).unsqueeze(1)
# one_hot = torch.zeros(len(SuitableObjeID), len(SuitableObjeID)).scatter_(1, LabelList, 1)

jsq = 0
for key, sublist in MeanShape_dict2.items():
    for i in range(len(sublist)):
        # mid = MyFunc.get_edges(torch.tensor(sublist[i]['Neighbor_Sem_Trans']))
        mid = MyFunc.pt_get_edges(torch.tensor(sublist[i]['Neighbor_Sem_Trans']))
        train_feature_list.append(mid)
        # label_list.append(one_hot[:, SuitableObjeID2ProjectedLabel[key]])
        label_list.append(torch.tensor([SuitableObjeID2ProjectedLabel[key]]).long())
        xl[SuitableObjeID2ProjectedLabel[key]].append(jsq)
        jsq += 1

#start devide dataset
tr_xl = []
val_xl = []
val_rate = 0.1
for key, sublist in xl.items():
    sub_len = len(sublist)
    val_len = int(val_rate * sub_len)
    tr_len = sub_len - val_len
    val_xl.extend(sublist[0: val_len])
    tr_xl.extend(sublist[val_len:])

tr_feature_list = []
vl_feature_list = []
tr_label_list = []
vl_label_list = []
for i in range(len(tr_xl)):
    tr_feature_list.append(train_feature_list[tr_xl[i]])
    tr_label_list.append(label_list[tr_xl[i]])

for i in range(len(val_xl)):
    vl_feature_list.append(train_feature_list[val_xl[i]])
    vl_label_list.append(label_list[val_xl[i]])


print('finish Transform')
input_nc = train_feature_list[0].size()[1]
pool_nd = 10

model1 = classifier_val.CLASSIFIER(tr_feature_list, tr_label_list, vl_feature_list, vl_label_list, input_nc, pool_nd, len(SuitableObjeID), 0.001, 0.5, 20, 1) #14
model1.fit()


#笔记：为进行val集的设定，可以在后续的实验中，改为验证集进行测试


if opt.gpu_ids[0] == 1:
    pretrain_classifier_path2 = 'ED_val_pretrained_classifier_gpu1.pkl'
elif opt.gpu_ids[0] == 0:
    pretrain_classifier_path2 = 'ED_val_pretrained_classifier_gpu0.pkl'
torch.save(model1, pretrain_classifier_path2)

model2 = torch.load(pretrain_classifier_path2)
acc2_single, acc2_whole = model2.cal_acc(vl_feature_list, vl_label_list)

# print('model1 accuracy:', acc1)
print('model2 accuracy:', acc2_single, acc2_whole)

acc2_single, acc2_whole = model1.cal_acc()

# print('model1 accuracy:', acc1)
print('model1 accuracy:', acc2_single, acc2_whole)

print('end!~')







