import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0


from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import scipy.io as io
import json
import MyFunc

# 上述描写中表明使用BCE_loss会导致不稳定的求导，这时使用BCEWithLogitsLoss()函数即可。
# This is change from JF's Mac
# This is change from JF's Ubuntu
opt = TrainOptions().parse()  # 进行参数的设置和写入这些参数去opt.txt文件中
# opt.use_full_data = False
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
	try:
		start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
	except:
		start_epoch, epoch_iter = 1, 0
	print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
	start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)  # 打印频率，乘积和最大公约数的商
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

# -------------------------------


removalOjbID = 'random'
randomSeedID = None
PredefinedSize = -1

# Start import my own scGraph for all images one by one
if opt.use_full_data:
	DesPath = './datasets2/cityscapes/train_scGraph/'
else:
	DesPath = './datasets/cityscapes/train_scGraph/'

if opt.is_scGraph == 2:
	FileName = 'train_scGraph' + str(dataset_size) + '.json'
	DesFile = DesPath + FileName
	ReadFile_scGraph = open(DesFile, 'r', encoding='utf-8')
	Read_scGraph = json.load(ReadFile_scGraph)

	BinAdjFileName = 'train_scAdj' + 'BinaryAdjSet' + str(dataset_size) + '.npy'
	BinDesAdjSetFile = DesPath + BinAdjFileName
	RealAdjFileName = 'train_scAdj' + 'RealAdjSet' + str(dataset_size) + '.npy'
	RealDesAdjSetFile = DesPath + RealAdjFileName

	BinaryAdjSet = np.load(BinDesAdjSetFile)
	RealAdjSet = np.load(RealDesAdjSetFile)
	print("Test Read Scene Graph Successfully!")

# AdjFileName = 'train_scAdj' + str(dataset_size) + '.mat'
# DesAdjSetFile = DesPath + AdjFileName
# AdjSet = io.loadmat(DesAdjSetFile)

classNum = 35
instNumEachClass = 25
ExcVal = 35
opt.numEdges = 3
opt.numNodes = classNum * instNumEachClass + opt.numEdges

SuitableObjeID = eval(opt.SuitableObjeID)
Neighbor_Sem_ObjShape_Set = []

use_prior_obj_feature = False
if not use_prior_obj_feature:
	InitNodeEmb = torch.cuda.LongTensor(range(opt.numNodes))

# Finish import my own scGraph for all images one by one
# -------------------------------

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

epoch = 1
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
	print(epoch)
	if epoch != start_epoch:
		epoch_iter = epoch_iter % dataset_size
	for i, data in enumerate(dataset, start=epoch_iter):
		if total_steps % opt.print_freq == print_delta:
			iter_start_time = time.time()
		total_steps += opt.batchSize
		epoch_iter += opt.batchSize

		if PredefinedSize == -1 and opt.objFilterCoe >= 0:
			InstSize = data['inst'].size()
			PredefinedSize = InstSize[2] * InstSize[3] * opt.objFilterCoe

		# whether to collect output images


		if opt.is_scGraph == 3:  # 使用手动输入一个item
			# ----------------------------------------------------------
			# Below is add by Jianfeng He
			try:
				InComLableMap, InComInstMap, InComImg, SelectedObj, SelectedCalss, SelectedBox, isNone = \
					MyFunc.SelfSupervisePrePro_Basic_temporary(data['label'], data['inst'], data['image'], data['feat'],
					                                 SuitableObjeID, ExcVal, PredefinedSize, removalOjbID, randomSeedID)
			except:
				continue
			if isNone == True:
				continue
			mid = data['image'][:,:, SelectedBox[0]: SelectedBox[1], SelectedBox[2]: SelectedBox[3]].tolist()
			Neighbor_Sem_ObjShape_Set.append([mid, int(SelectedCalss)])

print('Analysis finished! Start save!')

#save as json file
if opt.use_full_data:
    DesPath = './datasets2/cityscapes/train_scGraph/'
else:
    DesPath = './datasets/cityscapes/train_scGraph/'
# DesPath = './datasets2/cityscapes/train_scGraph/'
FileName = 'Neighbor_Sem_Transformed_ObjShape_Set'+str(dataset_size) +'.json'
DesFile = DesPath + FileName

MyFunc.mkdir1(DesPath)

# FinalShapeSet = np.array(Shape_dict)
#
#
# np.save(DesFile, FinalShapeSet)
#
# FinalShapeSet2 = np.load(DesFile)
#
# print('Finish test of CalShapeSet!!!')


SaveFile_shape = open(DesFile, 'w',encoding='utf-8')
json.dump(Neighbor_Sem_ObjShape_Set, SaveFile_shape, ensure_ascii=False)
SaveFile_shape.close()

#read saved json file
testRead = True
if testRead == True:
    ReadFile_scGraph = open(DesFile, 'r', encoding='utf-8')
    Read_scGraph = json.load(ReadFile_scGraph)
    print("Test Read Short Shape Successfully!")


