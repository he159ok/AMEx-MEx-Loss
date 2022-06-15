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

PredefinedSize = -1

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

Shape_dict = {}
SuitableObjeID = eval(opt.SuitableObjeID)
for ID in SuitableObjeID:
    Shape_dict[ID] = []

testScGraph = True
label_num = 35

f_tar_key = "/home/jfhe/Documents/MyPix2pixHD2/datasets2/cityscapes/train_img2labelcolor_MyOwn/"
f_part_key = './datasets2/cityscapes/train_label/'
len_fpt_key = len(f_part_key)
l_part_key = 'labelIds.png'
len_lpt_key = len(l_part_key)
l_tar_key = 'color.png'


for i, data in enumerate(dataset, start=epoch_iter):
    labelMap = data['label']
    labelMap = labelMap.squeeze(0)
    ini_path = data['path']
    ini_file_name = ini_path[0][len_fpt_key:-(len_lpt_key)]
    goal_file_path = f_tar_key + ini_file_name + l_tar_key

    ColorLableMap = util.tensor2label(labelMap, label_num)
    ColorLableMap = torch.from_numpy(ColorLableMap)
    if i%100 == 0:
        print(i)
    #存储图片过程
    # print('current image path:', ini_path)
    MyFunc.save_image(ColorLableMap, goal_file_path)

print('finish transformed')



# #save as json file
# if opt.use_full_data:
#     DesPath = './datasets2/cityscapes/train_scGraph/'
# else:
#     DesPath = './datasets/cityscapes/train_scGraph/'
# # DesPath = './datasets2/cityscapes/train_scGraph/'
# FileName = 'ShortShapeSet'+str(dataset_size) +'.json'
# DesFile = DesPath + FileName
#
# MyFunc.mkdir1(DesPath)
#
# # FinalShapeSet = np.array(Shape_dict)
# #
# #
# # np.save(DesFile, FinalShapeSet)
# #
# # FinalShapeSet2 = np.load(DesFile)
# #
# # print('Finish test of CalShapeSet!!!')
#
#
# SaveFile_shape = open(DesFile, 'w',encoding='utf-8')
# json.dump(Shape_dict, SaveFile_shape, ensure_ascii=False)
# SaveFile_shape.close()
#
# #read saved json file
# testRead = True
# if testRead == True:
#     ReadFile_scGraph = open(DesFile, 'r', encoding='utf-8')
#     Read_scGraph = json.load(ReadFile_scGraph)
#     print("Test Read Short Shape Successfully!")









