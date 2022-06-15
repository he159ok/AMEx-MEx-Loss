import scipy.io as sio
import mat73
import os
import util.util as util
import torch
import copy
from PIL import Image

import numpy as np
dataroot = '/home/jfhe/Documents/MyPix2pixHD2/datasets2/NYU/'
colorSegFileName = 'Processed_Color_Seg_Img/'
# #
is_Toy = 0
if is_Toy == 1:
	fileName = 'nyu_depth_v2_toy_labeled.mat'
	dir_B = os.path.join(dataroot, fileName)
	mat = sio.loadmat(dir_B)
else:
	fileName = 'nyu_depth_v2_labeled.mat'
	dir_B = os.path.join(dataroot, fileName)
	mat = mat73.loadmat(dir_B)
	mat['labels'] = mat['labels'].transpose()
	mat['instances'] = mat['instances'].transpose()
	mat['images'] = mat['images'].transpose()

if is_Toy == 1:
	SampleNum = 8
else:
	SampleNum = 1449
label_nc = 896

if is_Toy == 1:
	desName = 'nyu_depth_v2_labeled_processed_toy.npy'
	matName = 'nyu_d_v2_labeled_colored_segmentation_toy.mat'
else:
	desName = 'nyu_depth_v2_labeled_processed.npy'
	matName = 'nyu_d_v2_labeled_colored_segmentation.mat'
matPath = dataroot + matName
# #

desPath = dataroot + desName
colorSegFilePath = dataroot + colorSegFileName
H = 480
W = 640
N = SampleNum
mat['seg_image'] = np.zeros((H, W, 3, N))

for index in range(SampleNum):
	A = mat['labels'][:, :, range(SampleNum)[index]]
	B = mat['images'][:, :, :, range(SampleNum)[index]]
	inst = (mat['instances'][:, :, range(SampleNum)[index]] + A * 1000)

	A2 = copy.deepcopy(A)

	B_seg = util.tensor2label((torch.tensor(A2.astype(float))).unsqueeze(0), label_nc)
	B_seg = np.array(B_seg)
	mat['seg_image'][:, :, :, index] = B_seg


np.save(desPath, mat['seg_image'])

print('end')



#mat['images'].shape (1449, 3, 640, 480)  0-255
#mat['instances'].shape (1449, 640, 480)   0-37
# mat['labels'].shape (1449, 640, 480)    0-894
# mat['rawDepths'].shape (1449, 640, 480) 0.0-10.0

