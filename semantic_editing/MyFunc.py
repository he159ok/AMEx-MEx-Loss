import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
import math
import copy
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import re


import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image


# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage(mode='RGB')


def CalBoundry(data):
	# labelIDindex = torch.unique(data['label'])
	instIDindex = torch.unique(data['inst'])
	# print(labelIDindex)
	# print(instIDindex)
	# print('%s-th iter sample finished' % i)

	boxes = []
	object_centers = []
	instIDs = []
	for l in range(len(instIDindex)):
		instMask = torch.eq(data['inst'], float(instIDindex[l]))
		# cal boundary for every single object
		boundry_LabelMask = instMask.nonzero()
		x_axis = boundry_LabelMask[:, 2]
		y_axis = boundry_LabelMask[:, 3]
		x_axis_min = int(x_axis.min())
		x_axis_max = int(x_axis.max())
		y_axis_min = int(y_axis.min())
		y_axis_max = int(y_axis.max())
		box = [x_axis_min, y_axis_min, x_axis_max, y_axis_max]
		# cal object center
		x_axis_mean = (x_axis_min + x_axis_max) / 2
		y_axis_mean = (y_axis_min + y_axis_max) / 2
		object_center = [x_axis_mean, y_axis_mean]
		# save data
		boxes.append(box)
		object_centers.append(object_center)
		instIDs.append(int(instIDindex[l]))

	return boxes, object_centers, instIDs

def CalSolarVal(obj_center_s, obj_cener_o):
	d = torch.Tensor(obj_center_s) - torch.Tensor(obj_cener_o)
	theta = math.atan2(d[1], d[0])
	distance = int(math.sqrt( math.pow(d[1], 2) + math.pow(d[0], 2) ))
	return theta, distance

def CalSceneGraph(boxes, object_centers, instIDs, method = 'Distance'):
	vocab = {
		'__in_image__': 0,
		'right of': 1,
		'below': 2,
		'surrounding': 3,
	}
	NoneInstID = {1: -999, 2: -998, 3: -997} #表示不存在的关系的 对应虚拟物体的编号ID

	p_order = {2: 'right of', 1: 'below', 0:'surrounding'}
	res = []
	if method == 'Distance':
		for cur_XlId, cur_instId in enumerate(instIDs):
			# 寻找最近的几个objects
			if len(instIDs) == 0:
				continue

			choices = [obj for obj in range(len(instIDs)) if obj != cur_XlId]
			s = cur_XlId
			#{'distance': 'instID'}
			r_choices = {} #right
			b_choices = {} #below
			s_choices = {} #surrounding
			mid_res = []
			for o in choices:
				sx0, sy0, sx1, sy1 = boxes[s]
				ox0, oy0, ox1, oy1 = boxes[o]
				if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
					p = 'surrounding'
					p = vocab[p]
					_, distance = CalSolarVal(object_centers[s], object_centers[o])
					s_choices[distance] = o
					continue
				# elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
				# 	p = 'inside'
				# 	p = vocab[p]
				# 	mid_res.append([int(instIDs[s]), p, int(instIDs[o])])
				# 	continue
				else:
					theta, distance = CalSolarVal(object_centers[s], object_centers[o])
					# if theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
					# 	p = 'left of'
					# 	l_choices[distance] = o
					if -3 * math.pi / 4 <= theta < -math.pi / 4:
						p = 'below'
						b_choices[distance] = o
					elif -math.pi / 4 <= theta < math.pi / 4:
						p = 'right of'
						r_choices[distance] = o
					# elif math.pi / 4 <= theta < 3 * math.pi / 4:
					# 	p = 'above'
					# 	a_choices[distance] = o
			whole_choices = [s_choices, b_choices, r_choices]

			for p_id, one_choices in enumerate(whole_choices):
				if len(one_choices.keys()) == 0:
					p = p_order[p_id]
					p = vocab[p]
					mid_res.append([int(instIDs[s]), p, NoneInstID[p]])
					continue
				one_choices_distances = torch.Tensor(list(one_choices.keys()))
				min_distance = one_choices_distances.min()
				o = one_choices[int(min_distance)]
				p = p_order[p_id]
				p = vocab[p]
				mid_res.append([int(instIDs[s]), p, int(instIDs[o])])
			res.append(mid_res)
	return res


def transferScGraph2AdjMat(ScGraph):
	res = []
	for NodeSetScGraph in ScGraph:
		if len(NodeSetScGraph) == 0:
			continue
		for SinglePart in NodeSetScGraph:
			if len(SinglePart) == 0:
				continue
			res.append(SinglePart)
	return res

def CalTwoKindsAdjs(singleBracket_scGraph, classNum, instNumEaClass):
	Dim = classNum*instNumEaClass + 3
	ResBinary = torch.zeros(Dim, Dim)
	ResReal = torch.zeros(Dim, Dim)
	for rel in singleBracket_scGraph:
		# 按照  行-关系-列 转换 【node-edge-node】
		# label是0-34理解的，所以有0号label，所以每个node的物体种类label都不用减少1
		classID1, instanceID1 = transfer45Digits(rel[0], classNum, instNumEaClass)
		classID2, instanceID2 = transfer45Digits(rel[2], classNum, instNumEaClass)
		assert (classID1)*instNumEaClass+instanceID1 >= 0
		assert (classID2)*instNumEaClass+instanceID2 >= 0
		assert (classID1)*instNumEaClass+instanceID1 <= (classNum * instNumEaClass+2)
		assert (classID2) * instNumEaClass + instanceID2 <= (classNum * instNumEaClass+2)
		ResBinary[(classID1)*instNumEaClass+instanceID1, (classID2)*instNumEaClass+instanceID2] = 1
		ResReal[(classID1)*instNumEaClass+instanceID1, (classID2)*instNumEaClass+instanceID2] = rel[1]
	ResBinary = ResBinary.t()
	ResReal = ResReal.t()
	return ResBinary, ResReal


def transfer45Digits(num, classNum, instNumEaClass):
	numStr = str(num)
	instanID_inThatClass = 0
	if len(numStr) > 2:
		if numStr == '-999' or numStr == '-998' or numStr == '-997':                #考虑不是一个物体的情形，这样701（classNum * instNumEaClass）维度的情况下则 应该最后一维度表达这个不是物体的虚拟物体
			classID = classNum
			instanID_inThatClass = 999 + int(numStr)
			return classID, instanID_inThatClass
		else:
			classID = int(numStr[:-3])
			instanID_inThatClass = int(numStr[-3:])

	else:
		classID = num
	if instanID_inThatClass >= instNumEaClass:
		print('There is a instanceID >= %d, you cal instanceID starting from 0', ((instNumEaClass)))
	if classID >= classNum:
		print('There is a object category >= 20, you cal object category starting from 0')
	return classID, instanID_inThatClass


def SelfSupervisePrePro_Advanced(labelMap, instMap, img, featMap, TriScGraphList, classNum, instNumEachClass, ExcVal, removalOjbID, randomSeedID):
	sizeOfImg = img.shape
	sizeOfInstMap = instMap.shape
	if removalOjbID == 'random':
		uniIDLable = labelMap.unique()
		uniIDInst = instMap.unique()
		instIDLen = len(uniIDInst)
		np.random.seed(randomSeedID)
		RankArr = np.arange(0, instIDLen)
		np.random.shuffle(RankArr)
		SelectedObj = uniIDInst[RankArr[0]]
	else:
		SelectedObj = removalOjbID

	#produce mask
	SelectObjMap = torch.eq(instMap, SelectedObj)
	# MaskObjMap = torch.ones(sizeOfInstMap) - SelectObjMap

	if 1:
		boundry_LabelMask = SelectObjMap.nonzero()
		x_axis = boundry_LabelMask[:, 2]
		y_axis = boundry_LabelMask[:, 3]
		x_axis_min = int(x_axis.min())
		x_axis_max = int(x_axis.max())
		y_axis_min = int(y_axis.min())
		y_axis_max = int(y_axis.max())
		SelectObjMap[:, :, x_axis_min:(x_axis_max), y_axis_min:(y_axis_max)] = 1

	# produce InComLableMap, InComInstMap

	InComLableMap = copy.deepcopy(labelMap)
	InComInstMap = copy.deepcopy(instMap)

	InComLableMap[SelectObjMap] = ExcVal
	InComInstMap[SelectObjMap] = ExcVal

	if 1:
		assert ExcVal in InComInstMap.unique()
		assert SelectedObj in instMap.unique()
		assert SelectedObj not in InComInstMap.unique()
		assert ExcVal not in instMap.unique()


	#produce InComScGraph
	InComScGraph = copy.deepcopy(TriScGraphList)
	for i in range(len(InComScGraph)):
		objTriSet = InComScGraph[i]
		if len(objTriSet) == 0:
			continue
		if objTriSet[0][0] == SelectedObj:
			InComScGraph.pop(i)
			break

	#produce InComBinAdj, InComRealAdj

	singleBracket_scGraph = transferScGraph2AdjMat(InComScGraph)
	InComBinAdj, InComRealAdj = CalTwoKindsAdjs(singleBracket_scGraph, classNum, instNumEachClass)

	#produce InComImg for check
	InComImg = copy.deepcopy(img)
	SelectImgMap = SelectObjMap.expand(sizeOfImg)
	InComImg[SelectImgMap] = 0
	#change on ThursDay

	return InComLableMap, InComInstMap, InComScGraph, InComBinAdj, InComRealAdj, InComImg, SelectedObj



def SelfSupervisePrePro_Basic(labelMap, instMap, img, feaMap, SuitableObjeID, ExcVal, PredefinedSize, removalOjbID, randomSeedID=None):
	sizeOfImg = img.shape
	sizeOfInstMap = instMap.shape
	if removalOjbID == 'random':
		uniIDLable = labelMap.unique()
		uniIDInst = instMap.unique()
		instIDLen = len(uniIDInst)
		if randomSeedID != None:
			np.random.seed(randomSeedID)
		RankArr = np.arange(0, instIDLen)
		np.random.shuffle(RankArr)
		SelectedObj = None
		isNone = True
		for i in range(instIDLen):
			CandidateClass = uniIDInst[RankArr[i]].item()
			numStr = str(CandidateClass)
			if len(numStr) > 2:
				CandidateClass = int(numStr[:-3])
				instanID_inThatClass = int(numStr[-3:])
				numStr = numStr[:-3]
			if CandidateClass in SuitableObjeID:
				SelectedObj = uniIDInst[RankArr[i]].item()   #此处应该加入 sizeFilter
				# produce mask
				SelectObjMap = torch.eq(instMap, SelectedObj)

				boundry_LabelMask = SelectObjMap.nonzero()
				x_axis = boundry_LabelMask[:, 2]
				y_axis = boundry_LabelMask[:, 3]
				x_axis_min = int(x_axis.min()) #- 1
				x_axis_max = int(x_axis.max()) #+ 1
				y_axis_min = int(y_axis.min()) #- 1
				y_axis_max = int(y_axis.max()) #+ 1
				SelectObjMap[:, :, x_axis_min:(x_axis_max + 1), y_axis_min:(y_axis_max + 1)] = 1
				SelectedBox = [x_axis_min, x_axis_max, y_axis_min, y_axis_max]
				SelectedSize = (x_axis_max - x_axis_min) * (y_axis_max - y_axis_min)
				if SelectedSize <= PredefinedSize:
					SelectedObj = None
					continue
				else:
					isNone = False     #做对后才置为False
					break
		#一遍循环结束后，说明该图片中没有合适的item,返回这种事实，并开始下一个循环
		if isNone == True:
			return None, None, None, None, isNone, None
	else:
		SelectedObj = removalOjbID
		# produce mask
		SelectObjMap = torch.eq(instMap, SelectedObj) #其实这个 就是 物体的形状speci_obj

		boundry_LabelMask = SelectObjMap.nonzero()  #其实这个 就是 物体的形状的4维图
		x_axis = boundry_LabelMask[:, 2]
		y_axis = boundry_LabelMask[:, 3]
		x_axis_min = int(x_axis.min()) #-1
		x_axis_max = int(x_axis.max()) #+ 1
		y_axis_min = int(y_axis.min()) #-1
		y_axis_max = int(y_axis.max()) #+ 1
		SelectObjMap[:, :, x_axis_min:(x_axis_max+1), y_axis_min:(y_axis_max+1)] = 1
		SelectedBox = [x_axis_min, x_axis_max, y_axis_min, y_axis_max]


	#produce mask
	# SelectObjMap = torch.eq(instMap, SelectedObj)

	# if 1:
	# 	boundry_LabelMask = SelectObjMap.nonzero()
	# 	x_axis = boundry_LabelMask[:, 2]
	# 	y_axis = boundry_LabelMask[:, 3]
	# 	x_axis_min = int(x_axis.min())
	# 	x_axis_max = int(x_axis.max())
	# 	y_axis_min = int(y_axis.min())
	# 	y_axis_max = int(y_axis.max())
	# 	SelectObjMap[:, :, x_axis_min:(x_axis_max), y_axis_min:(y_axis_max)] = 1
	# 	SelectedBox = [x_axis_min, x_axis_max+1, y_axis_min, y_axis_max+1]

	# produce InComLableMap, InComInstMap

	InComLableMap = copy.deepcopy(labelMap)           #可能需要 detach操作
	InComInstMap = copy.deepcopy(instMap)

	InComLableMap[SelectObjMap] = ExcVal
	InComInstMap[SelectObjMap] = ExcVal

	if 1:
		assert ExcVal in InComInstMap.unique()
		assert SelectedObj in instMap.unique()
		assert SelectedObj not in InComInstMap.unique()
		assert ExcVal not in instMap.unique()


	#produce InComImg for check
	InComImg = copy.deepcopy(img)
	SelectImgMap = SelectObjMap.expand(sizeOfImg)
	InComImg[SelectImgMap] = 0
	#change on ThursDay

	return InComLableMap, InComInstMap, InComImg, SelectedObj, numStr, SelectedBox, isNone, SelectObjMap  #numStr是所挑选类别的字符串， SelectedCalss


def SelfSupervise_RemovalCertainArea(labelMap, instMap, img, featMap, RevidedBoundBox, ExcVal, RemVal):
	sizeOfImg = img.shape
	SelectObjMap = torch.zeros_like(instMap)

	SelectObjMap[:, :, RevidedBoundBox[0]:RevidedBoundBox[1], RevidedBoundBox[2]:RevidedBoundBox[3]] = 1

	InComLableMapShow = copy.deepcopy(labelMap)           #可能需要 detach操作
	InComInstMapShow = copy.deepcopy(instMap)
	InComLableMapTrain = copy.deepcopy(labelMap)           #可能需要 detach操作
	InComInstMapTrain = copy.deepcopy(instMap)

	SelectObjMap = SelectObjMap.byte()

	InComLableMapShow[SelectObjMap] = RemVal
	InComInstMapShow[SelectObjMap] = RemVal
	InComLableMapTrain[SelectObjMap] = ExcVal
	InComInstMapTrain[SelectObjMap] = ExcVal


	#produce InComImg for check
	InComImg = copy.deepcopy(img)
	SelectImgMap = SelectObjMap.expand(sizeOfImg)
	InComImg[SelectImgMap] = RemVal  #-1

	return InComLableMapShow, InComInstMapShow, InComLableMapTrain, InComInstMapTrain, InComImg

def CalBoundry2(data, PredifinedSize):   #将inst map根据其中值的不同分离出来
	# labelIDindex = torch.unique(data['label'])
	instIDindex = torch.unique(data['inst'])


	boxes = []
	object_centers = []
	instIDs = []
	size = data['inst'].shape
	MaskSet = torch.zeros((size[0], 1, size[2], size[3]))
	jsq = 0
	for l in range(len(instIDindex)):
		instMask = torch.eq(data['inst'], float(instIDindex[l]))
		subMaskSet = instMask
		# cal boundary for every single object
		boundry_LabelMask = instMask.nonzero()
		x_axis = boundry_LabelMask[:, 2]
		y_axis = boundry_LabelMask[:, 3]
		x_axis_min = int(x_axis.min())
		x_axis_max = int(x_axis.max())
		y_axis_min = int(y_axis.min())
		y_axis_max = int(y_axis.max())
		box = [x_axis_min, y_axis_min, x_axis_max, y_axis_max]
		if PredifinedSize > 0:
			SelectedSize = (x_axis_max - x_axis_min) * (y_axis_max - y_axis_min)
			if SelectedSize < PredifinedSize:
				continue
		# cal object center
		x_axis_mean = (x_axis_min + x_axis_max) / 2
		y_axis_mean = (y_axis_min + y_axis_max) / 2
		object_center = [x_axis_mean, y_axis_mean]
		# save data
		jsq = jsq + 1
		boxes.append(box)
		object_centers.append(object_center)
		instIDs.append(int(instIDindex[l]))
		if jsq == 1:
			MaskSet[:, 0, :, :] = subMaskSet.type(torch.FloatTensor)
		else:
			MaskSet = torch.cat((MaskSet, subMaskSet.type(torch.FloatTensor)), 1)
	return boxes, object_centers, instIDs, MaskSet



def PickupShape(SelectedShape, SelectedBox, MapSize):
	Shape = np.array(SelectedShape).squeeze()
	Shape = Image.fromarray(Shape)
	size = (SelectedBox[1]-SelectedBox[0], SelectedBox[3]-SelectedBox[2])
	transform = transforms.Compose([
		transforms.Resize(size),
	])
	new_Shape = transform(Shape)
	TransformRes = torch.tensor(np.array(new_Shape))
	initLayer = torch.zeros(MapSize)
	initLayer[:, :, SelectedBox[0]:SelectedBox[1], SelectedBox[2]:SelectedBox[3]] = TransformRes
	return initLayer

def PickupShape2(SelectedShape, SelectedBox, MapSize):
	Shape = np.array(SelectedShape).squeeze()
	Shape = Image.fromarray(Shape)
	size = (SelectedBox[1]-SelectedBox[0], SelectedBox[3]-SelectedBox[2])
	transform = transforms.Compose([
		transforms.Resize(size),
	])
	new_Shape = transform(Shape)
	TransformRes = torch.tensor(np.array(new_Shape))
	initLayer = torch.zeros(MapSize)
	initLayer[:, :, SelectedBox[0]:SelectedBox[1], SelectedBox[2]:SelectedBox[3]] = TransformRes
	return initLayer

def FillShapeAll0(SelectedBox, MapSize):
	initLayer = torch.zeros(MapSize)
	initLayer[:, :, SelectedBox[0]:SelectedBox[1], SelectedBox[2]:SelectedBox[3]] = 1
	return initLayer

def FillShapeAllSameVal(SelectedBox, MapSize, Val):
	initLayer = torch.zeros(MapSize)
	initLayer[:, :, SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]:SelectedBox[3]+1] = Val
	return initLayer


def ToSem(Atensor, ref):
	instMask = torch.gt(Atensor, ref)
	mid = torch.ones_like(Atensor)
	mid[instMask] = ref
	Atensor = int(torch.div(Atensor, mid))
	Atensor = torch.floor(Atensor)
	return Atensor


#make a new path
def mkdir1(path):
	folder = os.path.exists(path)

	if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder...  ---")
		print(path)
		print("---  OK  ---")



def SelfSupervisePrePro_Basic_temporary(labelMap, instMap, img, featMap, SuitableObjeID, ExcVal, PredefinedSize, removalOjbID, randomSeedID):
	sizeOfImg = img.shape
	sizeOfInstMap = instMap.shape
	if removalOjbID == 'random':
		uniIDLable = labelMap.unique()
		uniIDInst = instMap.unique()
		instIDLen = len(uniIDInst)
		if randomSeedID != None:
			np.random.seed(randomSeedID)
		RankArr = np.arange(0, instIDLen)
		np.random.shuffle(RankArr)
		SelectedObj = None
		isNone = True
		for i in range(instIDLen):
			CandidateClass = uniIDInst[RankArr[i]].item()
			numStr = str(CandidateClass)
			if len(numStr) > 2:
				CandidateClass = int(numStr[:-3])
				instanID_inThatClass = int(numStr[-3:])
				numStr = numStr[:-3]
			if CandidateClass in SuitableObjeID:
				SelectedObj = uniIDInst[RankArr[i]].item()   #此处应该加入 sizeFilter
				# produce mask
				SelectObjMap = torch.eq(instMap, SelectedObj)

				boundry_LabelMask = SelectObjMap.nonzero()
				x_axis = boundry_LabelMask[:, 2]
				y_axis = boundry_LabelMask[:, 3]
				x_axis_min = int(x_axis.min())
				x_axis_max = int(x_axis.max())
				y_axis_min = int(y_axis.min())
				y_axis_max = int(y_axis.max())
				SelectObjMap[:, :, x_axis_min:(x_axis_max), y_axis_min:(y_axis_max)] = 1
				SelectedBox = [x_axis_min, x_axis_max, y_axis_min, y_axis_max]
				SelectedSize = (x_axis_max - x_axis_min) * (y_axis_max - y_axis_min)
				if SelectedSize <= PredefinedSize:
					SelectedObj = None
					continue
				else:
					isNone = False     #做对后才置为False
					break
		#一遍循环结束后，说明该图片中没有合适的item,返回这种事实，并开始下一个循环
		if isNone == True:
			return None, None, None, None, isNone
	else:
		SelectedObj = removalOjbID
		# produce mask
		SelectObjMap = torch.eq(instMap, SelectedObj)

		boundry_LabelMask = SelectObjMap.nonzero()
		x_axis = boundry_LabelMask[:, 2]
		y_axis = boundry_LabelMask[:, 3]
		x_axis_min = int(x_axis.min())
		x_axis_max = int(x_axis.max())
		y_axis_min = int(y_axis.min())
		y_axis_max = int(y_axis.max())
		SelectObjMap[:, :, x_axis_min:(x_axis_max), y_axis_min:(y_axis_max)] = 1
		SelectedBox = [x_axis_min, x_axis_max, y_axis_min, y_axis_max]


	#produce mask
	# SelectObjMap = torch.eq(instMap, SelectedObj)

	# if 1:
	# 	boundry_LabelMask = SelectObjMap.nonzero()
	# 	x_axis = boundry_LabelMask[:, 2]
	# 	y_axis = boundry_LabelMask[:, 3]
	# 	x_axis_min = int(x_axis.min())
	# 	x_axis_max = int(x_axis.max())
	# 	y_axis_min = int(y_axis.min())
	# 	y_axis_max = int(y_axis.max())
	# 	SelectObjMap[:, :, x_axis_min:(x_axis_max), y_axis_min:(y_axis_max)] = 1
	# 	SelectedBox = [x_axis_min, x_axis_max+1, y_axis_min, y_axis_max+1]

	# produce InComLableMap, InComInstMap

	InComLableMap = copy.deepcopy(labelMap)           #可能需要 detach操作
	InComInstMap = copy.deepcopy(instMap)

	InComLableMap[SelectObjMap] = ExcVal
	InComInstMap[SelectObjMap] = ExcVal

	if 1:
		assert ExcVal in InComInstMap.unique()
		assert SelectedObj in instMap.unique()
		assert SelectedObj not in InComInstMap.unique()
		assert ExcVal not in instMap.unique()


	#produce InComImg for check
	InComImg = copy.deepcopy(img)
	SelectImgMap = SelectObjMap.expand(sizeOfImg)
	InComImg[SelectImgMap] = 0
	#change on ThursDay

	return InComLableMap, InComInstMap, InComImg, SelectedObj, numStr, SelectedBox, isNone  #numStr是所挑选类别的字符串， SelectedCalss



def get_edges(t):   #Prewitt edge detector
	edge = torch.ByteTensor(t.size()).zero_()
	# edge = torch.zeros(t.size())
	edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
	edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
	edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
	edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])

	return edge.float()

def pt_get_edges(t):
	#define laplacian
	fil = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).float() #别忘了，pytorch里的是cross-correlation 而不是卷积操作，得上下颠倒
	fil2 = fil.expand(3, 3, 3, 3)
	res = torch.nn.functional.conv2d(t, fil2, stride=1, padding=1)
	# res = torch.clamp(res, -1, 1)
	# res2 = (res + 1) / 2
	# show_from_tensor(res2)
	return res

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor_to_np(img)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def save_image(tensor, path):
    dir = path
    image = tensor.permute(2, 0, 1).cpu().clone()
    image2 = unloader(image)
    image2.save(path, quality=95, subsampling=0)
# image2 = unloader(image).transpose(Image.FLIP_LEFT_RIGHT)



def cosine_distance(matrix1,matrix2):
        matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())
        matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
        matrix1_norm = matrix1_norm[:, np.newaxis]
        matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
        matrix2_norm = matrix2_norm[:, np.newaxis]
        cosine_distance = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))
        return cosine_distance


# cite from # cited from  https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
def iou(pred, target, n_classes, IgnoreZero = True):
  ious = []
  ious_NoNan = []
  pred = pred.view(-1)
  target = target.view(-1)

  if IgnoreZero == True:
	  start = 1
	  ious.append(0)
  else:
	  start = 0
  # Ignore IoU for background class ("0")
  for cls in range(start, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum()  # Cast to long to prevent overflows
    union = pred_inds.long().sum() + target_inds.long().sum() - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
      if cls in target:
        ious_NoNan.append(float(intersection) / float(max(union, 1)))
  return np.array(ious), np.array(ious_NoNan)


def findmykey2(keyword1, head=os.path.abspath('.')):
    dirs=set()
    files=[]
    pattern='.*'+keyword1 + '.*'
    dirs.add(head)
    while(len(dirs)!=0):
        mydir=dirs.pop()
        for x in os.listdir(mydir):
          #  print(x)
            absp=os.path.join(mydir,x)
            if os.path.isdir(absp):
                dirs.add(absp)
            elif os.path.isfile(absp):
            #    print(pattern+' '+keyword)
             #   print("isfile")
                if re.match(pattern,x):
                    files.append(absp)
    return files


# Applied for Interface, to generate a data with related process
def GetDataByNamePre(namePre, opt):
	if opt.dataName == 'cityscape':
		PrePath = "/home/jfhe/Documents/MyPix2pixHD2/static/images/cityscapes/"
		subPath1 = PrePath + "test_label/"
		subPath2 = PrePath + "test_img2labelcolor/"
		subPath3 = PrePath + "test_inst/"
	elif opt.dataName == 'NYU':
		PrePath = "/home/jfhe/Documents/MyPix2pixHD2/static/images/NYU/"
		subPath1 = PrePath + "test_label/"
		subPath2 = PrePath + "test_img2labelcolor/"
		subPath3 = PrePath + "test_inst/"
		BWLabelEnd = "_gtFine_labelIds"
		ColorLabelEnd = "_gtFine_labelIds"
		BWInstEnd = "_gtFine_instanceIds"
	elif opt.dataName == 'helen':
	#if opt.dataName == 'cityscape':
		PrePath = "/home/jfhe/Documents/MyPix2pixHD2/static/images/HelenFace/"
		subPath1 = PrePath + "test_label/"
		subPath2 = PrePath + "test_img2labelcolor/"
		subPath3 = PrePath + "test_inst/"


	BWLabelEnd = "_gtFine_labelIds"
	ColorLabelEnd = "_gtFine_color"
	BWInstEnd = "_gtFine_instanceIds"

	BWLabelFileName = namePre + BWLabelEnd
	BWColorLabelFileName = namePre + ColorLabelEnd
	BWInstFileName = namePre + BWInstEnd


	BWLabelFilePath = findmykey2(BWLabelFileName, subPath1)[0]
	ColorLabelFilePath = findmykey2(BWColorLabelFileName, subPath2)[0]
	BWInstFilePath = findmykey2(BWInstFileName, subPath3)[0]

	PathSet = {}
	PathSet['BWLabel'] = BWLabelFilePath
	PathSet['ColorLabel'] = ColorLabelFilePath
	PathSet['BWInst'] = BWInstFilePath


	### input A (label maps)
	A_path = BWLabelFilePath
	A = Image.open(A_path)
	params = get_params(opt, A.size)
	if opt.label_nc == 0:
		transform_A = get_transform(opt, params)
		A_tensor = transform_A(A.convert('RGB'))
	else:
		transform_A = get_transform(opt, params, method=Image.NEAREST, normalize=False)
		A_tensor = transform_A(A) * 255.0

	B_tensor = inst_tensor = feat_tensor = 0
	### input B (real images)
	if opt.isTrain or opt.use_encoded_image or opt.Te:
		B_path = ColorLabelFilePath
		B = Image.open(B_path).convert('RGB')
		transform_B = get_transform(opt, params)
		B_tensor = transform_B(B)

	### if using instance maps
	if not opt.no_instance:
		inst_path = BWInstFilePath
		inst = Image.open(inst_path)
		inst_tensor = transform_A(inst)

		if opt.load_features:
			# feat_path = feat_paths[index]
			# feat = Image.open(feat_path).convert('RGB')
			# norm = normalize()
			# feat_tensor = norm(transform_A(feat))
			feat_tensor = torch.tensor([0, 1])

	if len(A_tensor.shape) == 3:
		A_tensor = A_tensor.unsqueeze(0)
	if len(inst_tensor.shape) == 3:
		inst_tensor = inst_tensor.unsqueeze(0)
	if len(B_tensor.shape) == 3:
		B_tensor = B_tensor.unsqueeze(0)
	feat_tensor = torch.tensor([0, 1])

	input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
	              'feat': feat_tensor, 'path': PathSet}

	return input_dict




