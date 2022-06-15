import os
import re
import shutil
import MyFunc
from PIL import Image
import numpy as np
import util.util as util
import torch
import scipy.misc

def findmykey(keyword1, keyword2, head=os.path.abspath('.')):
	dirs = set()
	files = []
	pattern = '.*' + keyword1 + keyword2 + '.*'
	dirs.add(head)
	while (len(dirs) != 0):
		mydir = dirs.pop()
		for x in os.listdir(mydir):
			#  print(x)
			absp = os.path.join(mydir, x)
			if os.path.isdir(absp):
				dirs.add(absp)
			elif os.path.isfile(absp):
				#    print(pattern+' '+keyword)
				#   print("isfile")
				if re.match(pattern, x):
					files.append(absp)
	return files



IsExtractLabel = False

mode = 'train'

basePath1 = "/home/jfhe/Documents/MyPix2pixHD2/datasets2/HelenFace/"
basePath2 = "helenstar_release/" + mode + "/"
DesName = mode + "_label/"

keyword1 = "_label"
keyword2 = ".png"

labelBehLen = len("_gtFine_labelIds.png")
colorlabelBeh = "_gtFine_color.png"
instBeh = "_gtFine_instanceIds.png"
colorlabelFileFolder = basePath1 + mode + "_img2labelcolor/"
instFileFolder = basePath1 + mode + "_inst/"


if IsExtractLabel:

	keyword1 = "_label"
	keyword2 = ".png"

	source = basePath1 + basePath2
	goal = basePath1 + DesName

	ret = findmykey(keyword1, keyword2, source)
	fileNum = len(ret)
	for i in range(fileNum):
		CurFileName = os.path.split(ret[i])[1]
		NewFileName = CurFileName[:-10] + "_gtFine_labelIds.png"
		pa = os.path.join(goal, NewFileName)
		shutil.copyfile(ret[i], pa)

else:
	label_nc = 12
	source = basePath1 + DesName
	keyword1 = "_gtFine_labelIds"
	keyword2 = ".png"
	ret = findmykey(keyword1, keyword2, source)
	fileNum = len(ret)
	for i in range(fileNum):
		if i%10 == 0:
			print(i)
		CurFileName = os.path.split(ret[i])[1]
		FileFullPath = source + CurFileName
		LabelImg = np.array(Image.open(FileFullPath))
		LabelImgT = torch.tensor(LabelImg)
		LabelImgT = LabelImgT.unsqueeze(0)
		LabelImgT = LabelImgT.unsqueeze(0)
		ColorLabelT = util.tensor2label(LabelImgT[0], label_nc)
		ColorLabel = np.array(ColorLabelT)
		InstImg = LabelImg * 1000

		ColorLabelFileName = CurFileName[:(-1*labelBehLen)] + colorlabelBeh
		InstFileName = CurFileName[:(-1*labelBehLen)] + instBeh

		ColorLabelFilePath = colorlabelFileFolder + ColorLabelFileName
		InstFilePath = instFileFolder + InstFileName

		scipy.misc.imsave(ColorLabelFilePath, ColorLabel)
		scipy.misc.imsave(InstFilePath, InstImg)




print("finish !!!")
