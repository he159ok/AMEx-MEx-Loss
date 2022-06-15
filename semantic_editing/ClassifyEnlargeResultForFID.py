# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:50:09 2018

@author: zhkgo
"""

import os
import re
import shutil
import MyFunc
def findmykey(keyword1, keyword2, head=os.path.abspath('.')):
    dirs=set()
    files=[]
    pattern='.*'+keyword1 + keyword2 + '.*'
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
            
        
 
#shutil.copyfile(r'F:\华研外语\2.六级2套预测\Model Test 2.lrc','F:\\fas\\'+'madel.lrc')
basePath = "/home/jfhe/Desktop/ECCV_AMT/"
basePath2 = "web_test/images/"


datasetName = "Helen_For_FID/" #"NYU_For_FID/" cityscape_For_FID
if datasetName == "cityscape_For_FID/":
    methodList = ["BasicPipeline/", "BasicPipeline_BothLoss/", "BasicPipeline_MulExp/", "BasicPipeline_Shape/", "LSGAN/"]
    imgLen = 500
elif datasetName == "NYU_For_FID/":
    methodList = ["BasicPipeline/", "BasicPipeline_BothLoss/", "BasicPipeline_MulExp/", "LSGAN/", "BasicPipeline_Shape/"]
    imgLen = 249
elif datasetName == "Helen_For_FID/":
    methodList = ["BasicPipeline/", "BasicPipeline_BothLoss/", "BasicPipeline_MulExp/", "LSGAN/", "BasicPipeline_Shape/"]
    imgLen = 299
    basePath2 = "web_test_realImage/images/"

methodList = ["BasicPipeline/"]
imgLen = 299

# methodList = ["Train_Real/"]
# imgLen = 1200

#"BasicPipeline_Shape/", "BasicPipeline/", "BasicPipeline_BothLoss/", "BasicPipeline_MulExp/", "LSGAN/"

Turn = 10
# DesName = "FID_whole_TR_Real/"
# DesName = "FID_local_TE/"
DesName = "FID_local_TR/"


# keyword2 = "_SynMask_image_colormap" #For Whole Test
# keyword2 = "_SynLocal_colormap" # For Local Test
keyword2 = "_OriLocal_Map"  #For Local Train
# keyword1 = 'ColorSegProByJianfengHe_' #For NYU Training set
# keyword1 = "_leftImg8bit"  For global Real
# keyword1 = "_gtFine_color"   For global Seg

IsSeveralPart = True

if IsSeveralPart:
    for i in range(len(methodList)):
        source = basePath + datasetName + methodList[i] + basePath2
        goal = basePath + datasetName + DesName + methodList[i] + str(Turn) + "/"
        MyFunc.mkdir1(goal)
        for j in range(imgLen):

            keyword1 = str(j).zfill(3)
            # keyword2 = str(j).zfill(4)

            # keyword1 = "%03d" % j
            ret = findmykey(keyword1, keyword2, source)
            assert len(ret) <= 1
            if len(ret) == 1:
                for t in range(Turn):
                    CurFileName = os.path.split(ret[0])[1]
                    NewFileName = str(t) + '_' + CurFileName
                    pa=os.path.join(goal, NewFileName)
                    shutil.copyfile(ret[0], pa)


#For training
# if IsSeveralPart:
#     for i in range(len(methodList)):
#         source = basePath + datasetName + methodList[i] + basePath2
#         goal = basePath + datasetName + DesName + methodList[i] + str(Turn) + "/"
#         MyFunc.mkdir1(goal)
#         for j in range(imgLen):
#
#             # keyword1 = str(j).zfill(3)
#             keyword2 = '.png'
#
#             # keyword1 = "%03d" % j
#             ret = findmykey(keyword1, keyword2, source)
#             for mm in range(len(ret)):
#                 for t in range(Turn):
#                     CurFileName = os.path.split(ret[mm])[1]
#                     NewFileName = str(t) + '_' + CurFileName
#                     pa=os.path.join(goal, NewFileName)
#                     shutil.copyfile(ret[mm], pa)

print('复制完成')