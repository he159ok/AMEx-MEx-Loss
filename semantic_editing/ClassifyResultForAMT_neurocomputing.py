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
datasetName = "NYU_GenRealImage/" #"NYU/"  "Helen_GenRealImage/"
methodList = ["BasicPipeline_MulExp/", "LSGAN/"]
#"BasicPipeline_Shape/", "BasicPipeline/", "BasicPipeline_BothLoss/", "BasicPipeline_MulExp/", "LSGAN/"
imgLen = 249 # 249 NYU # 500 Cityscape
DesName = "Sum_aaai/"

keyword2 = "_SynMask_image_colormap"

IsSeveralPart = True

IsGroundTruth = True

IsIncompletPart = True
if IsSeveralPart:
    for i in range(len(methodList)):
        source = basePath + datasetName + methodList[i] + basePath2

        for j in range(imgLen):

            keyword1 = str(j).zfill(3)
            goal = basePath + datasetName + DesName + keyword1 +'/'

            # keyword1 = "%03d" % j
            ret = findmykey(keyword1, keyword2, source)
            assert len(ret) <= 1
            if len(ret) == 1:
                MyFunc.mkdir1(goal)
                CurFileName = os.path.split(ret[0])[1]
                NewFileName = methodList[i][:-1] + '_' + CurFileName
                pa=os.path.join(goal, NewFileName)
                shutil.copyfile(ret[0], pa)





if IsGroundTruth:
    ConcreteMethodList = "BasicPipeline_BothLoss/"
    keyword_GT = "_Real_image"
    source = basePath + datasetName + ConcreteMethodList + basePath2

    for j in range(imgLen):

        keyword1 = str(j).zfill(3)
        goal = basePath + datasetName + DesName + keyword1 +'/'
        MyFunc.mkdir1(goal)
        # keyword1 = "%03d" % j
        ret = findmykey(keyword1, keyword_GT, source)
        assert len(ret) <= 1
        if len(ret) == 1:
            CurFileName = os.path.split(ret[0])[1]
            NewFileName = "GroundTruth" + '_' + CurFileName
            pa=os.path.join(goal, NewFileName)
            shutil.copyfile(ret[0], pa)


if IsIncompletPart:
    ConcreteMethodList = "BasicPipeline_BothLoss/"
    keyword_GT = "_InCom_img"
    source = basePath + datasetName + ConcreteMethodList + basePath2

    for j in range(imgLen):

        keyword1 = str(j).zfill(3)
        goal = basePath + datasetName + DesName + keyword1 +'/'
        MyFunc.mkdir1(goal)
        # keyword1 = "%03d" % j
        ret = findmykey(keyword1, keyword_GT, source)
        assert len(ret) <= 1
        if len(ret) == 1:
            CurFileName = os.path.split(ret[0])[1]
            NewFileName = "RemovalPart" + '_' + CurFileName
            pa=os.path.join(goal, NewFileName)
            shutil.copyfile(ret[0], pa)



print('复制完成')