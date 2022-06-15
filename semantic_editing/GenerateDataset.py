# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:50:09 2018

@author: zhkgo
"""

import os
import re
import shutil
def findmykey(keyword,head=os.path.abspath('.')):
    dirs=set()
    files=[]
    pattern='.*'+keyword
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
goal = "/home/jfhe/Documents/MyPix2pixHD2/datasets2/cityscapes/test_img2labelgray/"
keyword = "gtFine_labelIds.png"
source = "/home/jfhe/Documents/pix2pixHD/datasets/cityscapes/gtFine/val/"
ret=findmykey(keyword, source)
#print(ret)

try:
    for x in ret:
        pa=os.path.join(goal,os.path.split(x)[1])
   # print(pa)
        shutil.copyfile(x,pa)
except:
    print("复制失败")
else:
    print('复制完成')