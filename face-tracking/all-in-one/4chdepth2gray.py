# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:42:44 2019

@author: b18.nagamatsu
"""

import cv2
import numpy as np
import sys
import os
import glob

def img2depth(img):
    height,width = img.shape[:2]
    depth = np.zeros([height,width]).astype('float64')
    img = img.astype(np.float64)
    depth = img[:,:,1]*256.0*256.0
    depth += img[:,:,2]*256.0
    depth += img[:,:,3]
    validMask = np.where(img[:,:,0]>0,1,0)
    img[:,:,0] = img[:,:,0] - (128.0+24.0)
    lx = np.where(validMask==1,np.ldexp(np.zeros([height,width])+1.0,img[:,:,0].astype('int32')),0.0)
    depth = (depth.astype('float64')+0.5)*lx    
    return depth.astype('float32')

def depth2img(depth):
    height,width = depth.shape[:2]
    depth = np.reshape(depth,[height,width])
    m, e = np.frexp(depth)
    m -= 0.5
    m *= 256*256*256
    img = np.zeros((height, width, 4), np.uint8)
    img[:,:,0] = (e+128).astype('uint8')
    img[:,:,1] = np.right_shift(np.bitwise_and(m.astype('uint64'), 0x00ff0000), 16)+128
    img[:,:,2] = np.right_shift(np.bitwise_and(m.astype('uint64'), 0x0000ff00), 8)
    img[:,:,3] = np.bitwise_and(m.astype('uint64'), 0x000000ff)
    #outImg[:,:,1] = np.where(outImg[:,:,1]-outImg[:,:,0] == 24,0,outImg[:,:,1])
    unvalidMask = np.where(img[:,:,0]==128,1,0) - np.where(img[:,:,1]+img[:,:,2]+img[:,:,3]>0,1,0)
    img[:,:,0] = np.where(unvalidMask==1,0,img[:,:,0])   
    return img

basedir = sys.argv[1]
outputdir = sys.argv[2]

if (os.path.exists(outputdir)==False):
    os.makedirs(outputdir)

minDist = 0.2
maxDist = 1.0

  
imgList = glob.glob(basedir+"/*.png")
img_total_num = len(imgList)
    
    
for imgNum in range(img_total_num):
    string = "img:%05d"%(imgNum)
    print(string)
        
    img = cv2.imread(imgList[imgNum],-1)
    depth = img2depth(img)
    depth_gray = (depth-minDist)/(maxDist-minDist)*255.0
    depth_gray = np.where(depth_gray<0.001,0.0,depth_gray)
    cv2.imwrite(outputdir+"/image%05d.png"%(imgNum),depth_gray)
    #np.savetxt("depth_mod/subject%02d/csv/image%05d.csv"%(dirNum,imgNum),depth,delimiter=",")
    cv2.imshow("test",depth_gray.astype('uint8'))
    cv2.waitKey(1)
