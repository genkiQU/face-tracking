import os
import sys
import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import open3d
import copy
import requests
import base64
import json
import math


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

def get_all_topics(argv):
    bag_name = argv[1]
    bag = rosbag.Bag(bag_name)
    topics = bag.get_type_and_topic_info()[1].keys()
    types = []

    f = open("rosbag_topic.txt","w")
    f.close()
    f = open("rosbag_topic.txt","a")
  
    for i in range(0,len(bag.get_type_and_topic_info()[1].values())):
        types.append(bag.get_type_and_topic_info()[1].values()[i][0])

    print("bag topics and types")
    for i in range(0,len(topics)):
        print(types[i]+"  :  "+topics[i])
        string = ""+types[i]+" : "+topics[i]+"\n"
        f.write(string)

    f.close()

def get_depth_camera_params(argv):
    print("\ndepth_camera_paramters")
    bag_name = argv[1]
    bag = rosbag.Bag(bag_name)
    depth_cam_info_topic = "/device_0/sensor_0/Depth_0/info/camera_info"
    for topic, msg, t in bag.read_messages(topics=depth_cam_info_topic):
        print(msg)

    return msg.width,msg.height,msg.K,msg.D

def Q2RM(RQ):
    RM = np.zeros([3,3])
    x = RQ[0]
    y = RQ[1]
    z = RQ[2]
    w = RQ[3]
    RM[0,0] = 1.0 - 2.0*y*y -2.0*z*z
    RM[0,1] = 2.0*x*y + 2.0*w*z
    RM[0,2] = 2.0*x*z - 2.0*w*y
    RM[1,0] = 2.0*x*y - 2.0*w*z
    RM[1,1] = 1.0 - 2.0*x*x - 2.0*z*z
    RM[1,2] = 2.0*y*z + 2.0*w*x
    RM[2,0] = 2.0*x*z + 2.0*w*y
    RM[2,1] = 2.0*y*z - 2.0*w*x
    RM[2,2] = 1.0 - 2.0*x*x - 2.0*y*y
    return RM

def get_color_camera_params(argv):
    print("\ncolor_camera_parameters")
    bag_name = argv[1]
    bag = rosbag.Bag(bag_name)
    color_cam_trans_topic = "/device_0/sensor_1/Color_0/tf/0"
    color_cam_info_topic = "/device_0/sensor_1/Color_0/info/camera_info"
    T = np.zeros([3])# Translation
    RQ = np.zeros([4])# rotation quatnion
    for topic, msg, t in bag.read_messages(topics=color_cam_trans_topic):
        #print(msg)
        print("reading TR matrix")
    T[0] = msg.translation.x
    T[1] = msg.translation.y
    T[2] = msg.translation.z
    RQ[0] = msg.rotation.x
    RQ[1] = msg.rotation.y
    RQ[2] = msg.rotation.z
    RQ[3] = msg.rotation.w
    RM = Q2RM(RQ)

    print("T : "+str(T))
    print("RQ : "+str(RQ))
    print("RM : "+str(RM)) 

    for topic, msg, t in bag.read_messages(topics=color_cam_info_topic):
        print("reading camera instric parameter")
    K = msg.K
    D = msg.D
    #K = K.reshape(K,[3,3])
    height = msg.height
    width = msg.width
    print("width:"+str(width))
    print("height:"+str(height))
    print("K : "+str(K))
    return T,RM,width,height,K,D

def get_depth_camera_info(argv):
    bag_name = argv[1]
    bag = rosbag.Bag(bag_name)
    sensor0_info_topic = "/device_0/sensor_0/info"
    for topic, msg, t in bag.read_messages(topics=sensor0_info_topic):
        print(msg)

def get_depth_motion_range(argv):
    bag_name = argv[1]
    bag = rosbag.Bag(bag_name)
    motion_range_topic = "/device_0/sensor_0/option/Motion Range/value"
    for topic, msg, t in bag.read_messages(topics=motion_range_topic):
        print(msg)

def get_depth_units_value(argv):
    bag_name = argv[1]
    bag = rosbag.Bag(bag_name)
    units_value_topic = "/device_0/sensor_0/option/Depth Units/value"
    for topic, msg, t in bag.read_messages(topics=units_value_topic):
        print(msg)
    return msg.data

def get_msg(argv):
    bag_name = argv[1]
    bag = rosbag.Bag(bag_name)
    topic = argv[2]
    for topic, msg, t in bag.read_messages(topics=topic):
        print(msg)

def bag2imgs(argv):
    bag_name = argv[1]
    
    # All image output
    # folder_name : 
    folder_name = bag_name[:bag_name.rfind('.bag')]
    if (os.path.exists(folder_name)==False):
        os.mkdir(folder_name)


    depth_scale = get_depth_units_value(argv)
    width,height,K,D = get_depth_camera_params(argv)
    
    depth_folder_name = 'depth'
    color_folder_name = 'color'
    ply_folder_name = "ply"
    if (os.path.exists(folder_name+"/"+depth_folder_name)==False):
        os.makedirs(folder_name+"/"+depth_folder_name)
    if (os.path.exists(folder_name+"/"+color_folder_name)==False):
        os.makedirs(folder_name+"/"+color_folder_name)
    if (os.path.exists(folder_name+"/"+ply_folder_name)==False):
        os.makedirs(folder_name+"/"+ply_folder_name)

    suffix = '.png'

    depth_topic = '/device_0/sensor_0/Depth_0/image/data'
    color_topic = '/device_0/sensor_1/Color_0/image/data'
    camera_topic = '/device_0/sensor_0/Depth_0/info/camera_info'

    depth_log_file = folder_name+"/"+depth_folder_name+"/log.csv"
    color_log_file = folder_name+"/"+color_folder_name+"/log.csv"
    
    depth_frame_list = []
    color_frame_list = []

    
    bag = rosbag.Bag(bag_name, "r")
    bridge = CvBridge()

    depthNum = 0
    colorNum = 0

    # depth mat
    print("depth image extract")
    i = 0
    string = ""
    for topic, msg, t in bag.read_messages(topics=depth_topic):

        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv_img = cv_img.astype('float32')*depth_scale
        cv2.imwrite(folder_name+"/"+depth_folder_name+"/image%05d.png"%i,depth2img(cv_img))
        i += 1
        if (False):
            break
    print('%d depth mat detected.' % i)
    depthTotalNum = i


    # save color iamge
    print("color image extract")
    i = 0
    for topic, msg, t in bag.read_messages(topics=color_topic):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite(folder_name+"/"+color_folder_name+"/image%05d.png"%i,cv_img)
        #l,t,w,h = face_detection(cv_img)
        #cv2.imshow("test",cv_img)
        #cv2.waitKey(0)
        if (False):
            break
        i += 1
    print('%d color mat detected.' % i)
    colorTotalNum = i
    return
    
if __name__ == '__main__':
    bag2imgs(sys.argv)


























