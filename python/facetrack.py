# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:05:20 2019

@author: genki
"""

import os
import numpy as np
import cv2
import sys
import subprocess
import glob
import requests
import base64
import json
import math
import open3d
import copy


def face_detection(img):
    # setting API key
    API_KEY = ''
    API_SECRET = ''
    
    # set image file and save file
    cv2.imwrite("detect_temp.jpg",img)
    file_path = "detect_temp.jpg"
    
    # open imagefile with binary
    with open(file_path,"rb") as f:
        img_in = f.read()
    img_file = base64.encodebytes(img_in)
    
    # URL for web api
    url = 'https://api-us.faceplusplus.com/facepp/v3/detect'

    # set configuration
    config = {
        'api_key':API_KEY,
        'api_secret':API_SECRET,
        'image_base64':img_file,
        'return_landmark':0#,
        #'return_attributes':'gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus'
        }
    
    # post to web api
    while(1):
        res = requests.post(url,data=config)
        print(res.status_code)
        if (res.status_code == 200):
            break
    
    # load json data
    data = json.loads(res.text)
    
    for face in data['faces']:
        face_rectangle = face['face_rectangle']
    
    w = face_rectangle['width']
    h = face_rectangle['height']
    t = face_rectangle['top']
    l = face_rectangle['left']
    
    return l,t,w,h

def preprocess_point_cloud(pointcloud, voxel_size):

    keypoints = open3d.voxel_down_sample(pointcloud, voxel_size)
    radius_normal = voxel_size * 2
    view_point = np.array([0., 10., 10.], dtype="float64")
    open3d.estimate_normals(
        keypoints,
        search_param = open3d.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30))
    open3d.orient_normals_towards_camera_location(keypoints, camera_location = view_point)

    radius_feature = voxel_size * 5
    fpfh = open3d.compute_fpfh_feature(
        keypoints,
        search_param = open3d.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))

    return keypoints, fpfh

def execute_global_registration(kp1, kp2, fpfh1, fpfh2, voxel_size):
    distance_threshold = voxel_size * 2.5
    result = open3d.registration_ransac_based_on_feature_matching(
        kp1, kp2, fpfh1, fpfh2, distance_threshold,
        open3d.TransformationEstimationPointToPoint(False), 4,
        [open3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         open3d.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        open3d.RANSACConvergenceCriteria(500000, 1000))
    return result

def refine_registration(scene1, scene2, trans, voxel_size,max_iteration):
    distance_threshold = voxel_size * 0.4
    result = open3d.registration_icp(
        scene1, scene2, distance_threshold, trans,
        open3d.TransformationEstimationPointToPoint(),
        open3d.ICPConvergenceCriteria(max_iteration=100))
    return result

def registration_point_to_point(scene1,scene2,voxel_size):

    # scene1 and 2 are point cloud data
    # voxel_size is grid size

    #draw_registration_result(scene1,scene2,np.identity(4))

    # voxel down sampling
    scene1_down = open3d.voxel_down_sample(scene1,voxel_size)
    scene2_down = open3d.voxel_down_sample(scene2,voxel_size)

    #draw_registration_result(scene1_down,scene2_down,np.identity(4))

    # estimate normal with search radius voxel_size*2.0
    radius_normal = voxel_size*2.0
    open3d.estimate_normals(scene1,
        open3d.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30))
    open3d.estimate_normals(scene2,
        open3d.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30))
    open3d.estimate_normals(scene1_down,
        open3d.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30))
    open3d.estimate_normals(scene2_down,
        open3d.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30))

    # compute fpfh feature with search radius voxel_size/2.0
    radius_feature = voxel_size*2.0
    scene1_fpfh = open3d.compute_fpfh_feature(
        scene1_down,
        search_param = open3d.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
    scene2_fpfh = open3d.compute_fpfh_feature(
        scene2_down,
        search_param = open3d.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))

    # compute ransac registration
    ransac_result = execute_global_registration(scene1_down, scene2_down, scene1_fpfh, scene2_fpfh, voxel_size)
    #draw_registration_result(scene1,scene2,ransac_result.transformation)
    
    # point to point ICP resigtration
    distance_threshold = voxel_size * 0.4
    result = open3d.registration_icp(
        scene1, scene2, distance_threshold,ransac_result.transformation,
        open3d.TransformationEstimationPointToPoint(),
        open3d.ICPConvergenceCriteria(max_iteration=1000))

    #draw_registration_result(scene1,scene2,result.transformation)
    print(result)

    return result

def registration_point_to_plane(scene1,scene2,voxel_size):

    # scene1 and 2 are point cloud data
    # voxel_size is grid size

    #draw_registration_result(scene1,scene2,np.identity(4))

    # voxel down sampling
    scene1_down = open3d.voxel_down_sample(scene1,voxel_size)
    scene2_down = open3d.voxel_down_sample(scene2,voxel_size)

    #draw_registration_result(scene1_down,scene2_down,np.identity(4))

    # estimate normal with search radius voxel_size*2.0
    radius_normal = voxel_size*2.0
    open3d.estimate_normals(scene1,
        open3d.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30))
    open3d.estimate_normals(scene2,
        open3d.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30))
    open3d.estimate_normals(scene1_down,
        open3d.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30))
    open3d.estimate_normals(scene2_down,
        open3d.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30))

    # compute fpfh feature with search radius voxel_size/2.0
    radius_feature = voxel_size*2.0
    scene1_fpfh = open3d.compute_fpfh_feature(
        scene1_down,
        search_param = open3d.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
    scene2_fpfh = open3d.compute_fpfh_feature(
        scene2_down,
        search_param = open3d.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))

    # compute ransac registration
    ransac_result = execute_global_registration(scene1_down, scene2_down, scene1_fpfh, scene2_fpfh, voxel_size)
    #draw_registration_result(scene1,scene2,ransac_result.transformation)
    
    # point to point ICP resigtration
    distance_threshold = voxel_size *10.0
    result = open3d.registration_icp(
        scene1, scene2, distance_threshold,ransac_result.transformation,
        open3d.TransformationEstimationPointToPlane(),
        open3d.ICPConvergenceCriteria(max_iteration=1000))

    #draw_registration_result(scene1,scene2,result.transformation)
    print(result)

    return result


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.draw_geometries([source_temp, target_temp])


# base dirctory
basedir = "data/subject00"

color_file = basedir+"/color/image%05d.png"
depth_file = basedir+"/depth/image%05d.png"

colorfileList = glob.glob(basedir+"/color/*.png")
depthfileList = glob.glob(basedir+"/depth/*.png")
colorImageNum = len(colorfileList)
depthImageNum = len(depthfileList)


# align depth data
if (os.path.exists(basedir+"/depth_align")==False):
    os.makedirs(basedir+"/depth_align")
depth_align_file = basedir+"/depth_align/image%05d.png"

# rect depth data
if (os.path.exists(basedir+"/depth_rect")==False):
    os.makedirs(basedir+"/depth_rect")
depth_rect_file = basedir+"/depth_rect/image%05d.png"

# rect color data
if (os.path.exists(basedir+"/color_rect")==False):
    os.makedirs(basedir+"/color_rect")
color_rect_file = basedir+"/color_rect/image%05d.png"

# color ply data
if (os.path.exists(basedir+"/color_icp")==False):
    os.makedirs(basedir+"/color_icp")
color_icp_file = basedir+"/color_icp/image%05d.png"

# depth ply data
if (os.path.exists(basedir+"/depth_icp")==False):
    os.makedirs(basedir+"/depth_icp")
depth_icp_file = basedir+"/depth_icp/image%05d.png"

# ply file Name
if (os.path.exists(basedir+"/ply")==False):
    os.makedirs(basedir+"/ply")
ply_file = basedir+"/ply/ply%05d.ply"

# icp ply file
if (os.path.exists(basedir+"/ply_icp")==False):
    os.makedirs(basedir+"/ply_icp")
ply_icp_file = basedir+"/ply_icp/ply%05d.ply"

# temp folder
if (os.path.exists(basedir+"/temp")==False):
    os.makedirs(basedir+"/temp")

indexS = 0
indexL = colorImageNum
height,width = 480,640
l2,t2,w2,h2 = width/4,height/4,width/4,height/4
for imgNum in range(indexS,indexL):
    string = "image%05d"%(imgNum)
    print(string)
    
    # align image
    print("align")
    alignExe = "alignDepth.exe"
    subprocess.call("%s %s %s"%(alignExe,depth_file%imgNum,depth_align_file%imgNum))
    
    # face area detection
    print("face rect")
    face_area_file = basedir+"/temp/face_area.txt"
    color_image = cv2.imread(color_file%imgNum,1)
    depth_align_image = cv2.imread(depth_align_file%imgNum,-1)
    l1,t1,w1,h1 = face_detection(color_image)
    if (l1>0):
        l2,t2,w2,h2 = l1,t1,w1,h1
    depth_rect_image = np.zeros(depth_align_image.shape)
    depth_rect_image[t2:t2+h2,l2:l2+w2] = depth_align_image[t2:t2+h2,l2:l2+w2]
    color_rect_image = cv2.rectangle(color_image,(l2,t2),(l2+w2,t2+h2),(0,255,0),10)
    cv2.imwrite(depth_rect_file%imgNum,depth_rect_image)
    cv2.imwrite(color_rect_file%imgNum,color_rect_image)
    
    # convert ply
    print("convert ply")
    plyExe = "depthImg2ply.exe"
    subprocess.call("%s %s %s %s"%(plyExe,color_file%imgNum,depth_rect_file%imgNum
                                   ,ply_file%imgNum))

    # icp
    print("registration")
    if (imgNum==0):
        pc1 = open3d.read_point_cloud(ply_file%imgNum)
        open3d.write_point_cloud(ply_icp_file%imgNum,pc1)
        color_icp_image = cv2.imread(color_file%imgNum,1)
        cv2.imwrite(color_icp_file%imgNum,color_icp_image)
        depth_icp_image = cv2.imread(depth_rect_file%imgNum,-1)
        cv2.imwrite(depth_icp_file%imgNum,depth_icp_image)
        continue
    
    voxel_size = 0.01
    pc1 = open3d.read_point_cloud(ply_file%(imgNum-1))
    pc2 = open3d.read_point_cloud(ply_file%(imgNum))
    result = registration_point_to_point(pc2,pc1,voxel_size)
    
    pc2_tf = copy.deepcopy(pc2)
    pc2_tf.transform(result.transformation)
    
    open3d.write_point_cloud(ply_icp_file%imgNum,pc2_tf)
    os.remove(ply_file%(imgNum-1))
    os.remove(ply_icp_file%(imgNum-1))

    print("projection")
    points = np.asarray(pc2_tf.points)
    colors = np.asarray(pc2_tf.colors)*255.0
    points_num = points.shape[0]
    points_and_colors = np.zeros([points_num,6])
    points_and_colors[:,0:3] = points
    points_and_colors[:,3:6] = colors
    np.savetxt(basedir+"/temp/points.txt",points_and_colors)
    
    plyExe = "projection.exe"
    subprocess.call("%s %s %s %s"%(plyExe,basedir+"/temp/points.txt"
                                   ,color_icp_file%imgNum
                                   ,depth_icp_file%imgNum))

    print("comp image output")
    
    print("section finish")
    print(" ")
    
print("all finish")