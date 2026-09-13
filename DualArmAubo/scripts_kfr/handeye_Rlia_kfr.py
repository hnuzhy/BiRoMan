
import os
import sys  
import json  
import cv2  
import numpy as np  
import yaml
from scipy.spatial.transform import Rotation as R

import rlia  # by dexforce
import open3d as o3d  
import open3d.core as o3c

sys.path.insert(0, os.getcwd())

#################################################################
### pip install open3d pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple/
### This script can only be run in python 3.9 for dexforce rlia requirement
### conda acitvate py39  (we cannot install rlia in windows, conda remove -n py39 --all)
### conda create --name embodychain python=3.9 (under the WSL mode with linux kernel)
### python handeye_Rlia_kfr.py

change_left_right = False  # True or False. reverse left and right camere or not
cam_yaml_path = "binocularCam/cameraConfigs/calib_kfr_250221.yaml"  # for the new kingfisher-R-6000
imgs_eeps_path = "binocularCam/example_kfr/saved_imgs_eeps/"
#################################################################

if __name__ == "__main__":

    
    with open(cam_yaml_path, "r") as f:
        intrinsic = yaml.load(f, Loader=yaml.Loader)
        scale_ratio = 4.0  # for low res (960, 540), we should adjust the calib_file /4.0
        for para_key in ["cam1_k", "cam2_k"]:
            for [loc_i, loc_j] in [[0,0], [1,1], [0,2], [1,2]]:
                intrinsic[para_key][loc_i][loc_j] /= scale_ratio
    
    k_list = [np.asarray(intrinsic["cam1_k"]), np.asarray(intrinsic["cam2_k"])]
    d_list = [np.asarray(intrinsic["dist_1"])[0], np.asarray(intrinsic["dist_2"])[0]]
    CamR = np.asarray(intrinsic["R_l_r"])
    CamT = np.asarray(intrinsic["t_l_r"])[:, 0] / 1000
    
    # print(list(k_list[0]), "\n", list(k_list[1]), "\n", list(CamR), "\n", list(CamT))
    # sys.exit()
    
    files_list = os.listdir(imgs_eeps_path)
    files_list.sort()
    eep_json_list = [i for i in files_list if ".json" in i]
    img_l_list = [i for i in files_list if "img_L.jpg" in i]
    img_r_list = [i for i in files_list if "img_R.jpg" in i]
    
    concentric_circles_pose_list = []
    robot_end_effector_pose_list = []
    for eep_name, img_l_name, img_r_name in zip(eep_json_list, img_l_list, img_r_list):
        eep_json_path = os.path.join(imgs_eeps_path, eep_name)
        with open(eep_json_path, "r") as json_file:
            ee_pose = json.load(json_file)
        
        assert img_l_name[:3]==img_r_name[:3], "image not matched!!!"
        
        img_l_cv2 = cv2.imread(os.path.join(imgs_eeps_path, img_l_name))
        img_r_cv2 = cv2.imread(os.path.join(imgs_eeps_path, img_r_name))
        
        if change_left_right:
            rgb_list = [img_r_cv2, img_l_cv2]
        else:
            rgb_list = [img_l_cv2, img_r_cv2]
        detect_report = rlia.calibration.stereo_ellipse_detect(
            rgb_list, k_list, d_list, CamR, CamT)
        
        # Store the detected board pose
        concentric_circles_pose_list.append(detect_report.board_pose)
        
        ee_pose_4x4 = np.identity(4)
        ee_pose_4x4[:3, :3] = R.from_euler("xyz", ee_pose[3:], degrees=True).as_matrix()  # rotation
        ee_pose_4x4[:3, 3] = np.asarray(ee_pose[:3])  # translation
        # Convert the end-effector pose to a tensor and store it  
        ee_pose_tensor = o3c.Tensor(ee_pose_4x4, dtype=o3c.Dtype.Float64)  
        robot_end_effector_pose_list.append(ee_pose_tensor)  
    
    # Perform calibration using the collected board and end-effector poses  
    calibrate_report = rlia.calibration.CalibrateAxyb(
        concentric_circles_pose_list, robot_end_effector_pose_list, False) 
    
    camera_pose = calibrate_report.pose_calibrate.numpy()
    
    # Print calibration results in a structured format
    print("Calibration Results:")
    print("\tCalibrated success:\t", calibrate_report.status == 0) 
    print("\tReprojection Error:\t", calibrate_report.reproject_err)
    print("\tCalibrated Pose:\n", camera_pose)

    # base_location = np.array([[0, 1, 0, 0.2], [-1, 0, 0, 0.6], [0, 0, 1, 0.827], [0, 0, 0, 1]])
    # world_camera = base_location @ camera_pose
    # position = camera_pose[:3, 3:]
    # look_at = position + camera_pose[:3, 2:3]`
    # print("postion", position, "look at", look_at, sep="\n")
    
    
    '''
    for the calib_kfr_250221.yaml file in 2025-02-25 (aubo arm 1) 
    Calibration Results: 
        Calibrated success:	 True
        Reprojection Error:	 0.001282136685200912
        Calibrated Pose:
    [[ 0.02826124 -0.82213902  0.56858485 -0.5574288 ]
    [-0.99958334 -0.01990389  0.020904   -0.7053366 ]
    [-0.00586894 -0.56893871 -0.82235898  0.79531344]
    [ 0.          0.         -0.          1.        ]]
    for the calib_kfr_250221.yaml file in 2025-02-25 (aubo arm 2) 
    Calibration Results:
        Calibrated success:	 True
        Reprojection Error:	 0.0011884976743340655
        Calibrated Pose:
    [[-0.01459961  0.82190626 -0.56943564  0.58294993]
    [ 0.99988048  0.01489787 -0.00413253 -0.63574671]
    [ 0.00508683 -0.56942791 -0.82202553  0.79036275]
    [ 0.          0.          0.          1.        ]]
    
    '''
    
    
