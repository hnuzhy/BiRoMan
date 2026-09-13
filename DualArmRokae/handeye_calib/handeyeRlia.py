
import os
import sys  
import json  
import cv2  
import numpy as np  
import yaml
from scipy.spatial.transform import Rotation as R

# http://rjyfb:123456@69.235.177.182:10801/simple/
# http://rjyfb:123456@69.235.177.182:10801/packages/open3d-0.18.0-cp310-cp310-manylinux_2_27_x86_64.whl#md5=0da06d37e4c7f2b47c43ca9c798a1670
# http://rjyfb:123456@69.235.177.182:10801/packages/rlia-0.3.4-cp310-cp310-manylinux_2_31_x86_64.whl#md5=86ae50524941513f28743dcf2c2e4197
# sudo apt-get update
# sudo apt-get install libc++abi1
import rlia  # by dexforce
import open3d as o3d  
import open3d.core as o3c

sys.path.insert(0, os.getcwd())

#################################################################
### https://github.com/hnuzhy/handeye_calib/blob/main/handeye_calibrate_C.ipynb
### pip install open3d pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple/
### This script can only be run in python 3.9 for dexforce rlia requirement
### conda acitvate py39  (we cannot install rlia in windows, conda remove -n py39 --all)
### conda create --name embodychain python=3.9 (under the WSL mode with linux kernel)
### python handeyeRlia.py

change_left_right = False  # True or False. reverse left and right camere or not
cam_yaml_path = "/home/dex/zhouhuayi/rokaeDemo/handeye_calib/calib_kfr_250916.yaml"  # for the new kingfisher
imgs_eeps_path = "/home/dex/zhouhuayi/rokaeDemo/handeye_calib/saved_imgs_eeps_01_R/"
#imgs_eeps_path = "/home/dex/zhouhuayi/rokaeDemo/handeye_calib/saved_imgs_eeps_02_L/"
#################################################################

if __name__ == "__main__":

    with open(cam_yaml_path, "r") as f:
        intrinsic = yaml.load(f, Loader=yaml.Loader)
        scale_ratio = 4.0  # for low res (960, 540), we should adjust the calib_file /4.0
        for para_key in ["cam1_k", "cam2_k"]:
            for [loc_i, loc_j] in [[0,0], [1,1], [0,2], [1,2]]:
                intrinsic[para_key][loc_i][loc_j] /= scale_ratio
    
    #with open(cam_yaml_path, "r") as f:
    #    intrinsic = yaml.load(f, Loader=yaml.Loader)

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
    for the calib_0904.yaml file in 2024-09-10
    Calibration Results:
        Calibrated success:	 True
        Reprojection Error:	 0.0007894053513491935
        Calibrated Pose:
     [[-0.5715738   0.39565333 -0.71886148  0.77725031]
     [ 0.82053757  0.28054541 -0.49800839 -0.15809828]
     [ 0.00463461 -0.87450141 -0.48500084  0.52394919]
     [ 0.          0.          0.          1.        ]]
    
    for the calib_0904.yaml file in 2024-09-24
    Calibration Results:
        Calibrated success:	 True
        Reprojection Error:	 0.0006568647129589484
        Calibrated Pose:
     [[ 0.99404971  0.03466854  0.1032631  -0.02890503]
     [-0.03611898 -0.78945868  0.61274009 -1.01781871]
     [ 0.10276475 -0.61282386 -0.78350898  0.60853942]
     [-0.          0.         -0.          1.        ]]
     

    for the calib_1106.yaml file in 2024-11-06 (aubo arm 1) 
     Calibration Results:
	Calibrated success:	 True
	Reprojection Error:	 0.0005599788680339257
	Calibrated Pose:
     [[-0.14041544 -0.77986075  0.61000059 -0.51553392]
     [-0.98988171  0.0978588  -0.10275144 -0.4792901 ]
     [ 0.02043788 -0.61825632 -0.78571077  0.71895748]
     [ 0.          0.          0.          1.        ]]
    for the calib_1106.yaml file in 2024-11-06 (aubo arm 2)
    Calibration Results:
	Calibrated success:	 True
	Reprojection Error:	 0.0007476927370270208
	Calibrated Pose:
     [[ 0.15344779  0.77205559 -0.61675274  0.54489893]
     [ 0.98764351 -0.09971405  0.12090245 -0.86202114]
     [ 0.0318445  -0.62768405 -0.7778166   0.71312324]
     [-0.          0.         -0.          1.        ]]
     
     
    for the calib_1106.yaml file in 2024-11-20 (aubo arm 1) 
    Calibration Results:
        Calibrated success:	 True
        Reprojection Error:	 0.000659358326417085
        Calibrated Pose:
    [[-0.14495955 -0.82672254  0.54361435 -0.46128096]
    [-0.98826554  0.09424301 -0.120206   -0.55922609]
    [ 0.04814516 -0.55466034 -0.83068282  0.77215299]
    [ 0.          0.          0.          1.        ]]
    for the calib_1106.yaml file in 2024-11-20 (aubo arm 2) 
    Calibration Results:
        Calibrated success:	 True
        Reprojection Error:	 0.0008349667015098737
        Calibrated Pose:
    [[ 0.15464175  0.82413502 -0.54487375  0.48699159]
    [ 0.98677    -0.10165977  0.12629433 -0.77214491]
    [ 0.04869184 -0.55719545 -0.82895256  0.76490652]
    [-0.          0.         -0.          1.        ]]



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
       



    for the calib_kfr_250916.yaml file in 2025-09-16 (rokae armR) 
    Calibration Results:
        Calibrated success:	 True
        Reprojection Error:	 0.0005828607147676769
        Calibrated Pose:
    [[ 6.34004343e-03 -7.87023936e-01  6.16889884e-01  4.46400294e-02]
    [ 4.94210047e-03 -6.16870087e-01 -7.87049472e-01  2.09050255e-01]
    [ 9.99967689e-01  8.03865962e-03 -2.14317819e-05 -1.86870870e-01]
    [-0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

    for the calib_kfr_250916.yaml file in 2025-09-16 (rokae armL) 
    Calibration Results:
        Calibrated success:	 True
        Reprojection Error:	 0.0007287035472574598
        Calibrated Pose:
    [[ 4.60052229e-03 -7.92004605e-01  6.10497781e-01  4.78865870e-02]
    [-3.76087529e-04  6.10502829e-01  7.92013987e-01 -2.15856101e-01]
    [-9.99989347e-01 -3.87327860e-03  2.51076882e-03 -3.78393681e-02]
    [ 0.00000000e+00 -0.00000000e+00 -0.00000000e+00  1.00000000e+00]]

    '''
    
    
