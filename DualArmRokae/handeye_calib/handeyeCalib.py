
import os
import sys
import shutil
import cv2
import json
import time
import platform
import numpy as np
from tqdm import tqdm

import kingfisher
sys.path.insert(0, os.getcwd())

from robots.demo_xMateCR7 import rokaeRobotBase
from src.config import cfg_dict_init as cfg_dict

#################################################################
### This script can only be run in python3 for aubo robot's requirement
### conda acitvate embodychain
### python handeyeCalib.py

#save_results_path = "/home/dex/zhouhuayi/rokaeDemo/handeye_calib/saved_imgs_eeps_01_R/"
save_results_path = "/home/dex/zhouhuayi/rokaeDemo/handeye_calib/saved_imgs_eeps_02_L/"

#################################################################

if __name__ == "__main__":
    
    if os.path.exists(save_results_path):
        shutil.rmtree(save_results_path)
    os.mkdir(save_results_path)

    # kingfisher-6000 camera initilization
    print("\n")
    c = kingfisher.connect(cfg_dict["kfr_ip"])   # connect the camera
    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']  # 540, 960
    kingfisher.SetAUTO_EXPOSURE()

    #robotBase = rokaeRobotBase(cfg_dict["arm2"]["robot_ip_add"])  # for the right arm
    robotBase = rokaeRobotBase(cfg_dict["arm1"]["robot_ip_add"])  # for the left arm
    robotBase.moving_pre_op()
    robotBase.set_motion_speed_ratio(0.2)  # 20% speed

    # robot movements list
    delta_xyz = 0.1  # in median
    delta_rpy = 30  # in degree
    robot_move_list = np.array([
        # only moving xyz (6 smaples)
        [delta_xyz, 0, 0, 0, 0, 0],
        [0, delta_xyz, 0, 0, 0, 0],
        [0, 0, delta_xyz, 0, 0, 0],
        [delta_xyz, 0, delta_xyz, 0, 0, 0],
        [0, delta_xyz, delta_xyz, 0, 0, 0],
        [delta_xyz, delta_xyz, 0, 0, 0, 0],
        # only moving rpy (6 smaples)
        [0, 0, 0, delta_rpy, delta_rpy, 0],
        [0, 0, 0, 0, delta_rpy, delta_rpy],
        [0, 0, 0, delta_rpy, 0, delta_rpy],
        [0, 0, 0, -delta_rpy, -delta_rpy, 0],
        [0, 0, 0, 0, -delta_rpy, -delta_rpy],
        [0, 0, 0, -delta_rpy, 0, -delta_rpy],
        # moving xyz and rpy (12 smaples)
        [delta_xyz, 0, 0, delta_rpy, delta_rpy, 0],
        [0, delta_xyz, 0, 0, delta_rpy, delta_rpy],
        [0, 0, delta_xyz, delta_rpy, 0, delta_rpy],
        [delta_xyz, 0, delta_xyz, -delta_rpy, -delta_rpy, 0],
        [0, delta_xyz, delta_xyz, 0, -delta_rpy, -delta_rpy],
        [delta_xyz, delta_xyz, 0, -delta_rpy, 0, -delta_rpy],
        [delta_xyz, 0, 0, -delta_rpy, -delta_rpy, 0],
        [0, delta_xyz, 0, 0, -delta_rpy, -delta_rpy],
        [0, 0, delta_xyz, -delta_rpy, 0, -delta_rpy],
        [delta_xyz, 0, delta_xyz, delta_rpy, delta_rpy, 0],
        [0, delta_xyz, delta_xyz, 0, delta_rpy, delta_rpy],
        [delta_xyz, delta_xyz, 0, delta_rpy, 0, delta_rpy],
    ])
    
    cur_pose_deg, cur_pose_rad = robotBase.get_current_pose()
    print("[Current robot pose]", cur_pose_deg)
    
    xyz_vars = np.random.rand(len(robot_move_list), 3) * 0.04 - 0.02  # [-0.02, 0.02], in median
    rpy_vars = np.random.rand(len(robot_move_list), 3) * 6 - 3  # [-3, 3], in degree
    for idx, delta_xyz_rpy in enumerate(tqdm(robot_move_list)):
        eep_name = str(idx).zfill(3) + "_eep.json"
        delta_xyz_rpy[:3] += xyz_vars[idx, :]  # add noise into xyz
        delta_xyz_rpy[3:] += rpy_vars[idx, :]  # add noise into rpy
        target_pose = np.array(cur_pose_deg) + delta_xyz_rpy  # move to target
        robotBase.move_to_a_waypoint(target_pose.tolist())
        time.sleep(1.5)  # wait until the board is stable
        temp_pose_deg, _ = robotBase.get_current_pose()
        
        img_name_prefix = os.path.join(save_results_path, str(idx).zfill(3) + "_img")
        img_cv2_L, img_cv2_R = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
        if img_cv2_L is not None and img_cv2_R is not None:
            with open(os.path.join(save_results_path, eep_name), "w") as json_file:
                json.dump(temp_pose_deg, json_file)
            cv2.imwrite(img_name_prefix + "_L.jpg", img_cv2_L)
            cv2.imwrite(img_name_prefix + "_R.jpg", img_cv2_R)
            print("[Succeed]", idx, delta_xyz_rpy)
        else:
            print("[Failed]", idx, delta_xyz_rpy)
            
        target_pose = target_pose - delta_xyz_rpy  # state return back
        robotBase.move_to_a_waypoint(target_pose.tolist())
        
        # exit()
    
