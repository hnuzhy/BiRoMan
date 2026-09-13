
import os
import sys
import shutil
import cv2
import json
import copy
import time
import datetime
import platform
import argparse
import numpy as np

sys.path.insert(0, os.getcwd())

if platform.system() == "Linux":
    from auboRobotArmLinux.demo import RobotBase
    from auboRobotArmLinux.robotcontrol import Auboi5Robot
if platform.system() == "Windows":
    from auboRobotArmWin.demo import RobotBase
    from auboRobotArmWin.robotcontrol import Auboi5Robot 

from gripperTools.dh_modbus_gripper import dh_device, dh_modbus_gripper

import kingfisher
from src.utils import cfg_dict_init, polygon2mask
from src.utils import compute_rotation_by_image_moments
from kingFisherR.utils_vfm import ov_det_seg_yoloe_slim
from kingFisherR.utils_vfm import post_processing_yoloe_results
from kingFisherR.utils_vfm import plot_processed_results_vis
from ultralytics import YOLOE


#################################################################
### This script can only be run in python3 for aubo robot's requirement
### conda acitvate embodychain
'''
# task_names: plugpen / reorient / unscrew / pouring / inserting / pressing
# keyframes: 7 actions / 8 actions / 12 actions / 10 actions / 9 actions / 7 actions
# demonstrations: 1944 + 720 + 576 + 2592 + 3888 + 1296 = 11016

# 2 * (3*3*6 * 1*3*6) = 2 * 972 = 1944
python scripts_kfr/run_Data_Collect.py --task_name plugpen --marker_id 2 --orient_id 1 --given_tag LpenXxxYxxRcapX05Y04

# (2 + 2) * (5*6*6) = 4 * 180 = 720
python scripts_kfr/run_Data_Collect.py --task_name reorient --anyobj_id 1 --orient_id 1 --given_tag LRspoonX010203Yxx

# 4 * (1*12*12) = 4 * 144 = 576
python scripts_kfr/run_Data_Collect.py --task_name unscrew --bottle_id 3 --given_tag LRbottleX010203Yxx

# (4 * 2) * (1*3*6 * 1*3*6) = 8 * 324 = 2592
python scripts_kfr/run_Data_Collect.py --task_name pouring --bottle_id 3 --mugcup_id 1 --orient_id 1 --given_tag LbottleXxxYxxRmugcapX05Y04

# (2 * 2) * (1*3*6 * 3*3*6) = 4 * 972 = 3888
python scripts_kfr/run_Data_Collect.py --task_name inserting --ordcup_id 1 --marker_id 2 --orient_id 1 --given_tag LordcupX02Y04RmarkerXxxYxx

# (2 * 2) * (1*3*6 * 1*3*6) = 4 * 324 = 1296
python scripts_kfr/run_Data_Collect.py --task_name pressing --ordcup_id 1 --nozzle_id 1 --given_tag LordcupX02Y04RnozzleXxxYxx

'''
#################################################################



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm_type', default="both", help="given_execute_arm_name. It can be arm1 arm2 or both")
    parser.add_argument('--root_dir_path', default="/home/dexforce/zhouhuayi/projects/WiLoR_CL/results", 
                        help="path to all processed dual-arm robot actions")  # the given one-shot demonstration
    parser.add_argument('--task_name', default="", help="string of bimanual task name.")
    parser.add_argument('--kfr_ip', default="192.168.9.111", help="the ip address of the kingfisher-R-6000")
    parser.add_argument('--save_imgs_dir', default="/home/dexforce/zhouhuayi/auboHandeyeCalib/scripts_kfr/BiDemoSyn/", 
                        help="path to save all initial seeding images")
    
    parser.add_argument('--debug_close_loop_vis', action='store_true', help="default is False")
    parser.add_argument('--is_rollout', action='store_true', help="is dual-arm running/replaying for checking. default is False")
    
    parser.add_argument('--marker_id', type=int, default=1, help="marker_id is selected from 1 ~ 8")  # for task plugpen
    parser.add_argument('--anyobj_id', type=int, default=1, help="anyobj_id is selected from 1 ~ 8")  # for task reorient
    parser.add_argument('--bottle_id', type=int, default=6, help="bottle_id is selected from 1 ~ 8")  # for task unscrew and pouring
    parser.add_argument('--mugcup_id', type=int, default=1, help="mugcup_id is selected from 1 ~ 4")  # for task pouring
    parser.add_argument('--ordcup_id', type=int, default=1, help="ordcup_id is selected from 1 ~ 2")  # for task inserting
    parser.add_argument('--nozzle_id', type=int, default=1, help="nozzle_id is selected from 1 ~ 2")  # for task pressing
    
    parser.add_argument('--orient_id', type=int, default=1, help="orient_id is selected from 1 ~ 4")  # useful for tasks plugpen / reorient / inserting
    parser.add_argument('--given_tag', default="", help="tag string of different epoch of demo collection.")   # e.g., LpenXxxYxxRcapX05Y04
    
    args = parser.parse_args()
    
    args.debug_close_loop_vis = True  # always set it as True for better collection demonstrations

    ############################################################################
    assert args.task_name in ["plugpen", "reorient", "unscrew", "pouring", "inserting", "pressing"], "Please give a valid task name !!!"
    assert args.marker_id >=1 and args.marker_id <= 8, "Please note that marker_id is selected from 1 ~ 8 !!!"  # [BiDemoSyn] only for the marker with id [2,4]
    assert args.anyobj_id >=1 and args.anyobj_id <= 8, "Please note that anyobj_id is selected from 1 ~ 8 !!!"  # [BiDemoSyn] only for the spoon/shovel with id [1,3,4,8]
    assert args.bottle_id >=1 and args.bottle_id <= 8, "Please note that bottle_id is selected from 1 ~ 8 !!!"  # [BiDemoSyn] only for the bottle with id [3,4,6,7]
    assert args.mugcup_id >=1 and args.mugcup_id <= 4, "Please note that mugcup_id is selected from 1 ~ 4 !!!"  # [BiDemoSyn] only for the mugcup with id [1,4]
    assert args.ordcup_id >=1 and args.ordcup_id <= 2, "Please note that ordcup_id is selected from 1 ~ 2 !!!"  # [BiDemoSyn] for all of the ordcup with id [1,2]
    assert args.nozzle_id >=1 and args.nozzle_id <= 2, "Please note that nozzle_id is selected from 1 ~ 2 !!!"  # [BiDemoSyn] for all of the nozzle with id [1,2]
    
    if args.task_name == "reorient_unscrew":  # for this task with lying down bottles, not all of them are usable.
        assert args.bottle_id in [2, 3, 4, 6, 7], "Please note that bottle_id is selected from [2, 3, 4, 6, 7] !!!"
    
    cfg_dict = cfg_dict_init
    execute_arm_names = ["arm1", "arm2"]  # arm1 (left) / arm2 (right)
    if args.arm_type == "both": given_execute_arm_name = None
    elif args.arm_type == "arm1": given_execute_arm_name = "arm1"
    elif args.arm_type == "arm2": given_execute_arm_name = "arm2"
    else: print("Please give the right string of robot type!!!"); sys.exit()
    
    c = kingfisher.connect(cfg_dict["kfr_ip"])   # connect the camera
    # kingfisher.SetExposure(40000)  # set exposure as 40000 for specular objects (reorient -> plastic_red_blade / wooden_long_blade / metal_long_blade)
    kingfisher.SetAUTO_EXPOSURE()
    kfr_height, kfr_width = 540, 960

    # top-left, top-right, bottom-right, bottom-left --> [260, 86], [652, 84], [761, 535], [168, 538]. length*width --> 616mm * 675mm
    four_pts_list = [[260, 86], [652, 84], [761, 535], [168, 538]]  # (pts_tl, pts_tr, pts_br, pts_bl) in the affine transformed camera-view image 
    rect_l, rect_w = 616, 675  # the length / width of the top-viewd rectangle
    tgt_pts_list = [[0, 0], [rect_l-1, 0], [rect_l-1, rect_w-1], [0, rect_w-1]]  # (pts_tl, pts_tr, pts_br, pts_bl) in top-viewed rectangle platform
    pts_polygon = np.array(four_pts_list, np.int32).reshape((-1, 1, 2))  # polygon corner points coordinates
    colors_list = [(0,255,128), (255,255,0), (128,0,255), (0,128,128), (0,0,255)]  # lawn green, cyan, light magenta, light yellow, red

    selected_four_corners = np.array(four_pts_list, dtype=np.float32)
    show_window_corners = np.array(tgt_pts_list, dtype=np.float32)
    transform_mat = cv2.getPerspectiveTransform(show_window_corners, selected_four_corners)  # obtain the transform function
    print("top-view reporjection transform_mat:\n", transform_mat)
    transform_mat_inv = np.linalg.inv(transform_mat)

    if args.task_name in ["unscrew", "pouring", "pressing"]:
        roi_bbox = [120, 0, 760, 540]  # the [x1, y1, x2, y2] (960*540 --> 640*540; 16:9 --> 32:27)
    if args.task_name in ["plugpen", "reorient", "inserting"]:
        roi_bbox = [140, 90, 740, 540]  # the [x1, y1, x2, y2] (960*540 --> 600*450; 16:9 --> 4:3)

    ############################################################################
    os.makedirs(args.save_imgs_dir, exist_ok=True)
    task_folder_dir = os.path.join(args.save_imgs_dir, args.task_name)
    os.makedirs(task_folder_dir, exist_ok=True)
    
    # the init pose for dual robot arms in each task
    if args.task_name == "plugpen":
        json_file_name = "plugpen_01_20250226160908_L_eep_new.json"
        saved_seed_img_path = os.path.join(task_folder_dir, f"marker{str(args.marker_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"pen":[], "cap":[]}, {"pen":[], "cap":[]}  # first grasp pen (left), then grasp cap (right)
        prev_delta_xy_list = [[0, 0, None], [0, 0, None]]
        # vfms_conf = {"fastsam": 0.5}; radius_ratio = 0.2
        radius_ratio = 0.2; conf_yoloe = 0.1; prompts = None  # Note, the rare category pen/cap cannot be detected stablely
    if args.task_name == "reorient":
        json_file_name = "reorient_01_20250226161104_L_eep_new.json"
        saved_seed_img_path = os.path.join(task_folder_dir, f"anyobj{str(args.anyobj_id).zfill(2)}_seed.jpg")
        if args.anyobj_id in [1, 2, 3, 5, 6]: anyobj_cls = "spoon"
        if args.anyobj_id in [4, 7, 8]: anyobj_cls = "shovel" 
        seed_ref_pts_dict, test_ref_pts_dict = {anyobj_cls:[]}, {anyobj_cls: []}
        prev_delta_xy_list = [[0, 0, None]]
        # vfms_conf = {"fastsam": 0.1}; radius_ratio = 0.2
        radius_ratio = 0.2; conf_yoloe = 0.01; prompts = [anyobj_cls]  # Note, the rare category spoon/shovel cannot be detected stablely
    if args.task_name == "unscrew":
        json_file_name = f"unscrew_01_20250226162314_L_eep_new_LR{args.bottle_id}.json"
        saved_seed_img_path = os.path.join(task_folder_dir, f"bottle{str(args.bottle_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"bottle":[]}, {"bottle":[]}
        prev_delta_xy_list = [[0, 0, None]]
        # vfms_conf = {"yoloworld": 0.5, "fastsam": 0.5}; radius_ratio = 0.2
        radius_ratio = 0.2; conf_yoloe = 0.2; prompts = ["bottle"]
    if args.task_name == "pouring":
        json_file_name = f"pouring_01_20250226162008_L_eep_new_L{args.bottle_id}_R{args.mugcup_id}.json"
        saved_seed_img_path = os.path.join(task_folder_dir, f"bottle{str(args.bottle_id).zfill(2)}-mugcup{str(args.mugcup_id).zfill(2)}_seed.jpg") 
        seed_ref_pts_dict, test_ref_pts_dict = {"cup":[], "bottle":[]}, {"cup":[], "bottle":[]}  # first grasp cup (right), then grasp bottle (left)
        prev_delta_xy_list = [[0, 0, None], [0, 0, None]]
        # vfms_conf = {"yoloworld": 0.5, "fastsam": 0.5}; radius_ratio = 0.2
        radius_ratio = 0.2; conf_yoloe = 0.2; prompts = ["bottle", "cup"]
        
    if args.task_name == "inserting":
        json_file_name = f"inserting_01_20250402202534_L_eep_new.json"
        saved_seed_img_path = os.path.join(task_folder_dir, f"ordcup{str(args.ordcup_id).zfill(2)}-marker{str(args.marker_id).zfill(2)}_seed.jpg") 
        seed_ref_pts_dict, test_ref_pts_dict = {"cup":[], "pen":[]}, {"cup":[], "pen":[]}  # first grasp cup (left), then grasp pen (right)
        prev_delta_xy_list = [[0, 0, None], [0, 0, None]]
        # vfms_conf = {"yoloworld": 0.5, "fastsam": 0.5}; radius_ratio = 0.2
        radius_ratio = 0.2; conf_yoloe = 0.01; prompts = ["cup", "pen"]  # Note, the up-side-down cup cannot be detected stablely
    if args.task_name == "pressing":
        json_file_name = f"pressing_01_20250404154346_L_eep_new_LR{args.nozzle_id}.json"
        saved_seed_img_path = os.path.join(task_folder_dir, f"ordcup{str(args.ordcup_id).zfill(2)}-nozzle{str(args.nozzle_id).zfill(2)}_seed.jpg") 
        seed_ref_pts_dict, test_ref_pts_dict = {"bottle":[], "cup":[]}, {"bottle":[], "cup":[]}  # first approach bottle (right), then grasp cup (left)
        prev_delta_xy_list = [[0, 0, None], [0, 0, None]]
        # vfms_conf = {"yoloworld": 0.5, "fastsam": 0.5}; radius_ratio = 0.2
        radius_ratio = 0.2; conf_yoloe = 0.1; prompts = ["bottle", "cup"]
     
    if not os.path.exists(saved_seed_img_path):
        left_image_seed, _ = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
        cv2.imwrite(saved_seed_img_path, left_image_seed)
        sys.exit()
    else:
        left_image_seed = cv2.imread(saved_seed_img_path)

    save_demos_sub_dir = saved_seed_img_path.replace("_seed.jpg", f"_orient{str(args.orient_id).zfill(2)}_temp")  # saving recently collected demonstrations
    os.makedirs(save_demos_sub_dir, exist_ok=True)
    grid_plot_img_path = os.path.join(args.save_imgs_dir, "assets", args.task_name + "_grid_cells.jpg")

    ############################################################################
    ''' Initialize a YOLOE model'''
    if prompts is None:  # (PF) Prompt-Free
        # model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11s-seg-pf.pt")
        model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11m-seg-pf.pt")
        # model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11l-seg-pf.pt")
    else:  # with Prompt (language/text or vision/image)
        # need to install https://github.com/ultralytics/CLIP and  https://github.com/apple/ml-mobileclip
        # model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11s-seg.pt")
        model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11m-seg.pt")
        # model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11l-seg.pt")
        model.set_classes(prompts, model.get_text_pe(prompts))  # Set text prompt. e.g., prompts = ["person", "bus"]
        
    ############################################################################  
    # split the platform into dense grids for fast and automatic demonstration collection
    if args.task_name in ["plugpen", "inserting"]: 
        grid_x1, grid_y1, grid_x2, grid_y2 = 20, 50, rect_l-20, rect_w-50
    if args.task_name in ["reorient"]:
        grid_x1, grid_y1, grid_x2, grid_y2 = 80, 120, rect_l-80, rect_w-120
    if args.task_name in ["unscrew", "pouring", "pressing"]:
        grid_x1, grid_y1, grid_x2, grid_y2 = 80, 200, rect_l-80, rect_w-20
    if args.task_name in ["inserting"]: 
        grid_x1, grid_y1, grid_x2, grid_y2 = 40, 180, rect_l-40, rect_w-20
        
    if args.task_name == "plugpen": 
        L_orient_max, L_posx_max, L_posy_max, R_orient_max, R_posx_max, R_posy_max = 3, 3, 6, 1, 3, 6
        grid_ptx_num, grid_pty_num = L_posx_max + R_posx_max, L_posy_max
    if args.task_name == "reorient": 
        LR_orient_max, LR_posx_max, LR_posy_max = 5, 6, 6
        grid_ptx_num, grid_pty_num = LR_posx_max, LR_posy_max
    if args.task_name == "unscrew": 
        LR_orient_max, LR_posx_max, LR_posy_max = 1, 12, 12
        grid_ptx_num, grid_pty_num = LR_posx_max, LR_posy_max
    if args.task_name == "pouring": 
        L_orient_max, L_posx_max, L_posy_max, R_orient_max, R_posx_max, R_posy_max = 1, 3, 6, 1, 3, 6
        grid_ptx_num, grid_pty_num = L_posx_max + R_posx_max, L_posy_max
    if args.task_name == "inserting": 
        L_orient_max, L_posx_max, L_posy_max, R_orient_max, R_posx_max, R_posy_max = 1, 3, 6, 3, 3, 6
        grid_ptx_num, grid_pty_num = L_posx_max + R_posx_max, L_posy_max
    if args.task_name == "pressing": 
        L_orient_max, L_posx_max, L_posy_max, R_orient_max, R_posx_max, R_posy_max = 1, 3, 6, 1, 3, 6
        grid_ptx_num, grid_pty_num = L_posx_max + R_posx_max, L_posy_max
        
    grid_top_xy_list_tgt, grid_bottom_xy_list_tgt = np.zeros([grid_ptx_num+1, 2]), np.zeros([grid_ptx_num+1, 2])  # shape (w, 2)
    grid_top_xy_list_src, grid_bottom_xy_list_src = np.zeros([grid_ptx_num+1, 2]), np.zeros([grid_ptx_num+1, 2])  # shape (w, 2)
    for idx in range(grid_ptx_num + 1):
        grid_top_tgt_x = grid_x1 + (grid_x2 - grid_x1) * idx / grid_ptx_num
        grid_top_tgt_y = grid_y1
        grid_top_xy_list_tgt[idx, :] = [grid_top_tgt_x, grid_top_tgt_y]
        temp_pt = np.dot(transform_mat, np.array([grid_top_tgt_x, grid_top_tgt_y, 1]).T)
        grid_top_xy_list_src[idx, :] = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
        
        grid_bottom_tgt_x = grid_x1 + (grid_x2 - grid_x1) * idx / grid_ptx_num
        grid_bottom_tgt_y = grid_y2
        grid_bottom_xy_list_tgt[idx, :] = [grid_bottom_tgt_x, grid_bottom_tgt_y]
        temp_pt = np.dot(transform_mat, np.array([grid_bottom_tgt_x, grid_bottom_tgt_y, 1]).T)
        grid_bottom_xy_list_src[idx, :] = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
    
    grid_left_xy_list_tgt, grid_right_xy_list_tgt = np.zeros([grid_pty_num+1, 2]), np.zeros([grid_pty_num+1, 2])  # shape (h, 2)   
    grid_left_xy_list_src, grid_right_xy_list_src = np.zeros([grid_pty_num+1, 2]), np.zeros([grid_pty_num+1, 2])  # shape (h, 2)   
    for idx in range(grid_pty_num + 1):
        grid_left_tgt_x = grid_x1
        grid_left_tgt_y = grid_y1 + (grid_y2 - grid_y1) * idx / grid_pty_num
        grid_left_xy_list_tgt[idx, :] = [grid_left_tgt_x, grid_left_tgt_y]
        temp_pt = np.dot(transform_mat, np.array([grid_left_tgt_x, grid_left_tgt_y, 1]).T)
        grid_left_xy_list_src[idx, :] = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
        
        grid_right_tgt_x= grid_x2
        grid_right_tgt_y = grid_y1 + (grid_y2 - grid_y1) * idx / grid_pty_num
        grid_right_xy_list_tgt[idx, :] = [grid_right_tgt_x, grid_right_tgt_y]
        temp_pt = np.dot(transform_mat, np.array([grid_right_tgt_x, grid_right_tgt_y, 1]).T)
        grid_right_xy_list_src[idx, :] = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
        
    grid_center_xy_list_tgt = np.zeros([grid_ptx_num, grid_pty_num, 2])  # shape (w, h, 2)
    grid_center_xy_list_src = np.zeros([grid_ptx_num, grid_pty_num, 2])  # shape (w, h, 2)
    for id_ptx in range(grid_ptx_num):
        for id_pty in range(grid_pty_num):
            ptx_end1 = grid_x1 + (grid_x2 - grid_x1) * (id_ptx) / grid_ptx_num
            ptx_end2 = grid_x1 + (grid_x2 - grid_x1) * (id_ptx + 1) / grid_ptx_num
            grid_center_tgt_x = (ptx_end1 + ptx_end2) / 2.0
            pty_end1 = grid_y1 + (grid_y2 - grid_y1) * (id_pty) / grid_pty_num
            pty_end2 = grid_y1 + (grid_y2 - grid_y1) * (id_pty + 1) / grid_pty_num
            grid_center_tgt_y = (pty_end1 + pty_end2) / 2.0
            grid_center_xy_list_tgt[id_ptx, id_pty, :] = [grid_center_tgt_x, grid_center_tgt_y]
            temp_pt = np.dot(transform_mat, np.array([grid_center_tgt_x, grid_center_tgt_y, 1]).T)
            grid_center_xy_list_src[id_ptx, id_pty, :] = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]

    if not os.path.exists(grid_plot_img_path):
        left_image, _ = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
        cv2.imwrite(grid_plot_img_path[:-4] + "_origin.jpg", left_image)
        cv2.polylines(left_image, [pts_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.imwrite(grid_plot_img_path[:-4] + "_range.jpg", left_image)
        for g_idx in range(grid_ptx_num + 1):
            px1, py1 = int(grid_top_xy_list_src[g_idx, 0]), int(grid_top_xy_list_src[g_idx, 1])
            px2, py2 = int(grid_bottom_xy_list_src[g_idx, 0]), int(grid_bottom_xy_list_src[g_idx, 1])
            cv2.line(left_image, (px1, py1), (px2, py2), color=(192,192,192), thickness=1, lineType=cv2.LINE_AA)
        for g_idx in range(grid_pty_num + 1):
            px1, py1 = int(grid_left_xy_list_src[g_idx, 0]), int(grid_left_xy_list_src[g_idx, 1])
            px2, py2 = int(grid_right_xy_list_src[g_idx, 0]), int(grid_right_xy_list_src[g_idx, 1])
            cv2.line(left_image, (px1, py1), (px2, py2), color=(192,192,192), thickness=1, lineType=cv2.LINE_AA)
        for id_ptx in range(grid_ptx_num):
            for id_pty in range(grid_pty_num):
                gcx, gcy = int(grid_center_xy_list_src[id_ptx, id_pty, 0]), int(grid_center_xy_list_src[id_ptx, id_pty, 1])
                cv2.circle(left_image, (gcx, gcy ), radius=5, color=(255,255,255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.imwrite(grid_plot_img_path, left_image)
        
    # sys.exit()
    
    ############################################################################  
    
    eef_pose_seq_json_path = os.path.join(args.root_dir_path, args.task_name + "_01", json_file_name)
    eef_pose_dict = json.load(open(eef_pose_seq_json_path, "r"))
    eef_pose_seqs_L, eef_pose_seqs_R = eef_pose_dict["L"], eef_pose_dict["R"]
    robot_init_pose_L, robot_init_gripper_L = eef_pose_seqs_L[0][1:7], eef_pose_seqs_L[0][7]
    robot_init_pose_R, robot_init_gripper_R = eef_pose_seqs_R[0][1:7], eef_pose_seqs_R[0][7]
    robot_arms, eef_pose_seqs = [], []
    max_step_id = max(eef_pose_seqs_L[-1][0], eef_pose_seqs_R[-1][0]) + 1  # step_id starts from 0
    steps_list_L, steps_list_R = [il[0] for il in eef_pose_seqs_L], [ir[0] for ir in eef_pose_seqs_R]
    for step_id in range(max_step_id):  # Now, we do not distinguish sync task and async task
        if step_id in steps_list_L:  # step_id, 7 dof = 3 location + 3 rotation + 1 gripper
            for eef_pose_l in eef_pose_seqs_L:  # one step may contain multiple actions
                if eef_pose_l[0] == step_id: robot_arms.append("L"); eef_pose_seqs.append(eef_pose_l[1:])        
        if step_id in steps_list_R:  # step_id, 7 dof = 3 location + 3 rotation + 1 gripper
            for eef_pose_r in eef_pose_seqs_R:  # one step may contain multiple actions
                if eef_pose_r[0] == step_id: robot_arms.append("R"); eef_pose_seqs.append(eef_pose_r[1:]) 
    print(max_step_id, "step_ids:", steps_list_L+steps_list_R, "\n robot_arms:", robot_arms)
    
    ############################################################################
    ''' aubo robot arm config '''
    if args.is_rollout:
        robotBase_L, robotBase_R = None, None
        if given_execute_arm_name == "arm1": execute_arm_names = ["arm1"]
        if given_execute_arm_name == "arm2": execute_arm_names = ["arm2"]
        for arm_id, execute_arm_name in enumerate(execute_arm_names):
            robot_ip = cfg_dict[execute_arm_name]["robot_ip_add"]
            robot_port = 8899  # robot port number
            Auboi5Robot.initialize()
            aubo_robot = Auboi5Robot()
            aubo_robot.create_context()
            print("[init] arm name:", execute_arm_name, ", aubo_robot.rshd:", aubo_robot.rshd)
            aubo_robot.connect(robot_ip, robot_port)
            robotBase = RobotBase(aubo_robot) 
            robotBase.speed_init()
            if execute_arm_name == "arm1": robotBase_L = robotBase
            if execute_arm_name == "arm2": robotBase_R = robotBase
    
    ############################################################################
    ''' gripper 串口配置 / 控制参数 (scissor_gripper / parallel_gripper) '''
    if args.is_rollout:
        m_gripper_L, m_gripper_R = None, None
        if given_execute_arm_name == "arm1": execute_arm_names = ["arm1"]
        if given_execute_arm_name == "arm2": execute_arm_names = ["arm2"]
        for execute_arm_name in execute_arm_names:
            port = cfg_dict[execute_arm_name]["gripper_port"]
            baudrate, initstate, gripper_ID, speed = 115200, 0, 0x01, 100
            force = 30  # default 100, values from 20 ~ 100 are valid
            print("[init] the gripper of", execute_arm_name, "gripper_ID:", gripper_ID, port)
            m_gripper = dh_modbus_gripper(dh_device(port), gripper_ID)
            m_gripper.open(baudrate)
            m_gripper.Initialization()
            while(initstate != 1) :
                initstate = m_gripper.GetInitState()
                # print(initstate)
                time.sleep(0.1)
            m_gripper.SetTargetForce(force); m_gripper.SetTargetSpeed(speed)
            if execute_arm_name == "arm1": m_gripper_L = m_gripper
            if execute_arm_name == "arm2": m_gripper_R = m_gripper
    
    ############################################################################
    if args.is_rollout:
        # robot init testing
        if given_execute_arm_name is None or given_execute_arm_name == "arm1":
            assert robot_init_pose_L is not None, "You should give the init_pose of left arm when collecting data!"
            robotBase_L.move_to_target_in_cartesian(robot_init_pose_L)
        cur_pose_deg_L, cur_pose_rad_L = robotBase_L.get_current_pose()
        print("[arm1 (left)][Current robot pose (robot view)]", cur_pose_deg_L)
        
        if given_execute_arm_name is None or given_execute_arm_name == "arm2":
            assert robot_init_pose_R is not None, "You should give the init_pose of right arm when collecting data!"
            robotBase_R.move_to_target_in_cartesian(robot_init_pose_R)
        cur_pose_deg_R, cur_pose_rad_R = robotBase_R.get_current_pose()
        print("[arm2 (right)][Current robot pose (robot view)]", cur_pose_deg_R)
        
        # gripper init testing
        if robot_init_gripper_L == 1 and robot_init_gripper_R == 1:
            cur_gripper_state_L, cur_gripper_state_R, openness_val = 1, 1, 995  # 1: opening gripper, values 0 ~ 1000
        if robot_init_gripper_L == 0 and robot_init_gripper_R == 0:
            cur_gripper_state_L, cur_gripper_state_R, openness_val = 0, 0, 5  # 0: closing gripper, values 0 ~ 1000
        if given_execute_arm_name is None: m_gripper_L.SetTargetPosition(openness_val); m_gripper_R.SetTargetPosition(openness_val)
        elif given_execute_arm_name == "arm1": m_gripper_L.SetTargetPosition(openness_val)
        elif given_execute_arm_name == "arm2": m_gripper_R.SetTargetPosition(openness_val)

        time.sleep(1)
        # sys.exit()
    
    ############################################################################

    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    left_image_seed_sub = left_image_seed[y1:y2, x1:x2]
    final_res_list, im_bgr = ov_det_seg_yoloe_slim(model, left_image_seed_sub, imgsz=x2-x1, conf=conf_yoloe)
    final_res_list = post_processing_yoloe_results(final_res_list, args.task_name, roi_bbox)
    if args.debug_close_loop_vis:
        left_image_vis = left_image_seed.copy()
        left_image_vis[y1:y2, x1:x2] = plot_processed_results_vis(left_image_seed_sub.copy(), final_res_list, list(seed_ref_pts_dict.keys()) )
  
    for [obj_name, conf, bbox, mask] in final_res_list:
        mask[:, 0] += x1; mask[:, 1] += y1  # remember add back the offsets in x / y
        obj_binary_mask = polygon2mask((kfr_height, kfr_width), [mask], color=255, downsample_ratio=1)
        if args.task_name in ["plugpen", "reorient"] or (args.task_name=="inserting" and obj_name=="pen"):
            rot_deg_init, pts_raw, pts_trans = compute_rotation_by_image_moments(mask, 
                radius_ratio=radius_ratio, trans_mat=transform_mat_inv, task_name=args.task_name)
            [c_point_raw, e_point_raw], [c_point_trans, e_point_trans] = pts_raw, pts_trans
            seed_ref_pts_dict[obj_name] = [c_point_raw, rot_deg_init, bbox, e_point_raw]
        if args.task_name in ["unscrew", "pouring", "pressing"] or (args.task_name=="inserting" and obj_name=="cup"):
            max_y_val = mask[:, 1].max()
            max_x_val = mask[list(mask[:, 1]).index(max_y_val), 0]
            seed_ref_pts_dict[obj_name] = [[max_x_val, max_y_val], 0.0, bbox, [max_x_val, max_y_val]]  # only one single point
    if args.debug_close_loop_vis:
        result_img_init = cv2.warpPerspective(left_image_seed, transform_mat_inv, (rect_l, rect_w))
        cv2.polylines(left_image_vis, [pts_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        
        for g_idx in range(grid_ptx_num + 1):
            px1, py1 = int(grid_top_xy_list_src[g_idx, 0]), int(grid_top_xy_list_src[g_idx, 1])
            px2, py2 = int(grid_bottom_xy_list_src[g_idx, 0]), int(grid_bottom_xy_list_src[g_idx, 1])
            cv2.line(left_image_vis, (px1, py1), (px2, py2), color=(192,192,192), thickness=1, lineType=cv2.LINE_AA)
        for g_idx in range(grid_pty_num + 1):
            px1, py1 = int(grid_left_xy_list_src[g_idx, 0]), int(grid_left_xy_list_src[g_idx, 1])
            px2, py2 = int(grid_right_xy_list_src[g_idx, 0]), int(grid_right_xy_list_src[g_idx, 1])
            cv2.line(left_image_vis, (px1, py1), (px2, py2), color=(192,192,192), thickness=1, lineType=cv2.LINE_AA)
        for id_ptx in range(grid_ptx_num):
            for id_pty in range(grid_pty_num):
                gcx, gcy = int(grid_center_xy_list_src[id_ptx, id_pty, 0]), int(grid_center_xy_list_src[id_ptx, id_pty, 1])
                cv2.circle(left_image_vis, (gcx, gcy ), radius=5, color=(255,255,255), thickness=-1, lineType=cv2.LINE_AA)
                    
    reprojected_seeding_pts = []
    for obj_name, obj_position in seed_ref_pts_dict.items():
        if len(obj_position) == 0: continue
        [x, y], rot_deg_init, obj_bbox, e_point_raw = obj_position[0], obj_position[1], obj_position[2], obj_position[3]
        temp_pt = np.dot(transform_mat_inv, np.array([x, y, 1]).T)
        new_x, new_y = temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]
        reprojected_seeding_pts.append([new_x, new_y, rot_deg_init, obj_bbox, [x, y]])
        if args.debug_close_loop_vis:
            cv2.circle(left_image_vis, (int(x), int(y)), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(left_image_vis, (int(x), int(y)), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
            if args.task_name in ["plugpen", "reorient"] or (args.task_name=="inserting" and obj_name=="pen"):    
                cv2.arrowedLine(left_image_vis, (int(x), int(y)), (e_point_raw[0], e_point_raw[1]), 
                            (0,255,255), thickness=2, tipLength = 0.25, line_type=cv2.LINE_AA)
            cv2.circle(result_img_init, (int(new_x), int(new_y)), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(result_img_init, (int(new_x), int(new_y)), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
    
    if args.debug_close_loop_vis:
        video_output_path = saved_seed_img_path.replace("_seed.jpg", f"_test_orient{args.orient_id}_tag{args.given_tag}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vout = cv2.VideoWriter(video_output_path, fourcc, 5, (1452, 1080))  # Create VideoWriter object, FPS = 5 
        video_max_frame_num = 600  # about 120 seconds / 2 minutes
        total_frame_index = 0
        saved_frame_count = 0
        start_time = time.time()
        if args.task_name in ["plugpen"]: 
            one_epoch_max_frame_number = L_posx_max * L_posy_max  # for example 3*6=18
            check_save_interval = 20  # checking and saving once every check_save_interval frames
        if args.task_name in ["reorient"]: 
            one_epoch_max_frame_number = LR_posx_max * 3 # for example 6*3=18
            check_save_interval = 15  # checking and saving once every check_save_interval frames
        if args.task_name in ["unscrew"]: 
            one_epoch_max_frame_number = LR_posx_max * 3 # for example 12*3=36
            check_save_interval = 10  # checking and saving once every check_save_interval frames
        if args.task_name in ["pouring", "inserting", "pressing"]: 
            one_epoch_max_frame_number = L_posx_max * L_posy_max  # for example 3*6=18
            check_save_interval = 15  # checking and saving once every check_save_interval frames
    ############################################################################

    
    
    
    
    
    
    
    
    
    
    
    ############################################################################
    if args.task_name == "plugpen":  # close-loop for the left-arm mand right-arm
        # original robot_arms: ['L', 'R', 'L', 'R', 'L', 'R', 'L']  # 7 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
        interp_num = 5  # after removing start_eef_pose and end_eef_pose, we actuall add (interp_num-2) new actions
    if args.task_name == "reorient":  # close-loop for the right-arm
        # original robot_arms: ['R', 'R', 'R', 'L', 'R', 'R', 'L', 'L'] 8 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:1], eef_pose_seqs[:1]
        robot_arms, eef_pose_seqs = robot_arms[1:], eef_pose_seqs[1:]
        interp_num = 6  # after removing start_eef_pose and end_eef_pose, we actuall add (interp_num-2) new actions
    if args.task_name == "unscrew":  # close-loop for the left-arm
        # original robot_arms: ['L', 'L', 'L', 'R', 'R', 'R', 'R', 'R', 'R', R', 'R', 'L'] 12 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:1], eef_pose_seqs[:1]
        robot_arms, eef_pose_seqs = robot_arms[1:], eef_pose_seqs[1:]
        interp_num = 6  # after removing start_eef_pose and end_eef_pose, we actuall add (interp_num-2) new actions
    if args.task_name == "pouring":  # close-loop for the right-arm and left-arm
        # original robot_arms: ['R', 'R', 'L', 'L', 'R', 'L', 'L', 'L', 'L', 'R'] 10 actions
        cl_robot_arms, cl_eef_pose_seqs = ['R', 'L'], [eef_pose_seqs[0], eef_pose_seqs[2]]
        robot_arms, eef_pose_seqs = ['R', 'L'] + robot_arms[4:], eef_pose_seqs[1:2] + eef_pose_seqs[3:]
        interp_num = 6  # after removing start_eef_pose and end_eef_pose, we actuall add (interp_num-2) new actions

    if args.task_name == "inserting":  # close-loop for the left-arm and right-arm 
        # original robot_arms: ['L', 'L', 'L', 'R', 'R', 'L', 'R', 'R', 'L'] 9 actions
        cl_robot_arms, cl_eef_pose_seqs = ['L', 'R'], [eef_pose_seqs[0], eef_pose_seqs[3]]
        robot_arms, eef_pose_seqs = ['L', 'L'] + robot_arms[4:], eef_pose_seqs[1:3] + eef_pose_seqs[4:]
        interp_num = 6  # after removing start_eef_pose and end_eef_pose, we actuall add (interp_num-2) new actions
    if args.task_name == "pressing":  # close-loop for the right-arm and left-arm
        # original robot_arms: ['R', 'L, 'L', 'L', 'R', 'R', 'L'] 7 actions
        cl_robot_arms, cl_eef_pose_seqs = ['R', 'L'], [eef_pose_seqs[0], eef_pose_seqs[1]]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
        interp_num = 6  # after removing start_eef_pose and end_eef_pose, we actuall add (interp_num-2) new actions
        
    ############################################################################
    while True:
        if saved_frame_count == one_epoch_max_frame_number or total_frame_index > video_max_frame_num: break  # stop this loop
        left_image_test, right_image_test = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
        if args.debug_close_loop_vis: total_frame_index += 1

        ##########################
        left_image_test_sub = left_image_test[y1:y2, x1:x2]
        try:
            final_res_list, im_bgr = ov_det_seg_yoloe_slim(model, left_image_test_sub, imgsz=x2-x1, conf=conf_yoloe)
            final_res_list = post_processing_yoloe_results(final_res_list, args.task_name, roi_bbox)
            assert len(final_res_list) != 0, "[Warning] at least one manipulated object should be detected."
        except:
            print("[Warning] No at least one object is detected in this frame!!! return back")
            continue
        if args.debug_close_loop_vis:
            left_image_test_vis = left_image_test.copy()
            # left_image_test_vis[y1:y2, x1:x2] = plot_processed_results_vis(left_image_test_sub.copy(), final_res_list, list(seed_ref_pts_dict.keys()) )
        
        obj_binary_mask_dict= {}   
        for [obj_name, conf, bbox, mask] in final_res_list:
            mask[:, 0] += x1; mask[:, 1] += y1  # remember add back the offsets in x / y
            obj_binary_mask = polygon2mask((kfr_height, kfr_width), [mask], color=255, downsample_ratio=1)
            adjusted_bbox = [bbox[0]+x1, bbox[1]+y1, bbox[2]+x1, bbox[3]+y1]
            obj_binary_mask_dict[obj_name] = [obj_binary_mask, mask.tolist(), adjusted_bbox]  # binary mask image, [N, 2] numpy array (tolist()), object bbox
            if args.task_name in ["plugpen", "reorient"] or (args.task_name=="inserting" and obj_name=="pen"):
                rot_deg_test, pts_raw, pts_trans = compute_rotation_by_image_moments(mask,
                    radius_ratio=radius_ratio, trans_mat=transform_mat_inv, task_name=args.task_name)
                [c_point_raw_t, e_point_raw_t], [c_point_trans_t, e_point_trans_t] = pts_raw, pts_trans
                test_ref_pts_dict[obj_name] = [c_point_raw_t, rot_deg_test, bbox, e_point_raw_t]
            if args.task_name in ["unscrew", "pouring", "pressing"] or (args.task_name=="inserting" and obj_name=="cup"):
                max_y_val = mask[:, 1].max()
                max_x_val = mask[list(mask[:, 1]).index(max_y_val), 0]
                test_ref_pts_dict[obj_name] = [[max_x_val, max_y_val], 0.0, bbox, [max_x_val, max_y_val]]  # only one single point
        if args.debug_close_loop_vis:
            result_img_test = cv2.warpPerspective(left_image_test, transform_mat_inv, (rect_l, rect_w))
            cv2.polylines(left_image_test_vis, [pts_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
            
            for g_idx in range(grid_ptx_num + 1):
                px1, py1 = int(grid_top_xy_list_src[g_idx, 0]), int(grid_top_xy_list_src[g_idx, 1])
                px2, py2 = int(grid_bottom_xy_list_src[g_idx, 0]), int(grid_bottom_xy_list_src[g_idx, 1])
                cv2.line(left_image_test_vis, (px1, py1), (px2, py2), color=(192,192,192), thickness=1, lineType=cv2.LINE_AA)
            for g_idx in range(grid_pty_num + 1):
                px1, py1 = int(grid_left_xy_list_src[g_idx, 0]), int(grid_left_xy_list_src[g_idx, 1])
                px2, py2 = int(grid_right_xy_list_src[g_idx, 0]), int(grid_right_xy_list_src[g_idx, 1])
                cv2.line(left_image_test_vis, (px1, py1), (px2, py2), color=(192,192,192), thickness=1, lineType=cv2.LINE_AA)
            for id_ptx in range(grid_ptx_num):
                for id_pty in range(grid_pty_num):
                    gcx, gcy = int(grid_center_xy_list_src[id_ptx, id_pty, 0]), int(grid_center_xy_list_src[id_ptx, id_pty, 1])
                    cv2.circle(left_image_test_vis, (gcx, gcy ), radius=5, color=(255,255,255), thickness=-1, lineType=cv2.LINE_AA)

        ##########################
        
        reprojected_testing_pts, left_cls_list = [], []
        for obj_name, obj_position in test_ref_pts_dict.items():
            if len(obj_position) == 0: continue
            [x, y], rot_deg_test, obj_bbox, e_point_raw_t = obj_position[0], obj_position[1], obj_position[2], obj_position[3]
            temp_pt = np.dot(transform_mat_inv, np.array([x, y, 1]).T)
            new_x, new_y = temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]

            min_dist, pos_x_idx, pos_y_idx = 10000, -1, -1
            for id_ptx in range(grid_ptx_num):
                for id_pty in range(grid_pty_num):
                    gcx, gcy = grid_center_xy_list_tgt[id_ptx, id_pty, 0], grid_center_xy_list_tgt[id_ptx, id_pty, 1]
                    temp_dist = (gcx - new_x)**2 + (gcy - new_y)**2
                    if temp_dist < min_dist: min_dist = temp_dist; pos_x_idx, pos_y_idx = id_ptx, id_pty
                    
            if args.debug_close_loop_vis:
                if pos_x_idx != -1 and pos_y_idx != -1:   # we have find the grid of one object, highlight this grid cell !!!
                    grid_cell_pts_list = [] 
                    temp_pt = np.dot(transform_mat, np.array([grid_top_xy_list_tgt[pos_x_idx, 0], grid_left_xy_list_tgt[pos_y_idx, 1], 1]).T)
                    grid_cell_pts_list.append([int(temp_pt[0]/temp_pt[2]), int(temp_pt[1]/temp_pt[2])])
                    temp_pt = np.dot(transform_mat, np.array([grid_top_xy_list_tgt[pos_x_idx+1, 0], grid_left_xy_list_tgt[pos_y_idx, 1], 1]).T)
                    grid_cell_pts_list.append([int(temp_pt[0]/temp_pt[2]), int(temp_pt[1]/temp_pt[2])])
                    temp_pt = np.dot(transform_mat, np.array([grid_top_xy_list_tgt[pos_x_idx+1, 0], grid_left_xy_list_tgt[pos_y_idx+1, 1], 1]).T)
                    grid_cell_pts_list.append([int(temp_pt[0]/temp_pt[2]), int(temp_pt[1]/temp_pt[2])])
                    temp_pt = np.dot(transform_mat, np.array([grid_top_xy_list_tgt[pos_x_idx, 0], grid_left_xy_list_tgt[pos_y_idx+1, 1], 1]).T)
                    grid_cell_pts_list.append([int(temp_pt[0]/temp_pt[2]), int(temp_pt[1]/temp_pt[2])])
                    grid_cell_polygon = np.array(grid_cell_pts_list, np.int32).reshape((-1, 1, 2))  # polygon corner points coordinates
                    if total_frame_index % 2 == 0: cv2.polylines(left_image_test_vis, [grid_cell_polygon], isClosed=True, color=(255,255,255), thickness=2)
                    if total_frame_index % 2 == 1: cv2.polylines(left_image_test_vis, [grid_cell_polygon], isClosed=True, color=(128,128,128), thickness=2)

                [x_val, y_val] = test_ref_pts_dict[obj_name][0]
                cv2.circle(left_image_test_vis, (int(x), int(y)), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(left_image_test_vis, (int(x), int(y)), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
                if args.task_name in ["plugpen", "reorient"] or (args.task_name=="inserting" and obj_name=="pen"):
                    cv2.arrowedLine(left_image_test_vis, (int(x), int(y)), (e_point_raw_t[0], e_point_raw_t[1]), 
                                    (0,255,255), thickness=2, tipLength = 0.25, line_type=cv2.LINE_AA)
                
                cv2.circle(result_img_test, (int(new_x), int(new_y)), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(result_img_test, (int(new_x), int(new_y)), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
                        
            reprojected_testing_pts.append([new_x, new_y, rot_deg_test, obj_bbox, [x, y], [pos_x_idx, pos_y_idx]])
            left_cls_list.append(obj_name)
        
        if args.debug_close_loop_vis and len(left_cls_list) != len(list(obj_binary_mask_dict.keys())): continue  # skip this frame for missing detection of some objects       
                
        if args.debug_close_loop_vis:
            left_image_test_vis[y1:y2, x1:x2] = plot_processed_results_vis(left_image_test_vis[y1:y2, x1:x2], final_res_list, list(seed_ref_pts_dict.keys()) )
            
            if total_frame_index % check_save_interval == 0:  # checking and saving once every check_save_interval frames
                cv2.rectangle(left_image_test_vis, (3,3), (kfr_width-6, kfr_height-6), color=(0,255,255), thickness=6)
                cv2.putText(left_image_test_vis, f'Checking & Saving...', (kfr_width//3, 36),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0,255,255), thickness=3, lineType=cv2.LINE_AA)
            
            left_imgs = np.vstack((left_image_vis, left_image_test_vis))   # shape (1080, 960, 3)
            cv2.putText(left_imgs, f'Frame Index {total_frame_index}', (10, kfr_height+30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(left_imgs, f'Examplar (static)', (int(0.72*kfr_width), 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(left_imgs, f'NewScene (dynamic)', (int(0.70*kfr_width), kfr_height+30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
            result_imgs = np.vstack((result_img_init, result_img_test))  # shape (rect_w*2, rect_l, 3) --> (1350, 616, 3)
            
            for idx, (pt_seeding, pt_testing) in enumerate(zip(reprojected_seeding_pts, reprojected_testing_pts)):
                delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
                print(total_frame_index, idx, "\n pt_seeding:", pt_seeding, "\n pt_testing:", pt_testing)
                print(f"Relative translation of two points (mm): (delta_x: {delta_x}, delta_y: {delta_y})")
                px1, py1, px2, py2 = int(pt_seeding[0]), int(pt_seeding[1]), int(pt_testing[0]), int(pt_testing[1])
                cv2.line(result_imgs, (px1, py1), (px2, py2+rect_w), color=colors_list[idx], thickness=2, lineType=cv2.LINE_AA)

                show_str, plot_xy = f'{left_cls_list[idx]}', (int(0.63*rect_l), int(0.2*rect_w+(4*idx-2)*30))
                cv2.putText(result_imgs, show_str, plot_xy, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=colors_list[idx], thickness=2, lineType=cv2.LINE_AA)
                show_str, plot_xy = f'Dx: {np.round(delta_x, 2)} mm', (int(0.63*rect_l), int(0.2*rect_w+(4*idx-1)*30))
                cv2.putText(result_imgs, show_str, plot_xy, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=colors_list[idx], thickness=2, lineType=cv2.LINE_AA)
                show_str, plot_xy = f'Dy: {np.round(delta_y, 2)} mm', (int(0.63*rect_l), int(0.2*rect_w+(4*idx)*30))
                cv2.putText(result_imgs, show_str, plot_xy, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=colors_list[idx], thickness=2, lineType=cv2.LINE_AA)

                rot_deg_init, rot_deg_test = pt_seeding[2], pt_testing[2]
                show_str, plot_xy = f'Rot: {np.round(rot_deg_test, 0)} deg', (int(0.63*rect_l), int(0.2*rect_w+(4*idx+1)*30))
                cv2.putText(result_imgs, show_str, plot_xy, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.85, color=colors_list[idx], thickness=3, lineType=cv2.LINE_AA)

                if abs(delta_x) > 10 or abs(delta_y) > 10:  # we think this object is moved
                    plot_x1y1 = ( int(0.62*rect_l), int(0.2*rect_w+(4*idx-3)*30)+6 )
                    plot_x2y2 = ( int(0.99*rect_l), int(0.2*rect_w+(4*idx+1)*30)+6 )
                    cv2.rectangle(result_imgs, plot_x1y1, plot_x2y2, color=colors_list[idx], thickness=2)
                    
            result_imgs = cv2.resize(result_imgs, (int(616*1080/1350.0), 1080))  # shape (1080, 493, 3)
            cv2.putText(result_imgs, f'Top-view Reprojection', (5, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.8, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
            final_imgs = np.hstack((left_imgs, result_imgs))  # shape (1080, 1452, 3)            
            
            cv2.namedWindow("MyDebugWindow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("MyDebugWindow", 1452, 1080)
            cv2.imshow("MyDebugWindow", final_imgs)
            print("[One Frame Finished]", total_frame_index)
            if total_frame_index % check_save_interval == 0:
                for _ in range(5): vout.write(final_imgs); time.sleep(0.2)
            else:
                vout.write(final_imgs)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # if video_frame_index == 1: cv2.imwrite(saved_seed_img_path.replace("_seed.jpg", "_test_frame1.jpg"), final_imgs)
        
        ##########################
        demo_tag_info = ""
        grid_xy_marker_list = []
        aligned_init_frames = []
        for step_id_record in range(len(cl_robot_arms)):
            if total_frame_index % check_save_interval != 0: continue  # checking and saving once every check_save_interval frames

            robot_arm_temp, eef_pose_temp = cl_robot_arms[step_id_record], cl_eef_pose_seqs[step_id_record]
            print(total_frame_index, "[initial frame alignment]", step_id_record, robot_arm_temp, eef_pose_temp)
            
            try:
                pt_seeding, pt_testing = reprojected_seeding_pts[step_id_record], reprojected_testing_pts[step_id_record]
            except:
                print(reprojected_seeding_pts, reprojected_testing_pts); sys.exit()
            delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
            rot_deg_init, rot_deg_test = pt_seeding[2], pt_testing[2]  # for tasks "plugpen", "reorient", "inserting"
            obj_bbox_init, obj_bbox_test = pt_seeding[3], pt_testing[3]  # for tasks with frequent occlusion between the robot-arm and targeted object
            [obj_pos_x, obj_pos_y] = pt_testing[4]  # for calculating the grid x/y ids of this object using its original [x, y] info
            
            [grid_cell_x_id, grid_cell_y_id] = pt_testing[5]
            grid_xy_marker_list.append([grid_cell_x_id, grid_cell_y_id])
            gcx_id, gcy_id = str(grid_cell_x_id + 1).zfill(2), str(grid_cell_y_id + 1).zfill(2)
            obj_name_str = left_cls_list[step_id_record]
            if args.task_name in ["plugpen", "inserting"]: # with two objects and arm order ['L', 'R']
                demo_tag_info = demo_tag_info + f"{robot_arm_temp}{obj_name_str}X{gcx_id}Y{gcy_id}-"
            if args.task_name in ["pouring", "pressing"]: # with two objects and arm order ['R', 'L']
                demo_tag_info = f"{robot_arm_temp}{obj_name_str}X{gcx_id}Y{gcy_id}-" + demo_tag_info
            if args.task_name in ["reorient", "unscrew"]:  # only one object
                demo_tag_info = f"LR{obj_name_str}X{gcx_id}Y{gcy_id}-"

            # get end_eef_pose
            cur_end_eef_pose = eef_pose_temp.copy()
            if robot_arm_temp == "L": cur_end_eef_pose[0] -= (delta_y / 1000); cur_end_eef_pose[1] -= (delta_x / 1000)
            if robot_arm_temp == "R": cur_end_eef_pose[0] += (delta_y / 1000); cur_end_eef_pose[1] += (delta_x / 1000)
            if args.task_name in ["plugpen", "reorient", "inserting"]:
                delta_rot_z = rot_deg_test - rot_deg_init
                if robot_arm_temp == "L": cur_end_eef_pose[5] = robot_init_pose_L[5] + delta_rot_z
                if robot_arm_temp == "R": cur_end_eef_pose[5] = robot_init_pose_R[5] + delta_rot_z
                
            # get start_eef_pose
            if robot_arm_temp == "L": cur_start_eef_pose = robot_init_pose_L
            if robot_arm_temp == "R": cur_start_eef_pose = robot_init_pose_R
            
            # pose_list = [cur_start_eef_pose, cur_end_eef_pose]
            # traj_poses = generate_interpolation_traj(pose_list, interp_num, is_6dof=True)  # length interp_num (not used this time)

            if args.is_rollout:
                if robot_arm_temp == "L": robotBase = robotBase_L; m_gripper = m_gripper_L; g_port = cfg_dict["arm1"]["gripper_port"]
                if robot_arm_temp == "R": robotBase = robotBase_R; m_gripper = m_gripper_R; g_port = cfg_dict["arm2"]["gripper_port"]
                m_gripper = dh_modbus_gripper(dh_device(g_port), gripper_ID)
                m_gripper.open(baudrate)
                robotBase.move_to_target_in_cartesian(cur_end_eef_pose[:6])
                if cur_end_eef_pose[-1] == 0: m_gripper.SetTargetPosition(5)  # close gripper, range is (0, 1000)
                if cur_end_eef_pose[-1] == 1: m_gripper.SetTargetPosition(995)  # close gripper, range is (0, 1000)
                g_state = 0
                while(g_state == 0):
                    g_state = m_gripper.GetGripState(); time.sleep(0.1)

            if args.task_name == "unscrew":
                eef_pose_seqs[0][:2] = cur_end_eef_pose[:2]  # modify the x and y lifting-up position of left-arm
            if args.task_name == "pressing":
                if robot_arm_temp == "R":  # for the right-arm, approaching and pressing the bottle
                    pt_seeding, pt_testing = reprojected_seeding_pts[0], reprojected_testing_pts[0]
                    delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
                    eef_pose_seqs[1][0] -= (delta_y / 1000)  # modify the x position of left-arm when moving ordcup
                    eef_pose_seqs[1][1] -= (delta_x / 1000)  # modify the y position of left-arm when moving ordcup
                    eef_pose_seqs[2][0] += (delta_y / 1000)  # modify the x position of right-arm when pressing nozzle
                    eef_pose_seqs[2][1] += (delta_x / 1000)  # modify the y position of right-arm when pressing nozzle
                    eef_pose_seqs[3][0] += (delta_y / 1000)  # modify the x position of right-arm when unpressing nozzle
                    eef_pose_seqs[3][1] += (delta_x / 1000)  # modify the y position of right-arm when unpressing nozzle
                if robot_arm_temp == "L":  # for the left-arm, grasping and lifting-up the ordcup
                    pt_seeding, pt_testing = reprojected_seeding_pts[1], reprojected_testing_pts[1]
                    delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
                    eef_pose_seqs[0][0] -= (delta_y / 1000)  # modify the x position of left-arm when lifting-up ordcup
                    eef_pose_seqs[0][1] -= (delta_x / 1000)  # modify the y position of left-arm when lifting-up ordcup
        
            # save the cur_end_eef_pose as a new target 6-DoF pose
            aligned_init_frames.append([robot_arm_temp, cur_end_eef_pose])
            
        ##########################
        if total_frame_index % check_save_interval == 0:  # checking and saving once every check_save_interval frames
            # save the left 6-DoF poses (eef_pose_seqs) as the total action labels
            print(total_frame_index, "[***Saving Demonstration***] save the left 6-DoF poses (eef_pose_seqs) as the total action labels.")
            save_imgL_raw_path = os.path.join(save_demos_sub_dir, demo_tag_info + "imgL_raw.jpg")
            save_imgR_raw_path = os.path.join(save_demos_sub_dir, demo_tag_info + "imgR_raw.jpg")
            save_imgL_mask_path = os.path.join(save_demos_sub_dir, demo_tag_info + "imgL_mask.jpg")
            save_imgL_mask_arr_path = os.path.join(save_demos_sub_dir, demo_tag_info + "imgL_mask_arr.json")
            save_json_labels_path = os.path.join(save_demos_sub_dir, demo_tag_info + "labels.json")

            print("Begin to save a new demonstration...")
            if args.debug_close_loop_vis:
                for [gcx_id, gcy_id] in grid_xy_marker_list:  # plot a X marker in the left_image_vis image for better collection
                    temp_pt = np.dot(transform_mat, np.array([grid_top_xy_list_tgt[gcx_id, 0], grid_left_xy_list_tgt[gcy_id, 1], 1]).T)
                    gc_tl = [int(temp_pt[0]/temp_pt[2]), int(temp_pt[1]/temp_pt[2])]
                    temp_pt = np.dot(transform_mat, np.array([grid_top_xy_list_tgt[gcx_id+1, 0], grid_left_xy_list_tgt[gcy_id+1, 1], 1]).T)
                    gc_br = [int(temp_pt[0]/temp_pt[2]), int(temp_pt[1]/temp_pt[2])]
                    cv2.line(left_image_vis, (gc_tl[0], gc_tl[1]), (gc_br[0], gc_br[1]), color=(0,0,255), thickness=3, lineType=cv2.LINE_AA)
                    temp_pt = np.dot(transform_mat, np.array([grid_top_xy_list_tgt[gcx_id+1, 0], grid_left_xy_list_tgt[gcy_id, 1], 1]).T)
                    gc_tr = [int(temp_pt[0]/temp_pt[2]), int(temp_pt[1]/temp_pt[2])]
                    temp_pt = np.dot(transform_mat, np.array([grid_top_xy_list_tgt[gcx_id, 0], grid_left_xy_list_tgt[gcy_id+1, 1], 1]).T)
                    gc_bl = [int(temp_pt[0]/temp_pt[2]), int(temp_pt[1]/temp_pt[2])]
                    cv2.line(left_image_vis, (gc_tr[0], gc_tr[1]), (gc_bl[0], gc_bl[1]), color=(0,0,255), thickness=3, lineType=cv2.LINE_AA)  
            
            if not os.path.exists(save_imgL_raw_path): saved_frame_count += 1
            
            cv2.imwrite(save_imgL_raw_path, left_image_test)
            cv2.imwrite(save_imgR_raw_path, right_image_test)
            
            imgL_mask_arr_dict = {}
            for obj_name_str in left_cls_list:
                [obj_binary_mask, mask_arr, obj_bbox] = obj_binary_mask_dict[obj_name_str]
                cv2.imwrite(save_imgL_mask_path[:-4] + f"_{obj_name_str}.jpg", obj_binary_mask)
                imgL_mask_arr_dict[obj_name_str] = [obj_bbox, mask_arr]
            with open(save_imgL_mask_arr_path, "w") as json_file:
                json.dump(imgL_mask_arr_dict, json_file)
                
            action_labels_json_list = []
            action_labels_json_list.append(["L", robot_init_pose_L])  # init L robot pose
            action_labels_json_list.append(["R", robot_init_pose_R])  # init R robot pose
            for robot_arm, eef_pose_seq in aligned_init_frames:  # first L/R/LR robot pose
                action_labels_json_list.append([robot_arm, eef_pose_seq])  
            for robot_arm, eef_pose_seq in zip(robot_arms, eef_pose_seqs):  # the left robot poses
                action_labels_json_list.append([robot_arm, eef_pose_seq])
            with open(save_json_labels_path, "w") as json_file:
                json.dump(action_labels_json_list, json_file)

        
        ##########################
        if args.is_rollout:
            ############################################################################
            for step_id, (robot_arm, eef_pose_seq) in enumerate(zip(robot_arms, eef_pose_seqs)):
                target_pose, gripper = eef_pose_seq[:6], eef_pose_seq[-1]
                print("\n", step_id, "robot:", robot_arm, "\t", "gripper:", gripper, "\t", "eef_pose:", target_pose)
                #######################################
                if robot_arm == "L":
                    m_gripper_L = dh_modbus_gripper(dh_device(cfg_dict["arm1"]["gripper_port"]), gripper_ID)
                    m_gripper_L.open(baudrate)
                    robotBase_L.move_to_target_in_cartesian(target_pose)
                    if gripper == 0: m_gripper_L.SetTargetPosition(5)  # close gripper, range is (0, 1000)
                    if gripper == 1: m_gripper_L.SetTargetPosition(995)  # close gripper, range is (0, 1000)
                    g_state = 0
                    while(g_state == 0):
                        g_state = m_gripper_L.GetGripState(); time.sleep(0.1)
                #######################################
                if robot_arm == "R":
                    m_gripper_R = dh_modbus_gripper(dh_device(cfg_dict["arm2"]["gripper_port"]), gripper_ID)
                    m_gripper_R.open(baudrate)
                    robotBase_R.move_to_target_in_cartesian(target_pose)
                    if gripper == 0: m_gripper_R.SetTargetPosition(5)  # close gripper, range is (0, 1000)
                    if gripper == 1: m_gripper_R.SetTargetPosition(995)  # close gripper, range is (0, 1000)
                    g_state = 0
                    while(g_state == 0):
                        g_state = m_gripper_R.GetGripState(); time.sleep(0.1)
                #######################################
            ############################################################################
            if given_execute_arm_name == "arm1":
                robotBase_L.move_to_target_in_cartesian(cur_pose_deg_L)
            if given_execute_arm_name == "arm2":
                robotBase_R.move_to_target_in_cartesian(cur_pose_deg_R)
            if given_execute_arm_name is None:
                robotBase_L.move_to_target_in_cartesian(cur_pose_deg_L)
                robotBase_R.move_to_target_in_cartesian(cur_pose_deg_R)
            ############################################################################
        ##########################
        
        print(time.time(), "\n\n\n")
        # time.sleep(1)  # do not update too fast !!!
        
    ############################################################################  
    if args.debug_close_loop_vis:
        video_output_path_last_frame = video_output_path.replace(".mp4", "_lastFrame.jpg")
        final_imgs[:kfr_height, :kfr_width] = left_image_vis
        cv2.imwrite(video_output_path_last_frame, final_imgs)
        
        cv2.destroyAllWindows()
        vout.release()  # Release everything ()
        
        end_time = time.time()
        print("Total time (seconds):", end_time-start_time)
    ############################################################################   
        
'''
##################################################################################################################
plugpen     marker02_test_orient1       LpenXxxYxxRcapX05Y04        18 demos        92 seconds    
plugpen     marker02_test_orient2       LpenXxxYxxRcapX05Y04        18 demos        91 seconds
plugpen     marker02_test_orient3       LpenXxxYxxRcapX05Y04        18 demos        93 seconds  
plugpen     marker02_test_orient4       LpenX02Y04RcapXxxYxx        18 demos        92 seconds
########################
plugpen     marker04_test_orient1       LpenXxxYxxRcapX05Y04        18 demos        93 seconds
plugpen     marker04_test_orient2       LpenXxxYxxRcapX05Y04        18 demos        94 seconds
plugpen     marker04_test_orient3       LpenXxxYxxRcapX05Y04        18 demos        92 seconds  
plugpen     marker04_test_orient4       LpenX02Y04RcapXxxYxx        18 demos        91 seconds

##################################################################################################################
reorient    anyobj01_test_orient1       LRspoonX010203Yxx           18 demos        70 seconds
reorient    anyobj01_test_orient1       LRspoonX040506Yxx           18 demos        71 seconds
reorient    anyobj01_test_orient2       LRspoonX010203Yxx           18 demos        72 seconds
reorient    anyobj01_test_orient2       LRspoonX040506Yxx           18 demos        73 seconds
reorient    anyobj01_test_orient3       LRspoonX010203Yxx           18 demos        76 seconds
reorient    anyobj01_test_orient3       LRspoonX040506Yxx           18 demos        78 seconds
reorient    anyobj01_test_orient4       LRspoonX010203Yxx           18 demos        76 seconds
reorient    anyobj01_test_orient4       LRspoonX040506Yxx           18 demos        78 seconds
reorient    anyobj01_test_orient5       LRspoonX010203Yxx           18 demos        81 seconds
reorient    anyobj01_test_orient5       LRspoonX040506Yxx           18 demos        82 seconds
########################
reorient    anyobj03_test_orient1       LRspoonX010203Yxx           18 demos        71 seconds
reorient    anyobj03_test_orient1       LRspoonX040506Yxx           18 demos        102 seconds
reorient    anyobj03_test_orient2       LRspoonX010203Yxx           18 demos        70 seconds
reorient    anyobj03_test_orient2       LRspoonX040506Yxx           18 demos        75 seconds
reorient    anyobj03_test_orient3       LRspoonX010203Yxx           18 demos        85 seconds
reorient    anyobj03_test_orient3       LRspoonX040506Yxx           18 demos        97 seconds
reorient    anyobj03_test_orient4       LRspoonX010203Yxx           18 demos        79 seconds
reorient    anyobj03_test_orient4       LRspoonX040506Yxx           18 demos        72 seconds
reorient    anyobj03_test_orient5       LRspoonX010203Yxx           18 demos        69 seconds
reorient    anyobj03_test_orient5       LRspoonX040506Yxx           18 demos        73 seconds
########################
reorient    anyobj04_test_orient1       LRshovelX010203Yxx          18 demos        78 seconds
reorient    anyobj04_test_orient1       LRshovelX040506Yxx          18 demos        75 seconds
reorient    anyobj04_test_orient2       LRshovelX010203Yxx          18 demos        72 seconds
reorient    anyobj04_test_orient2       LRshovelX040506Yxx          18 demos        74 seconds
reorient    anyobj04_test_orient3       LRshovelX010203Yxx          18 demos        71 seconds
reorient    anyobj04_test_orient3       LRshovelX040506Yxx          18 demos        80 seconds
reorient    anyobj04_test_orient4       LRshovelX010203Yxx          18 demos        85 seconds
reorient    anyobj04_test_orient4       LRshovelX040506Yxx          18 demos        95 seconds
reorient    anyobj04_test_orient5       LRshovelX010203Yxx          18 demos        78 seconds
reorient    anyobj04_test_orient5       LRshovelX040506Yxx          18 demos        72 seconds
########################
reorient    anyobj08_test_orient1       LRshovelX010203Yxx          18 demos        91 seconds
reorient    anyobj08_test_orient1       LRshovelX040506Yxx          18 demos        87 seconds
reorient    anyobj08_test_orient2       LRshovelX010203Yxx          18 demos        81 seconds
reorient    anyobj08_test_orient2       LRshovelX040506Yxx          18 demos        82 seconds
reorient    anyobj08_test_orient3       LRshovelX010203Yxx          18 demos        86 seconds
reorient    anyobj08_test_orient3       LRshovelX040506Yxx          18 demos        80 seconds
reorient    anyobj08_test_orient4       LRshovelX010203Yxx          18 demos        77 seconds
reorient    anyobj08_test_orient4       LRshovelX040506Yxx          18 demos        71 seconds
reorient    anyobj08_test_orient5       LRshovelX010203Yxx          18 demos        75 seconds
reorient    anyobj08_test_orient5       LRshovelX040506Yxx          18 demos        74 seconds

##################################################################################################################
unscrew     bottle03_test_orient1       LRbottleX010203Yxx          36 demos        116 seconds
unscrew     bottle03_test_orient1       LRbottleX040506Yxx          36 demos        115 seconds
unscrew     bottle03_test_orient1       LRbottleX070809Yxx          36 demos        139 seconds
unscrew     bottle03_test_orient1       LRbottleX101112Yxx          36 demos        132 seconds
########################
unscrew     bottle04_test_orient1       LRbottleX010203Yxx          36 demos        122 seconds
unscrew     bottle04_test_orient1       LRbottleX040506Yxx          36 demos        132 seconds
unscrew     bottle04_test_orient1       LRbottleX070809Yxx          36 demos        129 seconds
unscrew     bottle04_test_orient1       LRbottleX101112Yxx          36 demos        136 seconds
########################
unscrew     bottle06_test_orient1       LRbottleX010203Yxx          36 demos        122 seconds
unscrew     bottle06_test_orient1       LRbottleX040506Yxx          36 demos        125 seconds
unscrew     bottle06_test_orient1       LRbottleX070809Yxx          36 demos        132 seconds
unscrew     bottle06_test_orient1       LRbottleX101112Yxx          36 demos        122 seconds
########################
unscrew     bottle07_test_orient1       LRbottleX010203Yxx          36 demos        133 seconds
unscrew     bottle07_test_orient1       LRbottleX040506Yxx          36 demos        132 seconds
unscrew     bottle07_test_orient1       LRbottleX070809Yxx          36 demos        115 seconds
unscrew     bottle07_test_orient1       LRbottleX101112Yxx          36 demos        138 seconds

##################################################################################################################
pouring     bottle03-mugcup01_test_orient1      LbottleXxxYxxRmugcapX05Y04      18 demos        84 seconds
pouring     bottle04-mugcup01_test_orient1      LbottleXxxYxxRmugcapX05Y04      18 demos        77 seconds
pouring     bottle06-mugcup01_test_orient1      LbottleXxxYxxRmugcapX05Y04      18 demos        82 seconds
pouring     bottle07-mugcup01_test_orient1      LbottleXxxYxxRmugcapX05Y04      18 demos        77 seconds
########################
pouring     bottle03-mugcup04_test_orient1      LbottleXxxYxxRmugcapX05Y04      18 demos        77 seconds
pouring     bottle04-mugcup04_test_orient1      LbottleXxxYxxRmugcapX05Y04      18 demos        76 seconds
pouring     bottle06-mugcup04_test_orient1      LbottleXxxYxxRmugcapX05Y04      18 demos        77 seconds
pouring     bottle07-mugcup04_test_orient1      LbottleXxxYxxRmugcapX05Y04      18 demos        78 seconds
########################
pouring     bottle03-mugcup01_test_orient2      LbottleX02Y04RmugcapXxxYxx      18 demos        75 seconds
pouring     bottle03-mugcup04_test_orient2      LbottleX02Y04RmugcapXxxYxx      18 demos        78 seconds
########################
pouring     bottle04-mugcup01_test_orient2      LbottleX02Y04RmugcapXxxYxx      18 demos        76 seconds
pouring     bottle04-mugcup04_test_orient2      LbottleX02Y04RmugcapXxxYxx      18 demos        80 seconds
########################
pouring     bottle06-mugcup01_test_orient2      LbottleX02Y04RmugcapXxxYxx      18 demos        78 seconds
pouring     bottle06-mugcup04_test_orient2      LbottleX02Y04RmugcapXxxYxx      18 demos        77 seconds
########################
pouring     bottle07-mugcup01_test_orient2      LbottleX02Y04RmugcapXxxYxx      18 demos        76 seconds
pouring     bottle07-mugcup04_test_orient2      LbottleX02Y04RmugcapXxxYxx      18 demos        82 seconds

##################################################################################################################
inserting   ordcup01-marker02_test_orient4      LordcupXxxYxxRmarkerX05Y04      18 demos        84 seconds
inserting   ordcup01-marker02_test_orient1      LordcupX02Y04RmarkerXxxYxx      18 demos        96 seconds
inserting   ordcup01-marker02_test_orient2      LordcupX02Y04RmarkerXxxYxx      18 demos        70 seconds
inserting   ordcup01-marker02_test_orient3      LordcupX02Y04RmarkerXxxYxx      18 demos        82 seconds
########################
inserting   ordcup02-marker04_test_orient4      LordcupXxxYxxRmarkerX05Y04      18 demos        81 seconds
inserting   ordcup02-marker04_test_orient1      LordcupX02Y04RmarkerXxxYxx      18 demos        76 seconds
inserting   ordcup02-marker04_test_orient2      LordcupX02Y04RmarkerXxxYxx      18 demos        82 seconds
inserting   ordcup02-marker04_test_orient3      LordcupX02Y04RmarkerXxxYxx      18 demos        75 seconds

##################################################################################################################
pressing    ordcup01-nozzle01_test_orient1      LordcupX02Y04RnozzleXxxYxx      18 demos        77 seconds
pressing    ordcup02-nozzle02_test_orient1      LordcupX02Y04RnozzleXxxYxx      18 demos        121 seconds
########################
pressing    ordcup01-nozzle02_test_orient1      LordcupXxxYxxRnozzleX05Y04      18 demos        79 seconds
pressing    ordcup02-nozzle01_test_orient1      LordcupXxxYxxRnozzleX05Y04      18 demos        87 seconds

##################################################################################################################
[New Idea 1] Towards Collision-Aware Task-Oriented Grasping via Vision-Guided Motion Planning
[New Idea 2] ToMa: Learn to Touch and Manipulate without Seeing
##################################################################################################################
'''
        
        
        
        
        
        
        
        
        
        

    