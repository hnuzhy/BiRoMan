
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
from src.utils import cfg_dict_init, polygon2mask, calInsideRectIOU
# from utils import generate_interpolation_traj
from src.utils import interpolate_se3_bspline_startend_poses
from src.utils import compute_rotation_by_image_moments

import kingfisher
from kingFisherR.utils_vfm import ov_det_seg_yoloe_slim
from kingFisherR.utils_vfm import post_processing_yoloe_results
from kingFisherR.utils_vfm import plot_processed_results_vis
from ultralytics import YOLOE



#################################################################
### This script can only be run in python3 for aubo robot's requirement
### conda acitvate embodychain
'''
################################################
##### no_interaction = False  for VLBiMan
################################################
python scripts_kfr/run_Real_Rollout_v1.py --test_id 1 --task_name plugpen --marker_id 1
python scripts_kfr/run_Real_Rollout_v1.py --test_id 1 --task_name reorient --anyobj_id 1
python scripts_kfr/run_Real_Rollout_v1.py --test_id 1 --task_name unscrew --bottle_id 6
python scripts_kfr/run_Real_Rollout_v1.py --test_id 1 --task_name pouring --bottle_id 6 --mugcup_id 1
python scripts_kfr/run_Real_Rollout_v1.py --test_id 1 --task_name inserting --ordcup_id 1 --marker_id 1
python scripts_kfr/run_Real_Rollout_v1.py --test_id 1 --task_name pressing --ordcup_id 1 --nozzle_id 1

python scripts_kfr/run_Real_Rollout_v1.py --test_id 1 --task_name reorient_unscrew --bottle_id 6
python scripts_kfr/run_Real_Rollout_v1.py --test_id 1 --task_name unscrew_pouring --bottle_id 6 --mugcup_id 1

python scripts_kfr/run_Real_Rollout_v1.py --test_id 1 --task_name tool_spoon
python scripts_kfr/run_Real_Rollout_v1.py --test_id 1 --task_name tool_funnel

################################################
##### no_interaction = True for BiDemoSyn
################################################
python scripts_kfr/run_Real_Rollout_v1.py --no_interaction --task_name plugpen --marker_id 2
python scripts_kfr/run_Real_Rollout_v1.py --no_interaction --task_name reorient --anyobj_id 1
python scripts_kfr/run_Real_Rollout_v1.py --no_interaction --task_name unscrew --bottle_id 6
python scripts_kfr/run_Real_Rollout_v1.py --no_interaction --task_name pouring --bottle_id 6 --mugcup_id 1
python scripts_kfr/run_Real_Rollout_v1.py --no_interaction --task_name inserting --ordcup_id 1 --marker_id 2
python scripts_kfr/run_Real_Rollout_v1.py --no_interaction --task_name pressing --ordcup_id 1 --nozzle_id 1

'''
#################################################################



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm_type', default="both", help="given_execute_arm_name. It can be arm1 arm2 or both")
    parser.add_argument('--root_dir_path', default="/home/dexforce/zhouhuayi/auboHandeyeCalib/outputs/", 
                        help="path to all processed dual-arm robot actions")
    parser.add_argument('--task_name', default="", help="string of bimanual task name. It can be \
                        plugpen reorient unscrew pouring uncover openbox")
    parser.add_argument('--kfr_ip', default="192.168.9.111", help="the ip address of the kingfisher-R-6000")
    parser.add_argument('--save_imgs_dir', default="/home/dexforce/zhouhuayi/auboHandeyeCalib/seedinit/", 
                        help="path to save all initial seeding images")
    parser.add_argument('--debug_close_loop_vis', action='store_true', help="default is False")
    parser.add_argument('--no_interaction', action='store_true', help="default is False")
    parser.add_argument('--no_video_out', action='store_true', help="default is False")
    parser.add_argument('--test_id', type=int, default=1, help="we may record many times of one specific task")
    
    parser.add_argument('--marker_id', type=int, default=1, help="marker_id is selected from 1 ~ 8")  # for task plugpen
    parser.add_argument('--anyobj_id', type=int, default=1, help="anyobj_id is selected from 1 ~ 8")  # for task reorient
    parser.add_argument('--bottle_id', type=int, default=6, help="bottle_id is selected from 1 ~ 8")  # for task unscrew and pouring
    parser.add_argument('--mugcup_id', type=int, default=1, help="mugcup_id is selected from 1 ~ 4")  # for task pouring
    parser.add_argument('--ordcup_id', type=int, default=1, help="ordcup_id is selected from 1 ~ 2")  # for task inserting
    parser.add_argument('--nozzle_id', type=int, default=1, help="nozzle_id is selected from 1 ~ 2")  # for task pressing
    # parser.add_argument('--bowl_id', type=int, default=1, help="bowl_id is selected from 1 ~ 6")  # for task flipping (Bimanual Occluded Grasping)
    
    args = parser.parse_args()
    
    args.debug_close_loop_vis = True  # always set it as True for better debugging code and collecting demonstrations

    ############################################################################
    ''' demos collection for VLBiMan [see figures and videos saved in ./scripts_kfr/realcap_v3/]
    plugpen                 marker_id                   [1,2,3,4]marker                             04 demos * 1
    reorient                anyobj_id                   [1,3]spoon + [4,8]shovel                    04 demos * 1
    unscrew                 bottle_id                   [1,2,3,4,5,6,7,8]bottle                     08 demos * 1
    pouring                 bottle_id+mugcup_id         [3,4,6,7]bottle + [1,2,3,4]mugcup           16 demos * 1
    inserting               ordcup_id+marker_id         [1,2]ordcup + [1,2,3,4]marker               08 demos * 1
    pressing                ordcup_id+nozzle_id         [1,2]ordcup + [1,2]nozzle                   04 demos * 1
    reorient_unscrew        bottle_id                   [3,4,6,7]bottle                             04 demos * 1
    unscrew_pouring         bottle_id+mugcup_id         [3,6]bottle + [1,3]mugcup                   04 demos * 1
    tool_spoon              spoon+bowlL+bowlS           [3]spoon + [1]bowlL + [1]bowlS              01 demos * 2
    tool_funnel             funnel+bottle+mugcup        [1]funnel + [3]bottle + [1]mugcup           01 demos * 2
    '''
    ############################################################################
    assert args.task_name in ["plugpen", "reorient", "unscrew", "pouring", "inserting", "pressing", "uncover", "openbox",
                         "reorient_unscrew", "unscrew_pouring", "tool_spoon", "tool_funnel", "sweeping"], "Please give a valid task name !!!"
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

    # roi_bbox = [110, 90, 770, 530]  # the [x1, y1, x2, y2] (960*540 --> 660*440; 16:9 --> 3:2)
    if args.task_name in ["unscrew", "pouring", "pressing", "unscrew_pouring"]:   # we aplly YOLO-World + FastSAM / YOLOE
        roi_bbox = [120, 0, 760, 540]  # the [x1, y1, x2, y2] (960*540 --> 640*540; 16:9 --> 32:27)
    if args.task_name in ["plugpen", "reorient", "inserting", "reorient_unscrew", "tool_spoon", "tool_funnel"]:  # we may need to apply PerSAM or sth else / YOLOE
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
        mask_pixel_count_dict = {"pen": 0, "cap": 0}  ##### specially for the plugpen task
        prev_delta_xy_list = [[0, 0, None], [0, 0, None]]
        # vfms_conf = {"fastsam": 0.5}; radius_ratio = 0.2
        radius_ratio = 0.2; conf_yoloe = 0.1; prompts = None
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
        saved_seed_img_path = os.path.join(task_folder_dir, 
            f"bottle{str(args.bottle_id).zfill(2)}-mugcup{str(args.mugcup_id).zfill(2)}_seed.jpg") 
        seed_ref_pts_dict, test_ref_pts_dict = {"cup":[], "bottle":[]}, {"cup":[], "bottle":[]}  # first grasp cup (right), then grasp bottle (left)
        prev_delta_xy_list = [[0, 0, None], [0, 0, None]]
        # vfms_conf = {"yoloworld": 0.5, "fastsam": 0.5}; radius_ratio = 0.2
        radius_ratio = 0.2; conf_yoloe = 0.2; prompts = ["bottle", "cup"]

    if args.task_name == "inserting":
        json_file_name = f"inserting_01_20250402202534_L_eep_new.json"
        saved_seed_img_path = os.path.join(task_folder_dir, 
            f"ordcup{str(args.ordcup_id).zfill(2)}-marker{str(args.marker_id).zfill(2)}_seed.jpg") 
        seed_ref_pts_dict, test_ref_pts_dict = {"cup":[], "pen":[]}, {"cup":[], "pen":[]}  # first grasp cup (left), then grasp pen (right)
        prev_delta_xy_list = [[0, 0, None], [0, 0, None]]
        # vfms_conf = {"yoloworld": 0.5, "fastsam": 0.5}; radius_ratio = 0.2
        radius_ratio = 0.2; conf_yoloe = 0.01; prompts = ["cup", "pen"]  # Note, the up-side-down cup cannot be detected stablely
    if args.task_name == "pressing":
        json_file_name = f"pressing_01_20250404154346_L_eep_new_LR{args.nozzle_id}.json"
        saved_seed_img_path = os.path.join(task_folder_dir,
            f"ordcup{str(args.ordcup_id).zfill(2)}-nozzle{str(args.nozzle_id).zfill(2)}_seed.jpg") 
        seed_ref_pts_dict, test_ref_pts_dict = {"bottle":[], "cup":[]}, {"bottle":[], "cup":[]}  # first approach bottle (right), then grasp cup (left)
        prev_delta_xy_list = [[0, 0, None], [0, 0, None]]
        # vfms_conf = {"yoloworld": 0.5, "fastsam": 0.5}; radius_ratio = 0.2
        radius_ratio = 0.2; conf_yoloe = 0.1; prompts = ["bottle", "cup"]

    if args.task_name == "reorient_unscrew":
        json_file_name = f"reorient_unscrew_01_20250325142150_L_eep_new_LR{args.bottle_id}.json"
        saved_seed_img_path = os.path.join(task_folder_dir, f"bottle{str(args.bottle_id).zfill(2)}_seed.jpg") 
        seed_ref_pts_dict, test_ref_pts_dict = {"bottle":[]}, {"bottle":[]}
        prev_delta_xy_list = [[0, 0, None], [0, 0, None]]
        # vfms_conf = {"sam2": 0.1}; radius_ratio = 0.3  # SAM2 performs better than fastSAM
        radius_ratio = 0.2; conf_yoloe = 0.01; prompts = ["bottle"]  # Note, the lying down bottle cannot be detected stablely
    if args.task_name == "unscrew_pouring":
        json_file_name = f"unscrew_pouring_01_20250325143913_L_eep_new_L{args.bottle_id}_R{args.mugcup_id}.json"
        saved_seed_img_path = os.path.join(task_folder_dir, 
            f"bottle{str(args.bottle_id).zfill(2)}-mugcup{str(args.mugcup_id).zfill(2)}_seed.jpg") 
        seed_ref_pts_dict, test_ref_pts_dict = {"bottle":[], "cup":[]}, {"bottle":[], "cup":[]}  # first grasp bottle, then grasp cup
        prev_delta_xy_list = [[0, 0, None], [0, 0, None]]
        # vfms_conf = {"yoloworld": 0.5, "fastsam": 0.5}; radius_ratio = 0.2
        radius_ratio = 0.2; conf_yoloe = 0.2; prompts = ["bottle", "cup"]

    if args.task_name == "tool_spoon":
        json_file_name = f"tool_spoon_01_20250402205636_L_eep_new.json"
        saved_seed_img_path = os.path.join(task_folder_dir, f"bowlL-bowlS-spoon_seed.jpg") 
        seed_ref_pts_dict, test_ref_pts_dict = {"spoon":[], "bowlL":[], "bowlS":[]}, {"spoon":[], "bowlL":[], "bowlS":[]}  # first reorient spoon, then spooning
        prev_delta_xy_list = [[0, 0, None], [0, 0, None], [0, 0, None]]
        radius_ratio = 0.2; conf_yoloe = 0.1; prompts = ["bowl", "spoon"]
    if args.task_name == "tool_funnel":
        json_file_name = f"tool_funnel_01_20250402211746_L_eep_new.json"  
        saved_seed_img_path = os.path.join(task_folder_dir, f"bottle-cup-funnel_seed.jpg") 
        seed_ref_pts_dict, test_ref_pts_dict = {"funnel":[], "bottle":[], "cup":[]}, {"funnel":[], "bottle":[], "cup":[]}  # first reorient funnel, then pouring
        prev_delta_xy_list = [[0, 0, None], [0, 0, None], [0, 0, None]]
        radius_ratio = 0.2; conf_yoloe = 0.01; prompts = ["bottle", "cup", "funnel"]  # Note, funnel cannot be detected, we treat it as a bottle
    
    if not os.path.exists(saved_seed_img_path):
        left_image_seed, _ = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
        cv2.imwrite(saved_seed_img_path, left_image_seed)
        sys.exit()
    else:
        left_image_seed = cv2.imread(saved_seed_img_path)

    ############################################################################  
    ''' Initialize a YOLOE model'''
    if prompts is None:  # (PF) Prompt-Free
        # model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11s-seg-pf.pt")
        model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11m-seg-pf.pt")
        # model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11l-seg-pf.pt")
    else:  # with Prompt (language/text or vision/image)
        # need to install https://github.com/ultralytics/CLIP and https://github.com/apple/ml-mobileclip
        # model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11s-seg.pt")
        model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11m-seg.pt")
        # model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11l-seg.pt")
        model.set_classes(prompts, model.get_text_pe(prompts))  # Set text prompt. e.g., prompts = ["person", "bus"]
        
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
    
    '''v1: combine (yoloworld + fast_segment_anything + segment_anything_v2)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    if "yoloworld" in vfms_conf:
        conf_yolo = vfms_conf["yoloworld"] 
        final_res_list1, im_bgr1 = open_vocabulary_detector(left_image_seed[y1:y2, x1:x2], imgsz=x2-x1, conf=conf_yolo)  # bboxes w/ obj categories
    else:
        final_res_list1 = []
    if "fastsam" in vfms_conf:
        conf_fastsam = vfms_conf["fastsam"] 
        final_res_list2, im_bgr2 = fast_segment_anything(left_image_seed[y1:y2, x1:x2], imgsz=x2-x1, conf=conf_fastsam)  # bboxes+masks w/o categories
    if "sam2" in vfms_conf:
        conf_sam2 = vfms_conf["sam2"] 
        final_res_list2, im_bgr2 = segment_anything_v2(left_image_seed[y1:y2, x1:x2], imgsz=x2-x1, conf=conf_sam2)  # bboxes + masks w/o categories
    final_res_list = merge_yoloworld_fastsam_results(final_res_list1, final_res_list2, args.task_name)
    if args.debug_close_loop_vis:
        left_image_vis = left_image_seed.copy()
        if args.task_name in ["plugpen", "reorient", "reorient_unscrew"]: left_image_vis[y1:y2, x1:x2] = im_bgr2
        if args.task_name in ["unscrew", "pouring", "unscrew_pouring"]: left_image_vis[y1:y2, x1:x2] = im_bgr1
    '''
    
    '''v2: only use yoloe'''
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    left_image_seed_sub = left_image_seed[y1:y2, x1:x2]
    final_res_list_raw, im_bgr = ov_det_seg_yoloe_slim(model, left_image_seed_sub, imgsz=x2-x1, conf=conf_yoloe)
    final_res_list = post_processing_yoloe_results(final_res_list_raw, args.task_name, roi_bbox)
    if args.debug_close_loop_vis:
        left_image_vis = left_image_seed.copy()
        left_image_vis[y1:y2, x1:x2] = plot_processed_results_vis(left_image_seed_sub.copy(), final_res_list, list(seed_ref_pts_dict.keys()) )
    
    for [obj_name, conf, bbox, mask] in final_res_list:
        mask[:, 0] += x1; mask[:, 1] += y1  # remember add back the offsets in x / y
        if args.task_name in ["plugpen"]:
            obj_binary_mask = polygon2mask((kfr_height, kfr_width), [mask], color=255, downsample_ratio=1)
            mask_pixel_count_dict[obj_name] = cv2.countNonZero(obj_binary_mask)  # save obj_area in the seeding image
        if args.task_name in ["plugpen", "reorient", "reorient_unscrew"] or \
            (args.task_name=="inserting" and obj_name=="pen") or (args.task_name=="tool_spoon" and obj_name=="spoon"):
            # obj_binary_mask = polygon2mask((kfr_height, kfr_width), [mask], color=255, downsample_ratio=1)
            # rot_deg_init, pts_raw, pts_trans = compute_rotation_by_object_mask(mask, obj_binary_mask, transform_mat_inv, args.task_name)
            rot_deg_init, pts_raw, pts_trans = compute_rotation_by_image_moments(mask, 
                radius_ratio=radius_ratio, trans_mat=transform_mat_inv, task_name=args.task_name)
            [c_point_raw, e_point_raw], [c_point_trans, e_point_trans] = pts_raw, pts_trans
            seed_ref_pts_dict[obj_name] = [c_point_raw, rot_deg_init, bbox]
        if args.task_name in ["unscrew", "pouring", "pressing", "unscrew_pouring", "tool_funnel"] or \
            (args.task_name=="inserting" and obj_name=="cup") or (args.task_name=="tool_spoon" and "bowl" in obj_name):
            max_y_val = mask[:, 1].max()
            max_x_val = mask[list(mask[:, 1]).index(max_y_val), 0]
            seed_ref_pts_dict[obj_name] = [[max_x_val, max_y_val], 0.0, bbox]  # only one single point
        if args.debug_close_loop_vis:
            [x_val, y_val] = seed_ref_pts_dict[obj_name][0]
            cv2.circle(left_image_vis, (int(x_val), int(y_val)), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(left_image_vis, (int(x_val), int(y_val)), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
            if args.task_name in ["plugpen", "reorient", "reorient_unscrew"] or \
                (args.task_name=="inserting" and obj_name=="pen") or (args.task_name=="tool_spoon" and obj_name=="spoon"):    
                cv2.circle(left_image_vis, (e_point_raw[0], e_point_raw[1]), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(left_image_vis, (e_point_raw[0], e_point_raw[1]), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
                cv2.arrowedLine(left_image_vis, (c_point_raw[0], c_point_raw[1]), (e_point_raw[0], e_point_raw[1]), 
                            (0,255,255), thickness=3, tipLength = 0.25, line_type=cv2.LINE_AA)
    if args.debug_close_loop_vis:
        cv2.polylines(left_image_vis, [pts_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.arrowedLine(left_image_vis, (20, 20), (120, 20), (0,255,255), thickness=4, line_type=cv2.LINE_AA)
        cv2.arrowedLine(left_image_vis, (20, 20), (20, 120), (0,255,255), thickness=4, line_type=cv2.LINE_AA)
        cv2.putText(left_image_vis, "X", (125, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.8, color=(0,255,255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(left_image_vis, "Y", (30, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.8, color=(0,255,255), thickness=2, lineType=cv2.LINE_AA)
    
    if args.debug_close_loop_vis:
        result_img_init = cv2.warpPerspective(left_image_seed, transform_mat_inv, (rect_l, rect_w))
        
    reprojected_seeding_pts = []
    for obj_name, obj_position in seed_ref_pts_dict.items():
        if len(obj_position) == 0: continue
        [x, y], rot_deg_init, obj_bbox = obj_position[0], obj_position[1], obj_position[2]
        temp_pt = np.dot(transform_mat_inv, np.array([x, y, 1]).T)
        new_x, new_y = temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]
        reprojected_seeding_pts.append([new_x, new_y, rot_deg_init, obj_bbox])
        if args.debug_close_loop_vis:
            cv2.circle(result_img_init, (int(new_x), int(new_y)), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(result_img_init, (int(new_x), int(new_y)), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)

    if args.debug_close_loop_vis:
        if not args.no_video_out:
            video_output_path = saved_seed_img_path.replace("_seed.jpg", f"_test{args.test_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vout = cv2.VideoWriter(video_output_path, fourcc, 5, (1452, 1080))  # Create VideoWriter object
        video_frame_index = 0
    
    ############################################################################

    
    
    
    
    
    
    
    
    
    
    
    ############################################################################
    if args.task_name == "plugpen":  # close-loop for the left-arm mand right-arm
        # original robot_arms: ['L', 'R', 'L', 'R', 'L', 'R', 'L']  # 7 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    if args.task_name == "reorient":  # close-loop for the right-arm
        # original robot_arms: ['R', 'R', 'R', 'L', 'R', 'R', 'L', 'L'] 8 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:1], eef_pose_seqs[:1]
        robot_arms, eef_pose_seqs = robot_arms[1:], eef_pose_seqs[1:]
    if args.task_name == "unscrew":  # close-loop for the left-arm
        # original robot_arms: ['L', 'L', 'L', 'R', 'R', 'R', 'R', 'R', 'R', R', 'R', 'L'] 12 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:1], eef_pose_seqs[:1]
        robot_arms, eef_pose_seqs = robot_arms[1:], eef_pose_seqs[1:]
    if args.task_name == "pouring":  # close-loop for the right-arm and left-arm
        # original robot_arms: ['R', 'R', 'L', 'L', 'R', 'L', 'L', 'L', 'L', 'R'] 10 actions
        cl_robot_arms, cl_eef_pose_seqs = ['R', 'L'], [eef_pose_seqs[0], eef_pose_seqs[2]]
        robot_arms, eef_pose_seqs = ['R', 'L'] + robot_arms[4:], eef_pose_seqs[1:2] + eef_pose_seqs[3:]

    if args.task_name == "inserting":  # close-loop for the left-arm and right-arm 
        # original robot_arms: ['L', 'L', 'L', 'R', 'R', 'L', 'R', 'R', 'L'] 9 actions
        cl_robot_arms, cl_eef_pose_seqs = ['L', 'R'], [eef_pose_seqs[0], eef_pose_seqs[3]]
        robot_arms, eef_pose_seqs = ['L', 'L'] + robot_arms[4:], eef_pose_seqs[1:3] + eef_pose_seqs[4:]
    if args.task_name == "pressing":  # close-loop for the right-arm and left-arm
        # original robot_arms: ['R', 'L, 'L', 'L', 'R', 'R', 'L'] 7 actions
        cl_robot_arms, cl_eef_pose_seqs = ['R', 'L'], [eef_pose_seqs[0], eef_pose_seqs[1]]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
        
    if args.task_name == "reorient_unscrew":  # close-loop for the right-arm
        # original robot_arms: ['R', 'R', 'R', 'R', 'R', 'R',  'L', 'L', 'L', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L'] 6+12=18 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:1], eef_pose_seqs[:1]
        robot_arms, eef_pose_seqs = robot_arms[1:], eef_pose_seqs[1:]
    if args.task_name == "unscrew_pouring":  # close-loop for the left-arm
        # original robot_arms: ['L', 'L', 'L', 'R', 'R', 'R', 'R', 'R', 'R', R', 'R', 'L',  'R', 'R', 'R', 'L', 'L', 'L', 'L', 'R'] 12+8=20 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:1], eef_pose_seqs[:1]
        robot_arms, eef_pose_seqs = robot_arms[1:], eef_pose_seqs[1:]

    if args.task_name == "tool_spoon":  # close-loop for the right-arm
        # original robot_arms: ['R', 'R', 'R', 'L', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L'] 13 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:1], eef_pose_seqs[:1]
        robot_arms, eef_pose_seqs = robot_arms[1:], eef_pose_seqs[1:]
    if args.task_name == "tool_funnel":  # close-loop for the right-arm
        # original robot_arms: ['R', 'R', 'R', 'L', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'L'] 19 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:1], eef_pose_seqs[:1]
        robot_arms, eef_pose_seqs = robot_arms[1:], eef_pose_seqs[1:]
        
    if args.no_interaction: interp_num = 2  # do not add the interpolation step
    else: interp_num = 6  # after removing start_eef_pose and end_eef_pose, we actuall add (interp_num-2) new actions
    
    ############################################################################
    step_id_record = 0  # for recording close-loop step ids of left-arm / right-arm
    cl_substep_id_record = 0  # for recording close-loop substep ids for one specific arm
    while True:
        robot_arm_temp, eef_pose_temp = cl_robot_arms[step_id_record], cl_eef_pose_seqs[step_id_record]
        left_image_test, _ = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
        if args.debug_close_loop_vis: video_frame_index += 1
        
        ##########################

        '''v1: combine (yoloworld + fast_segment_anything + segment_anything_v2)
        if "yoloworld" in vfms_conf:
            conf_yolo = vfms_conf["yoloworld"] 
            final_res_list1, im_bgr1 = open_vocabulary_detector(left_image_test[y1:y2, x1:x2], imgsz=x2-x1, conf=conf_yolo)  # bboxes w/ obj categories
        else:
            final_res_list1 = []
        if "fastsam" in vfms_conf:
            conf_fastsam = vfms_conf["fastsam"] 
            final_res_list2, im_bgr2 = fast_segment_anything(left_image_test[y1:y2, x1:x2], imgsz=x2-x1, conf=conf_fastsam)  # bboxes+masks w/o categories
        if "sam2" in vfms_conf:
            conf_sam2 = vfms_conf["sam2"] 
            final_res_list2, im_bgr2 = segment_anything_v2(left_image_test[y1:y2, x1:x2], imgsz=x2-x1, conf=conf_sam2)  # bboxes + masks w/o categories
        final_res_list = merge_yoloworld_fastsam_results(final_res_list1, final_res_list2, args.task_name)
        if args.debug_close_loop_vis:
            left_image_test_vis = left_image_test.copy()
            if args.task_name in ["plugpen", "reorient", "reorient_unscrew"]: left_image_test_vis[y1:y2, x1:x2] = im_bgr2
            if args.task_name in ["unscrew", "pouring", "unscrew_pouring"]: left_image_test_vis[y1:y2, x1:x2] = im_bgr1
        '''
        
        '''v2: only use yoloe'''
        left_image_test_sub = left_image_test[y1:y2, x1:x2]
        try:  # at least one manipulated object is detected. we update the final_res_list and im_bgr
            ref_areas = mask_pixel_count_dict if args.task_name in ["plugpen"] else None
            print(video_frame_index, "ref_areas:", ref_areas)
            final_res_list_raw, im_bgr = ov_det_seg_yoloe_slim(model, left_image_test_sub, imgsz=x2-x1, conf=conf_yoloe)
            final_res_list = post_processing_yoloe_results(final_res_list_raw, args.task_name, roi_bbox, ref_areas=ref_areas)
        except:  # specifially for the task "inserting"
            im_bgr = left_image_test_sub.copy()
            print("[Warning][Continue] No one manipulated object is detected. We still keep moving with using previous final_res_list!!!")
        final_res_list_temp = copy.deepcopy(final_res_list)
        if args.debug_close_loop_vis:
            left_image_test_vis = left_image_test.copy()
            left_image_test_vis[y1:y2, x1:x2] = plot_processed_results_vis(left_image_test_sub.copy(), final_res_list_temp, list(seed_ref_pts_dict.keys()) )
                
        for [obj_name, conf, bbox, mask] in final_res_list_temp:
            mask[:, 0] += x1; mask[:, 1] += y1  # remember add back the offsets in x / y
            if args.task_name in ["plugpen", "reorient", "reorient_unscrew"] or \
                (args.task_name=="inserting" and obj_name=="pen") or (args.task_name=="tool_spoon" and obj_name=="spoon"):
                # obj_binary_mask = polygon2mask((kfr_height, kfr_width), [mask], color=255, downsample_ratio=1)
                # rot_deg_test, pts_raw, pts_trans = compute_rotation_by_object_mask(mask, obj_binary_mask, transform_mat_inv, args.task_name)
                rot_deg_test, pts_raw, pts_trans = compute_rotation_by_image_moments(mask,
                    radius_ratio=radius_ratio, trans_mat=transform_mat_inv, task_name=args.task_name)
                [c_point_raw_t, e_point_raw_t], [c_point_trans_t, e_point_trans_t] = pts_raw, pts_trans
                test_ref_pts_dict[obj_name] = [c_point_raw_t, rot_deg_test, bbox]
            if args.task_name in ["unscrew", "pouring", "pressing", "unscrew_pouring", "tool_funnel"] or \
                (args.task_name=="inserting" and obj_name=="cup") or (args.task_name=="tool_spoon" and "bowl" in obj_name):
                max_y_val = mask[:, 1].max()
                max_x_val = mask[list(mask[:, 1]).index(max_y_val), 0]
                test_ref_pts_dict[obj_name] = [[max_x_val, max_y_val], 0.0, bbox]  # only one single point
            if args.debug_close_loop_vis:
                [x_val, y_val] = test_ref_pts_dict[obj_name][0]
                cv2.circle(left_image_test_vis, (int(x_val), int(y_val)), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(left_image_test_vis, (int(x_val), int(y_val)), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
                if args.task_name in ["plugpen", "reorient", "reorient_unscrew"] or \
                    (args.task_name=="inserting" and obj_name=="pen") or (args.task_name=="tool_spoon" and obj_name=="spoon"):    
                    cv2.circle(left_image_test_vis, (e_point_raw_t[0], e_point_raw_t[1]), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(left_image_test_vis, (e_point_raw_t[0], e_point_raw_t[1]), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.arrowedLine(left_image_test_vis, (c_point_raw_t[0], c_point_raw_t[1]), (e_point_raw_t[0], e_point_raw_t[1]), 
                                    (0,255,255), thickness=3, tipLength = 0.25, line_type=cv2.LINE_AA)
        if args.debug_close_loop_vis:
            cv2.polylines(left_image_test_vis, [pts_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
            result_img_test = cv2.warpPerspective(left_image_test, transform_mat_inv, (rect_l, rect_w))

        #############
        #if args.task_name in ["plugpen"]:  # do not conduct close-loop of these tasks
            #if video_frame_index == 1: test_ref_pts_dict_first = test_ref_pts_dict.copy()       
            #else: test_ref_pts_dict = test_ref_pts_dict_first.copy()
        #############
        
        reprojected_testing_pts, left_cls_list = [], []
        for obj_name, obj_position in test_ref_pts_dict.items():
            if len(obj_position) == 0: continue
            [x, y], rot_deg_test, obj_bbox = obj_position[0], obj_position[1], obj_position[2]
            temp_pt = np.dot(transform_mat_inv, np.array([x, y, 1]).T)
            new_x, new_y = temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]
            reprojected_testing_pts.append([new_x, new_y, rot_deg_test, obj_bbox]); left_cls_list.append(obj_name)
            if args.debug_close_loop_vis:
                cv2.circle(result_img_test, (int(new_x), int(new_y)), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(result_img_test, (int(new_x), int(new_y)), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)

        if args.debug_close_loop_vis:
            left_imgs = np.vstack((left_image_vis, left_image_test_vis))   # shape (1080, 960, 3)
            cv2.putText(left_imgs, f'Frame Index {video_frame_index}', (5, kfr_height+35),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(left_imgs, f'seeding (static)', (int(0.75*kfr_width), 35),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(left_imgs, f'testing (dynamic)', (int(0.72*kfr_width), kfr_height+35),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
            result_imgs = np.vstack((result_img_init, result_img_test))  # shape (rect_w*2, rect_l, 3) --> (1350, 616, 3)
            
            for idx, (pt_seeding, pt_testing) in enumerate(zip(reprojected_seeding_pts, reprojected_testing_pts)):
                delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
                print(video_frame_index, idx, "\n pt_seeding:", pt_seeding, "\n pt_testing:", pt_testing)
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
            cv2.putText(result_imgs, f'Top-view Reprojection', (5, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.9, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
            final_imgs = np.hstack((left_imgs, result_imgs))  # shape (1080, 1452, 3)            
            
            cv2.namedWindow("MyDebugWindow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("MyDebugWindow", 1452, 1080)
            cv2.imshow("MyDebugWindow", final_imgs)
            print("[*****Finished*****]", video_frame_index)
            if not args.no_video_out:
                for _ in range(5): vout.write(final_imgs)

            if args.no_interaction:
                cv2.waitKey(0)  # wait until we close the plotted window (press Esc to continue)
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            if video_frame_index == 1: cv2.imwrite(saved_seed_img_path.replace("_seed.jpg", f"_test{args.test_id}_frame1.jpg"), final_imgs)
        ##########################
        try:
            pt_seeding, pt_testing = reprojected_seeding_pts[step_id_record], reprojected_testing_pts[step_id_record]
        except:
            print(reprojected_seeding_pts, reprojected_testing_pts); sys.exit()
        delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
        [p_delta_x, p_delta_y, prev_bbox] = prev_delta_xy_list[step_id_record]
        
        rot_deg_init, rot_deg_test = pt_seeding[2], pt_testing[2]  # for tasks "plugpen", "reorient", "inserting", "reorient_unscrew", "tool_spoon"
        obj_bbox_init, obj_bbox_test = pt_seeding[3], pt_testing[3]  # for tasks with frequent occlusion between the robot-arm and targeted object
        if prev_bbox is None: prev_bbox = obj_bbox_init
        
        if abs(delta_x - p_delta_x) > 10 or abs(delta_y - p_delta_y) > 10:  # we think this object may be moved
            inside_ratio = calInsideRectIOU(prev_bbox, obj_bbox_test)  # between 0 and 1
            if inside_ratio > 0.90: # obj_bbox_test is most likely inside the prev_bbox
                print("[Warning][Continue] targeted object is most likely occluced by robot arm now!!! Using prev_bbox and do not update prev_delta_xy_list.")
            else: # No occlusion happened. we need to start a new close-loop step
                cl_substep_id_record = 0; prev_delta_xy_list[step_id_record] = [delta_x, delta_y, obj_bbox_test]

        cl_substep_id_record += 1
        
        # get end_eef_pose
        cur_end_eef_pose = eef_pose_temp.copy()
        if robot_arm_temp == "L": cur_end_eef_pose[0] -= (delta_y / 1000); cur_end_eef_pose[1] -= (delta_x / 1000)
        if robot_arm_temp == "R": cur_end_eef_pose[0] += (delta_y / 1000); cur_end_eef_pose[1] += (delta_x / 1000)
        if args.task_name in ["plugpen", "reorient", "inserting", "reorient_unscrew", "tool_spoon"]:
            delta_rot_z = rot_deg_test - rot_deg_init
            if robot_arm_temp == "L": cur_end_eef_pose[5] = cur_pose_deg_L[5] + delta_rot_z
            if robot_arm_temp == "R": cur_end_eef_pose[5] = cur_pose_deg_R[5] + delta_rot_z
        
        # get start_eef_pose
        if robot_arm_temp == "L": robotBase = robotBase_L; m_gripper = m_gripper_L; g_port = cfg_dict["arm1"]["gripper_port"]
        if robot_arm_temp == "R": robotBase = robotBase_R; m_gripper = m_gripper_R; g_port = cfg_dict["arm2"]["gripper_port"]
        cur_start_eef_pose, _ = robotBase.get_current_pose()
        if args.task_name in ["unscrew", "pouring", "inserting", "pressing", "unscrew_pouring", "tool_funnel"]:
            if cl_substep_id_record != interp_num-1:
                cur_start_eef_pose[1] += 0.040  # move robot back slightly to avoid hitting objects
                cur_start_eef_pose[2] += 0.040  # move robot higher slightly to avoid hitting objects 
        if cl_substep_id_record == 1 and cur_start_eef_pose[1] < cur_end_eef_pose[1]:  # object is moved backward (not forward)
            cur_start_eef_pose[1] = cur_end_eef_pose[1] + 0.100  # move robot back slightly to avoid hitting objects
        
        
        ##########################
        # pose_list = [cur_start_eef_pose, cur_end_eef_pose]
        # traj_poses = generate_interpolation_traj(pose_list, interp_num, is_6dof=True)  # length interp_num
        traj_poses = interpolate_se3_bspline_startend_poses(cur_start_eef_pose[:6], cur_end_eef_pose[:6], num_steps=interp_num)
        target_6dof_vec = traj_poses[cl_substep_id_record]
        ##########################
            
        
        if args.task_name in ["plugpen", "reorient", "reorient_unscrew", "tool_spoon"]:
            if cl_substep_id_record != interp_num-1:  # the object is lying on the table, with a small height
                target_6dof_vec[2] += 0.080  # lifting-up robot in middle steps to avoid hitting objects
        
        robotBase.move_to_target_in_cartesian(target_6dof_vec)
        print("[***Close-Loop***]", robot_arm_temp, step_id_record, cl_substep_id_record, robotBase.get_current_pose()[0])
        
        if cl_substep_id_record == interp_num-1:  # this close-loop step is finished !!!
            m_gripper = dh_modbus_gripper(dh_device(g_port), gripper_ID)
            m_gripper.open(baudrate)
            if eef_pose_temp[-1] == 0: m_gripper.SetTargetPosition(5)  # close gripper, range is (0, 1000)
            if eef_pose_temp[-1] == 1: m_gripper.SetTargetPosition(995)  # close gripper, range is (0, 1000)
            g_state = 0
            while(g_state == 0):
                g_state = m_gripper.GetGripState(); time.sleep(0.1)
            
            if args.task_name == "unscrew":
                eef_pose_seqs[0][:2] = traj_poses[-1][:2]  # modify the x and y lifting-up position of left-arm
            if args.task_name == "unscrew_pouring":
                eef_pose_seqs[0][:2] = traj_poses[-1][:2]  # modify the x and y lifting-up position of left-arm
                # the mugcup for the right-arm may also be changed !!!
                pt_seeding, pt_testing = reprojected_seeding_pts[1], reprojected_testing_pts[1]
                delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
                eef_pose_seqs[9][0] += (delta_y / 1000)  # modify the x position of right-arm when grasping mugcup
                eef_pose_seqs[9][1] += (delta_x / 1000)  # modify the y position of right-arm when grasping mugcup
                eef_pose_seqs[11][0] += (delta_y / 1000)  # modify the x position of right-arm when lifting-up mugcup
                eef_pose_seqs[11][1] += (delta_x / 1000)  # modify the y position of right-arm when lifting-up mugcup
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
            if args.task_name == "tool_spoon":
                pt_seeding, pt_testing = reprojected_seeding_pts[1], reprojected_testing_pts[1]  # for the object bowlL
                delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
                eef_pose_seqs[6][0] -= (delta_y / 1000)  # modify the x position of left-arm when firstly approaching bowlL
                eef_pose_seqs[6][1] -= (delta_x / 1000)  # modify the y position of left-arm when firstly approaching bowlL
                eef_pose_seqs[7][0] -= (delta_y / 1000)  # modify the x position of left-arm when secondly approaching bowlL
                eef_pose_seqs[7][1] -= (delta_x / 1000)  # modify the y position of left-arm when secondly approaching bowlL
                eef_pose_seqs[10][0] -= (delta_y / 1000)  # modify the x position of left-arm when thirdly approaching bowlL
                eef_pose_seqs[10][1] -= (delta_x / 1000)  # modify the y position of left-arm when thirdly approaching bowlL
                eef_pose_seqs[11][0] -= (delta_y / 1000)  # modify the x position of left-arm when placing spoon into bowlL
                eef_pose_seqs[11][1] -= (delta_x / 1000)  # modify the y position of left-arm when placing spoon into bowlL
                pt_seeding, pt_testing = reprojected_seeding_pts[2], reprojected_testing_pts[2]  # for the object bowlS
                delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
                eef_pose_seqs[8][0] -= (delta_y / 1000)  # modify the x position of left-arm when approaching bowlS
                eef_pose_seqs[8][1] -= (delta_x / 1000)  # modify the y position of left-arm when approaching bowlS
                eef_pose_seqs[9][0] -= (delta_y / 1000)  # modify the x position of left-arm when pouring to bowlS
                eef_pose_seqs[9][1] -= (delta_x / 1000)  # modify the y position of left-arm when pouring to bowlS
            if args.task_name == "tool_funnel":
                pt_seeding, pt_testing = reprojected_seeding_pts[0], reprojected_testing_pts[0]  # for the object funnel
                delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
                eef_pose_seqs[0][0] += (delta_y / 1000)  # modify the x position of right-arm when lifting-up funnel
                eef_pose_seqs[0][1] += (delta_x / 1000)  # modify the y position of right-arm when lifting-up funnel
                pt_seeding, pt_testing = reprojected_seeding_pts[1], reprojected_testing_pts[1]  # for the object bottle
                delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
                eef_pose_seqs[5][0] -= (delta_y / 1000)  # modify the x position of left-arm when firstly approaching bottle
                eef_pose_seqs[5][1] -= (delta_x / 1000)  # modify the y position of left-arm when firstly approaching bottle
                eef_pose_seqs[6][0] -= (delta_y / 1000)  # modify the x position of left-arm when secondly inserting funnel into bottle
                eef_pose_seqs[6][1] -= (delta_x / 1000)  # modify the y position of left-arm when secondly inserting funnel into bottle
                eef_pose_seqs[7][0] -= (delta_y / 1000)  # modify the x position of left-arm when thirdly reorient the gripper
                eef_pose_seqs[7][1] -= (delta_x / 1000)  # modify the y position of left-arm when thirdly reorient the gripper
                eef_pose_seqs[8][0] -= (delta_y / 1000)  # modify the x position of left-arm when fourthly grasping bottle+funnel
                eef_pose_seqs[8][1] -= (delta_x / 1000)  # modify the y position of left-arm when fourthly grasping bottle+funnel
                eef_pose_seqs[9][0] -= (delta_y / 1000)  # modify the x position of left-arm when fifthly lifting-up bottle+funnel
                eef_pose_seqs[9][1] -= (delta_x / 1000)  # modify the y position of left-arm when fifthly lifting-up  bottle+funnel
                pt_seeding, pt_testing = reprojected_seeding_pts[2], reprojected_testing_pts[2]  # for the object cup
                delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
                eef_pose_seqs[4][0] += (delta_y * 0.5 / 1000)  # modify the x position of right-arm when firstly approaching cup
                eef_pose_seqs[4][1] += (delta_x * 0.5 / 1000)  # modify the y position of right-arm when firstly approaching cup
                eef_pose_seqs[10][0] += (delta_y / 1000)  # modify the x position of right-arm when secondly grasping cup
                eef_pose_seqs[10][1] += (delta_x / 1000)  # modify the y position of right-arm when secondly grasping cup
                eef_pose_seqs[11][0] += (delta_y / 1000)  # modify the x position of right-arm when thirdly lifting-up cup
                eef_pose_seqs[11][1] += (delta_x / 1000)  # modify the y position of right-arm when thirdly lifting-up cup
                
            step_id_record += 1
            cl_substep_id_record = 0
            if step_id_record == len(cl_robot_arms):  # stop the close-loop step or steps
                print("[***Close-Loop***] finished!!!")
                break
        ##########################
    
    if args.debug_close_loop_vis:
        cv2.destroyAllWindows() 
    ############################################################################   
        
        
        
        
        
        
        
        
        
        
        
        
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
        if args.debug_close_loop_vis:
            for frames_loop_idx in range(5):  # we collect 5 images (1 second) for each keyframe
                left_image_test, _ = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
                video_frame_index += 1
                cv2.polylines(left_image_test, [pts_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
                left_image_test = cv2.resize(left_image_test[:, 100:826], (1452, 1080)) # (960, 540) -> (726, 540) -> (1452, 1080) 
                cv2.putText(left_image_test, f'Frame Index {video_frame_index}', (5, kfr_height+35),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
                if not args.no_video_out: vout.write(left_image_test)
        #######################################
        
    ############################################################################
    if given_execute_arm_name == "arm1":
        robotBase_L.move_to_target_in_cartesian(cur_pose_deg_L)
    if given_execute_arm_name == "arm2":
        robotBase_R.move_to_target_in_cartesian(cur_pose_deg_R)
    if given_execute_arm_name is None:
        robotBase_L.move_to_target_in_cartesian(cur_pose_deg_L)
        robotBase_R.move_to_target_in_cartesian(cur_pose_deg_R)
        
    if args.debug_close_loop_vis:
        for frames_loop_idx in range(5):  # we collect 5 images for each keyframe
            left_image_test, _ = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
            video_frame_index += 1
            cv2.polylines(left_image_test, [pts_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
            left_image_test = cv2.resize(left_image_test[:, 100:826], (1452, 1080)) # (960, 540) -> (726, 540) -> (1452, 1080) 
            cv2.putText(left_image_test, f'Frame Index {video_frame_index}', (5, kfr_height+35),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
            if not args.no_video_out: vout.write(left_image_test)     
        if not args.no_video_out: vout.release()  # Release everything ()

    ############################################################################
    