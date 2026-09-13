
import os
import sys
import cv2
import time
import json
import threading

import kingfisher
sys.path.insert(0, os.getcwd())

from robots.demo_xMateCR7 import rokaeRobotBase
# from gripper.demo_jodell_rg75 import JodellRG75
from gripper.demo_jodell_rg75_v2 import JodellRG75

from src.config import cfg_dict_init as cfg_dict
from src.config import get_eef_keypose_dict
from src.states_v1 import get_llm_instance


##################################################################################################################################  
def parsing_eef_keypose_dict(eef_keypose_dict):
    eef_pose_seqs_L, eef_pose_seqs_R = eef_keypose_dict["L"], eef_keypose_dict["R"]
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
    print(max_step_id, "step_ids:", steps_list_L+steps_list_R, "\n robot_arms:", len(robot_arms), robot_arms)

    action_para_list = [robot_init_pose_L, robot_init_gripper_L, robot_init_pose_R, robot_init_gripper_R]
    return action_para_list, robot_arms, eef_pose_seqs

def drive_single_arm_to_run(target_pose_list, arm_type, robot_L, robot_R):
    if arm_type == "L": robot_L.move_by_trajectory(target_pose_list)
    if arm_type == "R": robot_R.move_by_trajectory(target_pose_list)

##################################################################################################################################  
def init_parameters_robots_grippers(args, is_reinit=False):

    ###################################################################################
    if args.llm_type != -1:
        llm_instance = get_llm_instance(args.llm_type, 0.0)  # 0.0 is temperature for LLM
    else:
        llm_instance = None  # LLM is not used
    ###################################################################################
    os.makedirs(args.save_imgs_dir, exist_ok=True)
    task_folder_dir = os.path.join(args.save_imgs_dir, args.task_name)  # "pouring" or "unscrew"
    os.makedirs(task_folder_dir, exist_ok=True)

    ################################
    if args.task_name == "pouring":  # first grasp cup (right), then grasp bottle (left)
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.bottle_id, args.mugcup_id])
        saved_seed_img_path = os.path.join(task_folder_dir, 
            f"bottle{str(args.bottle_id).zfill(2)}-mugcup{str(args.mugcup_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"cup":[], "bottle":[]}, {"cup":[], "bottle":[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None], [0, 0, None]]; detseg_prompts = "cup,bottle"; supp_prompts = None

    if args.task_name == "unscrew":  # only grasp bottle (left), then unscrew lid (right)
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.bottle_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"bottle{str(args.bottle_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"bottle":[]}, {"bottle":[]}
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = "bottle"; supp_prompts = None

    if args.task_name == "reorient":  # only grasp bottle (left / right)
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.bottle_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"bottle{str(args.bottle_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"bottle":[]}, {"bottle":[]}
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = "bottle"; supp_prompts = "bottle cap"

    if args.task_name == "grasping":  # only manipulate cup (left / right)
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.mugcup_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"mugcup{str(args.mugcup_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"cup":[]}, {"cup":[]}
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = "cup"; supp_prompts = None

    if args.task_name == "flatting":  # only grasp bottle (left / right)
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.bottle_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"bottle{str(args.bottle_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"bottle":[]}, {"bottle":[]}
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = "bottle"; supp_prompts = None

    if args.task_name == "flipping":  # only manipulate cup (left / right)
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.mugcup_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"mugcup{str(args.mugcup_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"cup":[]}, {"cup":[]}
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = "cup"; supp_prompts = None
    ################################

    ################################
    if args.task_name == "grasping_rectbox":  # only manipulate rectbox (left / right)
        cls_name_list = ["box", "box", "box"]
        cls_name = cls_name_list[args.rectbox_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.rectbox_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"rectbox{str(args.rectbox_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None
    if args.task_name == "grasping_cirbowl":  # only manipulate cirbowl (left / right)
        cls_name_list = ["white bowl", "green bowl", "transparent bowl", "gray bowl"]
        cls_name = cls_name_list[args.cirbowl_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.cirbowl_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"cirbowl{str(args.cirbowl_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None
    if args.task_name == "grasping_basket":  # only manipulate basket (bimanual skill)
        cls_name_list = ["pink basket", "green basket", "blue basket"]
        cls_name = cls_name_list[args.basket_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.basket_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"basket{str(args.basket_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None
    if args.task_name == "grasping_holder":  # only manipulate holder (left / right)
        cls_name_list = ["black cup", "black cup"]
        cls_name = cls_name_list[args.holder_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.holder_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"holder{str(args.holder_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None
    if args.task_name == "grasping_pencup":  # only manipulate holder (left / right)
        cls_name_list = ["black cup", "black cup"]
        cls_name = cls_name_list[args.pencup_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.pencup_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"pencup{str(args.pencup_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None
    ################################
    
    ################################
    if args.task_name == "unscrew-pouring":  # grasp bottle (left), unscrew lid (right), then grasp cup (right)
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.bottle_id, args.mugcup_id])
        saved_seed_img_path = os.path.join(task_folder_dir, 
            f"bottle{str(args.bottle_id).zfill(2)}-mugcup{str(args.mugcup_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"bottle":[], "cup":[]}, {"bottle":[], "cup":[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None], [0, 0, None]]; detseg_prompts = "bottle,cup"; supp_prompts = None
    if args.task_name == "flatting-reorient":  # grasp bottle (left/right), then flat and reorient bottle (left/right)
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.bottle_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"bottle{str(args.bottle_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"bottle":[]}, {"bottle":[]}
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = "bottle"; supp_prompts = None
    ################################

    ################################
    if args.task_name == "inserting":  # first grasp cup (right), then grasp pen (left)
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.marker_id, args.ordcup_id])
        saved_seed_img_path = os.path.join(task_folder_dir, 
            f"marker{str(args.marker_id).zfill(2)}-ordcup{str(args.ordcup_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"cup":[], "pen":[]}, {"cup":[], "pen":[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None], [0, 0, None]]; detseg_prompts = "cup,pen"; supp_prompts = None
    if args.task_name == "plugpen":  # first grasp pen cap (right), then grasp pen body (left) 
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.pencap_id, args.marker_id])
        saved_seed_img_path = os.path.join(task_folder_dir, 
            f"pencap{str(args.pencap_id).zfill(2)}-marker{str(args.marker_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"pen cap":[], "marker pen":[]}, {"pen cap":[], "marker pen":[]}
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None], [0, 0, None]]; detseg_prompts = "pen cap,marker pen"; supp_prompts = None
    if args.task_name == "handover":  # first grasp spoon / shovel
        cls_name_list = ["metal spoon", "plastic spoon", "plastic shovel", "metal shovel"]
        cls_name = cls_name_list[args.shovel_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.shovel_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"shovel{str(args.shovel_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None
        
    if args.task_name == "ppspoon":  # grasp spoon (left /right)
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.spoon_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"spoon{str(args.spoon_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"spoon":[]}, {"spoon":[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None], [0, 0, None]]; detseg_prompts = "spoon"; supp_prompts = None
    if args.task_name == "ppfork":  # grasp fork (left /right)
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.fork_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"fork{str(args.fork_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"fork":[]}, {"fork":[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None], [0, 0, None]]; detseg_prompts = "fork"; supp_prompts = None
    if args.task_name == "ppspoon-ppfork":  # grasp spoon / fork (left /right)
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.spoon_id, args.fork_id])
        saved_seed_img_path = os.path.join(task_folder_dir,
            f"spoon{str(args.spoon_id).zfill(2)}-fork{str(args.fork_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"spoon":[], "fork":[]}, {"spoon":[], "fork":[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None], [0, 0, None]]; detseg_prompts = "spoon,fork"; supp_prompts = None
    if args.task_name == "ppfork-ppspoon":  # grasp fork / spoon  (left /right)
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.spoon_id, args.fork_id])
        saved_seed_img_path = os.path.join(task_folder_dir,
            f"fork{str(args.fork_id).zfill(2)}-spoon{str(args.spoon_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {"fork":[], "spoon":[]}, {"fork":[], "spoon":[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None], [0, 0, None]]; detseg_prompts = "fork,spoon"; supp_prompts = None
    ################################

    ################################
    if args.task_name == "pivoting":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["white bowl", "green bowl", "transparent bowl", "gray bowl"]
        cls_name = cls_name_list[args.cirbowl_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.cirbowl_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"cirbowl{str(args.cirbowl_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None

    if args.task_name == "wrapping":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["pink basket", "green basket", "blue basket"]
        cls_name = cls_name_list[args.basket_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.basket_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"basket{str(args.basket_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None
    
    if args.task_name == "pivoting_rectbox":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["box", "box", "box"]
        cls_name = cls_name_list[args.rectbox_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.rectbox_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"rectbox{str(args.rectbox_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None
    if args.task_name == "pivoting_cirbowl":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["white bowl", "green bowl", "transparent bowl", "gray bowl"]
        cls_name = cls_name_list[args.cirbowl_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.cirbowl_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"cirbowl{str(args.cirbowl_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None

    if args.task_name == "flipping_basket":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["pink basket", "green basket", "blue basket"]
        cls_name = cls_name_list[args.basket_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.basket_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"basket{str(args.basket_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None
    if args.task_name == "flipping_block":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["green block", "purple block", "orange block"]
        cls_name = cls_name_list[args.block_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.block_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"block{str(args.block_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None

    if args.task_name == "pivoting_bigjar":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["plastic jar", "plastic jar", "plastic jar"]
        cls_name = cls_name_list[args.bigjar_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.bigjar_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"bigjar{str(args.bigjar_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None
    if args.task_name == "pivoting_block":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["green block", "purple block", "orange block"]
        cls_name = cls_name_list[args.block_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.block_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"block{str(args.block_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None

    if args.task_name == "toppling_holder":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["black cup", "black cup"]
        cls_name = cls_name_list[args.holder_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.holder_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"holder{str(args.holder_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None
    if args.task_name == "toppling_bigjar":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["plastic jar", "plastic jar"]
        cls_name = cls_name_list[args.bigjar_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.bigjar_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"bigjar{str(args.bigjar_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None 
        
    if args.task_name == "bilifting_bigjar":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["plastic jar", "plastic jar"]
        cls_name = cls_name_list[args.bigjar_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.bigjar_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"bigjar{str(args.bigjar_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None
    if args.task_name == "bilifting_block":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["green block", "purple block", "orange block"]
        cls_name = cls_name_list[args.block_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.block_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"block{str(args.block_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = cls_name; supp_prompts = None
    ################################

    ################################
    if args.task_name == "penbagzip":  # first grasp penbag (left), then grasp pen (right)
        cls_name_list1 = ["gray bag", "gray bag", "gray bag"]
        cls_name1 = cls_name_list1[args.penbag_id - 1]
        cls_name_list2 = ["pen", "pen", "pen"]
        cls_name2 = cls_name_list2[args.marker_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.penbag_id, args.marker_id])
        saved_seed_img_path = os.path.join(task_folder_dir, 
            f"penbag{str(args.penbag_id).zfill(2)}-marker{str(args.marker_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name1:[], cls_name2:[]}, {cls_name1:[], cls_name2:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None], [0, 0, None]]; detseg_prompts = f"{cls_name1},{cls_name2}"; supp_prompts = None

    if args.task_name == "foldtowel":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["blue towel", "towel", "blue towel", "towel"]
        cls_name = cls_name_list[args.towel_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.towel_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"towel{str(args.towel_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = f"{cls_name}"; supp_prompts = None
    if args.task_name == "foldpants":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["pants", "pants", "pants", "pants"]
        cls_name = cls_name_list[args.pants_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.pants_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"pants{str(args.pants_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = f"{cls_name}"; supp_prompts = None
    if args.task_name == "foldshirt":  # first contact left part (left), then contact right part (right)
        cls_name_list = ["T-shirt", "T-shirt", "T-shirt", "T-shirt"]
        cls_name = cls_name_list[args.shirt_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.shirt_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"shirt{str(args.shirt_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = f"{cls_name}"; supp_prompts = None
    
    if args.task_name == "coilcable":  # always using the right arm to operate the first grasping (right)
        cls_name_list = ["white cable", "black cable", "red cable"]
        cls_name = cls_name_list[args.cable_id - 1]
        eef_keypose_dict = get_eef_keypose_dict(args.task_name, object_ids=[args.cable_id])
        saved_seed_img_path = os.path.join(task_folder_dir, f"cable{str(args.cable_id).zfill(2)}_seed.jpg")
        seed_ref_pts_dict, test_ref_pts_dict = {cls_name:[]}, {cls_name:[]}  
        prev_delta_rot = 0; prev_delta_xy_list = [[0, 0, None]]; detseg_prompts = f"{cls_name}"; supp_prompts = None
        
    ################################
    
    torque_value = 32  # 0-->255 soft-->hard; the gripper should grasp the object loosely 
    task_para_list = [saved_seed_img_path, seed_ref_pts_dict, test_ref_pts_dict, prev_delta_rot, prev_delta_xy_list, detseg_prompts, supp_prompts]

    ###################################################################################
    # kingfisher-6000 camera initilization
    print("\n")
    c = kingfisher.connect(cfg_dict["kfr_ip"])   # connect the camera
    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']  # 540, 960
    kingfisher.SetAUTO_EXPOSURE()
   
    if not os.path.exists(saved_seed_img_path):
        left_image_seed, _ = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
        cv2.imwrite(saved_seed_img_path, left_image_seed); os._exit(0); sys.exit()
    #else:
    #    left_image_seed = cv2.imread(saved_seed_img_path)

    cam_para_list = [kingfisher, kfr_height, kfr_width]

    ############################################################################ 
    action_para_list, robot_arms, eef_pose_seqs = parsing_eef_keypose_dict(eef_keypose_dict)
    [robot_init_pose_L, robot_init_gripper_L, robot_init_pose_R, robot_init_gripper_R] = action_para_list
    ###################################################################################

    ################################
    if args.task_name == "pouring":  # (for a left bottle and a right mugcup) close-loop for the right-arm and left-arm (bimanual)
        # original robot_arms: ['R', 'R', 'L', 'L', 'R', 'L', 'L', 'L', 'L', 'R'] 10 actions
        cl_robot_arms, cl_eef_pose_seqs = ['R', 'L'], [eef_pose_seqs[0], eef_pose_seqs[2]]
        robot_arms, eef_pose_seqs = ['R', 'L'] + robot_arms[4:], eef_pose_seqs[1:2] + eef_pose_seqs[3:]
    if args.task_name == "unscrew":  # (for a upright bottle with cap) close-loop for the left-arm (bimanual)
        # original robot_arms: ['L', 'L', 'R', 'R', 'R', 'R', 'R', 'R', 'R', R', 'R',  R', 'L'] 13 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:1], eef_pose_seqs[:1]
        robot_arms, eef_pose_seqs = robot_arms[1:], eef_pose_seqs[1:]
    if args.task_name == "reorient":  # (for a lying down bottle) close-loop for the left-arm or right-arm
        # original robot_arms: ['L', 'L', 'L', 'L', 'L', 'L'] / ['R', 'R', 'R', 'R', 'R', 'R'] 6 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    if args.task_name == "grasping":  # (for a lying down mugcup) close-loop for the left-arm or right-arm
        # original robot_arms: ['L', 'L', 'L', 'L', 'L'] / ['R', 'R', 'R', 'R', 'R'] 5 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    if args.task_name == "flatting":  # (for an inverted bottle) close-loop for the left-arm or right-arm
        # original robot_arms: ['L', 'L', 'L', 'L', 'L'] / ['R', 'R', 'R', 'R', 'R'] 5 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    if args.task_name == "flipping":  # (for an inverted mugcup) close-loop for the left-arm or right-arm
        # original robot_arms: ['L', 'L', 'L', 'L', 'L', 'L'] / ['R', 'R', 'R', 'R', 'R', 'R'] 6 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    ################################

    ################################
    if args.task_name == "grasping_rectbox":  # (for a standing rectbox) close-loop for the left-arm or right-arm
        # original robot_arms: ['L', 'L', 'L', 'L', 'L', 'L'] / ['R', 'R', 'R', 'R', 'R', 'R'] 6 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    if args.task_name == "grasping_cirbowl":  # (for an upright cirbowl) close-loop for the left-arm or right-arm
        # original robot_arms: ['L', 'L', 'L', 'L', 'L', 'L'] / ['R', 'R', 'R', 'R', 'R', 'R'] 6 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    if args.task_name == "grasping_basket":  # (for an upright basket) close-loop for the left-arm and right-arm (bimanual)
        # original robot_arms: ['L', 'R', 'L', 'R', 'L', 'R', 'L', 'R', 'L', 'R', 'L', 'R'] 6 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    if args.task_name == "grasping_holder":  # (for a lying down pen holder) close-loop for the left-arm or right-arm
        # original robot_arms: ['L', 'L', 'L', 'L', 'L', 'L'] / ['R', 'R', 'R', 'R', 'R', 'R'] 6 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    if args.task_name == "grasping_pencup":  # (for an upright pen holder) close-loop for the left-arm or right-arm
        # original robot_arms: ['L', 'L', 'L', 'L', 'L', 'L'] / ['R', 'R', 'R', 'R', 'R', 'R'] 6 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    ################################

    ################################
    if args.task_name == "unscrew-pouring":  # close-loop for the left-arm and open-loop for the right-arm (bimanual + long-horizon)
        # original robot_arms: ['L', 'L', 'R', 'R', 'R', 'R', 'R', 'R', 'R', R', 'R',  R', 
        #   'R', 'L', 'R', 'L', 'L', 'L', 'L', 'R' ] 12 + 8 actions (unscrew + pouring)
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:1], eef_pose_seqs[:1]
        robot_arms, eef_pose_seqs = robot_arms[1:], eef_pose_seqs[1:]
    if args.task_name == "flatting-reorient":  # only close-loop for the left-arm / right-arm (bimanual + long-horizon)
        # original robot_arms: 5 + 1 + 6 actions (flatting + reorient)
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    ################################

    ################################
    if args.task_name == "inserting":  # close-loop for the right-arm and left-arm (bimanual)
        # original robot_arms: ['R', 'L', 'R', 'L', 'L', 'L', 'R', 'L'] 8 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    if args.task_name == "plugpen":  # close-loop for the right-arm and left-arm (bimanual)
        # original robot_arms: ['R', 'L', 'R', 'L', 'L', 'R', 'L', 'R'] 8 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    if args.task_name == "handover":  # close-loop for the right-arm
        # original robot_arms: ['R', 'R', 'R', 'R', 'L', 'L', 'R', 'R', 'L', 'L'] 10 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:1], eef_pose_seqs[:1]
        robot_arms, eef_pose_seqs = robot_arms[1:], eef_pose_seqs[1:] 
    if args.task_name == "ppspoon" or args.task_name == "ppfork" or \
        args.task_name == "ppspoon-ppfork" or args.task_name == "ppfork-ppspoon": # close-loop for the left-arm or right-arm
        # original robot_arms: ['L', 'L', 'L', 'L', 'L'] / ['R', 'R', 'R', 'R', 'R'] 5 actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    ################################

    ################################
    # (non-prehensile skills) close-loop for the left-arm and right-arm 
    if "pivoting" in args.task_name or "wrapping" in args.task_name or "toppling" in args.task_name or "bilifting" in args.task_name or \
        args.task_name == "flipping_basket" or args.task_name == "flipping_block":
        # original robot_arms: ['L', 'R', 'L', 'R', N*'R'] 4 + N actions
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    ################################

    ################################
    if args.task_name in ["penbagzip", "foldtowel", "foldpants", "foldshirt"]:  # close-loop for the left-arm and right-arm (bimanual)
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:2], eef_pose_seqs[:2]
        robot_arms, eef_pose_seqs = robot_arms[2:], eef_pose_seqs[2:]
    if args.task_name in ["coilcable", "coilrope", "coilbelt"]:  # close-loop for the right-arm (now only support the single-arm grasp)
        cl_robot_arms, cl_eef_pose_seqs = robot_arms[:1], eef_pose_seqs[:1]
        robot_arms, eef_pose_seqs = robot_arms[1:], eef_pose_seqs[1:] 
    ################################


    interp_num = 6  # after removing start_eef_pose and end_eef_pose, we actuall add (interp_num-2) new actions
    loop_para_list = [cl_robot_arms, cl_eef_pose_seqs, robot_arms, eef_pose_seqs, interp_num]

    ###################################################################################
    if not is_reinit:
        # grippers initilization
        print("\n")    
        gripper_L = JodellRG75(give_torque=torque_value)  # 1 close/ 0 open gripper
        gripper_L.connect(cfg_dict["arm1"]["gripper_port"], 9)  # for the left-arm, set tool_id=0
        if robot_init_gripper_L == 0 or robot_init_gripper_L == 1:
            # gripper_L.switch(robot_init_gripper_L, True); print("[gripper init][L] finished.")  # sudo chmod 777 /dev/ttyUSB0
            gripper_L.set_pos(int(255*robot_init_gripper_L)); print("[gripper init][L] finished.")  # sudo chmod 777 /dev/ttyUSB0
        else:
            # gripper_L.switch(0, True, int(255*robot_init_gripper_L)); robot_init_gripper_L = -1  # do not change it anymore
            gripper_L.set_pos(255 - int(255*robot_init_gripper_L)); robot_init_gripper_L = -1  # do not change it anymore
            action_para_list = [robot_init_pose_L, robot_init_gripper_L, robot_init_pose_R, robot_init_gripper_R]  # update gripper

        gripper_R = JodellRG75(give_torque=torque_value)  # 1 close/ 0 open gripper
        gripper_R.connect(cfg_dict["arm2"]["gripper_port"], 9)  # for the right-arm, set tool_id=0
        if robot_init_gripper_R == 0 or robot_init_gripper_R == 1:
            # gripper_R.switch(robot_init_gripper_R, True); print("[gripper init][R] finished.")  # sudo chmod 777 /dev/ttyUSB1
            gripper_R.set_pos(int(255*robot_init_gripper_R)); print("[gripper init][R] finished.")  # sudo chmod 777 /dev/ttyUSB1
        else:
            # gripper_R.switch(0, True, int(255*robot_init_gripper_R)); robot_init_gripper_R = -1  # do not change it anymore
            gripper_R.set_pos(255 - int(255*robot_init_gripper_R)); robot_init_gripper_R = -1  # do not change it anymore
            action_para_list = [robot_init_pose_L, robot_init_gripper_L, robot_init_pose_R, robot_init_gripper_R]  # update gripper

        ###################################################################################
        # robot arms initilization
        robot_L = rokaeRobotBase(cfg_dict["arm1"]["robot_ip_add"])  # for the left arm
        robot_L.moving_pre_op(); # robot_L.move_to_a_waypoint(robot_init_pose_L)
        robot_L.set_motion_speed_ratio(args.robot_speed)  # 20% speed
        print("[robot init][L] Current robot pose (robot view)", robot_init_pose_L)

        robot_R = rokaeRobotBase(cfg_dict["arm2"]["robot_ip_add"])  # for the right arm
        robot_R.moving_pre_op(); # robot_R.move_to_a_waypoint(robot_init_pose_R)
        robot_R.set_motion_speed_ratio(args.robot_speed)  # 20% speed
        print("[robot init][R] Current robot pose (robot view)", robot_init_pose_R)

        thread_list = []  # for saving multiple threadings for dual-arm coordination
        thread = threading.Thread(target=drive_single_arm_to_run, args=([robot_init_pose_L], "L", robot_L, robot_R, ))
        thread_list.append(thread); thread.start()
        thread = threading.Thread(target=drive_single_arm_to_run, args=([robot_init_pose_R], "R", robot_L, robot_R, ))
        thread_list.append(thread); thread.start()
        for thread in thread_list: thread.join()  # wait all threading to finish

        time.sleep(1)
        #sys.exit()

        robots_grippers_list = [robot_L, robot_R, gripper_L, gripper_R]
        
        return task_para_list, cam_para_list, action_para_list, loop_para_list, llm_instance, robots_grippers_list
    ###################################################################################

    return task_para_list, cam_para_list, action_para_list, loop_para_list
    
    
##################################################################################################################################  
def init_parameters_robots_grippers_urm(args, is_reinit=False):

    ###################################################################################
    if args.llm_type != -1:
        llm_instance = get_llm_instance(args.llm_type, 0.0)  # 0.0 is temperature for LLM
    else:
        llm_instance = None  # LLM is not used
    ###################################################################################
    os.makedirs(args.save_imgs_dir, exist_ok=True)
    task_folder_dir = os.path.join(args.save_imgs_dir, args.task_name)
    os.makedirs(task_folder_dir, exist_ok=True)

    ###################################################################################
    # kingfisher-6000 camera initilization
    print("\n")
    c = kingfisher.connect(cfg_dict["kfr_ip"])   # connect the camera
    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']  # 540, 960
    kingfisher.SetAUTO_EXPOSURE()
    cam_para_list = [kingfisher, kfr_height, kfr_width]
    ###################################################################################
    if not is_reinit:
        robot_init_pose_L, robot_init_gripper_L = [0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792], 0
        robot_init_pose_R, robot_init_gripper_R = [0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329], 0
        torque_value = 32  # 0-->255 soft-->hard; the gripper should grasp the object loosely

        robot_temp_pose_L = [0.508413994, 0.285476218, 0.352155227, -179.610881514, 60.474229046, -90.517829792]
        robot_temp_pose_R = [0.508413994, -0.285476218, 0.352155227, 179.732957532, 60.68246746, 89.739817245]
        
        ###################################################################################
        # grippers initilization
        print("\n")    
        gripper_L = JodellRG75(give_torque=torque_value)  # 1 close/ 0 open gripper
        gripper_L.connect(cfg_dict["arm1"]["gripper_port"], 9)  # for the left-arm, set tool_id=0
        gripper_L.set_pos(0); print("[gripper init][L] finished (fully open).")  # sudo chmod 777 /dev/ttyUSB0
        gripper_R = JodellRG75(give_torque=torque_value)  # 1 close/ 0 open gripper
        gripper_R.connect(cfg_dict["arm2"]["gripper_port"], 9)  # for the right-arm, set tool_id=0
        gripper_R.set_pos(0); print("[gripper init][R] finished (fully open).")  # sudo chmod 777 /dev/ttyUSB1
        
        ###################################################################################
        # robot arms initilization
        robot_L = rokaeRobotBase(cfg_dict["arm1"]["robot_ip_add"])  # for the left arm
        robot_L.moving_pre_op(); # robot_L.move_to_a_waypoint(robot_init_pose_L)
        robot_L.set_motion_speed_ratio(args.robot_speed)  # 20% speed
        print("[robot init][L] Current robot pose (robot view)", robot_init_pose_L)
        robot_R = rokaeRobotBase(cfg_dict["arm2"]["robot_ip_add"])  # for the right arm
        robot_R.moving_pre_op(); # robot_R.move_to_a_waypoint(robot_init_pose_R)
        robot_R.set_motion_speed_ratio(args.robot_speed)  # 20% speed
        print("[robot init][R] Current robot pose (robot view)", robot_init_pose_R)

        thread_list = []  # for saving multiple threadings for dual-arm coordination
        thread = threading.Thread(target=drive_single_arm_to_run, args=([robot_init_pose_L], "L", robot_L, robot_R, ))
        thread_list.append(thread); thread.start()
        thread = threading.Thread(target=drive_single_arm_to_run, args=([robot_init_pose_R], "R", robot_L, robot_R, ))
        thread_list.append(thread); thread.start()
        for thread in thread_list: thread.join()  # wait all threading to finish

        time.sleep(1)
        #sys.exit()

        robots_grippers_list = [robot_L, robot_R, gripper_L, gripper_R, robot_init_pose_L, robot_init_pose_R, robot_temp_pose_L, robot_temp_pose_R]
        
        return task_folder_dir, cam_para_list, llm_instance, robots_grippers_list
    ###################################################################################


    return task_folder_dir, cam_para_list
##################################################################################################################################  


if __name__ == '__main__':

    print("[initilization] robots and grippers.")
    