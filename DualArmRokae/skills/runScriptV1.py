
import os
import sys
import cv2
import time
import json
import copy
import threading
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from src.config import cfg_dict_init as cfg_dict
from src.init import init_parameters_robots_grippers
from src.vlms import calInsideRectIOU
from src.util import interpolate_se3_bspline_startend_poses as interpolate_pose_t1
from src.util import interpolate_se3_bspline_multiple_poses as interpolate_pose_t2
from src.func import processing_seed_image, processing_test_image
from src.action import actions_for_real_robot_rollouts

##################################################################################################################################  
#################################################################
'''
conda activate py310

#################################################################

##### prehensile skills (single-arm or dual-arm)
python skills/runScriptV1.py --task_name pouring --no_interaction --bottle_id 1 --mugcup_id 1 --test_id 1
python skills/runScriptV1.py --task_name unscrew --no_interaction --bottle_id 1 --test_id 1
python skills/runScriptV1.py --task_name reorient --no_interaction --bottle_id 1 --test_id 1
python skills/runScriptV1.py --task_name grasping --no_interaction --mugcup_id 1 --test_id 1
python skills/runScriptV1.py --task_name flatting --no_interaction --bottle_id 1 --test_id 1
python skills/runScriptV1.py --task_name flipping --no_interaction --mugcup_id 1 --test_id 1

python skills/runScriptV1.py --task_name grasping_rectbox --no_interaction --rectbox_id 1 --test_id 1
python skills/runScriptV1.py --task_name grasping_cirbowl --no_interaction --cirbowl_id 1 --test_id 1
python skills/runScriptV1.py --task_name grasping_basket --no_interaction --basket_id 1 --test_id 1
python skills/runScriptV1.py --task_name grasping_holder --no_interaction --holder_id 1 --test_id 1
python skills/runScriptV1.py --task_name grasping_pencup --no_interaction --pencup_id 1 --test_id 1

python skills/runScriptV1.py --task_name unscrew-pouring --no_interaction --bottle_id 1 --mugcup_id 1 --test_id 1
python skills/runScriptV1.py --task_name flatting-reorient --no_interaction --bottle_id 1 --test_id 1
python skills/runScriptV1.py --task_name flipping-grasping --no_interaction --mugcup_id 1 --test_id 1

python skills/runScriptV1.py --task_name inserting --no_interaction --marker_id 1 --ordcup_id 1 --test_id 1
python skills/runScriptV1.py --task_name plugpen --no_interaction --pencap_id 1 --marker_id 1 --test_id 1
python skills/runScriptV1.py --task_name handover --no_interaction --shovel_id 1 --test_id 1

python skills/runScriptV1.py --task_name ppspoon --no_interaction --spoon_id 1 --test_id 1
python skills/runScriptV1.py --task_name ppfork --no_interaction --fork_id 1 --test_id 1
python skills/runScriptV1.py --task_name ppspoon-ppfork --no_interaction --spoon_id 1 --fork_id 1 --test_id 1
python skills/runScriptV1.py --task_name ppfork-ppspoon --no_interaction --fork_id 1 --spoon_id 1 --test_id 1

#################################################################

##### non-prehensile skills (single-arm or dual-arm)
python skills/runScriptV1.py --task_name pivoting --no_interaction --cirbowl_id 1 --test_id 1
python skills/runScriptV1.py --task_name wrapping --no_interaction --basket_id 1 --test_id 1

python skills/runScriptV1.py --task_name pivoting_rectbox --no_interaction --rectbox_id 1 --test_id 1
python skills/runScriptV1.py --task_name pivoting_cirbowl --no_interaction --cirbowl_id 1 --test_id 1
python skills/runScriptV1.py --task_name flipping_basket --no_interaction --basket_id 1 --test_id 1
python skills/runScriptV1.py --task_name flipping_block --no_interaction --block_id 1 --test_id 1
python skills/runScriptV1.py --task_name pivoting_bigjar --no_interaction --bigjar_id 1 --test_id 1
python skills/runScriptV1.py --task_name pivoting_block --no_interaction --block_id 1 --test_id 1
python skills/runScriptV1.py --task_name toppling_holder --no_interaction --holder_id 1 --test_id 1
python skills/runScriptV1.py --task_name toppling_bigjar --no_interaction --bigjar_id 1 --test_id 1
python skills/runScriptV1.py --task_name bilifting_bigjar --no_interaction --bigjar_id 1 --test_id 1
python skills/runScriptV1.py --task_name bilifting_block --no_interaction --block_id 1 --test_id 1

#################################################################

##### complex and long-horizon tasks (single-arm or dual-arm)

========== this task is realted to deformable / articulated / oriented objects
python skills/runScriptV1.py --task_name penbagzip --no_interaction --penbag_id 1 --marker_id 1 --test_id 1
python skills/runScriptV1.py --task_name foldtowel --no_interaction --towel_id 1 --test_id 1
python skills/runScriptV1.py --task_name foldpants --no_interaction --pants_id 1 --test_id 1
python skills/runScriptV1.py --task_name foldshirt --no_interaction --shirt_id 1 --test_id 1
python skills/runScriptV1.py --task_name coilcable --no_interaction --cable_id 1 --test_id 1
python skills/runScriptV1.py --task_name coilrope --no_interaction --rope_id 1 --test_id 1
python skills/runScriptV1.py --task_name coilbelt --no_interaction --belt_id 1 --test_id 1

'''
#################################################################
##################################################################################################################################  

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_imgs_dir', default="/home/dex/zhouhuayi/rokaeDemo/results/", 
                        help="path to save all initial seeding / processed testing images")
    parser.add_argument('--task_name', default='', help="string of bimanual task name")
    parser.add_argument('--debug_close_loop_vis', action='store_true', help="default is False")
    parser.add_argument('--no_interaction', action='store_true', help="default is False")
    parser.add_argument('--no_video_out', action='store_true', help="default is False")
    parser.add_argument('--is_syn', action='store_true', help="default is False")
    parser.add_argument('--robot_speed', type=float, default=0.3, help="do not set it too large. DANGER!!!")
    parser.add_argument('--test_id', type=int, default=1, help="we may record many times of one specific task")
    parser.add_argument("--llm_type", type=int, default=-1, help="-1: not used; 0: doubao; 1: gpt-4o; 2: gemini-2.5-flash")

    parser.add_argument('--bottle_id', type=int, default=1, help="bottle_id is selected from 1 ~ 4")
    parser.add_argument('--mugcup_id', type=int, default=1, help="mugcup_id is selected from 1 ~ 4")
    parser.add_argument('--marker_id', type=int, default=1, help="marker_id is selected from 1 ~ 4")
    parser.add_argument('--ordcup_id', type=int, default=1, help="ordcup_id is selected from 1 ~ 4")
    parser.add_argument('--cirbowl_id', type=int, default=1, help="cirbowl_id is selected from 1 ~ 4")
    parser.add_argument('--basket_id', type=int, default=1, help="basket_id is selected from 1 ~ 4")

    parser.add_argument('--pencap_id', type=int, default=1, help="pencap_id is selected from 1 ~ 4")
    parser.add_argument('--shovel_id', type=int, default=1, help="shovel_id is selected from 1 ~ 4")

    parser.add_argument('--spoon_id', type=int, default=1, help="spoon_id is selected from 1 ~ 4")
    parser.add_argument('--fork_id', type=int, default=1, help="fork_id is selected from 1 ~ 4")

    parser.add_argument('--rectbox_id', type=int, default=1, help="rectbox_id is selected from 1 ~ 4")
    parser.add_argument('--holder_id', type=int, default=1, help="holder_id is selected from 1 ~ 4")
    parser.add_argument('--pencup_id', type=int, default=1, help="pencup_id is selected from 1 ~ 4")    
    parser.add_argument('--bigjar_id', type=int, default=1, help="bigjar_id is selected from 1 ~ 4")
    parser.add_argument('--block_id', type=int, default=1, help="block_id is selected from 1 ~ 4")

    parser.add_argument('--penbag_id', type=int, default=1, help="block_id is selected from 1 ~ 4")
    parser.add_argument('--towel_id', type=int, default=1, help="towel_id is selected from 1 ~ 4")
    parser.add_argument('--pants_id', type=int, default=1, help="pants_id is selected from 1 ~ 4")
    parser.add_argument('--shirt_id', type=int, default=1, help="shirt_id is selected from 1 ~ 4")
    
    parser.add_argument('--cable_id', type=int, default=1, help="cable_id is selected from 1 ~ 4")
    parser.add_argument('--rope_id', type=int, default=1, help="rope_id is selected from 1 ~ 4")
    parser.add_argument('--belt_id', type=int, default=1, help="belt_id is selected from 1 ~ 4")

    args = parser.parse_args()   
    args.debug_close_loop_vis = True  # always set it as True for better debugging code
    cv2_font = cv2.FONT_HERSHEY_SIMPLEX
    gripper_wait_time = 0.5

    ###################################################################################
    assert args.task_name in ["pouring", "unscrew", "reorient", "grasping", "flatting", "flipping",
        "flatting_flipping", "flatting-reorient", "flipping-grasping", "reorient_unscrew", "grasping_pouring", "unscrew-pouring",
        # "reorient_unscrew_pouring", "flatting_reorient_unscrew", 
        # "flatting_flipping_reorient_grasping", "reorient_grasping_unscrew_pouring",
        # "flatting_flipping_reorient_grasping_unscrew_pouring", 
        "grasping_rectbox", "grasping_cirbowl", "grasping_basket", "grasping_holder", "grasping_pencup",
        "inserting", "plugpen", "handover", "ppspoon", "ppfork", "ppspoon-ppfork", "ppfork-ppspoon",
        "pivoting", "wrapping", "pivoting_rectbox", "pivoting_cirbowl", "flipping_basket", "flipping_block", 
        "pivoting_bigjar", "pivoting_block", "toppling_holder", "toppling_bigjar", "bilifting_bigjar", "bilifting_block",
        "penbagzip", "foldtowel", "foldpants", "foldshirt", "coilcable", "coilrope", "coilbelt" ], "Please give a valid task name!!!"

    assert args.bottle_id >=1 and args.bottle_id <= 4, "Please note that bottle_id is selected from 1 ~ 4 !!!"
    assert args.mugcup_id >=1 and args.mugcup_id <= 4, "Please note that mugcup_id is selected from 1 ~ 4 !!!" 
    assert args.marker_id >=1 and args.marker_id <= 4, "Please note that marker_id is selected from 1 ~ 4 !!!" 
    assert args.ordcup_id >=1 and args.ordcup_id <= 4, "Please note that ordcup_id is selected from 1 ~ 4 !!!" 
    assert args.cirbowl_id >=1 and args.cirbowl_id <= 4, "Please note that cirbowl_id is selected from 1 ~ 4 !!!" 
    assert args.basket_id >=1 and args.basket_id <= 4, "Please note that basket_id is selected from 1 ~ 4 !!!" 

    ''' decrease the speed for these strictly synchronized dual-arm coordinated motion! '''
    # if args.task_name in ["grasping_basket", "bilifting_bigjar", "bilifting_block"]: args.robot_speed = 0.15
   
    task_para_list, cam_para_list, action_para_list, loop_para_list, llm_instance, robots_grippers_list = init_parameters_robots_grippers(args)
    [saved_seed_img_path, seed_ref_pts_dict, test_ref_pts_dict, prev_delta_rot, prev_delta_xy_list, detseg_prompts, supp_prompts] = task_para_list
    [kingfisher, kfr_height, kfr_width] = cam_para_list
    [robot_init_pose_L, robot_init_gripper_L, robot_init_pose_R, robot_init_gripper_R] = action_para_list
    [cl_robot_arms, cl_eef_pose_seqs, robot_arms, eef_pose_seqs, interp_num] = loop_para_list
    [robot_L, robot_R, gripper_L, gripper_R] = robots_grippers_list

    reprojected_seeding_pts, left_image_vis, result_img_init = processing_seed_image(args, saved_seed_img_path, detseg_prompts, supp_prompts)
    
    #os._exit(0)

    video_frame_index = 0
    if args.debug_close_loop_vis:
        if not args.no_video_out and not args.no_interaction:
            video_output_path = saved_seed_img_path.replace("_seed.jpg", f"_test{args.test_id}.mp4")
            # vout = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (1790, 1080))  # Create VideoWriter object
            vout = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (1253, 756))  # Create VideoWriter object

    ############################################################################
    cur_gripper_L, cur_gripper_R = robot_init_gripper_L, robot_init_gripper_R
    def drive_single_arm_to_run(target_pose_list, arm_type, sleep_time, gripper_value):
        if arm_type == "L":
            time.sleep(sleep_time); robot_L.move_by_trajectory(target_pose_list)
            global cur_gripper_L
            if gripper_value != -1 and cur_gripper_L != gripper_value:  # gripper = 1 close; gripper = 0 open
                gripper_L.switch(gripper_value, True); time.sleep(gripper_wait_time); cur_gripper_L = gripper_value          
        if arm_type == "R":
            time.sleep(sleep_time); robot_R.move_by_trajectory(target_pose_list)
            global cur_gripper_R
            if gripper_value != -1 and cur_gripper_R != gripper_value:  # gripper = 1 close; gripper = 0 open
                gripper_R.switch(gripper_value, True); time.sleep(gripper_wait_time); cur_gripper_R = gripper_value

    thread_list = []  # for saving multiple threadings for dual-arm coordination
    ###################################################################################

    #===================================
    save_temp_for_cap_path = saved_seed_img_path.replace("_seed.jpg", f"_test{args.test_id}_capStartL.json")
    if os.path.exists(save_temp_for_cap_path): os.remove(save_temp_for_cap_path)
    #===================================

    ###################################################################################
    step_id_record = 0  # for recording close-loop step ids of left-arm / right-arm
    cl_substep_id_record = 0  # for recording close-loop substep ids for one specific arm
    while True:
        robot_arm_temp, eef_pose_temp = cl_robot_arms[step_id_record], cl_eef_pose_seqs[step_id_record]
        left_image_test, _ = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
        video_frame_index += 1

        ##########################
        if video_frame_index == 1: final_res_list_pre = None
        reprojected_testing_pts, video_frame_index, final_res_list_pre, final_imgs = processing_test_image(args,
            left_image_test, detseg_prompts, supp_prompts, left_image_vis, result_img_init, 
            video_frame_index, kfr_height, kfr_width, reprojected_seeding_pts, final_res_list_pre)
            
        if reprojected_testing_pts is None: continue

        if args.debug_close_loop_vis:
            if not args.no_interaction:
                cv2.namedWindow("MyDebugWindow", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("MyDebugWindow", 1253, 756)  # (1790, 1080) --> (1253, 756)
                cv2.imshow("MyDebugWindow", final_imgs)
                if cv2.waitKey(1) & 0xFF == ord('q'): break  # or cv2.waitKey(0); wait until we close the plotted window (press Esc to continue)
                if not args.no_video_out:
                    for _ in range(5): vout.write(final_imgs)

            if video_frame_index == 1:
                cv2.imwrite(saved_seed_img_path.replace("_seed.jpg", f"_test{args.test_id}_frame1.jpg"), final_imgs)
        ##########################
        try:
            pt_seeding, pt_testing = reprojected_seeding_pts[step_id_record], reprojected_testing_pts[step_id_record]
            [p_delta_x, p_delta_y, prev_bbox] = prev_delta_xy_list[step_id_record]
        except:
            if (cl_substep_id_record >= interp_num-4) and (args.task_name in ["inserting", "plugpen"]):  # for the lost-detected marker pen
                pt_seeding = reprojected_seeding_pts[step_id_record]
                [p_delta_x, p_delta_y, prev_bbox] = prev_delta_xy_list[step_id_record]
                pt_testing = [p_delta_x+pt_seeding[0], p_delta_y+pt_seeding[1], prev_delta_rot+pt_seeding[2], prev_bbox]
            elif step_id_record == 1 and (args.task_name in ["pivoting", "wrapping", 
                "pivoting_rectbox", "pivoting_cirbowl", "flipping_basket", "flipping_block", "grasping_basket", 
                "pivoting_bigjar", "pivoting_block", "toppling_holder", "toppling_bigjar", 
                "bilifting_bigjar", "bilifting_block" ]):  # we only have one object for the second arm
                pt_seeding = reprojected_seeding_pts[0]
                [p_delta_x, p_delta_y, prev_bbox] = prev_delta_xy_list[0]
                pt_testing = [p_delta_x+pt_seeding[0], p_delta_y+pt_seeding[1], prev_delta_rot+pt_seeding[2], prev_bbox]
            elif step_id_record == 1 and (args.task_name in ["foldtowel", "foldpants", "foldshirt"]):
                pt_seeding, pt_testing = reprojected_seeding_pts[0], reprojected_testing_pts[0]
                [p_delta_x, p_delta_y, prev_bbox] = prev_delta_xy_list[0]
            else: 
                #print("ERROR:\t", video_frame_index, reprojected_seeding_pts, reprojected_testing_pts); os._exit(0)
                print("ERROR:\t", video_frame_index. reprojected_seeding_pts, reprojected_testing_pts); continue

        delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
        delta_rot_z = pt_testing[2] - pt_seeding[2]  # for tasks ["plugpen", "reorient", "inserting", "reorient_unscrew", "tool_spoon"]
        obj_bbox_init, obj_bbox_test = pt_seeding[3][:4], pt_testing[3][:4]  # for tasks with frequent occlusion between the arm and object
        if prev_bbox is None: prev_bbox = obj_bbox_init
        
        if abs(delta_x - p_delta_x) > 10 or abs(delta_y - p_delta_y) > 10:  # we think this object may be moved
            inside_ratio = calInsideRectIOU(prev_bbox, obj_bbox_test)  # between 0 and 1
            if inside_ratio > 0.90: # obj_bbox_test is most likely inside the prev_bbox
                print("[Warning][Continue] targeted object is most likely occluced!!! Using prev_bbox and do not update prev_delta_xy_list.")
            else: # No occlusion happened. we need to start a new close-loop step
                cl_substep_id_record = 0; prev_delta_xy_list[step_id_record] = [delta_x, delta_y, obj_bbox_test]

        if args.task_name in ["foldtowel", "foldpants", "foldshirt"]:  # we need to re-adjust the (delta_x, delta_y)
            if robot_arm_temp == "L": delta_x, delta_y = pt_testing[3][4] - pt_seeding[3][4], pt_testing[3][5] - pt_seeding[3][5]
            if robot_arm_temp == "R": delta_x, delta_y = pt_testing[3][6] - pt_seeding[3][6], pt_testing[3][7] - pt_seeding[3][7]

        cl_substep_id_record += 1
        ##########################
        # we need to decide to use left-arm / right-arm to manipulate
        if args.task_name in ["reorient", "grasping", "flatting", "flipping", "flatting-reorient", "flipping-grasping", 
            "grasping_rectbox", "grasping_cirbowl", "grasping_holder", "grasping_pencup", "ppspoon", "ppfork" ]:

            if args.task_name in ["reorient", "grasping", "flatting", "flatting-reorient", 
                "grasping_rectbox", "grasping_cirbowl", "grasping_holder", "grasping_pencup", "ppspoon", "ppfork" ]:
                robot_arm_temp = "R" if delta_x >= 0 else "L"

            # if args.task_name in ["flipping", "flipping-grasping"]: robot_arm_temp = "R" if pt_testing[2] < 90 else "L"  # not stable currently
            if args.task_name in ["flipping", "flipping-grasping"]: robot_arm_temp = "R" if delta_x >= 0 else "L"
            if args.task_name in ["flipping", "flipping-grasping"]: delta_rot_z = 0  # do not using this to adjust eep of mug

            if robot_arm_temp == "R":  # using armR to manipulate
                if not args.is_syn: robot_L.move_to_a_waypoint(robot_init_pose_L, is_waiting=False)
                eef_pose_temp = cl_eef_pose_seqs[1]; step_id_record = 0
            if robot_arm_temp == "L":  # using armL to manipulate
                if not args.is_syn: robot_R.move_to_a_waypoint(robot_init_pose_R, is_waiting=False)
                eef_pose_temp = cl_eef_pose_seqs[0]; step_id_record = 0
        ##########################
        # get end_eef_pose
        cur_end_eef_pose = eef_pose_temp.copy()
        if robot_arm_temp == "L": cur_end_eef_pose[0] -= (delta_y / 1000); cur_end_eef_pose[2] -= (delta_x / 1000)
        if robot_arm_temp == "R": cur_end_eef_pose[0] -= (delta_y / 1000); cur_end_eef_pose[2] += (delta_x / 1000)
        cur_end_eef_pose[4] -= delta_rot_z

        # get start_eef_pose
        if robot_arm_temp == "L": robotBase = robot_L; m_gripper = gripper_L; init_3d_pose = robot_init_pose_L[3:6]
        if robot_arm_temp == "R": robotBase = robot_R; m_gripper = gripper_R; init_3d_pose = robot_init_pose_R[3:6]
        cur_start_eef_pose, _ = robotBase.get_current_pose(); cur_eef_pose_temp = cur_start_eef_pose.copy()
        if cl_substep_id_record != interp_num-1:
            if robot_arm_temp == "L": cur_start_eef_pose[1] -= 0.060  # move robot higher slightly to avoid hitting objects 
            if robot_arm_temp == "R": cur_start_eef_pose[1] += 0.060  # move robot higher slightly to avoid hitting objects 
        if cl_substep_id_record == 1:  # object is moved backward (not forward)
            cur_start_eef_pose[2] = cur_end_eef_pose[2] + 0.120  # move robot back slightly to avoid hitting objects
            cur_start_eef_pose[0] = cur_end_eef_pose[0] + 0.000  # directly align the x-value of these two keyposes
        ##########################
        if args.task_name in ["reorient", "ppspoon", "ppfork", "ppspoon-ppfork", "ppfork-ppspoon", 
            "handover", "coilcable", "coilrope", "coilbelt"]: # avoid the singular pose.
            if robot_arm_temp == "L":  # degree range (-90, 270). 90 is dangerous for armL. -90 or 270 is dangerous for armR 
                rot_ad1 = delta_rot_z-360 if delta_rot_z >= 90 else delta_rot_z
                rot_ad0 = prev_delta_rot-360 if prev_delta_rot >= 90 else prev_delta_rot
            if (robot_arm_temp == "L" and abs(rot_ad1 - rot_ad0) > 175) or (robot_arm_temp == "R" and abs(delta_rot_z - prev_delta_rot) > 175):
                cur_start_eef_pose[3:6] = init_3d_pose
                traj_poses_list_adjusting = interpolate_pose_t1(cur_eef_pose_temp, cur_start_eef_pose, num_steps=5)
                robotBase.move_by_trajectory(traj_poses_list_adjusting)  # direct using move_to_a_waypoint() may have singular eep pose error
        ##########################
        traj_poses_list = interpolate_pose_t1(cur_start_eef_pose[:6], cur_end_eef_pose[:6], num_steps=interp_num)
        target_6dof_vec = traj_poses_list[cl_substep_id_record]
        ##########################

        #===================================
        if not args.no_interaction and video_frame_index == 1:
            with open(save_temp_for_cap_path, "w") as json_file: json.dump(["start moving"], json_file)
            print(f"\n[Start Moving] save state as a json file for starting the video capturing...")
            time.sleep(0.1)
        #===================================

        ##########################
        gripper_val = eef_pose_temp[-1]
        if args.no_interaction:
            thread = threading.Thread(target=drive_single_arm_to_run, args=(traj_poses_list, robot_arm_temp, 0.0, gripper_val, ))
            thread_list.append(thread); thread.start()
            print("[***Open-Loop***][no_interaction]", args.task_name, robot_arm_temp, step_id_record)
            cl_substep_id_record = interp_num-1
        elif robot_arm_temp == "R" and args.task_name in ["pivoting", "wrapping", 
            "pivoting_rectbox", "pivoting_cirbowl", "flipping_basket", "flipping_block", "grasping_basket", 
            "pivoting_bigjar", "pivoting_block", "toppling_holder", "toppling_bigjar", "bilifting_bigjar", "bilifting_block",
            "foldtowel", "foldpants", "foldshirt" ]: # the right arm should always be open-loop for non-prehensile tasks!!!"
            robotBase.move_by_trajectory(traj_poses_list)
            print("[***Open-Loop***][non-prehensile right-arm]", args.task_name, robot_arm_temp, step_id_record)
            cl_substep_id_record = interp_num-1
        else:
            robotBase.move_to_a_waypoint(target_6dof_vec)
            prev_delta_rot = delta_rot_z  # update the prev_delta_rot for adjusting eep pose
            print("[***Close-Loop***]", args.task_name, robot_arm_temp, step_id_record, cl_substep_id_record)
        ##########################
        if cl_substep_id_record == interp_num-1:  # this close-loop step is finished !!!
            if not args.no_interaction:
                if (robot_arm_temp == "L" and gripper_val != cur_gripper_L):
                    m_gripper.switch(gripper_val, True); time.sleep(gripper_wait_time*2.0); cur_gripper_L = gripper_val
                if (robot_arm_temp == "R" and gripper_val != cur_gripper_R):
                    m_gripper.switch(gripper_val, True); time.sleep(gripper_wait_time*2.0); cur_gripper_R = gripper_val

            if args.task_name in ["reorient", "ppspoon", "ppfork", "ppspoon-ppfork", "ppfork-ppspoon"]:
                if robot_arm_temp == "L": eef_pose_seqs[0][0] = traj_poses_list[-1][0]; eef_pose_seqs[0][2:6] = traj_poses_list[-1][2:6]
                if robot_arm_temp == "R": eef_pose_seqs[1][0] = traj_poses_list[-1][0]; eef_pose_seqs[1][2:6] = traj_poses_list[-1][2:6]
                if args.task_name in ["ppspoon", "ppfork", "ppspoon-ppfork", "ppfork-ppspoon"]:
                    if robot_arm_temp == "L": eef_pose_seqs[2][0] = traj_poses_list[-1][0]; eef_pose_seqs[2][2:6] = traj_poses_list[-1][2:6]
                    if robot_arm_temp == "R": eef_pose_seqs[3][0] = traj_poses_list[-1][0]; eef_pose_seqs[3][2:6] = traj_poses_list[-1][2:6]
                    if robot_arm_temp == "L" and (delta_rot_z >= 90 and delta_rot_z <= 180): eef_pose_seqs[2][4] = init_3d_pose[1]  # adjust pose for corner case
                    if robot_arm_temp == "R" and (delta_rot_z >= 180 and delta_rot_z <= 270): eef_pose_seqs[3][4] = init_3d_pose[1]  # adjust pose for corner case
            if args.task_name in ["grasping", "flatting", "flipping", "flatting-reorient", "flipping-grasping",
                "grasping_rectbox", "grasping_cirbowl", "grasping_basket", "grasping_holder", "grasping_pencup",
                "pivoting", "wrapping", "pivoting_rectbox", "pivoting_cirbowl", "flipping_basket", "flipping_block", 
                "pivoting_bigjar", "pivoting_block", "toppling_holder", "toppling_bigjar", "bilifting_bigjar", "bilifting_block", 
                "foldtowel", "foldpants", "foldshirt" ]:  # pre-grasp
                if robot_arm_temp == "L": eef_pose_seqs[0][0] -= (delta_y / 1000); eef_pose_seqs[0][2] -= (delta_x / 1000)
                if robot_arm_temp == "R": eef_pose_seqs[1][0] -= (delta_y / 1000); eef_pose_seqs[1][2] += (delta_x / 1000)
                if args.task_name in ["grasping_rectbox", "grasping_cirbowl", "grasping_basket", "grasping_pencup", "bilifting_bigjar", "bilifting_block"]:
                    if robot_arm_temp == "L": eef_pose_seqs[2][0] -= (delta_y / 1000); eef_pose_seqs[2][2] -= (delta_x / 1000)
                    if robot_arm_temp == "R": eef_pose_seqs[3][0] -= (delta_y / 1000); eef_pose_seqs[3][2] += (delta_x / 1000)
                if args.task_name in ["toppling_holder", "toppling_bigjar"]:
                    if robot_arm_temp == "L": eef_pose_seqs[2][0] -= (delta_y / 1000); eef_pose_seqs[2][2] -= (delta_x / 1000)
                    if robot_arm_temp == "R": eef_pose_seqs[3][0] -= (delta_y / 1000); eef_pose_seqs[3][2] += (delta_x / 1000)
                    if robot_arm_temp == "L": eef_pose_seqs[4][0] -= (delta_y / 1000); eef_pose_seqs[4][2] -= (delta_x / 1000)
                    if robot_arm_temp == "R": eef_pose_seqs[5][0] -= (delta_y / 1000); eef_pose_seqs[5][2] += (delta_x / 1000)
            if args.task_name in ["handover", "coilcable", "coilrope", "coilbelt"]:
                assert robot_arm_temp == "R", f"We now only support the right-arm to do the task {args.task_name}!!!"
                eef_pose_seqs[0][0] = traj_poses_list[-1][0]; eef_pose_seqs[0][2:6] = traj_poses_list[-1][2:6]  # grasping
                eef_pose_seqs[1][0] = traj_poses_list[-1][0]; eef_pose_seqs[1][2:6] = traj_poses_list[-1][2:6]  # lift-up
                if (delta_rot_z >= 180 and delta_rot_z <= 270): eef_pose_seqs[1][4] = init_3d_pose[1]  # adjust pose for corner case

            if args.task_name == "unscrew-pouring":  # for open-loop adjusting/grasping the mugcup
                pt_seeding, pt_testing = reprojected_seeding_pts[1], reprojected_testing_pts[1]
                delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
                eef_pose_seqs[12][0] -= (delta_y / 1000); eef_pose_seqs[12][2] += (delta_x / 1000)

            if args.task_name == "penbagzip":  # for open-loop adjusting/grasping the penbag and pen
                if robot_arm_temp == "L": eef_pose_seqs[0][0] -= (delta_y / 1000); eef_pose_seqs[0][2] -= (delta_x / 1000)  # for grasping penbag
                if robot_arm_temp == "R": eef_pose_seqs[1][0] = traj_poses_list[-1][0]; eef_pose_seqs[1][2:6] = traj_poses_list[-1][2:6] # for grasping pen

            step_id_record += 1
            cl_substep_id_record = 0
            if step_id_record == len(cl_robot_arms) or \
                args.task_name in ["reorient", "grasping", "flatting", "flipping", "flatting-reorient", "flipping-grasping",
                    "grasping_rectbox", "grasping_cirbowl", "grasping_holder", "grasping_pencup", "ppspoon", "ppfork", 
                    "handover", "coilcable", "coilrope", "coilbelt"]:
                print("[***Close-Loop***] finished!!!"); break  # stop the close-loop step or steps
        ##########################
    
    if args.no_interaction:
        #===================================
        with open(save_temp_for_cap_path, "w") as json_file: json.dump(["start moving"], json_file)
        print(f"\n[Start Moving] save state as a json file for starting the video capturing...")
        time.sleep(0.1)
        #===================================
        for thread in thread_list: thread.join()  # wait all threading to finish
        thread_list = []

    if args.debug_close_loop_vis:
        cv2.destroyAllWindows()
    #sys.exit()
    ###################################################################################
    vout_main = vout if args.debug_close_loop_vis and not args.no_video_out and not args.no_interaction else None

    cur_gripper_L, cur_gripper_R, video_frame_index, thread_list = actions_for_real_robot_rollouts(
        args, robot_arms, eef_pose_seqs, robot_L, robot_R, gripper_L, gripper_R,
        robot_init_pose_L, robot_init_pose_R, cur_gripper_L, cur_gripper_R,
        robot_arm_temp, video_frame_index, thread_list, vout=vout_main)

    #===================================
    if os.path.exists(save_temp_for_cap_path): os.remove(save_temp_for_cap_path)
    #===================================

    print("All has been done! Quit.")
    os._exit(0); sys.exit(0)  # the former often works better than the latter
    ###################################################################################


