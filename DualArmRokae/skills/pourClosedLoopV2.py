
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
from src.func import processing_test_image_full_stage_pouring
from src.func import processing_seed_image, processing_test_image
from src.func import llm_anchored_task_switching, llm_anchored_task_switching_for_pouring
from src.action import actions_for_real_robot_rollouts

##################################################################################################################################  
#################################################################
'''
conda activate py310

python skills/pourClosedLoopV2.py --task_name pouring_full_stage --bottle_id 1 --mugcup_id 1 --test_id 100
python skills/pourClosedLoopV2.py --task_name pouring_full_stage --no_interaction --bottle_id 1 --mugcup_id 1 --test_id 100

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
    parser.add_argument("--llm_type", type=int, default=1, help="-1: not used; 0: doubao; 1: gpt-4o; 2: gemini-2.5-flash")

    parser.add_argument('--bottle_id', type=int, default=1, help="bottle_id is selected from 1 ~ 4")
    parser.add_argument('--mugcup_id', type=int, default=1, help="mugcup_id is selected from 1 ~ 4")

    args = parser.parse_args()   
    args.debug_close_loop_vis = True  # always set it as True for better debugging code
    cv2_font= cv2.FONT_HERSHEY_SIMPLEX
    gripper_wait_time = 0.5

    ###################################################################################
    assert args.task_name == "pouring_full_stage", "This script is only for the full-stage pouring task. Please give a valid task name!!!"
    assert args.bottle_id >=1 and args.bottle_id <= 4, "Please note that bottle_id is selected from 1 ~ 4 !!!"
    assert args.mugcup_id >=1 and args.mugcup_id <= 4, "Please note that mugcup_id is selected from 1 ~ 4 !!!" 

    args.task_name = "pouring"  # the "pouring" name is the real meta-skill

    task_para_list, cam_para_list, action_para_list, loop_para_list, llm_instance, robots_grippers_list = init_parameters_robots_grippers(args)
    [saved_seed_img_path, seed_ref_pts_dict, test_ref_pts_dict, prev_delta_rot, prev_delta_xy_list, detseg_prompts, supp_prompts] = task_para_list
    [kingfisher, kfr_height, kfr_width] = cam_para_list
    [robot_init_pose_L, robot_init_gripper_L, robot_init_pose_R, robot_init_gripper_R] = action_para_list
    [cl_robot_arms, cl_eef_pose_seqs, robot_arms, eef_pose_seqs, interp_num] = loop_para_list
    [robot_L, robot_R, gripper_L, gripper_R] = robots_grippers_list
    
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
    ############################################################################

    just_switched = False
    llm_invoke_count = 0
    performed_task_list = []
    while True:
        print(f"\n**************************************************************************************************************************************")
        print(f"*************************************************[Switching into Task] {args.task_name}*************************************************")
        print(f"**************************************************************************************************************************************\n")

        task_para_list, cam_para_list, action_para_list, loop_para_list = init_parameters_robots_grippers(args, is_reinit=True)  # without robots_grippers_list
        [saved_seed_img_path, seed_ref_pts_dict, test_ref_pts_dict, prev_delta_rot, prev_delta_xy_list, detseg_prompts, supp_prompts] = task_para_list
        [kingfisher, kfr_height, kfr_width] = cam_para_list
        [robot_init_pose_L, robot_init_gripper_L, robot_init_pose_R, robot_init_gripper_R] = action_para_list
        [cl_robot_arms, cl_eef_pose_seqs, robot_arms, eef_pose_seqs, interp_num] = loop_para_list

        cur_gripper_L, cur_gripper_R = robot_init_gripper_L, robot_init_gripper_R
        reprojected_seeding_pts, left_image_vis, result_img_init = processing_seed_image(args, saved_seed_img_path, detseg_prompts, supp_prompts)

        ###################################################################################
        step_id_record = 0  # for recording close-loop step ids of left-arm / right-arm
        cl_substep_id_record = 0  # for recording close-loop substep ids for one specific arm
        video_frame_index = 0
        is_task_finished = True
        while True:
            ##########################
            robot_arm_temp, eef_pose_temp = cl_robot_arms[step_id_record], cl_eef_pose_seqs[step_id_record]
            left_image_test, _ = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
            video_frame_index += 1

            final_res_list_dict = processing_test_image_full_stage_pouring(left_image_test.copy())  # detection and segmentation only once

            if video_frame_index == 1: final_res_list_pre = None
            reprojected_testing_pts, video_frame_index, final_res_list_pre, final_imgs = processing_test_image(args,
                left_image_test, detseg_prompts, supp_prompts, left_image_vis, result_img_init, 
                video_frame_index, kfr_height, kfr_width, reprojected_seeding_pts,
                final_res_list_pre=final_res_list_pre, final_res_list_dict=final_res_list_dict)
                
            if reprojected_testing_pts is None: continue

            ##########################
            if (step_id_record == 0) and (cl_substep_id_record <= 1) and (not just_switched):
                #state_bottle, state_bottle_cap = llm_anchored_task_switching(args, llm_instance, 
                #    left_image_test.copy(), "bottle", final_res_list_dict=final_res_list_dict)
                #state_mugcup, _ = llm_anchored_task_switching(args, llm_instance, 
                #    left_image_test.copy(), "cup", final_res_list_dict=final_res_list_dict)

                llm_invoke_count += 1
                state_res_list = llm_anchored_task_switching_for_pouring(args, llm_instance,
                    left_image_test.copy(), final_res_list_dict=final_res_list_dict, llm_invoke_count=llm_invoke_count)
                [state_bottle, state_mugcup, state_bottle_cap] = state_res_list

                if state_bottle == "A" and state_mugcup == "A" and state_bottle_cap == "NO": new_task_name = "pouring"
                if state_bottle == "A" and state_mugcup == "A" and state_bottle_cap == "YES": new_task_name = "unscrew"
                if state_bottle == "A" and state_mugcup == "B": new_task_name = "grasping"
                if state_bottle == "A" and state_mugcup == "C": new_task_name = "flipping"
                if state_bottle == "B": new_task_name = "reorient"
                if state_bottle == "C": new_task_name = "flatting"

                if new_task_name != args.task_name:
                    args.task_name = new_task_name; is_task_finished = False; just_switched = True; break  # stop the second while loop
            
            just_switched = False
            ##########################
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
            try: pt_seeding, pt_testing = reprojected_seeding_pts[step_id_record], reprojected_testing_pts[step_id_record]
            except: print(reprojected_seeding_pts, reprojected_testing_pts); sys.exit()
            delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
            [p_delta_x, p_delta_y, prev_bbox] = prev_delta_xy_list[step_id_record]
            
            delta_rot_z = pt_testing[2] - pt_seeding[2]  # for tasks ["plugpen", "reorient", "inserting", "reorient_unscrew", "tool_spoon"]
            obj_bbox_init, obj_bbox_test = pt_seeding[3], pt_testing[3]  # for tasks with frequent occlusion between the arm and object
            if prev_bbox is None: prev_bbox = obj_bbox_init
            
            if abs(delta_x - p_delta_x) > 10 or abs(delta_y - p_delta_y) > 10:  # we think this object may be moved
                inside_ratio = calInsideRectIOU(prev_bbox, obj_bbox_test)  # between 0 and 1
                if inside_ratio > 0.90: # obj_bbox_test is most likely inside the prev_bbox
                    print("[Warning][Continue] targeted object is most likely occluced!!! Using prev_bbox and do not update prev_delta_xy_list.")
                else: # No occlusion happened. we need to start a new close-loop step
                    cl_substep_id_record = 0; prev_delta_xy_list[step_id_record] = [delta_x, delta_y, obj_bbox_test]

            cl_substep_id_record += 1
            ##########################
            # we need to decide to use left-arm / right-arm to manipulate
            if args.task_name in ["reorient", "grasping", "flatting", "flipped"]:
                if args.task_name in ["reorient", "grasping", "flatting"]: robot_arm_temp = "R" if delta_x >= 0 else "L"
                if args.task_name in ["flipped"]: robot_arm_temp = "R" if pt_testing[2] < 90 else "L"
                if robot_arm_temp == "R":  # using armR to manipulate
                    if not args.is_syn: robot_L.move_to_a_waypoint(robot_init_pose_L)
                    eef_pose_temp = cl_eef_pose_seqs[1]; step_id_record = 0
                if robot_arm_temp == "L":  # using armL to manipulate
                    if not args.is_syn: robot_R.move_to_a_waypoint(robot_init_pose_R)
                    eef_pose_temp = cl_eef_pose_seqs[0]; step_id_record = 0
                if args.task_name in ["flipped"]: delta_rot_z = 0  # do not using this to adjust eep
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
            if args.task_name == "reorient": # avoid the singular pose.
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
            if args.no_interaction:
                #robotBase.move_by_trajectory(traj_poses_list)
                #if cl_substep_id_record == 1:  # move forward only one step to try it
                #    robotBase.move_to_a_waypoint(target_6dof_vec); prev_delta_rot = delta_rot_z; continue  # let the Open-Loop way be safer
                thread = threading.Thread(target=drive_single_arm_to_run, args=(traj_poses_list, robot_arm_temp, 0.0, eef_pose_temp[-1], ))
                thread_list.append(thread); thread.start()
                print("[***Open-Loop***][no_interaction]", robot_arm_temp, step_id_record)
                cl_substep_id_record = interp_num-1
            else:
                robotBase.move_to_a_waypoint(target_6dof_vec)
                prev_delta_rot = delta_rot_z  # update the prev_delta_rot for adjusting eep pose
                print("[***Close-Loop***]", robot_arm_temp, step_id_record, cl_substep_id_record)
            ##########################
            if cl_substep_id_record == interp_num-1:  # this close-loop step is finished !!!
                if not args.no_interaction:
                    gripper = eef_pose_temp[-1]
                    if (robot_arm_temp == "L" and gripper != cur_gripper_L):
                        m_gripper.switch(gripper, True); time.sleep(gripper_wait_time*2.0); cur_gripper_L = gripper
                    if (robot_arm_temp == "R" and gripper != cur_gripper_R):
                        m_gripper.switch(gripper, True); time.sleep(gripper_wait_time*2.0); cur_gripper_R = gripper

                if args.task_name == "reorient":
                    if robot_arm_temp == "L": eef_pose_seqs[0][0] = traj_poses_list[-1][0]; eef_pose_seqs[0][2:6] = traj_poses_list[-1][2:6]
                    if robot_arm_temp == "R": eef_pose_seqs[1][0] = traj_poses_list[-1][0]; eef_pose_seqs[1][2:6] = traj_poses_list[-1][2:6]
                if args.task_name in ["grasping", "flatting", "flipped"]:
                    if robot_arm_temp == "L": eef_pose_seqs[0][0] -= (delta_y / 1000); eef_pose_seqs[0][2] -= (delta_x / 1000)
                    if robot_arm_temp == "R": eef_pose_seqs[1][0] -= (delta_y / 1000); eef_pose_seqs[1][2] += (delta_x / 1000)

                step_id_record += 1
                cl_substep_id_record = 0
                if step_id_record == len(cl_robot_arms) or args.task_name in ["reorient", "grasping", "flatting", "flipped"]:
                    print("[***Close-Loop***] finished!!!"); break  # stop the close-loop step or steps
            ##########################
        ###################################################################################
        if not is_task_finished: continue  # do not stop the first while loop

        if args.no_interaction:
            for thread in thread_list: thread.join()  # wait all threading to finish
            thread_list = []

        if args.debug_close_loop_vis:
            cv2.destroyAllWindows()
        #sys.exit()
        ############################################################################
        vout_main = vout if args.debug_close_loop_vis and not args.no_video_out and not args.no_interaction else None

        args.debug_close_loop_vis = False  # we do not want to set it as True for actions_for_real_robot_rollouts()
        cur_gripper_L, cur_gripper_R, video_frame_index, thread_list = actions_for_real_robot_rollouts(
            args, robot_arms, eef_pose_seqs, robot_L, robot_R, gripper_L, gripper_R,
            robot_init_pose_L, robot_init_pose_R, cur_gripper_L, cur_gripper_R,
            robot_arm_temp, video_frame_index, thread_list, vout=vout_main)
        args.debug_close_loop_vis = True

        performed_task_list.append(args.task_name)
        if "pouring" in performed_task_list:
            print("\n", performed_task_list,  "\n")
            break
        ###################################################################################
    if args.debug_close_loop_vis:
        if (not args.no_video_out) and (not args.no_interaction):
            vout.release()  # Release everything ()

    ###################################################################################
    print("All has been done! Quit.")
    os._exit(0); sys.exit(0)  # the former often works better than the latter
    ###################################################################################


