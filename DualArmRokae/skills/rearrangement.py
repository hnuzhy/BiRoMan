
import os
import sys
import cv2
import time
import json
import copy
import shutil
import threading
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from src.config import cfg_dict_init as cfg_dict
from src.init import init_parameters_robots_grippers
from src.init import init_parameters_robots_grippers_urm
from src.vlms import calInsideRectIOU
from src.util import interpolate_se3_bspline_startend_poses as interpolate_pose_t1
from src.util import interpolate_se3_bspline_multiple_poses as interpolate_pose_t2
from src.func import processing_test_image_full_stage_pouring
from src.func import processing_seed_image, processing_test_image

from src.func2 import processing_test_image_full_stage_urm_t1_box
from src.func2 import processing_test_image_full_stage_urm_t2_bowl
from src.func2 import processing_test_image_full_stage_urm_t3_basket
from src.func2 import processing_test_image_full_stage_urm_t4_holder
from src.func2 import processing_test_image_full_stage_urm_t5_bigjar
from src.func2 import processing_test_image_full_stage_urm_t6_block

from src.action import actions_for_real_robot_rollouts

from src.func2 import processing_test_image_full_stage_rarg_t1_dining
from src.func2 import processing_test_image_full_stage_rarg_t2_drinking


##################################################################################################################################  
#################################################################
'''
conda activate py310

python skills/rearrangement.py --task_name rarg_t1_dining --cirbowl_id 1 --spoon_id 1 --fork_id 1 --test_id 1
python skills/rearrangement.py --task_name rarg_t1_dining --no_interaction --cirbowl_id 1 --spoon_id 1 --fork_id 1 --test_id 1

python skills/rearrangement.py --task_name rarg_t2_drinking --bottle_id 1 --mugcup_id 1 --test_id 1
python skills/rearrangement.py --task_name rarg_t2_drinking --no_interaction --bottle_id 1 --mugcup_id 1 --test_id 1

python skills/rearrangement.py --task_name rarg_t3_sloting --bottle_id 1 --mugcup_id 1 --test_id 1
python skills/rearrangement.py --task_name rarg_t3_sloting --no_interaction --bottle_id 1 --mugcup_id 1 --test_id 1

'''
#################################################################
##################################################################################################################################  
if __name__ == "__main__":

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
    parser.add_argument("--llm_type", type=int, default=-1, help="-1: not used;  0: doubao;  1: chat-gpt-4o;  2: gemini-2.5-flash")

    parser.add_argument('--cirbowl_id', type=int, default=1, help="cirbowl_id is selected from 1 ~ 4")
    parser.add_argument('--spoon_id', type=int, default=1, help="spoon_id is selected from 1 ~ 4")
    parser.add_argument('--fork_id', type=int, default=1, help="fork_id is selected from 1 ~ 4")

    parser.add_argument('--bottle_id', type=int, default=1, help="bottle_id is selected from 1 ~ 4")
    parser.add_argument('--mugcup_id', type=int, default=1, help="mugcup_id is selected from 1 ~ 4")


    args = parser.parse_args()   
    args.debug_close_loop_vis = True  # always set it as True for better debugging code
    cv2_font= cv2.FONT_HERSHEY_SIMPLEX
    gripper_wait_time = 0.5
    global_interp_num = 4  # 6 --> 4

    ###################################################################################
    rearrangement_task_name = args.task_name

    task_folder_dir, cam_para_list, llm_instance, robots_grippers_list = init_parameters_robots_grippers_urm(args)
    [kingfisher, kfr_height, kfr_width] = cam_para_list
    [robot_L, robot_R, gripper_L, gripper_R, robot_init_pose_L, robot_init_pose_R, robot_temp_pose_L, robot_temp_pose_R] = robots_grippers_list
    

    if rearrangement_task_name == "rarg_t1_dining":
        cls_name_list = ["white bowl", "green bowl", "transparent bowl", "gray bowl"]  # four bowls have different colors and sizes
        cls_name = cls_name_list[args.cirbowl_id - 1]
        cls_name_list1 = ["spoon", "spoon", "spoon"]  # three spoons have different materials, shapes and sizes
        cls_name1 = cls_name_list1[args.spoon_id - 1]
        cls_name_list2 = ["fork", "fork", "fork"]  # three forks have different materials, shapes and sizes
        cls_name2 = cls_name_list2[args.fork_id - 1]
        oid_str = str(args.cirbowl_id).zfill(2) +"-"+ str(args.spoon_id).zfill(2) +"-"+ str(args.fork_id).zfill(2)
        video_name_str = f"{args.task_name}_test{args.test_id}_oid{oid_str}"

    if rearrangement_task_name == "rarg_t2_drinking":
        cls_name_list = ["bottle", "bottle", "bottle", "bottle"]  # four bottles have different colors and sizes
        cls_name = cls_name_list[args.bottle_id - 1]
        cls_name_list1 = ["cup", "cup", "cup"]  # three mug cups have different materials, shapes and sizes
        cls_name1 = cls_name_list1[args.mugcup_id - 1]
        oid_str = str(args.bottle_id).zfill(2) +"-"+ str(args.mugcup_id).zfill(2)
        video_name_str = f"{args.task_name}_test{args.test_id}_oid{oid_str}"


    video_frame_index = 0
    if args.debug_close_loop_vis:
        video_output_path = os.path.join(task_folder_dir, f"{video_name_str}.mp4")
        video_output_path_dir = os.path.join(task_folder_dir, video_name_str)
        if os.path.exists(video_output_path_dir): shutil.rmtree(video_output_path_dir)
        os.makedirs(video_output_path_dir, exist_ok=True)
        if not args.no_video_out and not args.no_interaction:
            # vout = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (1790, 1080))  # Create VideoWriter object
            vout = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (1253, 756))  # Create VideoWriter object

    ############################################################################
    cur_gripper_L, cur_gripper_R = 0, 0
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
    
    def drive_single_arm_to_run_slim(target_pose_list, arm_type, robot_L, robot_R):
        if arm_type == "L": robot_L.move_by_trajectory(target_pose_list)
        if arm_type == "R": robot_R.move_by_trajectory(target_pose_list)

    thread_list = []  # for saving multiple threadings for dual-arm coordination
    ############################################################################
    
    #===================================
    save_temp_for_cap_path = video_output_path.replace(".mp4", f"_capStartL.json")
    if os.path.exists(save_temp_for_cap_path): os.remove(save_temp_for_cap_path)
    #===================================
    
    # dual-arm pivoting --> single-arm grasping+placing --> stop
    is_task_finished = False 
    global_substep_id = 0
    saved_frame_count = 0
    while True:
        ###################################################################################
        left_image_test, right_image_test = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
        #####==================================================
        if rearrangement_task_name == "rarg_t1_dining":
            ##### for the urm_t2_bowl stage / steps (we now consider the closed-loop mode for stacking bowls)
            final_res_list_dict, cur_state, img_vis_cv2 = processing_test_image_full_stage_urm_t2_bowl(
                left_image_test.copy(), cls_name, right_image_test.copy())
            if cur_state == "A":
                [bx1, by1, bx2, by2] = final_res_list_dict[cls_name][1]  # [obj_name, bbox, obj_binary_mask, multi_masks]
                bcx, bcy, [x1, y1, x2, y2] = (bx1+bx2)*0.5, (by1+by2)*0.5, cfg_dict["detection_roi_bbox"]
                if bcx > (x2-x1)*(2/5.0) and bcx < (x2-x1)*(3/5.0) and bcy < (y2-y1)*(1/2.0):  # upright --> stop
                    is_task_finished = True
                else:  # upright --> single-arm grasping+placing (the bowl is not rightly placed!)
                    subtask_name = "grasping_cirbowl"; is_task_finished = False
            if cur_state == "B":  # inverted --> dual-arm pivoting (the bowl can not be grasped now)
                subtask_name = "pivoting_cirbowl"; is_task_finished = False
            ##### for the ppspoon + ppfork stage / steps (we now only consider the open-loop mode for ppspoon/ppfork)
            if is_task_finished == True and ("ppspoon" not in subtask_name and "ppfork" not in subtask_name):
                final_res_list_dict, cur_state, img_vis_cv2 = processing_test_image_full_stage_rarg_t1_dining(
                    left_image_test.copy(), cls_name1, cls_name2, right_image_test.copy())
                if cur_state == "A2": subtask_name = "ppspoon-ppfork"; is_task_finished = False
                if cur_state == "B2": subtask_name = "ppfork-ppspoon"; is_task_finished = False
                if cur_state == "C2": subtask_name = "ppspoon"; is_task_finished = False
                if cur_state == "D2": subtask_name = "ppfork"; is_task_finished = False
        #####==================================================
        if rearrangement_task_name == "rarg_t2_drinking":
            ##### for the reorient / unscrew / pouring stage / steps (we now consider the closed-loop mode for these stages)
            final_res_list_dict, bottle_state, mugcup_state, img_vis_cv2 = processing_test_image_full_stage_rarg_t2_drinking(
                left_image_test.copy(), cls_name, cls_name1, right_image_test.copy())
            if is_task_finished == False:
                if bottle_state == "C": subtask_name = "flatting"; is_task_finished = False
                if bottle_state == "B": subtask_name = "reorient"; is_task_finished = False
                if bottle_state == "A":
                    if mugcup_state == "C": subtask_name = "flipping"; is_task_finished = False
                    if mugcup_state == "B": subtask_name = "grasping"; is_task_finished = False
                    if mugcup_state == "A": subtask_name = "unscrew-pouring"; is_task_finished = True  # True for debug; Fasle for demo
        #####==================================================

        ##########################
        saved_frame_count += 1
        new_img_tag = f"frame{str(saved_frame_count).zfill(2)}_gid{str(global_substep_id).zfill(2)}_vid{str(video_frame_index).zfill(2)}.jpg"
        img_vis_cv2 = img_vis_cv2.astype(np.uint8)
        #####==================================================
        if rearrangement_task_name == "rarg_t1_dining":
            if ("ppspoon" not in subtask_name and "ppfork" not in subtask_name):
                obj_bbox = final_res_list_dict[cls_name][1]
                cv2.putText(img_vis_cv2, cur_state, (int(obj_bbox[0]), int(obj_bbox[1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
            elif is_task_finished == False:
                if cur_state in ["A2", "B2", "C2"]:  # for the spoon
                    obj_bbox = final_res_list_dict[cls_name1][1]
                    cv2.putText(img_vis_cv2, cur_state, (int(obj_bbox[0]), int(obj_bbox[1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
                if cur_state in ["A2", "B2", "D2"]:  # for the fork
                    obj_bbox = final_res_list_dict[cls_name2][1]
                    cv2.putText(img_vis_cv2, cur_state, (int(obj_bbox[0]), int(obj_bbox[1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
            else: print("[closed-loop + open-loop] all finished!!!")
        #####==================================================
        if rearrangement_task_name == "rarg_t2_drinking":
            obj_bbox = final_res_list_dict[cls_name][1]
            cv2.putText(img_vis_cv2, bottle_state, (int(obj_bbox[0]), int(obj_bbox[1])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
            obj_bbox = final_res_list_dict[cls_name1][1]
            cv2.putText(img_vis_cv2, mugcup_state, (int(obj_bbox[0]), int(obj_bbox[1])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
        #####==================================================
        cv2.imwrite(os.path.join(video_output_path_dir, new_img_tag), img_vis_cv2)
        ##########################

        ###################################################################################
        if is_task_finished:
            thread_list = []  # for saving multiple threadings for dual-arm coordination
            thread = threading.Thread(target=drive_single_arm_to_run_slim, args=([robot_init_pose_L], "L", robot_L, robot_R, ))
            thread_list.append(thread); thread.start()
            thread = threading.Thread(target=drive_single_arm_to_run_slim, args=([robot_init_pose_R], "R", robot_L, robot_R, ))
            thread_list.append(thread); thread.start()
            for thread in thread_list: thread.join()  # wait all threading to finish

            #===================================
            if os.path.exists(save_temp_for_cap_path): os.remove(save_temp_for_cap_path)
            #===================================
            print("All has been done! Quit.")

            os._exit(0); sys.exit(0)  # the former often works better than the latter
        else:
            global_substep_id += 1    
        args.task_name = subtask_name
        ###################################################################################
        task_para_list, cam_para_list, action_para_list, loop_para_list = init_parameters_robots_grippers(args, is_reinit=True)  # without robots_grippers_list
        [saved_seed_img_path, seed_ref_pts_dict, test_ref_pts_dict, prev_delta_rot, prev_delta_xy_list, detseg_prompts, supp_prompts] = task_para_list
        [robot_init_pose_L, robot_init_gripper_L, robot_init_pose_R, robot_init_gripper_R] = action_para_list
        [cl_robot_arms, cl_eef_pose_seqs, robot_arms, eef_pose_seqs, interp_num] = loop_para_list
        interp_num = global_interp_num

        if cur_gripper_L != robot_init_gripper_L:
            if robot_init_gripper_L == 0 or robot_init_gripper_L == 1: gripper_L.set_pos(int(255*robot_init_gripper_L))
            else: gripper_L.set_pos(255 - int(255*robot_init_gripper_L)); robot_init_gripper_L = -1  # do not change it anymore 
        if cur_gripper_R != robot_init_gripper_R:
            if robot_init_gripper_R == 0 or robot_init_gripper_R == 1: gripper_R.set_pos(int(255*robot_init_gripper_R))
            else: gripper_R.set_pos(255 - int(255*robot_init_gripper_R)); robot_init_gripper_R = -1  # do not change it anymore
        cur_gripper_L, cur_gripper_R = robot_init_gripper_L, robot_init_gripper_R
        reprojected_seeding_pts, left_image_vis, result_img_init = processing_seed_image(args, saved_seed_img_path, detseg_prompts, supp_prompts)

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
                    if not args.no_video_out: vout.write(final_imgs)
            ##########################
            saved_frame_count += 1
            new_img_tag = f"frame{str(saved_frame_count).zfill(2)}_gid{str(global_substep_id).zfill(2)}_vid{str(video_frame_index).zfill(2)}.jpg"
            cv2.imwrite(os.path.join(video_output_path_dir, new_img_tag), final_imgs)
            ##########################
            try:
                pt_seeding, pt_testing = reprojected_seeding_pts[step_id_record], reprojected_testing_pts[step_id_record]
                [p_delta_x, p_delta_y, prev_bbox] = prev_delta_xy_list[step_id_record]
            except:
                if (cl_substep_id_record >= interp_num-2) and (args.task_name in ["inserting"]):  # for the lost-detected marker pen
                    pt_seeding = reprojected_seeding_pts[step_id_record]
                    [p_delta_x, p_delta_y, prev_bbox] = prev_delta_xy_list[step_id_record]
                    pt_testing = [p_delta_x+pt_seeding[0], p_delta_y+pt_seeding[1], prev_delta_rot+pt_seeding[2], prev_bbox]
                elif step_id_record == 1 and (args.task_name in ["pivoting", "wrapping", 
                    "pivoting_rectbox", "pivoting_cirbowl", "flipping_basket", "flipping_block", "grasping_basket", 
                    "pivoting_bigjar", "pivoting_block", "toppling_holder", "toppling_bigjar", 
                    "bilifting_bigjar", "bilifting_block"]):  # we only have one object for the second arm
                    pt_seeding = reprojected_seeding_pts[0]
                    [p_delta_x, p_delta_y, prev_bbox] = prev_delta_xy_list[0]
                    pt_testing = [p_delta_x+pt_seeding[0], p_delta_y+pt_seeding[1], prev_delta_rot+pt_seeding[2], prev_bbox]
                else: 
                    #print("ERROR:\t", video_frame_index, reprojected_seeding_pts, reprojected_testing_pts); os._exit(0)
                    print("ERROR:\t", video_frame_index. reprojected_seeding_pts, reprojected_testing_pts); continue

            delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
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
            if args.task_name in ["reorient", "grasping", "flatting", "flipping", "flatting-reorient", "flipping-grasping", 
                "grasping_rectbox", "grasping_cirbowl", "grasping_holder", "grasping_pencup", "ppspoon", "ppfork" ]:

                if args.task_name in ["reorient", "grasping", "flatting", "flatting-reorient", 
                    "grasping_rectbox", "grasping_cirbowl", "grasping_holder", "grasping_pencup", "ppspoon", "ppfork" ]:
                    robot_arm_temp = "R" if delta_x >= 0 else "L"

                # if args.task_name in ["flipping", "flipping-grasping"]: robot_arm_temp = "R" if pt_testing[2] < 90 else "L"  # not stable currently
                if args.task_name in ["flipping", "flipping-grasping"]: robot_arm_temp = "R" if delta_x >= 0 else "L"
                if args.task_name in ["flipping", "flipping-grasping"]: delta_rot_z = 0  # do not using this to adjust eep of mug

                if robot_arm_temp == "R":  # using armR to manipulate
                    # if not args.is_syn: robot_L.move_to_a_waypoint(robot_init_pose_L, is_waiting=False)
                    if not args.is_syn: robot_L.move_to_a_waypoint(robot_temp_pose_L, is_waiting=False)
                    eef_pose_temp = cl_eef_pose_seqs[1]; step_id_record = 0
                if robot_arm_temp == "L":  # using armL to manipulate
                    # if not args.is_syn: robot_R.move_to_a_waypoint(robot_init_pose_R, is_waiting=False)
                    if not args.is_syn: robot_R.move_to_a_waypoint(robot_temp_pose_R, is_waiting=False)
                    eef_pose_temp = cl_eef_pose_seqs[0]; step_id_record = 0
            ##########################
            # get end_eef_pose
            cur_end_eef_pose = eef_pose_temp.copy()
            if robot_arm_temp == "L": cur_end_eef_pose[0] -= (delta_y / 1000); cur_end_eef_pose[2] -= (delta_x / 1000)
            if robot_arm_temp == "R": cur_end_eef_pose[0] -= (delta_y / 1000); cur_end_eef_pose[2] += (delta_x / 1000)
            cur_end_eef_pose[4] -= delta_rot_z

            # get start_eef_pose
            # if robot_arm_temp == "L": robotBase = robot_L; m_gripper = gripper_L; init_3d_pose = robot_init_pose_L[3:6]
            # if robot_arm_temp == "R": robotBase = robot_R; m_gripper = gripper_R; init_3d_pose = robot_init_pose_R[3:6]
            if robot_arm_temp == "L": robotBase = robot_L; m_gripper = gripper_L; init_3d_pose = robot_temp_pose_L[3:6]
            if robot_arm_temp == "R": robotBase = robot_R; m_gripper = gripper_R; init_3d_pose = robot_temp_pose_R[3:6]
            cur_start_eef_pose, _ = robotBase.get_current_pose(); cur_eef_pose_temp = cur_start_eef_pose.copy()
            if cl_substep_id_record != interp_num-1:
                if robot_arm_temp == "L": cur_start_eef_pose[1] -= 0.060  # move robot higher slightly to avoid hitting objects 
                if robot_arm_temp == "R": cur_start_eef_pose[1] += 0.060  # move robot higher slightly to avoid hitting objects 
            if cl_substep_id_record == 1:  # object is moved backward (not forward)
                cur_start_eef_pose[2] = cur_end_eef_pose[2] + 0.120  # move robot back slightly to avoid hitting objects
                cur_start_eef_pose[0] = cur_end_eef_pose[0] + 0.000  # directly align the x-value of these two keyposes
            ##########################
            if args.task_name in ["reorient", "ppspoon", "ppfork", "ppspoon-ppfork", "ppfork-ppspoon"]: # avoid the singular pose.
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
                if not os.path.exists(save_temp_for_cap_path):
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
                "pivoting_bigjar", "pivoting_block", "toppling_holder", "toppling_bigjar", "bilifting_bigjar", "bilifting_block"]:
                # the right arm should always be open-loop for non-prehensile tasks!!!"
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
                    "pivoting_bigjar", "pivoting_block", "toppling_holder", "toppling_bigjar", "bilifting_bigjar", "bilifting_block"]:  # pre-grasp
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
                        "grasping_rectbox", "grasping_cirbowl", "grasping_holder", "grasping_pencup", "ppspoon", "ppfork"]:
                    print("[***Close-Loop***] finished!!!"); break  # stop the close-loop step or steps
            ##########################


        if args.no_interaction:
            #===================================
            if not os.path.exists(save_temp_for_cap_path):
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
            # robot_init_pose_L, robot_init_pose_R, cur_gripper_L, cur_gripper_R,
            robot_temp_pose_L, robot_temp_pose_R, cur_gripper_L, cur_gripper_R,
            robot_arm_temp, video_frame_index, thread_list, vout=vout_main)

        ###################################################################################

        if rearrangement_task_name == "rarg_t1_dining":
            if ("ppspoon" in args.task_name) or ("ppfork" in args.task_name): is_task_finished = True

        if rearrangement_task_name == "rarg_t2_drinking":
            if ("unscrew" in args.task_name) or ("pouring" in args.task_name): is_task_finished = True

        ###################################################################################


