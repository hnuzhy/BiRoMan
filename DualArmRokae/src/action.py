
import os
import sys
import cv2
import time
import threading
from copy import copy

import kingfisher
sys.path.insert(0, os.getcwd())

from src.config import cfg_dict_init as cfg_dict
from src.util import interpolate_se3_bspline_startend_poses as interpolate_pose_t1
from src.BiNoMaP import pivot_around_spindle
from src.BiNoMaP import wrapping_with_coordination

#gripper_wait_time = 0.5  # for the "from gripper.demo_jodell_rg75 import JodellRG75" API
gripper_wait_time = 0.1  # for the "from gripper.demo_jodell_rg75_v2 import JodellRG75" API
cv2_font = cv2.FONT_HERSHEY_SIMPLEX

###################################################################################
global g_robot_L, g_robot_R, g_gripper_L, g_gripper_R, cur_gripper_L, cur_gripper_R

def drive_single_arm_to_run(target_pose_list, arm_type, sleep_time, gripper_value):
    if arm_type == "L":
        global g_robot_L, g_gripper_L, cur_gripper_L
        time.sleep(sleep_time); g_robot_L.move_by_trajectory(target_pose_list)
        if gripper_value != -1 and cur_gripper_L != gripper_value:  # gripper = 1 close; gripper = 0 open
            g_gripper_L.switch(gripper_value, True); time.sleep(gripper_wait_time); cur_gripper_L = gripper_value            
    if arm_type == "R":
        global g_robot_R, g_gripper_R, cur_gripper_R
        time.sleep(sleep_time); g_robot_R.move_by_trajectory(target_pose_list)
        if gripper_value != -1 and cur_gripper_R != gripper_value:  # gripper = 1 close; gripper = 0 open
            g_gripper_R.switch(gripper_value, True); time.sleep(gripper_wait_time); cur_gripper_R = gripper_value 

###################################################################################

def actions_for_real_robot_rollouts(
    args, robot_arms, eef_pose_seqs, robot_L, robot_R, gripper_L, gripper_R,
    robot_init_pose_L, robot_init_pose_R, robot_cur_gripper_L, robot_cur_gripper_R,
    robot_arm_temp, video_frame_index, thread_list, vout=None, is_back_home=True):
    # thread_list = [] is for saving multiple threadings for dual-arm coordination

    global g_robot_L, g_robot_R, g_gripper_L, g_gripper_R, cur_gripper_L, cur_gripper_R
    g_robot_L, g_robot_R, g_gripper_L, g_gripper_R = robot_L, robot_R, gripper_L, gripper_R
    cur_gripper_L, cur_gripper_R = robot_cur_gripper_L, robot_cur_gripper_R

    ############################################################################
    target_pose_list = []  # for saving some continous keyposes for smooth moving
    target_pose_list_L, target_pose_list_R = [], []  # dual-arm coordination skills (pivoting, wrapping, toppling, etc.)
    contact_pos_deg_L, contact_pos_deg_R = [], []
    for step_id, (robot_arm, eef_pose_seq) in enumerate(zip(robot_arms, eef_pose_seqs)):
        target_pose, gripper = eef_pose_seq[:6], eef_pose_seq[-1]
        print("\n", step_id, "robot:", robot_arm, "\t", "gripper:", gripper, "\t", "eef_pose:", target_pose)
        #######################################
        if args.debug_close_loop_vis and (not args.no_video_out) and (not args.no_interaction):
            left_image_test, _ = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
            video_frame_index += 1; text_str = f'Frame Index {video_frame_index}'
            cv2.polylines(left_image_test, [cfg_dict['pts_polygon']], isClosed=True, color=(255, 0, 0), thickness=2)
            left_image_test = cv2.resize(left_image_test[:, 65:960], (1253, 756)) # (960, 540) -> (895, 540) -> (1790, 1080) --> (1253, 756)
            cv2.putText(left_image_test, text_str, (5, 35), fontFace=cv2_font, fontScale=0.9, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
            vout.write(left_image_test)
        #######################################

        #####====================================================================
        if args.task_name == "pouring":
            if step_id in [0, 1]:
                if step_id == 0:  # for armR
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 1:  # for armL
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; continue
            if step_id in [2, 3, 4]:
                if step_id == 2:  # for armR
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id in [3, 4]:  # for armL. preparing + pouring steps
                    target_pose_list.append(target_pose)
                    if step_id != 4: continue
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list, "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list = []; continue
            if step_id in [5, 6, 7]:
                if step_id in [5, 6]:  # for armL. reorient-back + place-down steps
                    target_pose_list.append(target_pose)
                    if step_id != 6: continue
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list, "L", 2.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 7:  # for armR
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 3.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list = []; continue

        if args.task_name == "unscrew":
            if step_id in [0, 1]:
                if step_id == 0:  # for armL
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 1:  # for armR
                    target_pose_higher = target_pose.copy(); target_pose_higher[1] += 0.050  # enlarge the y-value
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose_higher, target_pose], "R", 0.2, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; continue
            if step_id == 8:  # pull the bottle cap using the right arm (set a new checking-regrasping-pulling loop)
                cur_pos_deg_R, _ = robot_R.get_current_pose()  # get the pregrasp pose
                pos_deg_R_untwist = cur_pos_deg_R.copy(); pos_deg_R_untwist[4] += 30  # still try to untwist the cap
                robot_R.move_to_a_waypoint(target_pose)  # the pull up pose/action
                g_gripper_R.switch(1, True); temp_gR_value = g_gripper_R.get_pos()  # close gripper; 0 ~ 255, 0-->fully open; 255-->fully closed
                while temp_gR_value > 256-32:  # if the gripper can be fully closed, we think that the bottle cap is clamped out of the bottle
                    g_gripper_R.switch(0, True); robot_R.move_to_a_waypoint(pos_deg_R_untwist)  # open gripper; regrasp-untwist action
                    g_gripper_R.switch(1, True); robot_R.move_by_trajectory([cur_pos_deg_R, target_pose])  # close gripper; the pull up pose/action
                    g_gripper_R.switch(1, True); temp_gR_value = g_gripper_R.get_pos()  # close gripper; 0 ~ 255, 0-->fully open; 255-->fully closed
                cur_gripper_R = gripper; continue 
            if step_id in [9, 10, 11]:  # [8, 9, 10, 11] --> [9, 10, 11]
                if step_id in [9, 10]:  # the armR pull-lift-place the cap steps, finally open the gripper, [8, 9, 10] --> [9, 10]
                    target_pose_list.append(target_pose)
                    if step_id != 10: continue
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list, "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 11:
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.5, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list = []; continue

        if args.task_name == "reorient":
            if robot_arm != robot_arm_temp: continue  # single arm task, the other arm will keep static
            if robot_arm_temp == "L" and step_id in [2, 4, 6]:  # armL lift-reorient-place steps, finally open gripper
                if step_id == 2: poses_list = interpolate_pose_t1(eef_pose_seqs[0][:6], target_pose, num_steps=5); target_pose_list += poses_list
                else: target_pose_list.append(target_pose)
                if step_id != 6: continue
                robot_L.move_by_trajectory(target_pose_list); gripper_L.switch(0, True); target_pose_list = []; continue
            if robot_arm_temp == "R" and step_id in [3, 5, 7]:  # armR lift-reorient-place steps, finally open gripper
                if step_id == 3: poses_list = interpolate_pose_t1(eef_pose_seqs[1][:6], target_pose, num_steps=5); target_pose_list += poses_list
                else: target_pose_list.append(target_pose)
                if step_id != 7: continue
                robot_R.move_by_trajectory(target_pose_list); gripper_R.switch(0, True); target_pose_list = []; continue
            if (robot_arm_temp == "L" and step_id == 8): robot_L.move_by_trajectory([target_pose, robot_init_pose_L]); continue
            if (robot_arm_temp == "R" and step_id == 9): robot_R.move_by_trajectory([target_pose, robot_init_pose_R]); continue
            
        if args.task_name == "grasping":
            if robot_arm != robot_arm_temp: continue  # single arm task, the other arm will keep static
            if robot_arm_temp == "L" and step_id in [2, 4, 6]:  # armL lift-reorient-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 6: continue
                time.sleep(gripper_wait_time); robot_L.move_by_trajectory(target_pose_list); gripper_L.switch(0, True); target_pose_list = []; continue
            if robot_arm_temp == "R" and step_id in [3, 5, 7]:  # armR lift-reorient-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 7: continue
                time.sleep(gripper_wait_time); robot_R.move_by_trajectory(target_pose_list); gripper_R.switch(0, True); target_pose_list = []; continue

        if args.task_name == "flatting":
            if robot_arm != robot_arm_temp: continue  # single arm task, the other arm will keep static
            if robot_arm_temp == "L" and step_id in [2, 4]:  # armL move&rotate-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 4: continue
                time.sleep(gripper_wait_time); robot_L.move_by_trajectory(target_pose_list); gripper_L.switch(0, True); target_pose_list = []; continue
            if robot_arm_temp == "R" and step_id in [3, 5]:  # armR move&rotate-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 5: continue
                time.sleep(gripper_wait_time); robot_R.move_by_trajectory(target_pose_list); gripper_R.switch(0, True); target_pose_list = []; continue
            if (robot_arm_temp == "L" and step_id == 6): robot_L.move_by_trajectory([target_pose, robot_init_pose_L]); continue
            if (robot_arm_temp == "R" and step_id == 7): robot_R.move_by_trajectory([target_pose, robot_init_pose_R]); continue

        if args.task_name == "flipping":
            if robot_arm != robot_arm_temp: continue  # single arm task, the other arm will keep static
            if robot_arm_temp == "L" and step_id in [2, 4, 6]:  # armL move&rotate-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 6: continue
                time.sleep(gripper_wait_time); robot_L.move_by_trajectory(target_pose_list); gripper_L.switch(0, True); target_pose_list = []; continue
            if robot_arm_temp == "R" and step_id in [3, 5, 7]:  # armR move&rotate-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 7: continue
                time.sleep(gripper_wait_time); robot_R.move_by_trajectory(target_pose_list); gripper_R.switch(0, True); target_pose_list = []; continue
            if (robot_arm_temp == "L" and step_id == 8): robot_L.move_by_trajectory([target_pose, robot_init_pose_L]); continue
            if (robot_arm_temp == "R" and step_id == 9): robot_R.move_by_trajectory([target_pose, robot_init_pose_R]); continue
        #####====================================================================

        #####====================================================================
        if args.task_name in ["grasping_rectbox", "grasping_cirbowl", "grasping_holder", "grasping_pencup"]:
            if robot_arm != robot_arm_temp: continue  # single arm task, the other arm will keep static
            if robot_arm_temp == "L" and step_id in [2, 4, 6]:  # armL lift-reorient-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 6: continue
                time.sleep(gripper_wait_time); robot_L.move_by_trajectory(target_pose_list); gripper_L.switch(gripper, True); target_pose_list = []; continue
            if robot_arm_temp == "R" and step_id in [3, 5, 7]:  # armR lift-reorient-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 7: continue
                time.sleep(gripper_wait_time); robot_R.move_by_trajectory(target_pose_list); gripper_R.switch(gripper, True); target_pose_list = []; continue
            if (robot_arm_temp == "L" and step_id == 8): robot_L.move_by_trajectory([target_pose, robot_init_pose_L]); continue
            if (robot_arm_temp == "R" and step_id == 9): robot_R.move_by_trajectory([target_pose, robot_init_pose_R]); continue

        if args.task_name == "grasping_basket":
            if step_id in [0, 1]:
                if step_id == 0:  # for armL (start grasp)
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 1:  # for armR (start grasp)
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; continue
            if step_id in [2, 3, 4, 5, 6, 7]:
                if step_id in [2, 4, 6]: target_pose_list_L.append(target_pose)  # for armL (lift-up --> move-to-top --> place-down)
                if step_id in [3, 5, 7]: target_pose_list_R.append(target_pose)  # for armR (lift-up --> move-to-top --> place-down)
                if step_id != 7: continue  # the final step
                else:
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_L, "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_R, "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list_L = []; target_pose_list_R = []; continue
            if step_id in [8, 9]:
                if step_id == 8:  # for armL (start ungrasp)
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose, robot_init_pose_L], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 9:  # for armR (start ungrasp)
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose, robot_init_pose_R], "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; continue
        #####====================================================================

        #####====================================================================
        if args.task_name == "unscrew-pouring":
            if step_id in [0, 1]:
                if step_id == 0:  # for armL
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 1:  # for armR
                    target_pose_higher = target_pose.copy(); target_pose_higher[1] += 0.050  # enlarge the y-value
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose_higher, target_pose], "R", 0.2, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; continue
            if step_id == 8:  # pull the bottle cap using the right arm (set a new checking-regrasping-pulling loop)
                cur_pos_deg_R, _ = robot_R.get_current_pose()  # get the pregrasp pose
                pos_deg_R_untwist = cur_pos_deg_R.copy(); pos_deg_R_untwist[4] += 30  # still try to untwist the cap
                robot_R.move_to_a_waypoint(target_pose)  # the pull up pose/action
                g_gripper_R.switch(1, True); temp_gR_value = g_gripper_R.get_pos()  # close gripper; 0 ~ 255, 0-->fully open; 255-->fully closed
                while temp_gR_value > 256-32:  # if the gripper can be fully closed, we think that the bottle cap is clamped out of the bottle
                    g_gripper_R.switch(0, True); robot_R.move_to_a_waypoint(pos_deg_R_untwist)  # open gripper; regrasp-untwist action
                    g_gripper_R.switch(1, True); robot_R.move_by_trajectory([cur_pos_deg_R, target_pose])  # close gripper; the pull up pose/action
                    g_gripper_R.switch(1, True); temp_gR_value = g_gripper_R.get_pos()  # close gripper; 0 ~ 255, 0-->fully open; 255-->fully closed
                cur_gripper_R = gripper; continue 
            if step_id in [9, 10, 11]:  # [8, 9, 10, 11] --> [9, 10, 11]
                if step_id in [9, 10]:  # the armR pull-lift-place the cap steps, finally open the gripper, [8, 9, 10] --> [9, 10]
                    target_pose_list.append(target_pose)
                    if step_id != 10: continue
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list, "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 11:  # the armL lift-up the capless bottle step.
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.5, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list = []; continue
            if step_id == 12:  # for armR grasp the mugcup step
                target_pose_higher = target_pose.copy(); target_pose_higher[1] += 0.050; target_pose_higher[2] += 0.050  # enlarge the y/z values
                thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose_higher, target_pose], "R", 0.0, gripper, ))
                thread_list.append(thread); thread.start()
                for thread in thread_list: thread.join()  # wait all threading to finish
                thread_list = []; continue
            if step_id in [13, 14, 15]:
                if step_id == 13:  # for armR. align mugcup step
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id in [14, 15]:  # for armL. preparing + pouring steps
                    target_pose_list.append(target_pose)
                    if step_id != 15: continue
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list, "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list = []; continue
            if step_id in [16, 17, 18]:
                if step_id in [16, 17]:  # for armL. reorient-back + place-down steps
                    target_pose_list.append(target_pose)
                    if step_id != 17: continue
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list, "L", 2.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 18:  # for armR
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 3.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list = []; continue
        
        if args.task_name == "flatting-reorient":
            if robot_arm != robot_arm_temp: continue  # single arm task, the other arm will keep static
            if robot_arm_temp == "L" and step_id in [2, 4]:  # armL move&rotate-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 4: continue
                time.sleep(gripper_wait_time); robot_L.move_by_trajectory(target_pose_list); gripper_L.switch(0, True); target_pose_list = []; continue
            if robot_arm_temp == "R" and step_id in [3, 5]:  # armR move&rotate-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 5: continue
                time.sleep(gripper_wait_time); robot_R.move_by_trajectory(target_pose_list); gripper_R.switch(0, True); target_pose_list = []; continue
            if (robot_arm_temp == "L" and step_id in [6, 8, 10, 12]): 
                target_pose_list.append(target_pose)
                if step_id != 12: continue
                robot_L.move_by_trajectory(target_pose_list); gripper_L.switch(1, True); target_pose_list = []; continue
            if (robot_arm_temp == "R" and step_id in [7, 9, 11, 13]):
                target_pose_list.append(target_pose)
                if step_id != 13: continue
                robot_R.move_by_trajectory(target_pose_list); gripper_R.switch(1, True); target_pose_list = []; continue
            if robot_arm_temp == "L" and step_id in [14, 16, 18]:  # armL lift-reorient-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 18: continue
                robot_L.move_by_trajectory(target_pose_list); gripper_L.switch(0, True); target_pose_list = []; continue
            if robot_arm_temp == "R" and step_id in [15, 17, 19]:  # armR lift-reorient-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 19: continue
                robot_R.move_by_trajectory(target_pose_list); gripper_R.switch(0, True); target_pose_list = []; continue
            if (robot_arm_temp == "L" and step_id == 20): robot_L.move_by_trajectory([target_pose, robot_init_pose_L]); continue
            if (robot_arm_temp == "R" and step_id == 21): robot_R.move_by_trajectory([target_pose, robot_init_pose_R]); continue
        #####====================================================================

        #####====================================================================
        if args.task_name == "inserting":
            if step_id in [0, 1, 2, 3]:
                if step_id == 0:  # for armR
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 0.5, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id in [1, 2, 3]:  # for armL. move & rotate + lift-up  --> move & rotate + reorient --> place-down
                    target_pose_list.append(target_pose)
                    if step_id != 3: continue
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list, "L", 0.5, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list = []; continue
            if step_id in [4, 5]:
                if step_id == 4:  # for armR
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 5:  # for armL
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; continue

        if args.task_name == "plugpen":
            if step_id in [0, 1, 2, 3]:
                if step_id in [1, 3]: target_pose_list_L.append(target_pose)  # for armL (lift-up --> approach-to-another-arm)
                if step_id in [0, 2]: target_pose_list_R.append(target_pose)  # for armR (lift-up --> approach-to-another-arm)
                if step_id != 3: continue  # the final step
                else:
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_L, "L", 0.0, 1, ))  # close gripper
                    thread_list.append(thread); thread.start()
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_R, "R", 0.0, 1, ))  # close gripper 
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list_L = []; target_pose_list_R = []
                    gripper_R.switch(0, True); cur_gripper_R = 0; time.sleep(gripper_wait_time); continue  # wait the armR opening its gripper
            if step_id in [4, 5]:
                if step_id == 4:  # for armR
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 5:  # for armL
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; continue

        if args.task_name == "handover":
            if step_id in [1, 2, 3, 4]:
                if step_id in [3, 4]: target_pose_list_L.append(target_pose)  # for armL (approach to object / left-arm --> grasp)
                if step_id in [1, 2]: target_pose_list_R.append(target_pose)  # for armR (lift-up + reorient --> move + reorient)
                if step_id != 4: continue  # the final step
                else:
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_L, "L", 1.0, 1, ))  # close gripper (1.0 seconds latency)
                    thread_list.append(thread); thread.start()
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_R, "R", 0.0, 1, ))  # close gripper 
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list_L = []; target_pose_list_R = []; continue
            if step_id in [6, 7, 8]:  # armR move-back; armL move & rotate + place-down
                if step_id == 6:  # for armR
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose, robot_init_pose_R], "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id in [7, 8]:  # for armL.
                    target_pose_list.append(target_pose)
                    if step_id != 8: continue
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list, "L", 0.5, gripper, ))
                    thread_list.append(thread); thread.start()  
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list = []; continue
                
        if args.task_name in ["ppspoon", "ppfork"]:
            if robot_arm != robot_arm_temp: continue  # single arm task, the other arm will keep static
            if robot_arm_temp == "L" and step_id in [2, 4, 6]:  # armL lift-reorient-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 6: continue
                time.sleep(gripper_wait_time); robot_L.move_by_trajectory(target_pose_list); gripper_L.switch(gripper, True); target_pose_list = []; continue
            if robot_arm_temp == "R" and step_id in [3, 5, 7]:  # armR lift-reorient-place steps, finally open gripper
                target_pose_list.append(target_pose)
                if step_id != 7: continue
                time.sleep(gripper_wait_time); robot_R.move_by_trajectory(target_pose_list); gripper_R.switch(gripper, True); target_pose_list = []; continue
        
        if args.task_name in ["ppspoon-ppfork", "ppfork-ppspoon"]:
            if step_id in [0, 1]:
                if step_id == 0:  # for armL (start grasp)
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 1:  # for armR (start grasp)
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; continue
            if step_id in [2, 3, 4, 5, 6, 7]:
                if step_id in [2, 4, 6]: target_pose_list_L.append(target_pose)  # for armL (lift-up --> move-to-top --> place-down)
                if step_id in [3, 5, 7]: target_pose_list_R.append(target_pose)  # for armR (lift-up --> move-to-top --> place-down)
                if step_id != 7: continue  # the final step
                else:
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_L, "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_R, "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list_L = []; target_pose_list_R = []; continue

        #####====================================================================

        #####====================================================================
        if "pivoting" in args.task_name:
            if args.task_name == "pivoting" or args.task_name == "pivoting_cirbowl": gripper_offset = 0.012
            if args.task_name == "pivoting_rectbox": gripper_offset = -0.008
            if args.task_name == "pivoting_bigjar" or args.task_name == "pivoting_block": gripper_offset = -0.005
            if step_id == 0:
                contact_pos_deg_L = target_pose; target_pose_list_L.append(target_pose); continue  # for armL (start contact)
            if step_id == 1:
                contact_pos_deg_R = target_pose; target_pose_list_R.append(target_pose)  # for armR (start contact)
                pivoting_eep_list = pivot_around_spindle(contact_pos_deg_L, contact_pos_deg_R, gripper_offset=gripper_offset)  # pivoting actions
                if args.task_name == "pivoting" or args.task_name == "pivoting_cirbowl":  # post-action: push for flipping
                    armL_latency = 0.0
                    target_pose_R1 = copy(pivoting_eep_list[-1]); target_pose_R1[1] += 0.003; target_pose_R1[2] -= 0.060
                    target_pose_R2 = copy(pivoting_eep_list[-1]); target_pose_R2[1] += 0.043; target_pose_R2[2] -= 0.080
                    target_pose_list_R += (pivoting_eep_list + [target_pose_R1, target_pose_R2])
                if args.task_name == "pivoting_rectbox":
                    armL_latency = 0.0
                    target_pose_list_R += pivoting_eep_list
                if args.task_name == "pivoting_bigjar" or args.task_name == "pivoting_block":
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([contact_pos_deg_L], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([contact_pos_deg_R], "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list_L = []; target_pose_list_R = []
                    if args.task_name == "pivoting_bigjar":     armL_latency = 1.0  # this time latency is very important
                    if args.task_name == "pivoting_block":      armL_latency = 0.5  # this time latency is very important
                    target_pose_L1 = copy(contact_pos_deg_L); target_pose_L1[2] += 0.100  # move back the left-arm slightly
                    target_pose_list_L += [target_pose_L1, robot_init_pose_L]
                    if args.task_name == "pivoting_bigjar" and args.bigjar_id == 3:  # this is for paper rebuttal using
                        armL_latency = 1.3  # update the time latency
                        target_pose_R1 = copy(pivoting_eep_list[-1]); target_pose_R1[1] += 0.003; target_pose_R1[2] -= 0.030
                        target_pose_list_R += (pivoting_eep_list + [target_pose_R1, robot_init_pose_R])  # update the target_pose_list_R
                    else:
                        target_pose_list_R += (pivoting_eep_list + [robot_init_pose_R])
                # conduct the dual-arm coordination for pivoting
                thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_L, "L", armL_latency, gripper, ))
                thread_list.append(thread); thread.start()
                thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_R, "R", 0.0, gripper, ))
                thread_list.append(thread); thread.start()
                for thread in thread_list: thread.join()  # wait all threading to finish
                thread_list = []; target_pose_list_L = []; target_pose_list_R = []; continue

        if args.task_name == "wrapping" or args.task_name == "flipping_basket" or args.task_name == "flipping_block":
            if step_id in [0, 2]:  # for armL (start contact + pushing-lifting-wrapping)
                if step_id == 0: contact_pos_deg_L = target_pose; target_pose_list_L.append(target_pose) # the contact eep_pose
                if step_id == 2: target_pose_list_L += wrapping_with_coordination(contact_pos_deg_L, target_pose, leftarm=True)  # wrapping_pose_list_L
            if step_id in [1, 3]:  # for armR (start contact + pushing-lifting-wrapping)
                if step_id == 1: contact_pos_deg_R = target_pose; target_pose_list_R.append(target_pose)  # the contact eep_pose
                if step_id == 3: target_pose_list_R += wrapping_with_coordination(contact_pos_deg_R, target_pose, leftarm=False)  # wrapping_pose_list_R
            if step_id != 3: continue  # the final step
            else:
                target_pose_list_L.append(robot_init_pose_L)
                thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_L, "L", 0.0, gripper, ))
                thread_list.append(thread); thread.start()
                target_pose_list_R.append(robot_init_pose_R)
                thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_R, "R", 0.0, gripper, ))
                thread_list.append(thread); thread.start()
                for thread in thread_list: thread.join()  # wait all threading to finish
                thread_list = []; target_pose_list_L = []; target_pose_list_R = []; continue

        if args.task_name == "toppling_holder" or args.task_name == "toppling_bigjar":
            if args.task_name == "toppling_holder": armL_latency = 0.3
            if args.task_name == "toppling_bigjar": armL_latency = 0.1
            if step_id % 2 == 0: target_pose_list_L.append(target_pose)  # step_id in [0, 2, 4], armL (contact --> static --> back)
            if step_id % 2 == 1: target_pose_list_R.append(target_pose)  # step_id in [1, 3, 5], armR (contact --> toppling --> back)
            if step_id != 5: continue  # the final step
            else:
                target_pose_list_L.append(robot_init_pose_L)
                thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_L, "L", armL_latency, gripper, ))
                thread_list.append(thread); thread.start()
                target_pose_list_R.append(robot_init_pose_R)
                thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_R, "R", 0.0, gripper, ))
                thread_list.append(thread); thread.start()
                for thread in thread_list: thread.join()  # wait all threading to finish
                thread_list = []; target_pose_list_L = []; target_pose_list_R = []; continue

        if args.task_name == "bilifting_bigjar" or args.task_name == "bilifting_block":
            if step_id % 2 == 0: target_pose_list_L.append(target_pose)  # step_id in [0, 2, 4, 6], armL (contact --> lift --> move -> place)
            if step_id % 2 == 1: target_pose_list_R.append(target_pose)  # step_id in [1, 3, 5, 7], armR (contact --> lift --> move -> place)
            if step_id != 7: continue  # the final step
            else:
                target_pose_list_L.append(robot_init_pose_L)
                thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_L, "L", 0.0, gripper, ))
                thread_list.append(thread); thread.start()
                target_pose_list_R.append(robot_init_pose_R)
                thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_R, "R", 0.0, gripper, ))
                thread_list.append(thread); thread.start()
                for thread in thread_list: thread.join()  # wait all threading to finish
                thread_list = []; target_pose_list_L = []; target_pose_list_R = []; continue

        #####====================================================================

        #####====================================================================
        if args.task_name in ["penbagzip"]:
            if step_id in [0, 1]:
                if step_id == 0:  # for armL. 
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 1:  # for armR.
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; continue 
            if step_id in [2, 3, 4, 5, 6]:
                if step_id in [2, 3]: target_pose_list_L.append(target_pose)  # for armL (grasp --> lift-up & rotate --> move & rotate)
                if step_id in [4, 5, 6]: target_pose_list_R.append(target_pose)  # for armR (grasp --> lift-up & rotate --> move & rotate --> place-down --> move back & pre-grasp)
                if step_id != 6: continue  # the final step
                else:
                    temp_gripper_L = 1; temp_gripper_R = 0  # do not use the gripper. because this stage the L and R are different
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_L, "L", 0.5, temp_gripper_L, ))
                    thread_list.append(thread); thread.start()
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_R, "R", 0.5, temp_gripper_R, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list_L = []; target_pose_list_R = []; continue
            if step_id in [7, 8, 9]:
                if step_id == 8:  # for armL. move & rotate (for adjusting the zipper)
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.5, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id in [7, 9]:  # for armR. grasp (for grasping the penbag)
                    target_pose_list.append(target_pose)
                    if step_id != 9: continue
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list, "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list = []; continue   
            # when step_id == 10, armL is keeping static (for open the gripper)
            if step_id in [11, 12]:
                if step_id == 11:  # for armL. move back (for adjusting the arm-L)
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 12:  # for armR. grasp move & rotate (for adjusting the zipper)
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 0.5, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; continue 
            # when step_id == 13, armL is grasping the zipper
            if step_id in [14, 15, 16]:
                target_pose_list_L.append(target_pose)
                if step_id != 16: continue  # the final step
                thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_L, "L", 0.0, gripper, ))
                thread_list.append(thread); thread.start()
                for thread in thread_list: thread.join()  # wait all threading to finish
                thread_list = []; target_pose_list_L = []; continue
            # when step_id == 17, armR is moving back & place-down the penbag

        #####====================================================================
        if args.task_name in [ "foldtowel", "foldpants", "foldshirt"]:
            if step_id in [0, 1]:
                if step_id == 0:  # for armL (start grasp)
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 1:  # for armR (start grasp)
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; continue
            if step_id in [2, 3, 4, 5, 6, 7, 8, 9]:
                if step_id in [2, 4, 6, 8]: target_pose_list_L.append(target_pose)  # armL (lift-up --> put-down --> lift-up --> put-down)
                if step_id in [3, 5, 7, 9]: target_pose_list_R.append(target_pose)  # armR (lift-up --> put-down --> lift-up --> put-down)
                if step_id != 9: continue  # the final step
                else:
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_L, "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list_R, "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list_L = []; target_pose_list_R = []; continue
        #####====================================================================
        if args.task_name in [ "coilcable", "coilrope", "coilbelt"]:
            if step_id in [2, 3]:
                if step_id == 2 and robot_arm == "R":  # for armR
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 0.2, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 3 and robot_arm == "L":  # for armL
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; continue
            if step_id in [7, 8, 9, 10]:
                if step_id == 8 and robot_arm == "L":  # for armL
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id in [7, 9, 10] and robot_arm == "R":  # for armR
                    target_pose_list.append(target_pose)
                    if step_id != 10: continue
                    thread = threading.Thread(target=drive_single_arm_to_run, args=(target_pose_list, "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; target_pose_list = []; continue
            if step_id in [13, 14]:
                if step_id == 13 and robot_arm == "L":  # for armL
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "L", 0.0, gripper, ))
                    thread_list.append(thread); thread.start(); continue
                if step_id == 14 and robot_arm == "R":  # for armR
                    thread = threading.Thread(target=drive_single_arm_to_run, args=([target_pose], "R", 0.0, gripper, ))
                    thread_list.append(thread); thread.start()
                    for thread in thread_list: thread.join()  # wait all threading to finish
                    thread_list = []; continue
        #####====================================================================
            
        ####################################### 
        if robot_arm == "L":
            robot_L.move_to_a_waypoint(target_pose)
            if cur_gripper_L != gripper:  # gripper = 1 close; gripper = 0 open
                gripper_L.switch(gripper, True); time.sleep(gripper_wait_time); cur_gripper_L = gripper
        #######################################
        if robot_arm == "R":
            robot_R.move_to_a_waypoint(target_pose)
            if cur_gripper_R != gripper:  # gripper = 1 close; gripper = 0 open
                gripper_R.switch(gripper, True); time.sleep(gripper_wait_time); cur_gripper_R = gripper
        #######################################

    ############################################################################
    if is_back_home:
        if args.task_name in ["reorient", "grasping", "flatting", "flipping", "flatting-reorient", "flipping-grasping",
            "grasping_rectbox", "grasping_cirbowl", "grasping_holder", "grasping_pencup", "ppspoon", "ppfork"]:   # single-arm
            if robot_arm_temp == "L": robot_L.move_to_a_waypoint(robot_init_pose_L)
            if robot_arm_temp == "R": robot_R.move_to_a_waypoint(robot_init_pose_R)
        if args.task_name in ["pouring", "unscrew", "unscrew-pouring", "inserting", "plugpen", "handover",
            "pivoting", "wrapping", "pivoting_rectbox", "pivoting_cirbowl", "flipping_basket", "flipping_block", "grasping_basket", 
            "pivoting_bigjar", "pivoting_block", "toppling_holder", "toppling_bigjar", "bilifting_bigjar", "bilifting_block", 
            "ppspoon-ppfork", "ppfork-ppspoon", "penbagzip", "foldtowel", "foldpants", "foldshirt", "coilcable", "coilrope", "coilbelt" ]:  # dual-arm
            thread = threading.Thread(target=drive_single_arm_to_run, args=([robot_init_pose_L], "L", 0.0, -1, ))
            thread_list.append(thread); thread.start()
            thread = threading.Thread(target=drive_single_arm_to_run, args=([robot_init_pose_R], "R", 0.0, -1, ))
            thread_list.append(thread); thread.start()
            for thread in thread_list: thread.join()  # wait all threading to finish

     ############################################################################
    if args.debug_close_loop_vis and (not args.no_video_out) and (not args.no_interaction):
        left_image_test, _ = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
        video_frame_index += 1; text_str = f'Frame Index {video_frame_index}'
        cv2.polylines(left_image_test, [cfg_dict['pts_polygon']], isClosed=True, color=(255, 0, 0), thickness=2)
        left_image_test = cv2.resize(left_image_test[:, 65:960], (1253, 756)) # (960, 540) -> (895, 540) -> (1790, 1080) --> (1253, 756)
        cv2.putText(left_image_test, text_str, (5, 35), fontFace=cv2_font, fontScale=0.9, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
        vout.write(left_image_test)    
        vout.release()  # Release everything ()

    return cur_gripper_L, cur_gripper_R, video_frame_index, thread_list
###################################################################################