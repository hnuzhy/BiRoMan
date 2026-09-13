
import sys
import os
import json
import time
import copy
import threading
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.getcwd())

from robots.demo_xMateCR7 import rokaeRobotBase
from gripper.demo_jodell_rg75_v2 import JodellRG75
from src.config import cfg_dict_init as cfg_dict
from src.util import tcp_fix_gripper

#########################################################################################################
if __name__ == "__main__":

    handeye_cam_L = cfg_dict["arm1"]["handeye_para"]
    handeye_cam_R = cfg_dict["arm2"]["handeye_para"]
    egoNMI_example_dir = "/home/dex/zhouhuayi/projects/HaWoR/example/EgotestVideos/"
    egoNMI_example_name = "2026-01-05_19-35-47-pick-place+clip_001/hand2gripper_labels.json"
    egoNMI_json_path = os.path.join(egoNMI_example_dir, egoNMI_example_name)
    with open(egoNMI_json_path, "r", encoding='utf-8') as f:
        egoNMI_example_dict = json.load(f)
    pose_cache_L_list = egoNMI_example_dict["handL"][8:-2]  # remove some unstable frames
    pose_cache_R_list = egoNMI_example_dict["handR"][8:-2]  # remove some unstable frames
    
    ################################################################################
    '''Method 1: Handeye-based 6-DoF Actions Transformation'''
    # pose_t_L_list, pose_t_R_list = [], []
    # for arm_type, pose_cache_list in [["L", pose_cache_L_list], ["R", pose_cache_R_list]]:
        # handeye_cam = handeye_cam_L if arm_type == "L" else handeye_cam_R
        # pose_t_list = pose_t_L_list if arm_type == "L" else pose_t_R_list
        # for idx, pose_vec in enumerate(pose_cache_list):
            # [g_rot, g_loc, g_state] = pose_vec
            # g_rot_arm = handeye_cam[:3, :3] @ np.array(g_rot)  # camera --> armL / armR
            # g_loc_arm = handeye_cam[:3, :3] @ np.array(g_loc) + handeye_cam[:3, -1]  # camera --> armL / armR
            # g_6dof_pose_arm = np.eye(4)
            # g_6dof_pose_arm[:3, :3] = g_rot_arm
            # g_6dof_pose_arm[:3, -1] = g_loc_arm[:3]
            # g_6dof_pose_arm = tcp_fix_gripper(g_6dof_pose_arm, is_back=False)  # is_back=False, add tool length
            # target_pose_back = [0,0,0, 0,0,0, g_state]  # position, rotation, gripper
            # target_pose_back[:3] = list(g_6dof_pose_arm[:3, -1])
            # target_pose_back[3:6] = Rotation.from_matrix(g_6dof_pose_arm[:3, :3]).as_euler("xyz", degrees=True)
            # pose_t_list.append(target_pose_back)
        # print("[Finished!]", len(pose_t_list), pose_t_list[:5])
    ################################################################################
    '''Method 2: Relative Trajectory Retargeting/Alignment'''
    pose_t_L_list, pose_t_R_list = [], []
    robot_start_pose_L = [0.558413994, 0.385476218, 0.402155227, -179.610881514, 75.474229046, -90.517829792]
    robot_start_pose_R = [0.558413994, -0.385476218, 0.402155227, 179.732957532, 75.68246746, 89.739817245]
    R_corr_grip_init = np.array([ [0, -1, 0], [1, 0, 0], [0, 0, 1] ])
    R_cam_to_torso = np.array([ [0, -1, 0], [0, 0, -1], [1, 0, 0] ])
    R_world_to_base_L = np.array([ [1, 0, 0], [0, -1, 0], [0, 0, -1] ]).T
    R_world_to_base_R = np.eye(3)
    scale_factor = 1.5
    for arm_type, pose_cache_list in [["L", pose_cache_L_list], ["R", pose_cache_R_list]]:
        pose_t_list = pose_t_L_list if arm_type == "L" else pose_t_R_list
        r_start_pose = robot_start_pose_L if arm_type == "L" else robot_start_pose_R
        R_world_to_base = R_world_to_base_L if arm_type == "L" else R_world_to_base_R
        R_corr_grip = R_corr_grip_init.T if arm_type == "L" else R_corr_grip_init
        r_start_pose_4x4 = np.eye(4)
        r_start_pose_4x4[:3, :3] = Rotation.from_euler("xyz", r_start_pose[3:], degrees=True).as_matrix()
        r_start_pose_4x4[:3, 3] = np.array(r_start_pose[:3])
        g_rot_0, g_loc_0 = np.array(pose_cache_list[0][0]), np.array(pose_cache_list[0][1]) @ R_cam_to_torso.T
        for idx, pose_vec in enumerate(pose_cache_list):
            [g_rot, g_loc, g_state, wrist_loc] = pose_vec  # using g_loc or wrist_loc
            delta_pos_world = (np.array(wrist_loc) @ R_cam_to_torso.T - g_loc_0) * scale_factor
            delta_pos_robot = np.dot(R_world_to_base, delta_pos_world)
            R_rel_human = np.dot(g_rot_0.T, np.array(g_rot))
            R_rel_robot = R_corr_grip @ R_rel_human @ R_corr_grip.T
            R_next = np.eye(4)
            R_next[:3, 3] = r_start_pose_4x4[:3, 3] + delta_pos_robot
            R_next[:3, :3] = np.dot(r_start_pose_4x4[:3, :3], R_rel_robot)
            R_next = tcp_fix_gripper(R_next, is_back=False)  # is_back=False, add tool length
            target_pose_back = [0,0,0, 0,0,0, g_state]  # position, rotation, gripper
            target_pose_back[:3] = list(R_next[:3, -1])
            target_pose_back[3:6] = Rotation.from_matrix(R_next[:3, :3]).as_euler("xyz", degrees=True)
            pose_t_list.append(target_pose_back)
        print("[Finished!]", len(pose_t_list), pose_t_list[:5])
    ################################################################################
    '''Method 3: Relative Trajectory Retargeting/Alignment of 3D Point, and Directly Use Transformed 3D Pose'''
    # pose_t_L_list, pose_t_R_list = [], []
    # robot_start_pose_L = [0.558413994, 0.385476218, 0.402155227, -179.610881514, 75.474229046, -90.517829792]
    # robot_start_pose_R = [0.558413994, -0.385476218, 0.402155227, 179.732957532, 75.68246746, 89.739817245]
    # R_corr_grip_init = np.array([ [0, -1, 0], [1, 0, 0], [0, 0, 1] ])
    # R_slam_to_cv = np.array([ [1,  0,  0], [0, 0, 1], [0, -1, 0] ])
    # R_cam_to_torso = np.array([ [0, -1, 0], [0, 0, -1], [1, 0, 0] ])
    # R_world_to_base_L = np.array([ [1, 0, 0], [0, -1, 0], [0, 0, -1] ]).T
    # R_world_to_base_R = np.eye(3)
    # scale_factor = 1.5
    # for arm_type, pose_cache_list in [["L", pose_cache_L_list], ["R", pose_cache_R_list]]:
        # pose_t_list = pose_t_L_list if arm_type == "L" else pose_t_R_list
        # r_start_position = robot_start_pose_L if arm_type == "L" else robot_start_pose_R
        # R_world_to_base = R_world_to_base_L if arm_type == "L" else R_world_to_base_R
        # R_corr_grip = R_corr_grip_init.T if arm_type == "L" else R_corr_grip_init
        # handeye_cam = handeye_cam_L if arm_type == "L" else handeye_cam_R
        # g_loc_0 = np.array(pose_cache_list[0][1]) @ R_cam_to_torso.T
        # for idx, pose_vec in enumerate(pose_cache_list):
            # [g_rot, g_loc, g_state, wrist_loc] = pose_vec  # using g_loc or wrist_loc
            # delta_pos_world = (np.array(g_loc) @ R_cam_to_torso.T - g_loc_0) * scale_factor
            # delta_pos_robot = np.dot(R_world_to_base, delta_pos_world)
            # g_rot_4x4, R_corr_grip_4x4, R_slam_to_cv_4x4 = np.eye(4), np.eye(4), np.eye(4)
            # g_rot_4x4[:3, :3], R_corr_grip_4x4[:3, :3], R_slam_to_cv_4x4[:3, :3] = np.array(g_rot), R_corr_grip, R_slam_to_cv
            # R_base_grip = np.dot(np.dot(handeye_cam, np.dot(R_slam_to_cv_4x4, g_rot_4x4)), R_corr_grip_4x4)  # note this transformation
            # R_base_grip = np.dot(handeye_cam, R_corr_grip_4x4 @ np.dot(R_slam_to_cv_4x4, g_rot_4x4) @ R_corr_grip_4x4.T)
            # R_base_grip = np.dot(handeye_cam, R_corr_grip_4x4 @ np.dot(R_slam_to_cv_4x4, g_rot_4x4) @ R_corr_grip_4x4.T)  ########## (better)
            # R_next = np.eye(4)
            # R_next[:3, 3] = np.array(r_start_position[:3]) + delta_pos_robot  # Relative Position
            # R_next[:3, :3] = R_base_grip[:3, :3]  # Direct 3D Orientation Mapper
            # R_next = tcp_fix_gripper(R_next, is_back=False)  # is_back=False, add tool length
            # target_pose_back = [0,0,0, 0,0,0, g_state]  # position, rotation, gripper
            # target_pose_back[:3] = list(R_next[:3, -1])
            # target_pose_back[3:6] = Rotation.from_matrix(R_next[:3, :3]).as_euler("xyz", degrees=True)
            # pose_t_list.append(target_pose_back)
        # print("[Finished!]", len(pose_t_list), pose_t_list[:5])    
    ################################################################################
    
    
    # os._exit(0)
    time.sleep(5)
    
    
    ################################################################################
    ip_armL = cfg_dict["arm1"]["robot_ip_add"]
    ip_armR = cfg_dict["arm2"]["robot_ip_add"]
    gripper_L = JodellRG75(give_torque=64, given_speed=255); gripper_R = JodellRG75(give_torque=64, given_speed=255)
    gripper_L.connect("/dev/ttyUSB1", 9); gripper_R.connect("/dev/ttyUSB0", 9)
    gripper_L.set_pos(0); gripper_R.set_pos(0)  # for pivoting cirbowl
    robotL = rokaeRobotBase(ip_armL)
    robotR = rokaeRobotBase(ip_armR)
    
    def drive_single_arm_to_run(target_pose_list, arm_type, sleep_time, gripper_value):
        if arm_type == "L":
            time.sleep(sleep_time); robotL.move_by_trajectory(target_pose_list)
            global cur_gripper_L
            if gripper_value != -1 and cur_gripper_L != gripper_value:  # gripper = 1 close; gripper = 0 open
                gripper_L.switch(gripper_value, True); time.sleep(gripper_wait_time); cur_gripper_L = gripper_value          
        if arm_type == "R":
            time.sleep(sleep_time); robotR.move_by_trajectory(target_pose_list)
            global cur_gripper_R
            if gripper_value != -1 and cur_gripper_R != gripper_value:  # gripper = 1 close; gripper = 0 open
                gripper_R.switch(gripper_value, True); time.sleep(gripper_wait_time); cur_gripper_R = gripper_value
    thread_list = []  # for saving multiple threadings for dual-arm coordination
    
    ################################################################################
    init_pos_deg_L = [0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792]
    cur_pos_deg_L, cur_pos_rad_L = robotL.get_current_pose(); print("cur_pos_deg_L:", cur_pos_deg_L)
    robotL.moving_pre_op(); robotL.set_motion_speed_ratio(0.1)
    robotL.move_to_a_waypoint(init_pos_deg_L)  
    # robotL.move_to_a_waypoint(pose_t_L_list[1][:6])
    # robotL.move_by_trajectory(pose_t_L_list)
    # for pose_t_L in pose_t_L_list: robotL.move_to_a_waypoint(pose_t_L[:6], is_waiting=False)
    # robotL.move_to_a_waypoint(init_pos_deg_L) 
    
    thread = threading.Thread(target=drive_single_arm_to_run, args=(pose_t_L_list+[init_pos_deg_L], "L", 0.0, -1, ))
    thread_list.append(thread); thread.start()
    
    # os._exit(0)

    ################################################################################
    init_pos_deg_R = [0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329]
    cur_pos_deg_R, cur_pos_rad_R = robotR.get_current_pose(); print("cur_pos_deg_R:", cur_pos_deg_R)
    robotR.moving_pre_op(); robotR.set_motion_speed_ratio(0.1)
    robotR.move_to_a_waypoint(init_pos_deg_R)  
    # robotR.move_to_a_waypoint(pose_t_R_list[1][:6])
    # robotR.move_by_trajectory(pose_t_R_list)
    # for pose_t_R in pose_t_R_list: robotR.move_to_a_waypoint(pose_t_R[:6], is_waiting=False)
    # robotR.move_to_a_waypoint(init_pos_deg_R)  

    thread = threading.Thread(target=drive_single_arm_to_run, args=(pose_t_R_list+[init_pos_deg_R], "R", 0.0, -1, ))
    thread_list.append(thread); thread.start()
    
    ################################################################################
    # os._exit(0)
    for thread in thread_list: thread.join()  # wait all threading to finish
    thread_list = []
