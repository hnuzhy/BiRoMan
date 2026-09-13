
import os
import sys
import numpy as np
from copy import copy
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.getcwd())

from src.util import tcp_fix_gripper
from src.util import interpolate_6dof_on_sphere
from src.config import cfg_dict_init as cfg_dict


#################################################################
def pivot_around_spindle(static_pose_L, start_pose_R,
                         delta_xy=[0, 0], adj_deg=50, rot_ratio=1.0, interpt=20, gripper_offset=0.012):
    # static_pose_L: static 6-DoF pose of left-arm end-effector, including position and orientation
    # start_pose_R: starting 6-DoF pose of right-arm end-effector, including position and orientation
    
    # delta_xy: the x and y offsets for the manipulated object, which are defined in camera wolrd (mm)
    # adj_deg: the degree of multiple move-and-rotate actions by adjusting the right-arm ry
    # rot_ratio: the ratio of rotation degree (default is 1.0 for rotating object 90 degrees)
    # interpt: the interpolation points number (the start point and end point are also inclued)
    
    target_pose_list_final = []
    
    static_pose_L_seed = copy(static_pose_L)
    static_pose_L_seed[0] += (delta_xy[1] / 1000.0)  # adjust back x value according to the object y-pixel position offset
    static_pose_L_seed[2] += (delta_xy[0] / 1000.0)  # adjust back z value according to the object x-pixel position offset
    start_pose_R_seed = copy(start_pose_R)
    start_pose_R_seed[0] += (delta_xy[1] / 1000.0)  # adjust back x value according to the object y-pixel position offset
    start_pose_R_seed[2] -= (delta_xy[0] / 1000.0)  # adjust back z value according to the object x-pixel position offset

    # step 1: calculate the 6-DoF pose after removing tool length
    poses_adjusted = []
    for start_pose in [static_pose_L, start_pose_R, static_pose_L_seed, start_pose_R_seed]:
        target_6dof_pose_back = np.eye(4)
        target_6dof_pose_back[:3, :3] = R.from_euler("xyz", start_pose[3:], degrees=True).as_matrix()
        target_6dof_pose_back[:3, -1] = start_pose[:3]
        target_6dof_pose_back = tcp_fix_gripper(target_6dof_pose_back, is_back=True)  # is_back=True, remove tool length
        target_pose_back = [0,0,0, 0,0,0]
        target_pose_back[:3] = list(target_6dof_pose_back[:3, -1])
        target_pose_back[3:] = R.from_matrix(target_6dof_pose_back[:3, :3]).as_euler("xyz", degrees=True)
        poses_adjusted.append(target_pose_back)
    [static_pose_L_ad, start_pose_R_ad, static_pose_L_seed_ad, start_pose_R_seed_ad] = poses_adjusted
    print(f"\n[Step 1] static_pose_L_ad: {np.array(static_pose_L_ad).tolist()} \n[Step 1] start_pose_R_ad: {np.array(start_pose_R_ad).tolist()}")
    
    # step 2: calculate the targeting 6-DoF pose with tool length removed
    handeye_camera_L = cfg_dict["arm1"]["handeye_para"]  # for kingfisher-R-6000 to robot left
    handeye_camera_R = cfg_dict["arm2"]["handeye_para"]  # for kingfisher-R-6000 to robot right
    static_loc_L_ad_cam = handeye_camera_L[:3, :3].T @ (static_pose_L_ad[:3] - handeye_camera_L[:3, -1])  # armL --> camera
    static_loc_L_ad_armR = handeye_camera_R[:3, :3] @ static_loc_L_ad_cam + handeye_camera_R[:3, -1]  # camera --> armR
    print(f"\n[Step 2] static_loc_L_ad_armR: {np.array(static_loc_L_ad_armR).tolist()} \n[Step 2] start_loc_R_ad: {np.array(start_pose_R_ad[:3]).tolist()}")
    start_pose_R_ad[0] = static_loc_L_ad_armR[0] + 0.0  # align the x-value of dual-arm

    ########## [start] calculate the refer placement
    static_loc_L_seed_ad_cam = handeye_camera_L[:3, :3].T @ (static_pose_L_seed_ad[:3] - handeye_camera_L[:3, -1])  # armL --> camera
    static_loc_L_seed_ad_armR = handeye_camera_R[:3, :3] @ static_loc_L_seed_ad_cam + handeye_camera_R[:3, -1]  # camera --> armR
    start_pose_R_seed_ad[0] = static_loc_L_seed_ad_armR[0] + 0.0  # align the x-value of dual-arm

    sphere_origin_center_ref = static_loc_L_seed_ad_armR.copy()
    sphere_origin_center_ref[2] = min(static_loc_L_seed_ad_armR[2], start_pose_R_seed_ad[2])  # adjust z value
    deltaX = sphere_origin_center_ref[0] - start_pose_R_seed_ad[0]
    deltaY = sphere_origin_center_ref[1] - start_pose_R_seed_ad[1]
    deltaZ = sphere_origin_center_ref[2] - start_pose_R_seed_ad[2]
    radius_ref = (deltaX**2 + deltaY**2 + deltaZ**2) ** (0.5)  # re-calculating the rotation/pivoting radius
    ########## [end] calculate the refer placement
    
    sphere_origin_center = static_loc_L_ad_armR.copy()
    sphere_origin_center[1] = min(static_loc_L_ad_armR[1], start_pose_R_ad[1])  # adjust y value
    deltaX = sphere_origin_center[0] - start_pose_R_ad[0]
    deltaY = sphere_origin_center[1] - start_pose_R_ad[1]
    deltaZ = sphere_origin_center[2] - start_pose_R_ad[2]  # always a negative number
    radius_new = (deltaX**2 + deltaY**2 + deltaZ**2) ** (0.5)  # re-calculating the rotation/pivoting radius
    print(f"[Step 2] re-calculated rotation radius_new | radius_ref: {radius_new} | {radius_ref} <<<<<====================")
    
    deltaZ_adjusted = - (radius_ref**2 - deltaX**2 - deltaY**2) ** (0.5)
    start_pose_R_ad[2] = sphere_origin_center[2] - deltaZ_adjusted  # re-calculating the z-value of armR <<<<<===================
    print(f"[Step 2] deltaZ newly | deltaZ_adjusted --> {deltaZ} | {deltaZ_adjusted}")
        
    target_pose_R_ad = [0,0,0, 0,0,0]
    target_pose_R_ad[:3] = static_loc_L_ad_armR[:3].copy()  # assign the x and z values
    target_pose_R_ad[1] = sphere_origin_center[1] + radius_ref  # adjusted y value
    target_pose_R_ad[3:] = start_pose_R_ad[3:]  # assign the pose value rx/ry/rz
    target_pose_R_ad[4] = start_pose_R_ad[4] + adj_deg  # adjusted ry value (e.g., from  25 to 65)
    
    # step 3: calculate the interpolation points with tool length removed
    print(f"\n[Step 3] target_pose_R_ad: {np.array(target_pose_R_ad).tolist()} \n[Step 3] start_pose_R_ad: {np.array(start_pose_R_ad).tolist()}")
    target_pose_list_temp = interpolate_6dof_on_sphere(sphere_origin_center, start_pose_R_ad, target_pose_R_ad, num_steps=interpt)
    print(f"[Step 3] target_pose_list_temp: {np.array(target_pose_list_temp[:3]).tolist()}")
    
    # step 4: adjust the interpolation points with adding the tool length
    g_off_z = gripper_offset / (interpt - 1)
    for idx, pose_temp in enumerate(target_pose_list_temp):  # idx is from 0 to (interpt - 1)
        if (idx+1)*1.0 / interpt > rot_ratio: break  # only keep the former rot_ratio waypoints
        target_6dof_pose_back = np.eye(4)
        target_6dof_pose_back[:3, :3] = R.from_euler("xyz", pose_temp[3:], degrees=True).as_matrix()
        target_6dof_pose_back[:3, -1] = pose_temp[:3]
        target_6dof_pose_back = tcp_fix_gripper(target_6dof_pose_back, is_back=False)  # is_back=False, add tool length
        target_pose_back = [0,0,0, 0,0,0]
        target_pose_back[:3] = list(target_6dof_pose_back[:3, -1])
        target_pose_back[3:] = R.from_matrix(target_6dof_pose_back[:3, :3]).as_euler("xyz", degrees=True)
        target_pose_back[1] += (g_off_z*idx)  # adjust z value by adding the gripper_offset
        target_pose_list_final.append(target_pose_back)
        
    print(f"\n[Step 4] target_pose_list_final: {np.array(target_pose_list_final[:3]).tolist()}\n")
    
    return target_pose_list_final

#################################################################
def wrapping_with_coordination(contact_pose, start_pose, leftarm=False,
                         delta_xy=[0, 0], adj_deg=-20, rot_ratio=1.0, interpt=20, shrink_offset=0.000, given_radius=None):
    # start_pose: starting 6-DoF pose of left/right-arm end-effector, including position and orientation
    
    # delta_xy: the x and y offsets for the manipulated object, which are defined in camera wolrd (mm)
    # adj_deg: the degree of multiple move-and-rotate actions by adjusting the right-arm ry
    # rot_ratio: the ratio of rotation degree (default is 1.0 for rotating object 90 degrees)
    # interpt: the interpolation points number (the start point and end point are also inclued)
    # given_radius: the given radius for rotation around the spindle (mm). for example 0.150
    
    target_pose_list_final = []
    
    if given_radius is None: given_radius = abs(contact_pose[1] - start_pose[1])  # compute the lifting height

    if leftarm:
        start_pose_seed = copy(start_pose)
        start_pose_seed[0] += (delta_xy[1] / 1000.0)  # adjust back x value according to the object y-pixel position offset
        start_pose_seed[2] += (delta_xy[0] / 1000.0)  # adjust back z value according to the object x-pixel position offset
    else:
        start_pose_seed = copy(start_pose)
        start_pose_seed[0] += (delta_xy[1] / 1000.0)  # adjust back x value according to the object y-pixel position offset
        start_pose_seed[2] -= (delta_xy[0] / 1000.0)  # adjust back z value according to the object x-pixel position offset

    # step 1: calculate the 6-DoF pose after removing tool length
    poses_adjusted = []
    for start_pose in [start_pose, start_pose_seed]:
        target_6dof_pose_back = np.eye(4)
        target_6dof_pose_back[:3, :3] = R.from_euler("xyz", start_pose[3:], degrees=True).as_matrix()
        target_6dof_pose_back[:3, -1] = start_pose[:3]
        target_6dof_pose_back = tcp_fix_gripper(target_6dof_pose_back, is_back=True)  # is_back=True, remove tool length
        target_pose_back = [0,0,0, 0,0,0]
        target_pose_back[:3] = list(target_6dof_pose_back[:3, -1])
        target_pose_back[3:] = R.from_matrix(target_6dof_pose_back[:3, :3]).as_euler("xyz", degrees=True)
        poses_adjusted.append(target_pose_back)
    [start_pose_ad, start_pose_seed_ad] = poses_adjusted
    print(f"\n[Step 1] start_pose_ad: {np.array(start_pose_ad).tolist()}")
    

    # step 2: calculate the sphere_origin_center and target_pose_ad
    sphere_origin_center = start_pose_ad[:3].copy()
    if leftarm:
        sphere_origin_center[1] += given_radius  # adjust the y-value
    else:
        sphere_origin_center[1] -= given_radius  # adjust the y-value
    target_pose_ad = [0,0,0, 0,0,0]
    target_pose_ad[0] = sphere_origin_center[0] - given_radius  # adjusted x value
    target_pose_ad[1:3] = sphere_origin_center[1:].copy()  # assign the y and z values
    target_pose_ad[2] -= shrink_offset  # adjust the z-value for shrink the distance of two arms
    target_pose_ad[3:] = start_pose_ad[3:]  # assign the pose value rx/ry/rz
    target_pose_ad[4] = start_pose_ad[4] + adj_deg  # adjusted ry value (e.g., from 45 to 25)
    target_pose_list_temp = interpolate_6dof_on_sphere(
        np.array(sphere_origin_center), np.array(start_pose_ad), np.array(target_pose_ad), num_steps=interpt)
    print(f"[Step 2] target_pose_list_temp: {np.array(target_pose_list_temp[:3]).tolist()}")
    
    
    # step 3: adjust the interpolation points with adding the tool length
    for idx, pose_temp in enumerate(target_pose_list_temp):  # idx is from 0 to (interpt - 1)
        if (idx+1)*1.0 / interpt > rot_ratio: break  # only keep the former rot_ratio waypoints
        target_6dof_pose_back = np.eye(4)
        target_6dof_pose_back[:3, :3] = R.from_euler("xyz", pose_temp[3:], degrees=True).as_matrix()
        target_6dof_pose_back[:3, -1] = pose_temp[:3]
        target_6dof_pose_back = tcp_fix_gripper(target_6dof_pose_back, is_back=False)  # is_back=False, add tool length
        target_pose_back = [0,0,0, 0,0,0]
        target_pose_back[:3] = list(target_6dof_pose_back[:3, -1])
        target_pose_back[3:] = R.from_matrix(target_6dof_pose_back[:3, :3]).as_euler("xyz", degrees=True)
        target_pose_list_final.append(target_pose_back)
        
    print(f"\n[Step 3] target_pose_list_final: {np.array(target_pose_list_final[:3]).tolist()}\n")
    
    return target_pose_list_final

#################################################################
    








