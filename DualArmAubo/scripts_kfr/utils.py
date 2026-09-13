
import cv2
import time
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp  # line interpolate in SO(3) manifold space
from scipy.interpolate import make_interp_spline  # Cubic B-spline in 3D Euclidean space

######################################################################################################################################
cfg_dict_init = {
    "kfr_ip": "192.168.4.94",  # the ip address of kingfisher-R-6000
    "arm1": {
        "gripper_port": '/dev/ttyUSB0',  # for Ubuntu, sudo chmod 666 /dev/ttyUSB0
        "robot_ip_add": "192.168.4.93",  # robot IP address "192.168.31.134"
        "handeye_para": np.array([
            [-0.14495955, -0.82672254,  0.54361435, -0.46128096],
            [-0.98826554,  0.09424301, -0.120206  , -0.55922609],
            [ 0.04814516, -0.55466034, -0.83068282,  0.77215299],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ]),  # camera calib in 2024-11-20 (for old kingfisher)
        "handeye_para_v2": np.array([
            [ 0.02826124, -0.82213902,  0.56858485, -0.5574288 ],
            [-0.99958334, -0.01990389,  0.020904  , -0.7053366 ],
            [-0.00586894, -0.56893871, -0.82235898,  0.79531344],
            [ 0.        ,  0.        , -0.        ,  1.        ]
        ]),  # camera calib in 2025-02-25 (for kingfisher-R-6000)
    },
    "arm2": {
        "gripper_port": '/dev/ttyUSB1',  # for Ubuntu, sudo chmod 666 /dev/ttyUSB1
        "robot_ip_add": "192.168.4.109",  # robot IP address "192.168.31.135"
        "handeye_para": np.array([
            [ 0.15464175,  0.82413502, -0.54487375,  0.48699159],
            [ 0.98677   , -0.10165977,  0.12629433, -0.77214491],
            [ 0.04869184, -0.55719545, -0.82895256,  0.76490652],
            [-0.        ,  0.        , -0.        ,  1.        ]
        ]),  # camera calib in 2024-11-20 (for old kingfisher)
        "handeye_para_v2": np.array([
            [-0.01459961,  0.82190626, -0.56943564,  0.58294993],
            [ 0.99988048,  0.01489787, -0.00413253, -0.63574671],
            [ 0.00508683, -0.56942791, -0.82202553,  0.79036275],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ]),  # camera calib in 2025-02-25 (for kingfisher-R-6000)
    },
    "detection_roi_bbox": [120, 0, 760, 540],  # the [x1, y1, x2, y2] (960*540 --> 640*540; 16:9 --> 32:27)
    "detection_roi_bbox_2": [200, 80, 680, 500],  # the [x1, y1, x2, y2] (960*540 --> 480*420; 16:9 --> 8:7)
}
######################################################################################################################################



######################################################################################################################################
#################################################################
'''
from rlia.robotics import TrajectoryConstraints
from rlia.robotics import PathBezierCurveSE3
from rlia.robotics import TrajectoryDoubleSSE3
from rlia.robotics import SE3

def generate_interpolation_traj(pose_list: np.ndarray, interp_num: int, is_6dof=False):
    """_summary_
    Args:
        pose_list (np.ndarray): [pose_num, 4, 4] of float
        interp_num (int): interpolation num
    """
    if is_6dof:
        pose_list_new = []
        for eef_pose in pose_list:
            target_pose_se3 = np.eye(4)
            target_pose_se3[:3, :3] = R.from_euler("xyz", eef_pose[3:6], degrees=True).as_matrix()
            target_pose_se3[:3, -1] = eef_pose[0:3]
            pose_list_new.append(target_pose_se3)
        pose_list = pose_list_new
    
    limits = np.ones((3, 7)) * 1
    traj_constraints = TrajectoryConstraints(limits[0] * 0.1, limits[1] * 5.0, limits[2] * 20)
    se3_arr = np.array([SE3(eef_pose) for eef_pose in pose_list])
    bezier_path = PathBezierCurveSE3(se3_arr, degree=5, is_cartersian_space=True, blend_tolerance=0.1)
    traj_doubleS = TrajectoryDoubleSSE3(bezier_path, traj_constraints)
    max_time = traj_doubleS.get_duration()
    traj_inters = np.linspace(0, max_time, interp_num)
    traj_poses = np.empty(shape=(interp_num, 4, 4), dtype=float)
    for i in range(interp_num):
        traj_poses[i] = traj_doubleS.get_position(traj_inters[i]).get_transform()
    
    if is_6dof:
        pose_list_new = []
        for eef_pose in traj_poses:
            target_pose_6dof = [0,0,0, 0,0,0]
            target_pose_6dof[:3] = list(eef_pose[:3, -1])
            target_pose_6dof[3:] = R.from_matrix(eef_pose[:3, :3]).as_euler("xyz", degrees=True)
            pose_list_new.append(target_pose_6dof)
        traj_poses = pose_list_new
    
    return traj_poses
'''

#################################################################
def interpolate_se3_bspline_startend_poses(start_pose_6dof, target_pose_6dof, num_steps=5):
    start_pos, end_pos = start_pose_6dof[:3], target_pose_6dof[:3]
    start_quat = R.from_euler("xyz", start_pose_6dof[3:], degrees=True).as_quat()
    end_quat = R.from_euler("xyz", target_pose_6dof[3:], degrees=True).as_quat()
    
    # === Position: B-spline interpolation ===
    control_positions = np.vstack([start_pos, end_pos])
    t_positions = [0, 1]  # parameter values
    t_dense = np.linspace(0, 1, num_steps)
    
    # Use a linear B-spline since only 2 points
    spline_pos = make_interp_spline(t_positions, control_positions, k=1)
    interp_positions = spline_pos(t_dense)

    # === Orientation: SLERP for simplicity (approximate B-spline behavior) ===
    slerp_pose = Slerp([0, 1], R.from_quat([start_quat, end_quat]))
    interp_rotmats = slerp_pose(t_dense)
    interp_quats = interp_rotmats.as_quat()
    
    # === Combine into SE(3) poses ===
    trajectory = []
    for position_res, quaternion_res in zip(interp_positions, interp_quats):
        target_pose_temp = [0,0,0, 0,0,0]
        target_pose_temp[:3] = list(position_res)
        target_pose_temp[3:] = R.from_quat(quaternion_res).as_euler("xyz", degrees=True)
        trajectory.append(target_pose_temp)

    return trajectory


#################################################################
# https://docs.ultralytics.com/reference/data/utils/#ultralytics.data.utils.polygon2mask
def polygon2mask(
    imgsz: Tuple[int, int], polygons: List[np.ndarray], color: int = 1, downsample_ratio: int = 1
) -> np.ndarray:
    """
    Convert a list of polygons to a binary mask of the specified image size.

    Args:
        imgsz (Tuple[int, int]): The size of the image as (height, width).
        polygons (List[np.ndarray]): A list of polygons. Each polygon is an array with shape (N, M), where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int, optional): The color value to fill in the polygons on the mask.
        downsample_ratio (int, optional): Factor by which to downsample the mask.

    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygons filled in.
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw, nh))

###################################################################
def calInsideRectIOU(rectLarge, rectSmall):
    # calculate two rectangles IOU(intersection-over-union)
    [Ax0, Ay0, Ax1, Ay1] = rectLarge[0:4]
    [Bx0, By0, Bx1, By1] = rectSmall[0:4]
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        return crossArea/areaB

def calTwoRectIOU(rectA, rectB):
    # calculate two rectangles IOU(intersection-over-union)
    [Ax0, Ay0, Ax1, Ay1] = rectA[0:4]
    [Bx0, By0, Bx1, By1] = rectB[0:4]
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        return crossArea/(areaA + areaB - crossArea)
#################################################################
######################################################################################################################################



######################################################################################################################################
#################################################################
def tcp_fix_gripper(pose: np.ndarray, is_back=False):
    gripper_offset_z, gripper_offset_y, gripper_offset_x = 0.161, 0.000, 0.000  # using legal 3dof pose, for arm2-drawer
    
    gripper_rot_offset = 0  # rad, for parallel_gripper, e.g., np.pi / 4
    
    if is_back:
        gripper_offset_z *= -1
        gripper_offset_y *= -1
        gripper_offset_x *= -1
    
    # Pose_ee_in_camera
    pose_ = np.linalg.inv(pose)
    pose_[0, 3] += gripper_offset_x # x (the gripper length)
    pose_[1, 3] += gripper_offset_y # y (the gripper length)
    pose_[2, 3] += gripper_offset_z # z (the gripper length)
    r_offset_1x3 = np.array([0, 0, -1]) * gripper_rot_offset
    r_offset_3x3 = R.from_rotvec(r_offset_1x3).as_matrix() # (rotate 45 degree)
    pose_r = np.eye(4)
    pose_r[:3, :3] = r_offset_3x3
    pose_ = np.matmul(np.linalg.inv(pose_), pose_r)
    return pose_

#################################################################
def apply_primitive_skill(target_pose, rot_deg, skill_name, rot_axis="X"):
    skills_list = ["twist", "pour"]
    assert skill_name in skills_list, "We now only support several limited skills!"
    
    target_6dof_pose_back = np.eye(4)
    target_6dof_pose_back[:3, :3] = R.from_euler("xyz", target_pose[3:], degrees=True).as_matrix()
    target_6dof_pose_back[:3, -1] = target_pose[:3]
    target_6dof_pose_back = tcp_fix_gripper(target_6dof_pose_back, is_back=True)  # remove tool length
    target_pose_back = [0,0,0, 0,0,0]
    target_pose_back[:3] = list(target_6dof_pose_back[:3, -1])
    target_pose_back[3:] = R.from_matrix(target_6dof_pose_back[:3, :3]).as_euler("xyz", degrees=True)
    
    if skill_name == "twist":  # (close - moved&rotate - open - back-rotate) (only rotate part)
        twist_deg = rot_deg
        if rot_axis == "X":  # rotate around X-axis
            temp_pose_deg = [target_pose_back[3] + twist_deg] + target_pose_back[4:]
        if rot_axis == "Y":  # rotate around Y-axis
            temp_pose_deg = [target_pose_back[3], target_pose_back[4]+twist_deg, target_pose_back[5]] 
        if rot_axis == "Z":  # rotate around X-axis
            temp_pose_deg = target_pose_back[3:5] + [target_pose_back[5] + twist_deg]

    if skill_name == "pour":  # (closed - moved&rotate - closed - back-rotate) (only moved&rotate part)
        pour_deg = rot_deg
        if rot_axis == "X":  # rotate around X-axis
            temp_pose_deg = [target_pose_back[3] - pour_deg] + target_pose_back[4:]
        if rot_axis == "Y":  # rotate around Y-axis
            temp_pose_deg = [target_pose_back[3], target_pose_back[4] - pour_deg, target_pose_back[5]]   
        if rot_axis == "Z":  # rotate around X-axis
            temp_pose_deg = target_pose_back[3:5] + [target_pose_back[5] - pour_deg]

    target_6dof_pose = np.eye(4)
    target_6dof_pose[:3, :3] = R.from_euler("xyz", temp_pose_deg, degrees=True).as_matrix()
    target_6dof_pose[:3, -1] = target_pose_back[:3]
    target_6dof_pose = tcp_fix_gripper(target_6dof_pose)  # add tool length
    temp_eef_pose_deg = list(target_6dof_pose[:3, -1]) + temp_pose_deg
    print(target_pose, "\n", target_pose_back, "\n", temp_eef_pose_deg)

    return temp_eef_pose_deg

#################################################################
def process_single_arm(eef_pose_seq, handeye_camera, cur_pose_deg):
    
    [pixel_3d, rot_mat, gripper_state, euler_angles, frame_id] = eef_pose_seq
    gripper, [pitch, yaw, roll] = int(gripper_state), euler_angles

    target_6dof_pose = np.eye(4)
    rot_mat_robot = R.from_euler("xyz", cur_pose_deg[3:], degrees=True).as_matrix()
    target_6dof_pose[:3, :3] = handeye_camera[:3, :3].T @ rot_mat_robot
    target_6dof_pose[:3, -1] = np.array(pixel_3d)

    target_pose = [0,0,0, 0,0,0]
    target_6dof_pose = tcp_fix_gripper(target_6dof_pose)
    target_pose[:3] = handeye_camera[:3, :3] @ target_6dof_pose[:3, -1] + handeye_camera[:3, -1]
    temp_pose_3x3 = handeye_camera[:3, :3] @ target_6dof_pose[:3, :3]
    target_pose[3:] = R.from_matrix(temp_pose_3x3).as_euler("xyz", degrees=True)

    return target_pose, gripper


#################################################################
def run_single_arm_gripper(
    robotBase, m_gripper, target_pose, gripper, cur_gripper_state, step_id, robot_arm,
    twist_deg=160, pour_deg=95):

    eef_pose_dict_new_temp_list = []
    
    if gripper == 0:
        g_state = 0
        m_gripper.SetTargetPosition(5)  # close gripper, range is (0, 1000)
        while(g_state == 0):
            g_state = m_gripper.GetGripState()
            # print(step_id, robot_arm, g_state)  # for debug only
            time.sleep(0.1)
        cur_gripper_state = 0
    if gripper == 1 or gripper == 2:  # 2 means the primitive skill: preopen (unclasp first, then move)
        g_state = 0
        m_gripper.SetTargetPosition(990)  # close gripper, range is (0, 1000)
        while(g_state == 0):
            g_state = m_gripper.GetGripState()
            # print(step_id, robot_arm, g_state)  # for debug only
            time.sleep(0.1)
        cur_gripper_state = 1
    #######################################
    if gripper == -1:  # primitive skill: twist (close - moved&rotate - open - rotate&back - close - lift)
        # twist_deg = 120  # the default rotated degrees for original YOTO paper
        # twist_deg = 240  # the enlarged rotated degrees for the close-loop version and reorient_unscrew task
        assert cur_gripper_state == 1, "before twisting, the gripper must be opened!"
        g_state = 0
        m_gripper.SetTargetPosition(5)  # close gripper, range is (0, 1000)
        while(g_state == 0):
            g_state = m_gripper.GetGripState()
            time.sleep(0.1)

        robot_cur_state = robotBase.get_current_pose()[0]
        print("******[saving labels]*****", step_id, "\t", robot_arm, "\t", robot_cur_state, 0)  # 0, close
        eef_pose_dict_new_temp_list.append([step_id] + robot_cur_state + [0]) 
        
        ############################
        if twist_deg==240 or twist_deg==180:  # this rotation angle is too large, we need to split it into two steps
            current_joint = robotBase.get_current_joint()
            target_joint = current_joint[:-1] + [current_joint[-1] - np.pi*(twist_deg*0.5/180.0)]
            robotBase.move_joint(target_joint)  # Rotate Counter-Clockwise (forward-rotate)

            g_state = 0
            m_gripper.SetTargetPosition(990)  # open gripper, range is (0, 1000)
            while(g_state == 0):
                g_state = m_gripper.GetGripState()
                time.sleep(0.1)
                
            robot_cur_state = robotBase.get_current_pose()[0]
            print("******[saving labels]*****", step_id, "\t", robot_arm, "\t", robot_cur_state, 1)  # 1, open
            eef_pose_dict_new_temp_list.append([step_id] + robot_cur_state + [1])

            robotBase.move_joint(current_joint)  # Rotate Clockwise (backward-rotate)
            current_pose = robotBase.get_current_pose()[0]
            g_state = 0
            m_gripper.SetTargetPosition(5)  # close gripper, range is (0, 1000)
            while(g_state == 0):
                g_state = m_gripper.GetGripState()
                time.sleep(0.1)
            
            robot_cur_state = robotBase.get_current_pose()[0]
            print("******[saving labels]*****", step_id, "\t", robot_arm, "\t", robot_cur_state, 0)  # 0, close
            eef_pose_dict_new_temp_list.append([step_id] + robot_cur_state + [0]) 
            
            twist_deg = twist_deg*0.5
        ############################
            
        current_joint = robotBase.get_current_joint()
        target_joint = current_joint[:-1] + [current_joint[-1] - np.pi*(twist_deg/180.0)]
        robotBase.move_joint(target_joint)  # Rotate Counter-Clockwise (forward-rotate)

        g_state = 0
        m_gripper.SetTargetPosition(990)  # open gripper, range is (0, 1000)
        while(g_state == 0):
            g_state = m_gripper.GetGripState()
            time.sleep(0.1)

        robot_cur_state = robotBase.get_current_pose()[0]
        print("******[saving labels]*****", step_id, "\t", robot_arm, "\t", robot_cur_state, 1)  # 1, open
        eef_pose_dict_new_temp_list.append([step_id] + robot_cur_state + [1])
        
        robotBase.move_joint(current_joint)  # Rotate Clockwise (backward-rotate)
        current_pose = robotBase.get_current_pose()[0]
        current_pose[2] += 0.006  # adjust it from 0.005 into 0.006 in 2025-03-28
        robotBase.move_to_target_in_cartesian(current_pose)  # lifting the lip smaller
        g_state = 0
        m_gripper.SetTargetPosition(5)  # close gripper, range is (0, 1000)
        while(g_state == 0):
            g_state = m_gripper.GetGripState()
            time.sleep(0.1)

        robot_cur_state = robotBase.get_current_pose()[0]
        print("******[saving labels]*****", step_id, "\t", robot_arm, "\t", robot_cur_state, 0)  # 0, close
        eef_pose_dict_new_temp_list.append([step_id] + robot_cur_state + [0])
        
        current_pose[2] += 0.025
        robotBase.move_to_target_in_cartesian(current_pose)  # lifting the lip larger
        cur_gripper_state = 0
    #######################################
    if gripper == -2:  # primitive skill: lifting (opened - close - moved-higher - closed)
        assert cur_gripper_state == 1, "before lifting, the gripper must be opened!"
        
        g_state = 0
        m_gripper.SetTargetPosition(5)  # close gripper, range is (0, 1000)
        while(g_state == 0):
            g_state = m_gripper.GetGripState()
            time.sleep(0.1)

        robot_cur_state = robotBase.get_current_pose()[0]
        print("******[saving labels]*****", step_id, "\t", robot_arm, "\t", robot_cur_state, 0)  # 0, close
        eef_pose_dict_new_temp_list.append([step_id] + robot_cur_state + [0])
        
        target_pose[2] += 0.05  # enlarge the z value for <lifting the cup>, change 4 into 5
        robotBase.move_to_target_in_cartesian(target_pose)
        cur_gripper_state = 0
    #######################################  
    if gripper == -3:  # primitive skill: pour (closed - moved&rotate - closed - back-rotate)
        # pour_deg = 95  # the default rotated degrees (a large angle may have IK bug)
        assert cur_gripper_state == 0, "before pouring, the gripper must be closed!"
        
        robot_cur_state = robotBase.get_current_pose()[0]
        print("******[saving labels]*****", step_id, "\t", robot_arm, "\t", robot_cur_state, 0)  # 0, close
        eef_pose_dict_new_temp_list.append([step_id] + robot_cur_state + [0])
        
        temp_eef_pose_deg = apply_primitive_skill(target_pose, pour_deg, "pour", rot_axis="X")
        robotBase.move_to_target_in_cartesian(temp_eef_pose_deg)  # moved&rotate
        time.sleep(1)  # keep pouring for a short time 
        # time.sleep(3)  # keep pouring for a long time 
        
        robot_cur_state = robotBase.get_current_pose()[0]
        print("******[saving labels]*****", step_id, "\t", robot_arm, "\t", robot_cur_state, 0)  # 0, close
        eef_pose_dict_new_temp_list.append([step_id] + robot_cur_state + [0])
        
        robotBase.move_to_target_in_cartesian(target_pose)   # back-rotate
        cur_gripper_state = 0
    #######################################
    if gripper == -4:   # primitive skill: uncover (opened - moved-higher - opened)
        assert cur_gripper_state == 1, "before uncovering, the gripper must be opened!"
        
        robot_cur_state = robotBase.get_current_pose()[0]
        print("******[saving labels]*****", step_id, "\t", robot_arm, "\t", robot_cur_state, 1)  # 1, open
        eef_pose_dict_new_temp_list.append([step_id] + robot_cur_state + [1])
        
        target_pose[1] -= 0.01  # shrink the y value for <lifting the lid>
        target_pose[2] += 0.06  # enlarge the z value for <lifting the lid>
        robotBase.move_to_target_in_cartesian(target_pose)
        cur_gripper_state = 1
    #######################################
    if gripper == -5:   # primitive skill: detach (opened - moved-faraway - opened)
        robot_cur_state = robotBase.get_current_pose()[0]
        print("******[saving labels]*****", step_id, "\t", robot_arm, "\t", robot_cur_state, 1)  # 1, open
        eef_pose_dict_new_temp_list.append([step_id] + robot_cur_state + [1])
        
        target_pose[1] += 0.03  # enlarge the y value for <detaching the lid>
        robotBase.move_to_target_in_cartesian(target_pose)
        cur_gripper_state = cur_gripper_state
    #######################################

    return robotBase, eef_pose_dict_new_temp_list, cur_gripper_state

#################################################################
######################################################################################################################################



######################################################################################################################################
###################################################################
def compute_orient_degree(s_point, e_point, trans_mat):
    temp_pt = np.dot(trans_mat, np.array([s_point[0], s_point[1], 1]).T)
    s_pt_trans = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
    temp_pt = np.dot(trans_mat, np.array([e_point[0], e_point[1], 1]).T)
    e_pt_trans = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
    
    # x_delta, y_delta = e_pt_trans[0] - s_pt_trans[0], e_pt_trans[1] - s_pt_trans[1]
    x_delta, y_delta = e_pt_trans[0] - s_pt_trans[0], s_pt_trans[1] - e_pt_trans[1]  # note this translation of y-axis direction
    rot_rad = math.atan2(y_delta, x_delta)  # the arc tangent of y/x in radians, (-pi, pi)
    rot_deg = 180 * (rot_rad / np.pi)
    return rot_deg, [s_pt_trans, e_pt_trans]

def compute_rotation_by_object_mask(obj_mask, mask_bi, trans_mat, task_name):
    # pts_num = obj_mask.shape[0]
    xmin, xmax = obj_mask[:, 0].min(), obj_mask[:, 0].max()
    ymin, ymax = obj_mask[:, 1].min(), obj_mask[:, 1].max()

    xmin_idxs, xmax_idxs = np.where(obj_mask[:, 0] == xmin), np.where(obj_mask[:, 0] == xmax)
    xmin_yval, xmax_yval = obj_mask[xmin_idxs, 1], obj_mask[xmax_idxs, 1]
    ymin_idxs, ymax_idxs = np.where(obj_mask[:, 1] == ymin), np.where(obj_mask[:, 1] == ymax)
    ymin_xval, ymax_xval = obj_mask[ymin_idxs, 0], obj_mask[ymax_idxs, 0]
    
    xmid, ymid = (xmin + xmax)/2.0, (ymin + ymax)/2.0
    x_left_pixels, x_right_pixels = np.sum(mask_bi[:, :int(xmid)] > 0), np.sum(mask_bi[:, int(xmid):] > 0)
    y_top_pixels, y_down_pixels = np.sum(mask_bi[:int(ymid), :] > 0), np.sum(mask_bi[int(ymid):, :] > 0)
    print("(xl, xr, yt, yd):", x_left_pixels, x_right_pixels, y_top_pixels, y_down_pixels)
    
    center_point = [obj_mask[:, 0].mean(), obj_mask[:, 1].mean()] 
    wh_ratio = (xmax-xmin) / (ymax-ymin)
    
    # "reorient": for these kitchen tools such as spoon, blade and fork
    # "reorient_unscrew": for these lying down bottles with different sizes (width and height)
    if task_name in ["reorient", "reorient_unscrew"]:
        if wh_ratio > 1.05:  # 1.05 is better than 1.00 for being able to avoid approaching 45 or 135 bugs
            print("This object is placed with orientation (-45 ~ 45; -180 ~ -135; 135 ~ 180) degree.")
            if x_left_pixels < x_right_pixels:  # -45 ~ 45
                if ymin_xval.max() > xmid and ymax_xval.max() > xmid:  # about 0
                    pt_tr, pt_dr = [ymin_xval.max(), ymin], [ymax_xval.max(), ymax]
                    end_point = [(pt_tr[0]+pt_dr[0])/2.0, (pt_tr[1]+pt_dr[1])/2.0]
                elif ymin_xval.max() > xmid:  # about 0 ~ 45
                    pt_tr, pt_rt = [ymin_xval.max(), ymin], [xmax, xmax_yval.min()]
                    end_point = [(pt_tr[0]+pt_rt[0])/2.0, (pt_tr[1]+pt_rt[1])/2.0]
                elif ymax_xval.max() > xmid:  # about -45 ~ 0
                    pt_dr, pt_rd = [ymax_xval.max(), ymax], [xmax, xmax_yval.max()]
                    end_point = [(pt_dr[0]+pt_rd[0])/2.0, (pt_dr[1]+pt_rd[1])/2.0]
                else:
                    pt_rt, pt_rd = [xmax, xmax_yval.min()], [xmax, xmax_yval.max()]
                    end_point = [(pt_rt[0]+pt_rd[0])/2.0, (pt_rt[1]+pt_rd[1])/2.0]
            else:  # -180 ~ -135; 135 ~ 180
                if ymin_xval.min() < xmid and ymax_xval.min() < xmid:  # about 180
                    pt_tl, pt_dl = [ymin_xval.min(), ymin], [ymax_xval.min(), ymax]
                    end_point = [(pt_tl[0]+pt_dl[0])/2.0, (pt_tl[1]+pt_dl[1])/2.0]
                elif ymin_xval.min() < xmid:  # about 135 ~ 180 
                    pt_tl, pt_lt = [ymin_xval.min(), ymin], [xmin, xmin_yval.min()]
                    end_point = [(pt_tl[0]+pt_lt[0])/2.0, (pt_tl[1]+pt_lt[1])/2.0]
                elif ymax_xval.min() < xmid:  # about -180 ~ -135
                    pt_dl, pt_ld = [ymax_xval.min(), ymax], [xmin, xmin_yval.max()]
                    end_point = [(pt_dl[0]+pt_ld[0])/2.0, (pt_dl[1]+pt_ld[1])/2.0]
                else:
                    pt_lt, pt_ld = [xmin, xmin_yval.min()], [xmin, xmin_yval.max()]
                    end_point = [(pt_lt[0]+pt_ld[0])/2.0, (pt_lt[1]+pt_ld[1])/2.0]
        else:
            print("This object is placed with orientation (45 ~ 135; -135 ~ -45) degree.")
            if y_top_pixels > y_down_pixels:  # 45 ~ 135
                if xmin_yval.min() < ymid and xmax_yval.min() < ymid:  # about 90
                    pt_lt, pt_rt = [xmin, xmin_yval.min()], [xmax, xmax_yval.min()]
                    end_point = [(pt_lt[0]+pt_rt[0])/2.0, (pt_lt[1]+pt_rt[1])/2.0]
                elif xmin_yval.min() < ymid:  # 90 ~ 135
                    pt_lt, pt_tl = [xmin, xmin_yval.min()], [ymin_xval.min(), ymin]
                    end_point = [(pt_lt[0]+pt_tl[0])/2.0, (pt_lt[1]+pt_tl[1])/2.0]
                elif xmax_yval.min() < ymid:  # 45 ~ 90
                    pt_rt, pt_tr = [xmax, xmax_yval.min()], [ymin_xval.max(), ymin]
                    end_point = [(pt_rt[0]+pt_tr[0])/2.0, (pt_rt[1]+pt_tr[1])/2.0]
                else:
                    pt_tl, pt_tr = [ymin_xval.min(), ymin], [ymin_xval.max(), ymin]
                    end_point = [(pt_tl[0]+pt_tr[0])/2.0, (pt_tl[1]+pt_tr[1])/2.0]
            else:  # -135 ~ -45
                if xmin_yval.max() > ymid and xmax_yval.max() > ymid:  # about -90
                    pt_ld, pt_rd = [xmin, xmin_yval.max()], [xmax, xmax_yval.max()]
                    end_point = [(pt_ld[0]+pt_rd[0])/2.0, (pt_ld[1]+pt_rd[1])/2.0]
                elif xmin_yval.max() > ymid:  # -135 ~ -90
                    pt_ld, pt_dl = [xmin, xmin_yval.max()], [ymax_xval.min(), ymax]
                    end_point = [(pt_ld[0]+pt_dl[0])/2.0, (pt_ld[1]+pt_dl[1])/2.0]
                elif xmax_yval.max() > ymid:  # -90 ~ -45
                    pt_rd, pt_dr = [xmax, xmax_yval.max()], [ymax_xval.max(), ymax]
                    end_point = [(pt_rd[0]+pt_dr[0])/2.0, (pt_rd[1]+pt_dr[1])/2.0]
                else:
                    pt_dl, pt_dr = [ymax_xval.min(), ymax], [ymax_xval.max(), ymax]
                    end_point = [(pt_dl[0]+pt_dr[0])/2.0, (pt_dl[1]+pt_dr[1])/2.0]
        est_ori, pts_trans = compute_orient_degree(center_point, end_point, trans_mat)  # [-180 ~180)
    
    # "plugpen": for these paired mugpens or mugcaps (we can only process degree range -90 ~ 90 now)
    if task_name in ["plugpen"]:
        if wh_ratio > 1.05:  # 1.05 is better than 1.00 for being able to avoid approaching 45 or 135 bugs
            print("This mugpen or mugcap is placed with orientation (-45 ~ 45) degree.")
            if ymin_xval.max() > xmid and ymax_xval.max() > xmid:  # about 0
                pt_tr, pt_dr = [ymin_xval.max(), ymin], [ymax_xval.max(), ymax]
                end_point = [(pt_tr[0]+pt_dr[0])/2.0, (pt_tr[1]+pt_dr[1])/2.0]
            elif ymin_xval.max() > xmid:  # about 0 ~ 45
                pt_tr, pt_rt = [ymin_xval.max(), ymin], [xmax, xmax_yval.min()]
                end_point = [(pt_tr[0]+pt_rt[0])/2.0, (pt_tr[1]+pt_rt[1])/2.0]
            elif ymax_xval.max() > xmid:  # about -45 ~ 0
                pt_dr, pt_rd = [ymax_xval.max(), ymax], [xmax, xmax_yval.max()]
                end_point = [(pt_dr[0]+pt_rd[0])/2.0, (pt_dr[1]+pt_rd[1])/2.0]
            else:
                pt_rt, pt_rd = [xmax, xmax_yval.min()], [xmax, xmax_yval.max()]
                end_point = [(pt_rt[0]+pt_rd[0])/2.0, (pt_rt[1]+pt_rd[1])/2.0]
        else:
            print("This mugpen or mugcap is placed with orientation (45 ~ 90; -90 ~ -45) degree.")
            if xmax_yval.max() < ymid:  # 45 ~ 90
                pt_rt, pt_tr = [xmax, xmax_yval.min()], [ymin_xval.max(), ymin]
                end_point = [(pt_rt[0]+pt_tr[0])/2.0, (pt_rt[1]+pt_tr[1])/2.0]
            elif xmax_yval.min() > ymid:  # -90 ~ -45 
                pt_rd, pt_dr = [xmax, xmax_yval.max()], [ymax_xval.max(), ymax]
                end_point = [(pt_rd[0]+pt_dr[0])/2.0, (pt_rd[1]+pt_dr[1])/2.0]
            else:  # about 90 or -90
                print("[Warning] we cannot process orientation cases about 90 or -90 degrees of mugpen / mugcap now!!!")
                sys.exit()
                
    print("Final computed orientation (degree) and wh_ratio:", est_ori, wh_ratio)
    pts_raw = [[int(center_point[0]), int(center_point[1])], [int(end_point[0]), int(end_point[1])]]
    [center_point_trans, end_point_trans] = pts_trans
    pts_trans = [[int(center_point_trans[0]), int(center_point_trans[1])], [int(end_point_trans[0]), int(end_point_trans[1])]]
    return est_ori, pts_raw, pts_trans

###################################################################

###################################################################
def compute_rotation_by_image_moments(obj_mask, radius_ratio=0.2,
    trans_mat=None, img_file_path=None, task_name=None):

    '''
    # 读取mask图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None  # 无轮廓
    
    # 取最大轮廓
    contour = max(contours, key=cv2.contourArea)
    contour_points = contour.squeeze()
    
    # 计算质心
    M = cv2.moments(contour)
    '''
    
    # remove the influence of perspective transformation for computing the object centroid
    if trans_mat is not None:  
        if task_name in ["unscrew", "pouring", "pressing", "unscrew_pouring", "sweeping"]:
            roi_bbox = [120, 0, 760, 540]  # the [x1, y1, x2, y2] (960*540 --> 640*540; 16:9 --> 32:27)
        if task_name in ["plugpen", "reorient", "inserting", "reorient_unscrew", "tool_spoon", "tool_funnel"]:
            roi_bbox = [140, 90, 740, 540]  # the [x1, y1, x2, y2] (960*540 --> 600*450; 16:9 --> 4:3)
        [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
        obj_mask[:, 0] += x1; obj_mask[:, 1] += y1  # remember add back the offsets in x / y
        
        obj_mask_trans = []
        for obj_pt in obj_mask:
            temp_pt = np.dot(trans_mat, np.array([obj_pt[0], obj_pt[1], 1]).T)
            obj_pt_trans = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
            obj_mask_trans.append(obj_pt_trans)
        
        obj_mask = obj_mask_trans
        trans_mat_t = np.linalg.inv(trans_mat)
      
      
    obj_mask = np.array(obj_mask, dtype=np.int32)  # the obj_mask is in contour_points format with shape (N, 2)
    obj_mask_contour = np.expand_dims(obj_mask, axis=1)  # this is very important (N, 2) --> (N, 1, 2)
    # print(obj_mask.shape, obj_mask_contour.shape)

    # compute the centroid (计算质心) (https://theailearner.com/tag/cv2-moments/)
    M = cv2.moments(obj_mask_contour)
    assert M['m00'] != 0, "The area of given mask should not be zero!!!"
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
    
    # compute the second-order central moment and covariance matrix (计算二阶中心矩和协方差矩阵)[deepseek]
    mu20 = M['mu20'] / M['m00']
    mu11 = M['mu11'] / M['m00']
    mu02 = M['mu02'] / M['m00']
    cov_mat = np.array([[mu20, mu11], [mu11, mu02]])
    
    # compute eigenvectors and direction angles (计算特征向量和方向角度)[deepseek]
    eigenvals, eigenvecs = np.linalg.eigh(cov_mat)
    eigenvec = eigenvecs[:, np.argmax(eigenvals)]  # spindle / main-axis direction (主轴方向)
    theta_rad = np.arctan2(eigenvec[1], eigenvec[0])  # the arc tangent of y/x in radians, (-pi, pi)
    theta_deg = np.degrees(theta_rad) % 360  # (0, 360)
    # print(theta_rad, theta_deg, eigenvals, eigenvecs)
    
    # determine the spindle endpoint (确定主轴端点)[deepseek]
    vx, vy = eigenvec
    projections = (obj_mask[:, 0] - cx) * vx + (obj_mask[:, 1] - cy) * vy
    max_idx, min_idx = np.argmax(projections), np.argmin(projections)
    endpoint1, endpoint2 = obj_mask[max_idx], obj_mask[min_idx]
    
    # compute the object length to determine the analysis radius (计算物体长度以确定分析半径)[deepseek]
    proj_range = np.max(projections) - np.min(projections)
    radius = proj_range * radius_ratio  # radius is the 10% of length (半径取长度的10%)
    
    # compute the width of the area around the endpoint (计算端点周围区域的宽度)[deepseek]
    def get_endpoint_width(endpoint):
        dx, dy = obj_mask[:, 0] - endpoint[0], obj_mask[:, 1] - endpoint[1]
        nearby = obj_mask[(dx**2 + dy**2) <= radius**2]
        if len(nearby) == 0: return np.inf
        perp_proj = (nearby[:, 0] - cx) * (-vy) + (nearby[:, 1] - cy) * vx  # vertical projection (垂直方向投影)
        return np.ptp(perp_proj)  # range as width (极差作为宽度)
    
    width1 = get_endpoint_width(endpoint1)
    width2 = get_endpoint_width(endpoint2)
    
    # determine the front-end endpoint (确定前端端点)[deepseek]
    # width1 > width2: pointing to the larger end; width1 < width2: pointing to the smaller end
    front_endpoint = endpoint1 if width1 < width2 else endpoint2
    
    # adjust angle direction (调整角度方向)[deepseek]
    dx_front = front_endpoint[0] - cx
    dy_front = front_endpoint[1] - cy
    dot_product = dx_front * vx + dy_front * vy
    if dot_product < 0:
        theta_deg = (theta_deg + 180) % 360
    
    # still adjust angle direction for personal use [add by myself]
    theta_deg = 360 - theta_deg  # up-side-down the direction. clockwise (0, 360) --> counter-clockwise (0, 360)
    theta_deg = theta_deg - 180  # adjust the direction from smaller end to larger end. (0, 360) --> (-180, 180)
    
    if task_name in ["plugpen", "inserting"]:  # always let the arrow point to right
        if theta_deg >= -90 and theta_deg < 0:
            theta_deg += 180  # (-90, 0) --> (90, 180)
            front_endpoint = endpoint2 if width1 < width2 else endpoint1  # adjust the end point
        if theta_deg >= 0 and theta_deg <= 90:
            theta_deg -= 180  # (0, 90) --> (-180, -90)
            front_endpoint = endpoint2 if width1 < width2 else endpoint1  # adjust the end point
    
    if trans_mat is not None:
        temp_pt = np.dot(trans_mat_t, np.array([cx, cy, 1]).T)
        s_pt_raw = [temp_pt[0]/temp_pt[2] - x1, temp_pt[1]/temp_pt[2] - y1] # remember subtract the offsets in x / y
        temp_pt = np.dot(trans_mat_t, np.array([front_endpoint[0], front_endpoint[1], 1]).T)
        e_pt_raw = [temp_pt[0]/temp_pt[2] - x1, temp_pt[1]/temp_pt[2] - y1]
     
    if img_file_path is not None:  # visualization for debugging [add by myself]
        img_raw = cv2.imread(img_file_path)
        if trans_mat is not None:
            start_x, start_y, end_x, end_y = int(s_pt_raw[0]), int(s_pt_raw[1]), int(e_pt_raw[0]), int(e_pt_raw[1])
        else:
            start_x, start_y, end_x, end_y = int(cx), int(cy), int(front_endpoint[0]), int(front_endpoint[1])
        cv2.arrowedLine(img_raw, (start_x, start_y), (end_x, end_y), (0,0,0), thickness=3, tipLength = 0.15)
        cv2.circle(img_raw, (start_x, start_y), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img_raw, (end_x, end_y), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(img_raw, str(np.round(theta_deg, 0)), (start_x+5, start_y-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.6, color=(255,255,0), thickness=2, lineType=cv2.LINE_AA)
        img_vis_save_path = img_file_path.replace("rect.jpg", "rect_vis.jpg")
        cv2.imwrite(img_vis_save_path, img_raw)
    
    if trans_mat is not None:
        pts_raw = [[int(s_pt_raw[0]), int(s_pt_raw[1])], [int(e_pt_raw[0]), int(e_pt_raw[1])]]
        pts_trans = [[int(cx), int(cy)], [int(front_endpoint[0]), int(front_endpoint[1])]]
    else:
        pts_raw = [[int(cx), int(cy)], [int(front_endpoint[0]), int(front_endpoint[1])]]
        pts_trans = [[-1, -1], [-1, -1]]
        
    return theta_deg, pts_raw, pts_trans

###################################################################
######################################################################################################################################


