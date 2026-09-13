
import numpy as np
from scipy.spatial.transform import Rotation as R

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
            temp_pose_deg = [target_pose_back[3], [target_pose_back[4] + twist_deg], target_pose_back[5]]   
        if rot_axis == "Z":  # rotate around X-axis
            temp_pose_deg = target_pose_back[3:5] + [target_pose_back[5] + twist_deg]

    if skill_name == "pour":  # (closed - moved&rotate - closed - back-rotate) (only moved&rotate part)
        pour_deg = rot_deg
        if rot_axis == "X":  # rotate around X-axis
            temp_pose_deg = [target_pose_back[3] - pour_deg] + target_pose_back[4:]
        if rot_axis == "Y":  # rotate around Y-axis
            temp_pose_deg = [target_pose_back[3], [target_pose_back[4] - pour_deg], target_pose_back[5]]   
        if rot_axis == "Z":  # rotate around X-axis
            temp_pose_deg = target_pose_back[3:5] + [target_pose_back[5] - pour_deg]

    target_6dof_pose = np.eye(4)
    target_6dof_pose[:3, :3] = R.from_euler("xyz", temp_pose_deg, degrees=True).as_matrix()
    target_6dof_pose[:3, -1] = target_pose_back[:3]
    target_6dof_pose = tcp_fix_gripper(target_6dof_pose)  # add tool length
    temp_eef_pose_deg = list(target_6dof_pose[:3, -1]) + temp_pose_deg
    print(target_pose, "\n", target_pose_back, "\n", temp_eef_pose_deg)

    return temp_eef_pose_deg