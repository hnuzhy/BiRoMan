
import os
import cv2
import time
import numpy as np
import math
from math import cos, sin
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp  # line interpolate in SO(3) manifold space
from scipy.interpolate import make_interp_spline  # Cubic B-spline in 3D Euclidean space

try:
    from config import cfg_dict_init
    from algs.slots_v1 import find_dense_corners_of_a_deformable_cloth
    from algs.rope_uncross import RopeSkeletonExtractor
except:
    from .config import cfg_dict_init
    from .algs.slots_v1 import find_dense_corners_of_a_deformable_cloth
    from .algs.rope_uncross import RopeSkeletonExtractor
    
###################################################################################################################################
#################################################################
def tcp_fix_gripper(pose: np.ndarray, is_back=False):
    # gripper_offset_z, gripper_offset_y, gripper_offset_x = 0.161, 0.000, 0.000  # using legal 3dof pose, for arm2-drawer
    gripper_offset_z, gripper_offset_y, gripper_offset_x = cfg_dict_init["tool_length"], 0.000, 0.000
    
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
#################################################################
###################################################################################################################################




###################################################################################################################################
#################################################################
def normalize(v):
    return v / np.linalg.norm(v)

def slerp_vectors(v0, v1, t_array):
    v0_norm = normalize(v0)
    v1_norm = normalize(v1)
    omega = np.arccos(np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0))
    sin_omega = np.sin(omega)
    
    if sin_omega < 1e-6:
        return np.outer(1 - t_array, v0) + np.outer(t_array, v1)
    
    vec_list = (np.sin((1 - t_array) * omega)[:, None] * v0 +
            np.sin(t_array * omega)[:, None] * v1) / sin_omega
    
    return vec_list

def interpolate_6dof_on_sphere(center, start_pose_R_ad, target_pose_R_ad, num_steps=5):
    
    start_pos, end_pos = start_pose_R_ad[:3], target_pose_R_ad[:3]
    start_quat = R.from_euler("xyz", start_pose_R_ad[3:], degrees=True).as_quat()
    end_quat = R.from_euler("xyz", target_pose_R_ad[3:], degrees=True).as_quat()

    # Relative 3d vector
    v0 = start_pos - center
    v1 = end_pos - center
    
    # Spherical interpolation position
    t_array = np.linspace(0, 1, num_steps)
    arc_positions = slerp_vectors(v0, v1, t_array) + center

    # Pose interpolation (quaternion)
    slerp_pose = Slerp([0, 1], R.from_quat([start_quat, end_quat]))
    interp_rotmats = slerp_pose(t_array)
    interp_quats = interp_rotmats.as_quat()

    # Merge location and pose
    trajectory = []
    for position_res, quaternion_res in zip(arc_positions, interp_quats):
        target_pose_temp = [0,0,0, 0,0,0]
        target_pose_temp[:3] = list(position_res)
        target_pose_temp[3:] = R.from_quat(quaternion_res).as_euler("xyz", degrees=True)
        trajectory.append(target_pose_temp)

    return trajectory

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
def interpolate_se3_bspline_multiple_poses(pose_6dof_list, num_steps=10):
    positions, quaternions = []
    for pose_6dof in pose_6dof_list:
        positions.append(pose_6dof[:3])
        quat_part = R.from_euler("xyz", pose_6dof[3:], degrees=True).as_quat()
        quaternions.append(quat_part)

    positions = np.array(positions)  # List or array of shape (N, 3)
    quaternions = np.array(quaternions)  # List or array of shape (N, 4) — should be normalized
    num_control_points = len(positions)
    assert num_control_points >= 2, "At least two control poses are required."

    # === Time parameterization ===
    t_control = np.linspace(0, 1, num_control_points)
    t_dense = np.linspace(0, 1, num_steps)  # Number of total interpolated steps along the spline

    # === Position B-spline ===
    spline_pos = make_interp_spline(t_control, positions, k=min(3, num_control_points - 1))
    interp_positions = spline_pos(t_dense)

    # === Orientation interpolation using Slerp ===
    rot_control = R.from_quat(quaternions)
    slerp_pose = Slerp(t_control, rot_control)
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
        if task_name in ["unscrew", "pouring", "pressing", "unscrew_pouring"]:
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

#################################################################
###################################################################################################################################



###################################################################################################################################
#################################################################
def compute_abstract_points(obj_mask, obj_name_str):
    roi_bbox = cfg_dict_init["detection_roi_bbox"]
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    
    trans_mat = cfg_dict_init["transform_mat_inv"]
    trans_mat_t = cfg_dict_init["transform_mat"]
    
    # remove the influence of perspective transformation for computing the object centroid
    obj_mask[:, 0] += x1; obj_mask[:, 1] += y1  # remember add back the offsets in x / y
    obj_mask_temp = obj_mask.copy(); obj_mask_trans = []
    for obj_pt in obj_mask_temp:
        temp_pt = np.dot(trans_mat, np.array([obj_pt[0], obj_pt[1], 1]).T)
        obj_pt_trans = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]; obj_mask_trans.append(obj_pt_trans)

    ##### point type 1: for the object centroid point
    obj_mask_trans = np.array(obj_mask_trans, dtype=np.int32)  # the obj_mask is in contour_points format with shape (N, 2)
    obj_mask_contour = np.expand_dims(obj_mask_trans, axis=1)  # this is very important (N, 2) --> (N, 1, 2)
    M = cv2.moments(obj_mask_contour)  # compute the centroid (https://theailearner.com/tag/cv2-moments/)
    assert M['m00'] != 0, "The area of given [obj_mask] should not be zero!!!"
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']; cpt_project = [cx, cy]
    temp_pt = np.dot(trans_mat_t, np.array([cx, cy, 1]).T)
    cpt_origin = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]

    ##### point type 2: for the bottom-grounding point
    if obj_name_str == "cirbowl" or obj_name_str == "ball":
        ref_y_val = obj_mask[:, 1].max()  # max y_val
        ref_x_val = obj_mask[list(obj_mask[:, 1]).index(ref_y_val), 0]  # corresponding x_val
    if obj_name_str == "rectbox" or obj_name_str == "bottle" or \
        obj_name_str == "ordcup" or obj_name_str == "mugcup" or obj_name_str == "basket" or obj_name_str == "cuboid":
        ref_x_val = (obj_mask[:, 0].min() + obj_mask[:, 0].max()) // 2  # middle x_val
        ref_y_val = obj_mask[:, 1].max()  # max y_val
    rpt_origin = [ref_x_val, ref_y_val]  # only one single representative point
    
    temp_pt = np.dot(trans_mat, np.array([ref_x_val, ref_y_val, 1]).T)
    rpt_project = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]

    return cpt_origin, cpt_project, rpt_origin, rpt_project, obj_mask, obj_mask_trans

#################################################################
def compute_pts_delta_xy(cur_o_mask, tgt_o_mask, task_name_str, supp_masks):
    try:
        cur_cpt_o, cur_cpt_p, cur_rpt_o, cur_rpt_p, cur_o_mask_o, cur_o_mask_p = compute_abstract_points(cur_o_mask, task_name_str)
    except:
        print(cur_o_mask.shape, cur_o_mask)  
    tgt_cpt_o, tgt_cpt_p, tgt_rpt_o, tgt_rpt_p, tgt_o_mask_o, tgt_o_mask_p = compute_abstract_points(tgt_o_mask, task_name_str)

    delta_xy_mm = [cur_rpt_p[0] - tgt_rpt_p[0], cur_rpt_p[1] - tgt_rpt_p[1]]
    cur_pt0, tgt_pt0 = cur_cpt_o, tgt_cpt_o  # the object centroid point
    cur_pt1, tgt_pt1 = cur_rpt_o, tgt_rpt_o  # the bottom-grounding point
    
    computed_res_list = [delta_xy_mm, cur_pt0, tgt_pt0, cur_pt1, tgt_pt1]
    return computed_res_list

#################################################################


##############################################################################################
##############################################################################################

#################################################################
def compute_abstract_points_mixed(obj_mask, task_name_str, obj_name_str):
    roi_bbox = cfg_dict_init["detection_roi_bbox"]
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    
    trans_mat = cfg_dict_init["transform_mat_inv"]
    trans_mat_t = cfg_dict_init["transform_mat"]
    
    # remove the influence of perspective transformation for computing the object centroid
    obj_mask[:, 0] += x1; obj_mask[:, 1] += y1  # remember add back the offsets in x / y
    obj_mask_temp = obj_mask.copy(); obj_mask_trans = []
    for obj_pt in obj_mask_temp:
        temp_pt = np.dot(trans_mat, np.array([obj_pt[0], obj_pt[1], 1]).T)
        obj_pt_trans = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]; obj_mask_trans.append(obj_pt_trans)

    ##### point type 1: for the object centroid point
    obj_mask_trans = np.array(obj_mask_trans, dtype=np.int32)  # the obj_mask is in contour_points format with shape (N, 2)
    obj_mask_contour = np.expand_dims(obj_mask_trans, axis=1)  # this is very important (N, 2) --> (N, 1, 2)
    M = cv2.moments(obj_mask_contour)  # compute the centroid (https://theailearner.com/tag/cv2-moments/)
    assert M['m00'] != 0, "The area of given [obj_mask] should not be zero!!!"
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']; cpt_project = [cx, cy]
    temp_pt = np.dot(trans_mat_t, np.array([cx, cy, 1]).T)
    cpt_origin = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]

    ##### point type 2: for the bottom-grounding point
    ref_x_val, ref_y_val= 0.0, 0.0
    if (task_name_str == "pouring" and obj_name_str == "bottle") or (task_name_str == "pouring" and obj_name_str == "cup") or \
        (task_name_str == "unscrew" and obj_name_str == "bottle") or (task_name_str == "flatting" and obj_name_str == "bottle") or \
        (task_name_str == "flipping" and obj_name_str == "cup") or \
        (task_name_str == "unscrew-pouring" and obj_name_str == "bottle") or (task_name_str == "unscrew-pouring" and obj_name_str == "cup") or \
        (task_name_str == "flatting-reorient" and obj_name_str == "bottle") or \
        (task_name_str == "grasping_rectbox" and "box" in obj_name_str) or \
        (task_name_str == "inserting" and obj_name_str == "cup") or \
        (task_name_str == "pivoting" and "bowl" in obj_name_str) or (task_name_str == "pivoting_cirbowl" and "bowl" in obj_name_str) :
        ref_y_val = obj_mask[:, 1].max()  # max y_val
        ref_x_val = obj_mask[list(obj_mask[:, 1]).index(ref_y_val), 0]  # corresponding x_val
    if (task_name_str == "reorient" and obj_name_str == "bottle") or (task_name_str == "grasping" and obj_name_str == "cup") or \
        (task_name_str == "grasping_cirbowl" and "bowl" in obj_name_str) or (task_name_str == "grasping_basket" and "basket" in obj_name_str) or \
        (task_name_str == "grasping_holder" and "cup" in obj_name_str) or (task_name_str == "grasping_pencup" and "cup" in obj_name_str) or \
        (task_name_str == "wrapping" and "basket" in obj_name_str) or (task_name_str == "pivoting_rectbox" and "box" in obj_name_str) or \
        (task_name_str == "flipping_basket" and "basket" in obj_name_str) or (task_name_str == "flipping_block" and "block" in obj_name_str) or \
        (task_name_str == "pivoting_bigjar" and "jar" in obj_name_str) or (task_name_str == "pivoting_block" and "block" in obj_name_str) or \
        (task_name_str == "toppling_holder" and "cup" in obj_name_str) or (task_name_str == "toppling_bigjar" and "jar" in obj_name_str) or \
        (task_name_str == "bilifting_bigjar" and "jar" in obj_name_str) or (task_name_str == "bilifting_block" and "block" in obj_name_str) :
        ref_x_val = (obj_mask[:, 0].min() + obj_mask[:, 0].max()) // 2  # middle x_val
        ref_y_val = obj_mask[:, 1].max()  # max y_val
    rpt_origin = [ref_x_val, ref_y_val]  # only one single representative point
    
    temp_pt = np.dot(trans_mat, np.array([ref_x_val, ref_y_val, 1]).T)
    rpt_project = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]

    return cpt_origin, cpt_project, rpt_origin, rpt_project, obj_mask, obj_mask_trans

#################################################################

def compute_orientation_by_image_moments_180(obj_mask):

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
    # theta_rad = np.arctan2(eigenvec[1], eigenvec[0])  # the arc tangent of y/x in radians, (-pi, pi)
    # theta_deg = np.degrees(theta_rad) % 360  # (0, 360)
    # print(theta_rad, theta_deg, eigenvals, eigenvecs)
    
    # determine the spindle endpoint (确定主轴端点)[deepseek]
    vx, vy = eigenvec
    projections = (obj_mask[:, 0] - cx) * vx + (obj_mask[:, 1] - cy) * vy
    max_idx, min_idx = np.argmax(projections), np.argmin(projections)
    endpoint1, endpoint2 = obj_mask[max_idx], obj_mask[min_idx]

    # always let the arrow point to right
    if endpoint1[0] >= endpoint2[0]: endpoint_L = endpoint2; endpoint_R = endpoint1
    if endpoint1[0] < endpoint2[0]: endpoint_L = endpoint1; endpoint_R = endpoint2

    offset_x = endpoint_R[0] - endpoint_L[0]; offset_y = endpoint_L[1] - endpoint_R[1]
    theta_rad = np.arctan2(offset_y, offset_x)  # the arc tangent of y/x in radians, (-pi, pi)
    theta_deg = np.degrees(theta_rad) % 360  # (0, 360)

    trans_mat_t = cfg_dict_init["transform_mat"]
    temp_pt = np.dot(trans_mat_t, np.array([endpoint_L[0], endpoint_L[1], 1]).T)
    endpoint_L = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
    temp_pt = np.dot(trans_mat_t, np.array([endpoint_R[0], endpoint_R[1], 1]).T)
    endpoint_R = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
         
    if theta_deg > 90 and theta_deg <= 180: theta_deg = 90 - theta_deg  # (90, 180] --> [-90, 0) 
    if theta_deg > 180 and theta_deg <= 270: theta_deg = theta_deg - 180  # (180, 270] --> (0, 90]
    if theta_deg > 270 and theta_deg <= 360: theta_deg = theta_deg - 360  # (270, 360] --> (-90, 0]

    return theta_deg, endpoint_L, endpoint_R


def compute_orientation_by_image_moments_360(obj_mask):

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
    radius_ratio = 0.1  # radius is the 10% of length (半径取长度的10%)
    radius = proj_range * radius_ratio
    
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

    if width1 < width2: endpoint_start = endpoint1; endpoint_end = endpoint2  # pointing to the smaller end
    if width1 >= width2: endpoint_start = endpoint2; endpoint_end = endpoint1  # pointing to the larger end
    
    # adjust angle direction (调整角度方向)[deepseek]
    dx_front = front_endpoint[0] - cx
    dy_front = front_endpoint[1] - cy
    dot_product = dx_front * vx + dy_front * vy
    if dot_product < 0:
        theta_deg = (theta_deg + 180) % 360

    # still adjust angle direction for personal use [add by myself]
    theta_deg = 360 - theta_deg  # up-side-down the direction. clockwise (0, 360) --> counter-clockwise (0, 360) (-x -> -y -> +x -> +y)
    theta_deg = (theta_deg - 180) % 360  # align with the direction in reorient task. counter-clockwise (0, 360) (+x -> +y -> -x -> -y)

    trans_mat_t = cfg_dict_init["transform_mat"]
    temp_pt = np.dot(trans_mat_t, np.array([endpoint_start[0], endpoint_start[1], 1]).T)
    endpoint_start = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
    temp_pt = np.dot(trans_mat_t, np.array([endpoint_end[0], endpoint_end[1], 1]).T)
    endpoint_end = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
         
    return theta_deg, endpoint_start, endpoint_end


#################################################################
def compute_pts_abstract_xy_relative_rot(obj_mask, task_name_str, obj_name_str, supp_mask=None, binary_mask=None):
    cpt_o, cpt_p, rpt_o, rpt_p, obj_mask_o, obj_mask_p = compute_abstract_points_mixed(obj_mask.copy(), task_name_str, obj_name_str)

    if (task_name_str == "pouring" and obj_name_str == "bottle") or (task_name_str == "pouring" and obj_name_str == "cup") or \
        (task_name_str == "unscrew" and obj_name_str == "bottle") or (task_name_str == "grasping" and obj_name_str == "cup") or \
        (task_name_str == "flatting" and obj_name_str == "bottle") or (task_name_str == "flipping" and obj_name_str == "cup") or \
        (task_name_str == "unscrew-pouring" and obj_name_str == "bottle") or (task_name_str == "unscrew-pouring" and obj_name_str == "cup") or \
        (task_name_str == "flatting-reorient" and obj_name_str == "bottle") or \
        (task_name_str == "grasping_rectbox" and "box" in obj_name_str) or (task_name_str == "grasping_cirbowl" and "bowl" in obj_name_str) or \
        (task_name_str == "grasping_basket" and "basket" in obj_name_str) or (task_name_str == "grasping_holder" and "cup" in obj_name_str) or \
        (task_name_str == "grasping_pencup" and "cup" in obj_name_str) or \
        (task_name_str == "inserting" and obj_name_str == "cup") or \
        (task_name_str == "pivoting" and "bowl" in obj_name_str) or (task_name_str == "wrapping" and "basket" in obj_name_str) or \
        (task_name_str == "pivoting_rectbox" and "box" in obj_name_str) or (task_name_str == "pivoting_cirbowl" and "bowl" in obj_name_str) or \
        (task_name_str == "flipping_basket" and "basket" in obj_name_str) or (task_name_str == "flipping_block" and "block" in obj_name_str) or \
        (task_name_str == "pivoting_bigjar" and "jar" in obj_name_str) or (task_name_str == "pivoting_block" and "block" in obj_name_str) or \
        (task_name_str == "toppling_holder" and "cup" in obj_name_str) or (task_name_str == "toppling_bigjar" and "jar" in obj_name_str) or \
        (task_name_str == "bilifting_bigjar" and "jar" in obj_name_str) or (task_name_str == "bilifting_block" and "block" in obj_name_str) :
        ref_pts_rot_list = [rpt_o, rpt_p, 0.0, None, None]  # only one single point

    if (task_name_str == "reorient" and obj_name_str == "bottle"):
        cpt_o_supp, cpt_p_supp, _, _, _, _ = compute_abstract_points_mixed(supp_mask, task_name_str, obj_name_str)  # the cap / handle mask for computing
        offset_y = cpt_p[1] - cpt_p_supp[1]  # the vector is from obj_center_point --> supp_center_point. the y-value should be reversed
        offset_x = cpt_p_supp[0] - cpt_p[0]  # the vector is from obj_center_point --> supp_center_point. the x-value does not need adjusting
        rel_theta_rad = np.arctan2(offset_y, offset_x)  # the arc tangent of y/x in radians, (-pi, pi)
        rel_theta_deg = np.degrees(rel_theta_rad) % 360  # (0, 360)
        ref_pts_rot_list = [cpt_o, cpt_p, rel_theta_deg, cpt_o_supp, cpt_p_supp]  # only one single point
    
    if (task_name_str == "flipping" and obj_name_str == "cup"):
        ##### WAY 1: check the relative position of px_rep to px_mid . failed
        #px_mid = (obj_mask_p[:, 0].min() + obj_mask_p[:, 0].max()) * 0.5
        #rel_theta_deg = 180 if rpt_p[0] > px_mid else 0  # 180 --> left handle; 0 --> right handle
        ##### WAY 2: count the point number in left/right skeleton. failed
        #pts_count_L = np.sum(obj_mask_p[:, 0] < rpt_p[0])
        #pts_count_R = np.sum(obj_mask_p[:, 0] > rpt_p[0])
        #rel_theta_deg = 180 if pts_count_L > pts_count_R else 0  # 180 --> left handle; 0 --> right handle
        ##### WAY 3: search-find-compare the min_line_L / min_line_R. better
        min_line_L, min_line_R, sample_num = 10000, 10000, 5
        px_min, px_max = obj_mask_p[:, 0].min(), obj_mask_p[:, 0].max()
        L_step_len, R_step_len = (rpt_p[0] - px_min)*1.0 / sample_num, (px_max - rpt_p[0])*1.0 / sample_num
        px_L_start, px_L_end = int(px_min + L_step_len), int(rpt_p[0] - L_step_len)
        for sx in range(px_L_start, px_L_end, 2):
            sy_array = obj_mask_p[obj_mask_p[:, 0] == sx, 1]
            if len(sy_array) <= 1: continue
            temp_line_L = sy_array.max() - sy_array.min();  # print("L\t", temp_line_L)
            if temp_line_L != 0 and temp_line_L < min_line_L: min_line_L = temp_line_L
        px_R_start, px_R_end = int(rpt_p[0] + R_step_len), int(px_max - R_step_len)
        for sx in range(px_R_start, px_R_end, 2):
            sy_array = obj_mask_p[obj_mask_p[:, 0] == sx, 1]
            if len(sy_array) <= 1: continue
            temp_line_R = sy_array.max() - sy_array.min();  # print("R\t", temp_line_R)
            if temp_line_R != 0 and temp_line_R < min_line_R: min_line_R = temp_line_R
        rel_theta_deg = 180 if min_line_L < min_line_R else 0  # 180 --> left handle; 0 --> right handle
        #####
        ref_pts_rot_list = [rpt_o, rpt_p, rel_theta_deg, None, None]

    if (task_name_str == "inserting" and obj_name_str == "pen") or (task_name_str == "penbagzip" and obj_name_str == "pen") or \
        (task_name_str == "plugpen" and "marker" in obj_name_str):
        rel_theta_deg, endpoint_L, endpoint_R = compute_orientation_by_image_moments_180(obj_mask_p)
        ref_pts_rot_list = [cpt_o, cpt_p, rel_theta_deg, endpoint_L, endpoint_R]  # only one single point

    if (task_name_str == "ppspoon" and obj_name_str == "spoon") or (task_name_str == "ppfork" and obj_name_str == "fork") or \
        (task_name_str == "ppspoon-ppfork" and obj_name_str == "spoon") or (task_name_str == "ppspoon-ppfork" and obj_name_str == "fork") or \
        (task_name_str == "ppfork-ppspoon" and obj_name_str == "spoon") or (task_name_str == "ppfork-ppspoon" and obj_name_str == "fork") or \
        (task_name_str == "handover" and "spoon" in obj_name_str) or (task_name_str == "handover" and "shovel" in obj_name_str):
        rel_theta_deg, endpoint1, endpoint2 = compute_orientation_by_image_moments_360(obj_mask_p)
        ref_pts_rot_list = [cpt_o, cpt_p, rel_theta_deg, endpoint1, endpoint2]  # only one single point

    if (task_name_str == "penbagzip" and "bag" in obj_name_str) or (task_name_str == "plugpen" and "cap" in obj_name_str):
        ref_pts_rot_list = [cpt_o, cpt_p, 0.0, None, None]  # only one single point
        
    if (task_name_str == "foldtowel") or (task_name_str == "foldpants") or (task_name_str == "foldshirt"):
        if "towel" in obj_name_str:  # a rectangle towel with four keypoints / corners
            return_res_list = find_dense_corners_of_a_deformable_cloth(obj_mask, 4, "towel", get_mid_anchor=True)
            [cpt_p, cpt_o, corners_p, corners_o, corners_p_mid, corners_o_mid] = return_res_list
            interp_alpha = 0.1
            # mid_12_p = np.array([(corners_p[1][0] + corners_p[2][0])*0.5, (corners_p[1][1] + corners_p[2][1])*0.5])
            # mid_03_p = np.array([(corners_p[0][0] + corners_p[3][0])*0.5, (corners_p[0][1] + corners_p[3][1])*0.5])
            # mid_12_o = np.array([(corners_o[1][0] + corners_o[2][0])*0.5, (corners_o[1][1] + corners_o[2][1])*0.5])
            # mid_03_o = np.array([(corners_o[0][0] + corners_o[3][0])*0.5, (corners_o[0][1] + corners_o[3][1])*0.5])
            mid_12_p, mid_03_p, mid_12_o, mid_03_o = corners_p_mid[1], corners_p_mid[3], corners_o_mid[1], corners_o_mid[3]
            if mid_12_p[0] <= mid_03_p[0]: s_pt_L_p, s_pt_R_p, s_pt_L_o, s_pt_R_o = mid_12_p, mid_03_p, mid_12_o, mid_03_o
            elif mid_12_p[0] > mid_03_p[0]: s_pt_L_p, s_pt_R_p, s_pt_L_o, s_pt_R_o = mid_03_p, mid_12_p, mid_03_o, mid_12_o
            else: print("We dot not consider other conditions now !!! quit the script!!!"); os._exit(0)

        if "shirt" in obj_name_str:  # a T-shirt with quite short sleeves (seven keypoints)
            return_res_list = find_dense_corners_of_a_deformable_cloth(obj_mask, 7, "T-shirt")
            [cpt_p, cpt_o, corners_p, corners_o, corners_p_mid, corners_o_mid] = return_res_list
            corners_p, corners_o = np.array(corners_p), np.array(corners_o)
            interp_alpha = 0.1
            if min(corners_p[1][1], corners_p[2][1]) > max(corners_p[4][1], corners_p[5][1], corners_p[6][1]):
                s_pt_L_p = corners_p[3] + 0.67 * (corners_p[2] - corners_p[3])
                s_pt_R_p = corners_p[0] + 0.67 * (corners_p[1] - corners_p[0])
                s_pt_L_o = corners_o[3] + 0.67 * (corners_o[2] - corners_o[3])
                s_pt_R_o = corners_o[0] + 0.67 * (corners_o[1] - corners_o[0])
            elif max(corners_p[1][1], corners_p[2][1]) < min(corners_p[4][1], corners_p[5][1], corners_p[6][1]):
                s_pt_L_p = corners_p[6] + interp_alpha * (corners_p[5] - corners_p[6])
                s_pt_R_p = corners_p[4] + interp_alpha * (corners_p[5] - corners_p[4])
                s_pt_L_o = corners_o[6] + interp_alpha * (corners_o[2] - corners_o[6])
                s_pt_R_o = corners_o[4] + interp_alpha * (corners_o[1] - corners_o[4])
            else: print("We dot not consider other conditions now !!! quit the script!!!"); os._exit(0)
            
        grasp_point_L_p = s_pt_L_p + interp_alpha * (np.array(cpt_p) - s_pt_L_p)
        grasp_point_R_p = s_pt_R_p + interp_alpha * (np.array(cpt_p) - s_pt_R_p) 
        grasp_point_L_o = s_pt_L_o + interp_alpha * (np.array(cpt_o) - s_pt_L_o)
        grasp_point_R_o = s_pt_R_o + interp_alpha * (np.array(cpt_o) - s_pt_R_o)  
        
        grasp_point_L = (grasp_point_L_o.tolist(), grasp_point_L_p.tolist())  # make it a tuple yet not list
        grasp_point_R = (grasp_point_R_o.tolist(), grasp_point_R_p.tolist())  # make it a tuple yet not list
        ref_pts_rot_list = [cpt_o, cpt_p, 0.0, grasp_point_L, grasp_point_R]  # one center point + two grasp points

    if (task_name_str == "coilcable") or (task_name_str == "coilrope") or (task_name_str == "coilbelt"):
        assert binary_mask is not None, f"You should have to give the [binary_mask] for this task {task_name_str}!!!"
        
        trans_mat = cfg_dict_init["transform_mat_inv"]
        kfr_height, kfr_width = cfg_dict_init['kfr_height'], cfg_dict_init['kfr_width']  # 540, 960
        [x1, y1, x2, y2] = cfg_dict_init["detection_roi_bbox"]  # for fast and stable detection and segmentation
        top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
        binary_mask = cv2.copyMakeBorder(binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
        
        if "cable" in obj_name_str:
            extractor = RopeSkeletonExtractor(blur_ksize=11, prune_threshold=15)
            path_xy, result_skel, origin_skel, smooth_mask = extractor.extract(binary_mask, cluster_thre=40)
            quantiles_anchors, quantiles_indices = extractor.get_quantiles(path_xy, segment_num=4)  # five anchors
            curve_middle_point = quantiles_anchors[2]  # also the grasp point, from the list (0, 1/4, 1/2, 3/4, 1)
            nearest_pt, tangent, angle, normal = extractor.get_grasp_info(path_xy, target_point=curve_middle_point)
            
            cpt_o = curve_middle_point.copy()
            temp_pt = np.dot(trans_mat, np.array([cpt_o[0], cpt_o[1], 1]).T)
            cpt_p = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
            rel_theta_rad = np.arctan2(-normal[1], normal[0])  # the arc tangent of y/x in radians, (-pi, pi)
            rel_theta_deg = np.degrees(rel_theta_rad) % 360  # (0, 360)
            start_tan = [int(cpt_o[0] - tangent[0] * 50), int(cpt_o[1] - tangent[1] * 50)]
            end_tan = [int(cpt_o[0] + tangent[0] * 50), int(cpt_o[1] + tangent[1] * 50)]
            ref_pts_rot_list = [cpt_o, cpt_p, rel_theta_deg, (start_tan, start_tan), (end_tan, end_tan)]
        
    return ref_pts_rot_list

###################################################################################################################################





###################################################################################################################################
##################################################################
# (right) method-2: https://github.com/hnuzhy/DirectMHP/blob/main/exps/compare_img2pose.py#L91
def plot_3axis_Zaxis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50., limited=True, thickness=2):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    
    if tdx != None and tdy != None:
        face_x = tdx
        face_y = tdy
    else:
        height, width = img.shape[:2]
        face_x = width / 2
        face_y = height / 2

    # X-Axis (pointing to right) drawn in red
    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    
    # Y-Axis (pointing to down) drawn in green
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Plot head oritation line in black
    # scale_ratio = 5
    scale_ratio = 2
    base_len = math.sqrt((face_x - x3)**2 + (face_y - y3)**2)
    if face_x == x3:
        endx = tdx
        if face_y < y3:
            if limited:
                endy = tdy + (y3 - face_y) * scale_ratio
            else:
                endy = img.shape[0]
        else:
            if limited:
                endy = tdy - (face_y - y3) * scale_ratio
            else:
                endy = 0
    elif face_x > x3:
        if limited:
            endx = tdx - (face_x - x3) * scale_ratio
            endy = tdy - (face_y - y3) * scale_ratio
        else:
            endx = 0
            endy = tdy - (face_y - y3) / (face_x - x3) * tdx
    else:
        if limited:
            endx = tdx + (x3 - face_x) * scale_ratio
            endy = tdy + (y3 - face_y) * scale_ratio
        else:
            endx = img.shape[1]
            endy = tdy - (face_y - y3) / (face_x - x3) * (tdx - endx)
    # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0,0,0), 2)
    # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (255,255,0), 2)
    # not plot the extend line
    # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0,255,255), thickness)

    # X-Axis pointing to right. drawn in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),thickness, lineType=cv2.LINE_AA)
    # Y-Axis pointing to down. drawn in green    
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,255,0),thickness, lineType=cv2.LINE_AA)
    # Z-Axis (out of the screen) drawn in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),thickness, lineType=cv2.LINE_AA)

    return img

##################################################################

def plot_primitive_skill_traj(img_canvas, armL_traj=[], armR_traj=[], fix_point=None, paper_flag=False):
    
    assert len(armL_traj) != 0 or len(armR_traj) != 0, "[ERROR] You should at least give one trajectory of either arm!!!"
    handeye_camera_L = cfg_dict_init["arm1"]["handeye_para_v2"]  # for kingfisher-R-6000 to robot left
    handeye_camera_R = cfg_dict_init["arm2"]["handeye_para_v2"]  # for kingfisher-R-6000 to robot right
    camL_para = cfg_dict_init["cam1_K"]; camR_para = cfg_dict_init["cam2_K"]  # for kingfisher-R-6000
    
    if len(armL_traj) != 0:
        # step 1: calculate the 6-DoF pose after removing tool length
        poses_adjusted = []
        for temp_pose in armL_traj:
            target_6dof_pose_back = np.eye(4)
            target_6dof_pose_back[:3, :3] = R.from_euler("xyz", temp_pose[3:], degrees=True).as_matrix()
            target_6dof_pose_back[:3, -1] = temp_pose[:3]
            target_6dof_pose_back = tcp_fix_gripper(target_6dof_pose_back, is_back=True)  # is_back=True, remove tool length
            target_pose_back_cam = [0,0,0, 0,0,0]
            target_pose_back_cam[:3] = handeye_camera_L[:3, :3].T @ (target_6dof_pose_back[:3, -1] - handeye_camera_L[:3, -1])  # armL --> camera
            rot_mat_camera_world = handeye_camera_L[:3, :3].T @ target_6dof_pose_back[:3, :3]
            target_pose_back_cam[3:] = R.from_matrix(rot_mat_camera_world).as_euler("xyz", degrees=True)
            poses_adjusted.append(target_pose_back_cam)

    if len(armR_traj) != 0:
        # step 1: calculate the 6-DoF pose after removing tool length
        poses_adjusted = []
        for temp_pose in armR_traj:
            target_6dof_pose_back = np.eye(4)
            target_6dof_pose_back[:3, :3] = R.from_euler("xyz", temp_pose[3:], degrees=True).as_matrix()
            target_6dof_pose_back[:3, -1] = temp_pose[:3]
            target_6dof_pose_back = tcp_fix_gripper(target_6dof_pose_back, is_back=True)  # is_back=True, remove tool length
            target_pose_back_cam = [0,0,0, 0,0,0]
            target_pose_back_cam[:3] = handeye_camera_R[:3, :3].T @ (target_6dof_pose_back[:3, -1] - handeye_camera_R[:3, -1])  # armR --> camera
            rot_mat_camera_world = handeye_camera_R[:3, :3].T @ target_6dof_pose_back[:3, :3]
            target_pose_back_cam[3:] = R.from_matrix(rot_mat_camera_world).as_euler("xyz", degrees=True)
            poses_adjusted.append(target_pose_back_cam)
            
        if fix_point is not None: 
            target_6dof_pose_back = np.eye(4)
            target_6dof_pose_back[:3, :3] = R.from_euler("xyz", fix_point[3:], degrees=True).as_matrix()
            target_6dof_pose_back[:3, -1] = fix_point[:3]
            target_6dof_pose_back = tcp_fix_gripper(target_6dof_pose_back, is_back=True)  # is_back=True, remove tool length
            adjusted_fix_point = [0,0,0, 0,0,0]            
            adjusted_fix_point[:3] = handeye_camera_L[:3, :3].T @ (target_6dof_pose_back[:3, -1] - handeye_camera_L[:3, -1])  # armL --> camera
            rot_mat_camera_world = handeye_camera_L[:3, :3].T @ target_6dof_pose_back[:3, :3]
            adjusted_fix_point[3:] = R.from_matrix(rot_mat_camera_world).as_euler("xyz", degrees=True)
        poses_adjusted.append(adjusted_fix_point)
    
    
    point_cam2d_list, euler_angle_list = [], []
    for idx, pose_temp in enumerate(poses_adjusted):
        [pt3d_x, pt3d_y, pt3d_z] = pose_temp[:3]
        [point_2d_x, point_2d_y, temp_scale] = camL_para @ np.array([pt3d_x/pt3d_z, pt3d_y/pt3d_z, 1.0]).T  # temp_scale is always 1.0
        point_cam2d_list.append([point_2d_x, point_2d_y])
        euler_angle_list.append(pose_temp[3:])

    if fix_point is not None:  # we now only care about pivoting skill for setting the left arm as a fixed point
        adjusted_fix_point = point_cam2d_list[-1]
        [ptx, pty] = adjusted_fix_point
        cv2.circle(img_canvas, (int(ptx), int(pty)), 8, (255,255,0), -1, lineType=cv2.LINE_AA)
        cv2.circle(img_canvas, (int(ptx), int(pty)), 4, (0,0,0), -1, lineType=cv2.LINE_AA)
        cv2.circle(img_canvas, (int(ptx), int(pty)), 2, (255,255,255), -1, lineType=cv2.LINE_AA)        
        point_cam2d_list = point_cam2d_list[:-1]
        euler_angle_list = euler_angle_list[:-1]

    color = (0, 255, 255) if len(armR_traj) != 0 else (255, 255, 0)
    for idx, [ptx, pty] in enumerate(point_cam2d_list):
        if idx == 0 or idx == len(point_cam2d_list)-1:
            cv2.circle(img_canvas, (int(ptx), int(pty)), 8, color, -1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(img_canvas, (int(ptx), int(pty)), 6, color, -1, lineType=cv2.LINE_AA)

    if paper_flag:
        img_canvas_point_only = img_canvas.copy()
        for idx, ([ptx, pty], _) in enumerate(zip(point_cam2d_list, euler_angle_list)):
            if idx == 0 or idx == len(point_cam2d_list)-1:
                cv2.circle(img_canvas_point_only, (int(ptx), int(pty)), 4, (0,0,0), -1, lineType=cv2.LINE_AA)
                cv2.circle(img_canvas_point_only, (int(ptx), int(pty)), 2, (255,255,255), -1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(img_canvas_point_only, (int(ptx), int(pty)), 2, (0,0,0), -1, lineType=cv2.LINE_AA)
                cv2.circle(img_canvas_point_only, (int(ptx), int(pty)), 1, (255,255,255), -1, lineType=cv2.LINE_AA)
    
    
    for idx, ([ptx, pty], angle) in enumerate(zip(point_cam2d_list, euler_angle_list)):
        # pitch, yaw, roll = angle[0], -angle[1], -angle[2]  # following our hand 3d pose extraction algorithm
        [pitch, yaw, roll] = angle
        if idx == 0 or idx == len(point_cam2d_list)-1:
            img_canvas = plot_3axis_Zaxis(img_canvas, yaw, pitch, roll, tdx=ptx, tdy=pty, size=60, thickness=4)
            cv2.circle(img_canvas, (int(ptx), int(pty)), 4, (0,0,0), -1, lineType=cv2.LINE_AA)
            cv2.circle(img_canvas, (int(ptx), int(pty)), 2, (255,255,255), -1, lineType=cv2.LINE_AA)
    
    img_canvas_sparse = img_canvas.copy()
    
    for idx, ([ptx, pty], angle) in enumerate(zip(point_cam2d_list, euler_angle_list)):
        # pitch, yaw, roll = angle[0], -angle[1], -angle[2]  # following our hand 3d pose extraction algorithm
        [pitch, yaw, roll] = angle
        if idx == 0 or idx == len(point_cam2d_list)-1:
            img_canvas = plot_3axis_Zaxis(img_canvas, yaw, pitch, roll, tdx=ptx, tdy=pty, size=60, thickness=4)
            cv2.circle(img_canvas, (int(ptx), int(pty)), 4, (0,0,0), -1, lineType=cv2.LINE_AA)
            cv2.circle(img_canvas, (int(ptx), int(pty)), 2, (255,255,255), -1, lineType=cv2.LINE_AA)
        else:
            img_canvas = plot_3axis_Zaxis(img_canvas, yaw, pitch, roll, tdx=ptx, tdy=pty, size=40, thickness=1)
            cv2.circle(img_canvas, (int(ptx), int(pty)), 2, (0,0,0), -1, lineType=cv2.LINE_AA)
            cv2.circle(img_canvas, (int(ptx), int(pty)), 1, (255,255,255), -1, lineType=cv2.LINE_AA)
    
    if paper_flag:
        return img_canvas_sparse, img_canvas, img_canvas_point_only
    else:
        return img_canvas_sparse, img_canvas
    
###################################################################################################################################


