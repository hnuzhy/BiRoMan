
import cv2
import numpy as np

#################################################################
# top-left, top-right, bottom-right, bottom-left. length*width --> 1000 mm * 650 mm
four_pts_list = [[249, 86], [814, 91], [943, 440], [145, 434]]  # (pts_tl, pts_tr, pts_br, pts_bl) in the affine transformed camera-view image 
rect_l, rect_w = 1000, 650  # the length / width of the top-viewd rectangle
tgt_pts_list = [[0, 0], [rect_l-1, 0], [rect_l-1, rect_w-1], [0, rect_w-1]]  # (pts_tl, pts_tr, pts_br, pts_bl) in top-viewed rectangle platform

selected_four_corners = np.array(four_pts_list, dtype=np.float32)
show_window_corners = np.array(tgt_pts_list, dtype=np.float32)
transform_mat = cv2.getPerspectiveTransform(show_window_corners, selected_four_corners)  # obtain the transform function
transform_mat_inv = np.linalg.inv(transform_mat)
print("[Init] top-view reporjection transform_mat:\n", transform_mat)

# For arm1/armL: +x/-x ==> front/back; +y/-y ==> down/up; +z/-z ==> left/right
# For arm2/armR: +x/-x ==> front/back; +y/-y ==> up/down; +z/-z ==> right/left
#################################################################
cfg_dict_init = {
    "tool_length": 0.242,
    "kfr_height": 540, 
    "kfr_width": 960,
    "kfr_ip": "192.168.11.27",  # the ip address of kingfisher-R-6000
    "cam1_K": np.array([  # calib_file=kingfisher.getCalibData()
            [2994.737267701302, 0.000000000000000, 1943.025478333062],
            [0.000000000000000, 2994.379926399410, 988.065021870106],
            [0.000000000000000, 0.000000000000000, 1.000000000000000]
        ]),
    "cam2_K": np.array([  # calib_file=kingfisher.getCalibData()
            [3000.245256290249, 0.000000000000000, 1920.371195566339],
            [0.000000000000000, 2998.806173245669, 974.199805498459],
            [0.000000000000000, 0.000000000000000, 1.000000000000000]
        ]),
    
    "arm1": {  # the armL
        "gripper_port": '/dev/ttyUSB1',  # for Ubuntu, sudo chmod 666 /dev/ttyUSB1
        "robot_ip_add": "192.168.11.220",  # robot IP address "192.168.31.134"
        "handeye_para": np.array([
            [ 4.60052229e-03, -7.92004605e-01,  6.10497781e-01,  4.78865870e-02],
            [-3.76087529e-04,  6.10502829e-01,  7.92013987e-01, -2.15856101e-01],
            [-9.99989347e-01, -3.87327860e-03,  2.51076882e-03, -3.78393681e-02],
            [ 0.00000000e+00, -0.00000000e+00, -0.00000000e+00,  1.00000000e+00]
        ]),  # camera calib in 2025-09-16 (for kingfisher-R-6000)
    },
    "arm2": {  # the armR
        "gripper_port": '/dev/ttyUSB0',  # for Ubuntu, sudo chmod 666 /dev/ttyUSB0
        "robot_ip_add": "192.168.11.221",  # robot IP address "192.168.31.135"
        "handeye_para": np.array([
            [ 6.34004343e-03, -7.87023936e-01,  6.16889884e-01,  4.46400294e-02],
            [ 4.94210047e-03, -6.16870087e-01, -7.87049472e-01,  2.09050255e-01],
            [ 9.99967689e-01,  8.03865962e-03, -2.14317819e-05, -1.86870870e-01],
            [-0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ]),  # camera calib in 2025-09-16 (for kingfisher-R-6000)
    },
    
    "four_pts_list": four_pts_list,  # top-left, top-right, bottom-right, bottom-left
    "rect_l": rect_l,  # the length of the top-viewd rectangle
    "rect_w": rect_w,  # the width of the top-viewd rectangle
    "pts_polygon": np.array(four_pts_list, np.int32).reshape((-1, 1, 2)),  # polygon corner points coordinates
    "colors_list": [(0,255,128), (255,255,0), (128,0,255), (0,128,128), (0,0,255)],  # lawn green, cyan, light magenta, light yellow, red
    "bl_pt_to_arm1": [0.435309198, 0.442306666, 0.389107154, -84.110881514, -88.474229046, -6.917829792],
    "br_pt_to_arm2": [0.440170147, -0.446313696, 0.391927453, 110.33173128, 88.814541726, 20.4457572178],
    "transform_mat": transform_mat,  # Eight-point homography transformation matrix
    "transform_mat_inv": transform_mat_inv, # Eight-point homography inversed transformation matrix
    # "detection_roi_bbox": [120, 20, 870, 520],  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)  # for dual-arm aubo
    # "detection_roi_bbox_2": [245, 90, 745, 450],  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)  # for dual-arm aubo
    "detection_roi_bbox": [140, 20, 940, 520],  # the [x1, y1, x2, y2] (960*540 --> 800*500; 16:9 --> 8:5)  # for dual-arm rokae
    "detection_roi_bbox_2": [250, 90, 810, 450],  # the [x1, y1, x2, y2] (960*540 --> 560*360; 16:9 --> 14:9)  # for dual-arm rokae
}

#  for the function plot_primitive_skill_traj() in src/util.py
scale_ratio = 2.0  # for using low res image (960, 540), we should adjust the calib_file with dividing 2.0
for para_key in ["cam1_K", "cam1_K"]:
    for [loc_i, loc_j] in [[0,0], [1,1], [0,2], [1,2]]:
        cfg_dict_init[para_key][loc_i][loc_j] /= scale_ratio

#################################################################

# For arm1/armL: +x/-x ==> front/back;   +y/-y ==> down/up;   +z/-z ==> left/right
# For arm2/armR: +x/-x ==> front/back;   +y/-y ==> up/down;   +z/-z ==> right/left
def get_eef_keypose_dict(task_name, object_ids=[]):
    
    assert task_name in ["pouring", "unscrew", "reorient", "grasping", "flatting", "flipping",
        "unscrew-pouring", "flatting-reorient", "flipping-grasping",
        "grasping_rectbox", "grasping_cirbowl", "grasping_basket", "grasping_holder", "grasping_pencup",
        "inserting", "plugpen", "handover", "ppspoon", "ppfork", "ppspoon-ppfork", "ppfork-ppspoon",
        "pivoting", "wrapping", "pivoting_rectbox", "pivoting_cirbowl", "flipping_basket", "flipping_block", "pivoting_bigjar", "pivoting_block", 
        "toppling_holder", "toppling_bigjar", "bilifting_bigjar", "bilifting_block",
        "penbagzip", "foldtowel", "foldpants", "foldshirt", "coilcable", "coilrope", "coilbelt" ], "Please give a valid task name!!!"
    
    ################################################################################################
    if task_name == "pouring":
        [bottle_id, mugcup_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [1, 0.468171478, 0.502483409, 0.233157832, -179.610881514, 30.474229046, -90.517829792, 1],  # grasp
                [1, 0.518171478, 0.452483409, 0.291157832, -179.610881514, 30.474229046, -90.517829792, 1],  # lift-up
                [3, 0.568171478, 0.352483409, 0.291157832, -179.610881514, 30.474229046, -90.517829792, 1],  # pre-pour 
                [3, 0.423237851, 0.322091502, 0.053208860, 99.061818632, 35.288564357, 143.229796841, 1],  # pouring
                [3, 0.568171478, 0.352483409, 0.291157832, -179.610881514, 30.474229046, -90.517829792, 1],  # arm-back
                [4, 0.568171478, 0.499483409, 0.291157832, -179.610881514, 30.474229046, -90.517829792, 0]  # place-down
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0, 0.475196555, -0.423439014, 0.20591368, 179.732957532, 60.68246746, 89.739817245, 1],  # grasp
                [0, 0.525196555, -0.368439014, 0.25891368, 179.732957532, 60.68246746, 89.739817245, 1],  # lift-up
                [2, 0.475197799, -0.373521134, -0.04414114, 179.723716105, 60.686202994, 89.725888528, 1],  # align
                [5, 0.575196555, -0.420439014, 0.15891368, 179.732957532, 60.68246746, 89.739817245, 0]  # place-down
            ]
        }
        if bottle_id != 1:  # mo-li-hua-cha bottle
            if bottle_id == 2: height_offset = 0.014; width_offset = -0.001  # dong-fang-shu-ye bottle
            if bottle_id == 3: height_offset = 0.060; width_offset = 0.005  # wu-long-cha bottle
            if bottle_id == 4: height_offset = 0.060; width_offset = 0.001  # hong-dou-yi-mi bottle
            raw_eef_keypose_dict["L"][1][3] -= width_offset  # shrink the z-value
            raw_eef_keypose_dict["L"][4][1] += (1.732 * 0.5 * height_offset)  # enlarge the x-value
            raw_eef_keypose_dict["L"][4][3] += (1.000 * 0.5 * height_offset)  # enlarge the z-value
        if mugcup_id != 1:  # blue plastic mug
            if mugcup_id == 2: height_offset = 0.015; width_offset = 0.004  # yellow plastic mug
            raw_eef_keypose_dict["R"][1][1] += width_offset  # enlarge the x-value
            raw_eef_keypose_dict["R"][1][2] += height_offset * 0.5  # enlarge the y-value
            raw_eef_keypose_dict["R"][1][3] += width_offset  # enlarge the z-value
            raw_eef_keypose_dict["R"][3][2] -= height_offset * 0.5  # shrink the y-value
            raw_eef_keypose_dict["R"][3][3] += width_offset  # enlarge the z-value
            raw_eef_keypose_dict["R"][4][2] += height_offset * 0.5  # enlarge the y-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    ################################
    if task_name == "unscrew":
        [bottle_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.468171478, 0.502483409, 0.233157832, -179.610881514, 30.474229046, -90.517829792, 1],  # grasp
                [1, 0.568171478, 0.402483409, 0.035157832, -179.610881514, 30.474229046, -90.517829792, 1],  # lift-up
                [5, 0.568171478, 0.502483409, 0.233157832, -179.610881514, 30.474229046, -90.517829792, 0]  # place-down
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [2, 0.573413994, -0.206476218, -0.072155227, -90.310744971, 175.516336991, -179.648604329, 1],  # grasp
                [2, 0.573425291, -0.206635897, -0.07210559, -90.310744971, 5.516336991, -179.648604329, 0],  # untwist
                [2, 0.573413994, -0.206476218, -0.072155227, -90.310744971, 175.516336991, -179.648604329, 1],  # back
                [2, 0.573425291, -0.206635897, -0.07210559, -90.310744971, 5.516336991, -179.648604329, 0],  # untwist
                [2, 0.573413994, -0.206476218, -0.072155227, -90.310744971, 175.516336991, -179.648604329, 1],  # back
                [2, 0.573425291, -0.206635897, -0.07210559, -90.310744971, 5.516336991, -179.648604329, 0],  # untwist 
                [2, 0.573425291, -0.200635897, -0.07210559, -90.310744971, 10.516336991, -179.648604329, 1],  # re-grasp
                [2, 0.573425291, -0.106635897, -0.07210559, -90.310744971, 5.516336991, -179.648604329, 1],  # pull
                [3, 0.573413994, -0.056635897, 0.032155227, -90.310744971, 45.516336991, -179.648604329, 1],  # lift-up
                [4, 0.42413994, -0.426635897, 0.282155227, -90.310744971, 90.516336991, -179.648604329, 0]  # place-down
            ]
        }
        if bottle_id != 1:  # mo-li-hua-cha bottle
            if bottle_id == 2: height_offset = 0.012; width_offset = -0.001; rot_deg_offset = 60  # dong-fang-shu-ye bottle
            if bottle_id == 3: height_offset = 0.064; width_offset = 0.008; rot_deg_offset = 0  # wu-long-cha bottle
            if bottle_id == 4: height_offset = 0.058; width_offset = 0.000; rot_deg_offset = 60  # hong-dou-yi-mi bottle
            if bottle_id == 5: height_offset = -0.020; width_offset = 0.000; rot_deg_offset = 60  # markerpen (a fake bottle)

            raw_eef_keypose_dict["L"][1][3] -= width_offset  # shrink the z-value
            raw_eef_keypose_dict["L"][1][2] -= height_offset * 0.5  # shrink the y-value
            raw_eef_keypose_dict["L"][2][2] += height_offset * 0.5  # enlarge the y-value
            raw_eef_keypose_dict["L"][3][2] -= height_offset * 0.5  # shrink the y-value
            for step_id in [2, 4, 6, 7, 8, 9]: raw_eef_keypose_dict["R"][step_id][5] += rot_deg_offset  # adjust the untwist degree
                
        final_eef_keypose_dict = raw_eef_keypose_dict
    ################################
    if task_name == "reorient":
        [bottle_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.488413994, 0.330476218, -0.112155227, -89.610881514, 0.474229046, -0.517829792, 0],  # pre-grasp
                [1, 0.488413994, 0.430476218, -0.112155227, -89.610881514, 0.474229046, -0.517829792, 1],  # grasp
                [2, 0.558413994, 0.285476218, 0.252155227, -89.610881514, 90.474229046, -0.517829792, 1],  # lift-up
                [3, 0.558413994, 0.465476218, 0.402155227, -179.610881514, 0.474229046, -90.517829792, 1],  # reorient
                [4, 0.558413994, 0.615476218, 0.352155227, -179.610881514, 0.474229046, -90.517829792, 0],  # place-down
                [5, 0.558413994, 0.415476218, 0.452155227, -179.610881514, 45.474229046, -90.517829792, 0]  # back (avoid collision)
            ],
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0, 0.488413994, -0.330476218, -0.112155227, -90.310744971, 180.516336991, -179.648604329, 0],  # pre-grasp
                [1, 0.488413994, -0.430476218, -0.112155227, -90.310744971, 180.516336991, -179.648604329, 1],  # grasp
                [2, 0.528413994, -0.285476218, 0.252155227, -90.310744971, 90.516336991, -179.648604329, 1],  # lift-up
                [3, 0.528413994, -0.465476218, 0.402155227, 179.732957532, 0.68246746, 89.739817245, 1],  # reorient
                [4, 0.528413994, -0.615476218, 0.352155227, 179.732957532, 0.68246746, 89.739817245, 0],  # place-down
                [5, 0.528413994, -0.415476218, 0.452155227, 179.732957532, 45.68246746, 89.739817245, 0]  # back (avoid collision)
            ],
        }
        if bottle_id != 1:  # mo-li-hua-cha bottle
            if bottle_id == 2: height_offset = 0.010  # dong-fang-shu-ye bottle
            if bottle_id == 3: height_offset = 0.060  # wu-long-cha bottle
            if bottle_id == 4: height_offset = 0.060  # hong-dou-yi-mi bottle
            raw_eef_keypose_dict["L"][1][1] += height_offset * 0.5; raw_eef_keypose_dict["R"][1][1] += height_offset * 0.5  # enlarge x-value
            raw_eef_keypose_dict["L"][2][1] += height_offset * 0.5; raw_eef_keypose_dict["R"][2][1] += height_offset * 0.5  # enlarge x-value
            raw_eef_keypose_dict["L"][5][2] -= height_offset * 0.5; raw_eef_keypose_dict["R"][5][2] += height_offset * 0.5  # shrink / enlarge y-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    ################################
    if task_name == "grasping":  # grasping is better and more robust than poking
        [mugcup_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.463142021, 0.53252716, 0.202131294, -160.523111098, -0.31602756, -0.347483597, 0],  # pre-grasp
                [1, 0.463142021, 0.53252716, 0.138131294, -165.523111098, -0.31602756, -0.347483597, 1],  # grasp
                [2, 0.513142021, 0.33252716, 0.192131294, -165.523111098, -0.31602756, -0.347483597, 1],  # lift-up
                [3, 0.563142021, 0.23252716, 0.242131294, -89.610881514, -0.474229046, -0.517829792, 1],  # reorient
                [4, 0.563142021, 0.35052716-0.005, 0.142131294, -89.610881514, -65.474229046, -0.517829792, 0],  # place-down
            ],
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0, 0.463142021, -0.53252716, 0.202131294, -160.523111098, 0.403818601, 179.557424883, 0],  # pre-grasp
                [1, 0.463142021, -0.53252716, 0.138131294, -165.523111098, 0.403818601, 179.557424883, 1],  # grasp
                [2, 0.513142021, -0.33252716, 0.192131294, -165.523111098, 0.403818601, 179.557424883, 1],  # lift-up
                [3, 0.563142021, -0.23252716, 0.242131294, -90.310744971, 0.516336991, -179.648604329, 1],  # reorient
                [4, 0.563142021, -0.35052716, 0.142131294, -90.310744971, 65.516336991, -179.648604329, 0],  # place-down
            ],
        }
        if mugcup_id != 1:  # blue plastic mug
            if mugcup_id == 2: height_offset = 0.016; width_offset = 0.003  # yellow plastic mug
            raw_eef_keypose_dict["L"][1][2] -= width_offset * 2.0; raw_eef_keypose_dict["R"][1][2] += width_offset * 2.0
            raw_eef_keypose_dict["L"][2][2] -= width_offset * 2.0; raw_eef_keypose_dict["R"][2][2] += width_offset * 2.0
            raw_eef_keypose_dict["L"][5][2] -= height_offset * 0.5; raw_eef_keypose_dict["R"][5][2] += height_offset * 0.5
        final_eef_keypose_dict = raw_eef_keypose_dict
    ################################
    if task_name == "flatting":
        [bottle_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.458171478, 0.382483409, 0.180157832, -179.610881514, 30.474229046, -90.517829792, 0],  # pre-grasp
                [1, 0.458171478, 0.482483409, 0.080157832, -179.610881514, 30.474229046, -90.517829792, 1],  # grasp
                [2, 0.575793464, 0.252759091, 0.158338589, -100.056926667, -15.029180955, 30.020883987, 1],  # move & rotate
                [3, 0.625793464, 0.442759091, 0.158338589, -100.056926667, -30.029180955, 30.020883987, 0],  # place-down
                [4, 0.625793464, 0.335759091, 0.158338589, -100.056926667, -30.029180955, 30.020883987, 0],  # back (avoid collision)
            ],
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0, 0.458171478, -0.382483409, 0.180157832, 179.732957532, 30.68246746, 89.739817245, 0],  # pre-grasp
                [1, 0.458171478, -0.482483409, 0.080157832, 179.732957532, 30.68246746, 89.739817245, 1],  # grasp
                [2, 0.575793464, -0.252759091, 0.158338589, 100.056926667, -15.029180955, -30.020883987, 1],  # move & rotate
                [3, 0.625793464, -0.442759091, 0.158338589, 100.056926667, -30.029180955, -30.020883987, 0],  # place-down
                [4, 0.625793464, -0.335759091, 0.158338589, 100.056926667, -30.029180955, -30.020883987, 0],  # back (avoid collision)
            ],
        }
        if bottle_id != 1:  # mo-li-hua-cha bottle
            if bottle_id == 2: height_offset = 0.010  # dong-fang-shu-ye bottle
            if bottle_id == 3: height_offset = 0.060  # wu-long-cha bottle
            if bottle_id == 4: height_offset = 0.060  # hong-dou-yi-mi bottle
            raw_eef_keypose_dict["L"][1][2] -= height_offset * 0.5; raw_eef_keypose_dict["R"][1][2] += height_offset * 0.5
            raw_eef_keypose_dict["L"][2][2] -= height_offset * 0.5; raw_eef_keypose_dict["R"][2][2] += height_offset * 0.5
        final_eef_keypose_dict = raw_eef_keypose_dict
    ################################
    if task_name == "flipping":
        [mugcup_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.463171478, 0.352483409, 0.150157832, -179.610881514, 60.474229046, -90.517829792, 0],  # pre-grasp
                [1, 0.463171478, 0.452483409, 0.050157832, -179.610881514, 60.474229046, -90.517829792, 1],  # grasp
                [2, 0.619887079, 0.287784377, 0.072699376, 160.093396242, 70.031665426, -89.518932206, 1],  # move & rotate 1
                [3, 0.587550346, 0.391322418, 0.046802817, 0.536984102, 30.005960171, 130.098467944, 1],  # move & rotate 2
                [4, 0.557550346, 0.507322418, -0.006802817, 0.536984102, 30.005960171, 130.098467944, 0],  # place-down
                [5, 0.587550346, 0.391322418, -0.096802817, 0.536984102, 30.005960171, 130.098467944, 0],  # back (avoid collision)
            ],
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0 ,0.463171478, -0.352483409, 0.150157832, 179.732957532, 60.68246746, 89.739817245, 0],  # pre-grasp
                [1, 0.463171478, -0.452483409, 0.050157832, 179.732957532, 60.68246746, 89.739817245, 1],  # grasp
                [2, 0.619887079, -0.287784377, 0.072699376, -90.09575228, 70.171391397, 160.565267583, 1],  # move & rotate 1
                [3, 0.587550346, -0.391322418, 0.046802817, -0.536984102, 30.005960171, -130.098467944, 1],  # move & rotate 2
                [4, 0.557550346, -0.507322418, -0.006802817, -0.536984102, 30.005960171, -130.098467944, 0],  # place-down
                [5, 0.587550346, -0.391322418, -0.056802817, -0.536984102, 30.005960171, -130.098467944, 0],  # back (avoid collision)
            ],
        }
        if mugcup_id != 1:  # blue plastic mug
            if mugcup_id == 2: height_offset = 0.016; width_offset = 0.003  # yellow plastic mug
        final_eef_keypose_dict = raw_eef_keypose_dict
    ################################################################################################


    ################################################################################################
    if task_name == "grasping_rectbox":
        [rectbox_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.500413994, 0.250476218, -0.115155227, -89.610881514, 0.474229046, -0.517829792, 0],  # pre-grasp
                [1, 0.500413994, 0.320476218, -0.115155227, -89.610881514, 0.474229046, -0.517829792, 0.64],  # grasp
                [2, 0.500413994, 0.250476218, -0.115155227, -89.610881514, 0.474229046, -0.517829792, 0.64],  # lift-up
                [3, 0.750413994, 0.250476218, -0.115155227, -89.610881514, 90.474229046, -0.517829792, 0.64],  # move & rotate
                [4, 0.750413994, 0.320476218, -0.115155227, -89.610881514, 90.474229046, -0.517829792, 0],  # place-down
                [5, 0.750413994, 0.250476218, -0.115155227, -89.610881514, 90.474229046, -0.517829792, 0],  # back (avoid collision)
            ],
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0, 0.500413994, -0.250476218, -0.115155227, -90.310744971, 180.516336991, -179.648604329, 0],  # pre-grasp
                [1, 0.500413994, -0.320476218, -0.115155227, -90.310744971, 180.516336991, -179.648604329, 0.64],  # grasp
                [2, 0.500413994, -0.250476218, -0.115155227, -90.310744971, 180.516336991, -179.648604329, 0.64],  # lift-up
                [3, 0.750413994, -0.250476218, -0.115155227, -90.310744971, 90.516336991, -179.648604329, 0.64],  # reorient
                [4, 0.750413994, -0.320476218, -0.115155227, -90.310744971, 90.516336991, -179.648604329, 0],  # place-down
                [5, 0.750413994, -0.250476218, -0.115155227, -90.310744971, 90.516336991, -179.648604329, 0],  # back (avoid collision)
            ],
        }
        if rectbox_id == 1: height_offset = 0.000; width_offset = 0.000  # the smallest gray rectbox
        if rectbox_id == 2: height_offset = 0.015; width_offset = 0.120  # the middle size gray rectbox
        if rectbox_id == 3: height_offset = 0.065; width_offset = 0.150  # the largest gray rectbox
        for step_id in [1, 2, 3]: raw_eef_keypose_dict["L"][step_id][1] += width_offset*0.5; raw_eef_keypose_dict["R"][step_id][1] += width_offset*0.5
        for step_id in [1, 2, 3, 4, 5, 6]: raw_eef_keypose_dict["L"][step_id][2] -= height_offset; raw_eef_keypose_dict["R"][step_id][2] += height_offset
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "grasping_cirbowl":
        [cirbowl_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.495413994, 0.270476218, 0.035155227, -89.610881514, 180.474229046, -0.517829792, 0],  # pre-grasp
                [1, 0.495413994, 0.420476218, -0.015155227, -79.610881514, 180.474229046, -0.517829792, 0.05],  # grasp
                [2, 0.495413994, 0.270476218, -0.015155227, -79.610881514, 180.474229046, -0.517829792, 0.05],  # lift-up
                [3, 0.750413994, 0.270476218, -0.115155227, -89.610881514, 90.474229046, -10.517829792, 0.05],  # move & rotate
                [4, 0.750413994, 0.410476218, -0.115155227, -89.610881514, 90.474229046, -10.517829792, 0.2],  # place-down
                [5, 0.750413994, 0.320476218, -0.115155227, -89.610881514, 90.474229046, -0.517829792, 0],  # back (avoid collision)
            ],
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0, 0.495413994, -0.270476218, 0.035155227, -90.310744971, 0.516336991, -179.648604329, 0],  # pre-grasp
                [1, 0.495413994, -0.420476218, -0.015155227, -100.310744971, 0.516336991, -179.648604329, 0.05],  # grasp
                [2, 0.495413994, -0.270476218, -0.015155227, -100.310744971, 0.516336991, -179.648604329, 0.05],  # lift-up
                [3, 0.750413994, -0.270476218, -0.115155227, -90.310744971, 90.516336991, -169.648604329, 0.05],  # move & rotate
                [4, 0.750413994, -0.410476218, -0.115155227, -90.310744971, 90.516336991, -169.648604329, 0.2],  # place-down
                [5, 0.750413994, -0.320476218, -0.115155227, -90.310744971, 90.516336991, -179.648604329, 0],  # back (avoid collision)
            ],
        }
        if cirbowl_id == 1: height_offset = 0.000; width_offset = 0.000; gripper_val = 0.05  # the small white paper bowl
        if cirbowl_id == 2: height_offset = 0.008; width_offset = -0.015; gripper_val = 0.01  # the small green plastic bowl 
        if cirbowl_id == 3: height_offset = 0.016; width_offset = -0.016; gripper_val = 0.05  # the middle transparent plastic bowl 
        if cirbowl_id == 4: height_offset = 0.016; width_offset = +0.016; gripper_val = 0.01  # the large gray paper bowl     
        for step_id in [1, 2, 3]: raw_eef_keypose_dict["L"][step_id][1] += width_offset*0.5; raw_eef_keypose_dict["R"][step_id][1] += width_offset*0.5
        for step_id in [1, 2, 3, 4, 5, 6]: raw_eef_keypose_dict["L"][step_id][2] -= height_offset; raw_eef_keypose_dict["R"][step_id][2] += height_offset
        for step_id in [2, 3, 4]: raw_eef_keypose_dict["L"][step_id][-1] = gripper_val; raw_eef_keypose_dict["R"][step_id][-1] = gripper_val
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "grasping_basket":
        [basket_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.540413994, 0.285476218, 0.048155227, -89.610881514, 180.474229046, -0.517829792, 0],  # pre-grasp
                [2, 0.540413994, 0.385476218, 0.048155227, -89.610881514, 180.474229046, -0.517829792, 1],  # start grasp
                [4, 0.540413994, 0.235476218, 0.048155227, -89.610881514, 180.474229046, -0.517829792, 1],  # lift-up
                [6, 0.780413994, 0.235476218, 0.048155227, -89.610881514, 180.474229046, -0.517829792, 1],  # move & rotate
                [8, 0.780413994, 0.380476218, 0.048155227, -89.610881514, 180.474229046, -0.517829792, 0],  # place-down
                [10, 0.780413994, 0.285476218, 0.048155227, -89.610881514, 180.474229046, -0.517829792, 0],  # back (avoid collision)
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [1, 0.540413994, -0.285476218, 0.048155227, -90.310744971, 0.516336991, -179.648604329, 0],  # pre-grasp
                [3, 0.540413994, -0.385476218, 0.048155227, -90.310744971, 0.516336991, -179.648604329, 1],  # start grasp
                [5, 0.540413994, -0.235476218, 0.048155227, -90.310744971, 0.516336991, -179.648604329, 1],  # lift-up
                [7, 0.780413994, -0.235476218, 0.048155227, -90.310744971, 0.516336991, -179.648604329, 1],  # move & rotate
                [9, 0.780413994, -0.380476218, 0.048155227, -90.310744971, 0.516336991, -179.648604329, 0],  # place-down
                [11, 0.780413994, -0.285476218, 0.048155227, -90.310744971, 0.516336991, -179.648604329, 0],  # back (avoid collision)
            ]
        }
        if basket_id == 1: width_offset = -0.032; length_offset = -0.072; height_offset = +0.010  # the small pink mesh basket
        if basket_id == 2: width_offset = +0.016; length_offset = -0.012; height_offset = +0.025  # the large green mesh basket
        if basket_id == 3: width_offset = +0.000; length_offset = +0.000; height_offset = +0.000  # the large blue mesh basket
        for step_id in [1, 2, 3]: raw_eef_keypose_dict["L"][step_id][1] += width_offset*0.5; raw_eef_keypose_dict["R"][step_id][1] += width_offset*0.5  # adjust x-value
        for step_id in [1, 2, 3, 4, 5, 6]: raw_eef_keypose_dict["L"][step_id][3] += length_offset*0.5; raw_eef_keypose_dict["R"][step_id][3] += length_offset*0.5  # adjust z-value
        for step_id in [1, 2, 3, 4, 5, 6]: raw_eef_keypose_dict["L"][step_id][2] -= height_offset; raw_eef_keypose_dict["R"][step_id][2] += height_offset  # adjust y-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "grasping_holder":
        [holder_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.463142021, 0.53252716, 0.202131294, -160.523111098, -0.31602756, -0.347483597, 0],  # pre-grasp
                [1, 0.463142021, 0.53252716, 0.138131294, -165.523111098, -0.31602756, -0.347483597, 1],  # grasp
                [2, 0.513142021, 0.33252716, 0.192131294, -165.523111098, -0.31602756, -0.347483597, 1],  # lift-up and prepare
                [3, 0.563142021, 0.23252716, 0.242131294, -89.610881514, -0.474229046, -0.517829792, 1],  # reorient
                [4, 0.563142021, 0.35052716, 0.142131294, -89.610881514, -65.474229046, -0.517829792, 0],  # place-down
                [5, 0.563142021, 0.30052716, 0.142131294, -89.610881514, -65.474229046, -0.517829792, 0],  # back (avoid collision)
            ],
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0, 0.463142021, -0.53252716, 0.202131294, -160.523111098, 0.403818601, 179.557424883, 0],  # pre-grasp
                [1, 0.463142021, -0.53252716, 0.138131294, -165.523111098, 0.403818601, 179.557424883, 1],  # grasp
                [2, 0.513142021, -0.33252716, 0.192131294, -165.523111098, 0.403818601, 179.557424883, 1],  # lift-up and prepare
                [3, 0.563142021, -0.23252716, 0.242131294, -90.310744971, 0.516336991, -179.648604329, 1],  # reorient
                [4, 0.563142021, -0.35052716, 0.142131294, -90.310744971, 65.516336991, -179.648604329, 0],  # place-down
                [5, 0.563142021, -0.30052716, 0.142131294, -90.310744971, 65.516336991, -179.648604329, 0],  # back (avoid collision)
            ],
        }
        if holder_id == 1: height_offset = 0.000; width_offset = 0.000  # black pen holder (cylinder)
        if holder_id == 2: height_offset = +0.010; width_offset = 0.000  # black pen holder (cuboid)
        raw_eef_keypose_dict["L"][1][2] -= width_offset * 2.0; raw_eef_keypose_dict["R"][1][2] += width_offset * 2.0
        raw_eef_keypose_dict["L"][2][2] -= width_offset * 2.0; raw_eef_keypose_dict["R"][2][2] += width_offset * 2.0
        raw_eef_keypose_dict["L"][5][2] -= height_offset * 0.5; raw_eef_keypose_dict["R"][5][2] += height_offset * 0.5
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "grasping_pencup":
        [pencup_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.475413994, 0.220476218, 0.012155227, -89.610881514, 180.474229046, -0.517829792, 0],  # pre-grasp
                [1, 0.475413994, 0.370476218, -0.038155227, -79.610881514, 180.474229046, -0.517829792, 0.05],  # grasp
                [2, 0.475413994, 0.220476218, -0.038155227, -79.610881514, 180.474229046, -0.517829792, 0.05],  # lift-up
                [3, 0.750413994, 0.220476218, -0.115155227, -89.610881514, 90.474229046, -10.517829792, 0.05],  # move & rotate
                [4, 0.750413994, 0.365476218, -0.115155227, -89.610881514, 90.474229046, -10.517829792, 0],  # place-down
                [5, 0.750413994, 0.270476218, -0.115155227, -89.610881514, 90.474229046, -0.517829792, 0],  # back (avoid collision)
            ],
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0, 0.475413994, -0.220476218, 0.012155227, -90.310744971, 0.516336991, -179.648604329, 0],  # pre-grasp
                [1, 0.475413994, -0.370476218, -0.038155227, -100.310744971, 0.516336991, -179.648604329, 0.05],  # grasp
                [2, 0.475413994, -0.220476218, -0.038155227, -100.310744971, 0.516336991, -179.648604329, 0.05],  # lift-up
                [3, 0.750413994, -0.220476218, -0.115155227, -90.310744971, 90.516336991, -169.648604329, 0.05],  # move & rotate
                [4, 0.750413994, -0.365476218, -0.115155227, -90.310744971, 90.516336991, -169.648604329, 0],  # place-down
                [5, 0.750413994, -0.270476218, -0.115155227, -90.310744971, 90.516336991, -179.648604329, 0],  # back (avoid collision)
            ],
        }
        if pencup_id == 1: height_offset = +0.000; width_offset = +0.000; gripper_val = 1  # black pen holder (cylinder)
        if pencup_id == 2: height_offset = -0.008; width_offset = -0.010; gripper_val = 1  # black pen holder (cuboid)
        for step_id in [1, 2, 3]: raw_eef_keypose_dict["L"][step_id][1] += width_offset*0.5; raw_eef_keypose_dict["R"][step_id][1] += width_offset*0.5
        for step_id in [1, 2, 3, 4, 5, 6]: raw_eef_keypose_dict["L"][step_id][2] -= height_offset; raw_eef_keypose_dict["R"][step_id][2] += height_offset
        for step_id in [2, 3, 4]: raw_eef_keypose_dict["L"][step_id][-1] = gripper_val; raw_eef_keypose_dict["R"][step_id][-1] = gripper_val
        final_eef_keypose_dict = raw_eef_keypose_dict    
    
    ################################################################################################


    ################################################################################################
    if task_name == "unscrew-pouring":
        [bottle_id, mugcup_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.468171478, 0.502483409, 0.233157832, -179.610881514, 30.474229046, -90.517829792, 1],  # grasp
                [1, 0.568171478, 0.402483409, 0.035157832, -179.610881514, 30.474229046, -90.517829792, 1],  # lift-up

                [5, 0.518171478, 0.452483409, 0.291157832, -179.610881514, 30.474229046, -90.517829792, 1],  # lift-up
                [8, 0.568171478, 0.352483409, 0.291157832, -179.610881514, 30.474229046, -90.517829792, 1],  # pre-pour 
                [8, 0.423237851, 0.322091502, 0.053208860, 99.061818632, 35.288564357, 143.229796841, 1],  # pouring
                [8, 0.568171478, 0.352483409, 0.291157832, -179.610881514, 30.474229046, -90.517829792, 1],  # arm-back
                [9, 0.568171478, 0.499483409, 0.291157832, -179.610881514, 30.474229046, -90.517829792, 0]  # place-down
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [2, 0.573413994, -0.206476218, -0.072155227, -90.310744971, 175.516336991, -179.648604329, 1],  # grasp
                [2, 0.573425291, -0.206635897, -0.07210559, -90.310744971, 5.516336991, -179.648604329, 0],  # untwist
                [2, 0.573413994, -0.206476218, -0.072155227, -90.310744971, 175.516336991, -179.648604329, 1],  # back
                [2, 0.573425291, -0.206635897, -0.07210559, -90.310744971, 5.516336991, -179.648604329, 0],  # untwist
                [2, 0.573413994, -0.206476218, -0.072155227, -90.310744971, 175.516336991, -179.648604329, 1],  # back
                [2, 0.573425291, -0.206635897, -0.07210559, -90.310744971, 5.516336991, -179.648604329, 0],  # untwist 
                [2, 0.573425291, -0.200635897, -0.07210559, -90.310744971, 10.516336991, -179.648604329, 1],  # re-grasp
                [2, 0.573425291, -0.106635897, -0.07210559, -90.310744971, 5.516336991, -179.648604329, 1],  # pull
                [3, 0.573413994, -0.056635897, 0.032155227, -90.310744971, 45.516336991, -179.648604329, 1],  # lift-up
                [4, 0.42413994, -0.426635897, 0.282155227, -90.310744971, 90.516336991, -179.648604329, 0],  # place-down

                [6, 0.475196555, -0.423439014, 0.20591368, 179.732957532, 60.68246746, 89.739817245, 1],  # grasp
                [7, 0.475197799, -0.373521134, -0.04414114, 179.723716105, 60.686202994, 89.725888528, 1],  # align
                [10, 0.575196555, -0.420439014, 0.15891368, 179.732957532, 60.68246746, 89.739817245, 0]  # place-down
            ]
        }
        if bottle_id != 1:  # mo-li-hua-cha bottle
            if bottle_id == 2: height_offset = 0.010; width_offset = -0.003; rot_deg_offset = 60  # dong-fang-shu-ye bottle
            if bottle_id == 3: height_offset = 0.060; width_offset = 0.008; rot_deg_offset = 0  # wu-long-cha bottle
            if bottle_id == 4: height_offset = 0.058; width_offset = 0.000; rot_deg_offset = 60  # hong-dou-yi-mi bottle
            raw_eef_keypose_dict["L"][1][3] -= width_offset  # shrink the z-value
            raw_eef_keypose_dict["L"][1][2] -= height_offset * 0.5  # shrink the y-value
            raw_eef_keypose_dict["L"][2][2] += height_offset * 0.5  # enlarge the y-value
            for step_id in [2, 4, 6, 7, 8, 9]: raw_eef_keypose_dict["R"][step_id][5] += rot_deg_offset  # adjust the untwist degree
            raw_eef_keypose_dict["L"][5][1] += (1.732 * 0.5 * height_offset * 0.5)  # enlarge the x-value
            raw_eef_keypose_dict["L"][5][3] += (1.000 * 0.5 * height_offset * 0.5)  # enlarge the z-value
            raw_eef_keypose_dict["L"][7][2] -= height_offset * 0.5  # shrink the y-value
        if mugcup_id != 1:  # blue plastic mug
            if mugcup_id == 2: height_offset = 0.015; width_offset = 0.004  # yellow plastic mug
            raw_eef_keypose_dict["R"][11][1] += width_offset  # enlarge the x-value
            raw_eef_keypose_dict["R"][11][2] += height_offset * 0.5  # enlarge the y-value
            raw_eef_keypose_dict["R"][11][3] += width_offset  # enlarge the z-value
            raw_eef_keypose_dict["R"][12][2] -= height_offset * 0.5  # shrink the y-value
            raw_eef_keypose_dict["R"][12][3] += width_offset  # enlarge the z-value
            raw_eef_keypose_dict["R"][13][2] += height_offset * 0.5  # enlarge the y-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "flatting-reorient":
        [bottle_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.458171478, 0.382483409, 0.180157832, -179.610881514, 30.474229046, -90.517829792, 0],  # pre-grasp
                [1, 0.458171478, 0.482483409, 0.080157832, -179.610881514, 30.474229046, -90.517829792, 1],  # grasp
                [2, 0.575793464, 0.252759091, 0.158338589, -100.056926667, -15.029180955, 30.020883987, 1],  # move & rotate
                [3, 0.625793464, 0.442759091, 0.158338589, -100.056926667, -30.029180955, 30.020883987, 0],  # place-down
                [4, 0.625793464, 0.335759091, 0.158338589, -100.056926667, -30.029180955, 30.020883987, 0],  # back (avoid collision)

                [5, 0.625793464, 0.285759091, 0.158338589, -89.610881514, 60.474229046, -0.517829792, 0] , # move & rotate

                [6, 0.545793464, 0.330476218, 0.128338589, -89.610881514, 150.474229046, -0.517829792, 0],  # pre-grasp
                [7, 0.545793464, 0.430476218, 0.128338589, -89.610881514, 150.474229046, -0.517829792, 1],  # grasp
                [8, 0.558413994, 0.285476218, 0.252155227, -89.610881514, 90.474229046, -0.517829792, 1],  # lift-up
                [9, 0.558413994, 0.465476218, 0.402155227, -179.610881514, 0.474229046, -90.517829792, 1],  # reorient
                [10, 0.558413994, 0.615476218, 0.352155227, -179.610881514, 0.474229046, -90.517829792, 0],  # place-down
                [11, 0.558413994, 0.415476218, 0.452155227, -179.610881514, 45.474229046, -90.517829792, 0]  # back (avoid collision)
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0, 0.458171478, -0.382483409, 0.180157832, 179.732957532, 30.68246746, 89.739817245, 0],  # pre-grasp
                [1, 0.458171478, -0.482483409, 0.080157832, 179.732957532, 30.68246746, 89.739817245, 1],  # grasp
                [2, 0.575793464, -0.252759091, 0.158338589, 100.056926667, -15.029180955, -30.020883987, 1],  # move & rotate
                [3, 0.625793464, -0.442759091, 0.158338589, 100.056926667, -30.029180955, -30.020883987, 0],  # place-down
                [4, 0.625793464, -0.335759091, 0.158338589, 100.056926667, -30.029180955, -30.020883987, 0],  # back (avoid collision)

                [5, 0.625793464, -0.285759091, 0.158338589, 90.732957532, 60.516336991, 0.648604329, 0],  # move & rotate

                [6, 0.545793464, -0.330476218, 0.128338589, 90.732957532, 150.516336991, 0.648604329, 0],  # pre-grasp
                [7, 0.545793464, -0.430476218, 0.128338589, 90.732957532, 150.516336991, 0.648604329, 1],  # grasp
                [8, 0.528413994, -0.285476218, 0.252155227, 90.732957532, 90.516336991, 0.648604329, 1],  # lift-up
                [9, 0.528413994, -0.465476218, 0.402155227, 179.732957532, 0.68246746, 89.739817245, 1],  # reorient
                [10, 0.528413994, -0.615476218, 0.352155227, 179.732957532, 0.68246746, 89.739817245, 0],  # place-down
                [11, 0.528413994, -0.415476218, 0.452155227, 179.732957532, 45.68246746, 89.739817245, 0]  # back (avoid collision)
            ]
        }
        if bottle_id != 1:  # mo-li-hua-cha bottle
            if bottle_id == 2: height_offset = 0.010  # dong-fang-shu-ye bottle
            if bottle_id == 3: height_offset = 0.060  # wu-long-cha bottle
            if bottle_id == 4: height_offset = 0.060  # hong-dou-yi-mi bottle
            raw_eef_keypose_dict["L"][1][2] -= height_offset * 0.5; raw_eef_keypose_dict["R"][1][2] += height_offset * 0.5
            raw_eef_keypose_dict["L"][2][2] -= height_offset * 0.5; raw_eef_keypose_dict["R"][2][2] += height_offset * 0.5
            raw_eef_keypose_dict["L"][11][2] -= height_offset * 0.5; raw_eef_keypose_dict["R"][11][2] += height_offset * 0.5  # shrink / enlarge y-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "flipping-grasping":
        [mugcup_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
            ]
        }
        final_eef_keypose_dict = raw_eef_keypose_dict
    ################################################################################################


    ################################################################################################
    if task_name == "inserting":
        [marker_id, ordcup_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [1, 0.498413994, 0.440476218, 0.047155227, -89.610881514, 180.474229046, -0.517829792, 1],  # grasp
                [3, 0.548413994, 0.290476218, 0.247155227, -89.610881514, 90.474229046, -0.517829792, 1],  # move & rotate + lift-up 
                [4, 0.548413994, 0.290476218, 0.247155227, -179.610881514, -15.474229046, -90.517829792, 1],  # move & rotate + reorient
                [5, 0.498413994, 0.390476218, 0.187155227, -179.610881514, -15.474229046, -90.517829792, 0],  # place-down
                [7, 0.548413994, 0.290476218, 0.297155227, -179.610881514, 15.474229046, -90.517829792, 0],  # back (avoid collision)
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0, 0.443413994, -0.370476218, 0.077155227, -90.310744971, 45.516336991, -179.648604329, 1],  # grasp
                [2, 0.433413994, -0.220476218, -0.107155227, -105.310744971, 45.516336991, -179.648604329, 1],  # lift-up
                [6, 0.443413994, -0.365476218, 0.077155227, -90.310744971, 45.516336991, -179.648604329, 0],  # place-down
            ]
        }
        if marker_id != 1:  # black marker pen
            if marker_id == 2: length_offset = 0.000  # red marker pen
            if marker_id == 3: length_offset = -0.004  # green marker pen
            if marker_id == 4: length_offset = 0.000  # blue marker pen
            raw_eef_keypose_dict["L"][1][1] += (length_offset * 0.5)  # enlarge the x-value
            raw_eef_keypose_dict["L"][3][2] -= (length_offset * 0.5)  # shrink the y-value
            raw_eef_keypose_dict["L"][4][2] -= (length_offset * 0.5)  # shrink the y-value
        if ordcup_id != 1:  # yellow soft plastic cup
            if ordcup_id == 2: height_offset = 0.010; width_offset = -0.005  # blue hard plastic cup
            raw_eef_keypose_dict["R"][1][1] -= (width_offset * 0.5 * 0.707)  # shrink the x-value
            raw_eef_keypose_dict["R"][1][1] += width_offset  # enlarge the x-value
            raw_eef_keypose_dict["R"][1][2] += height_offset  # enlarge the y-value
            raw_eef_keypose_dict["R"][1][3] += (width_offset * 0.5 * 0.707)  # enlarge the z-value
            raw_eef_keypose_dict["R"][3][2] += height_offset  # enlarge the y-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "plugpen":
        [pencap_id, marker_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [1, 0.493413994, 0.440476218, 0.047155227, -89.610881514, 180.474229046, -0.517829792, 1],  # grasp
                [3, 0.493413994, 0.245476218, 0.047155227, -89.610881514, 90.474229046, -0.517829792, 1],  # lift-up
                [5, 0.493413994, 0.245476218, -0.062155227, -179.610881514, 84.474229046, -90.517829792, 1],  # move to armR
                [7, 0.493413994, 0.440476218, 0.007155227, -89.610881514, 90.474229046, -0.517829792, 0],  # place-down
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0, 0.448413994, -0.440476218, 0.047155227, -90.310744971, 90.516336991, -179.648604329, 1],  # grasp
                [2, 0.498413994, -0.240476218, 0.047155227, -90.310744971, 90.516336991, -179.648604329, 1],  # lift-up
                [4, 0.498413994, -0.240476218, -0.062155227, 179.732957532, 84.68246746, 89.739817245, 0],  # move to armL
                [6, 0.498413994, -0.190476218, 0.097155227, -90.310744971, 90.516336991, -179.648604329, 0],  # move back
            ]
        }
        assert marker_id == pencap_id, "We now only support matched [marker pen] and [pen cap] (with the same id)!!!"
        if pencap_id != 1:  # the black marker pen
            if pencap_id == 2: length_offset = 0.000  # the green marker pen
            if pencap_id == 3: length_offset = 0.000  # the red marker pen
            if pencap_id == 4: length_offset = 0.000  # the blue marker pen
            raw_eef_keypose_dict["R"][1][1] += length_offset * 0.5  # enlarge x-value
        if marker_id != 1:  # the black marker pen
            if marker_id == 2: length_offset = 0.000  # the green marker pen
            if marker_id == 3: length_offset = 0.000  # the red marker pen
            if marker_id == 4: length_offset = 0.000  # the blue marker pen
            raw_eef_keypose_dict["L"][1][1] += length_offset * 0.5
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "handover":
        [shovel_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [4, 0.433413994, 0.344476218, 0.213155227, 179.466997012, -0.524006675, -0.457821821, 0],  # pre-grasp
                [5, 0.533413994, 0.344476218, 0.113155227, 179.466997012, -0.524006675, -0.457821821, 1],  # grasp
                [8, 0.533413994, 0.344476218, -0.113155227, -89.610881514, 90.474229046, -0.517829792, 1],  # move & rotate
                [9, 0.533413994, 0.444476218, -0.113155227, -89.610881514, 90.474229046, -0.517829792, 0],  # place-down
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0, 0.603413994, -0.344476218, -0.113155227, -90.310744971, 180.516336991, -179.648604329, 0],  # pre-grasp
                [1, 0.603413994, -0.444476218, -0.113155227, -90.310744971, 180.516336991, -179.648604329, 1],  # grasp
                [2, 0.603413994, -0.240476218, -0.113155227, -90.310744971, 180.516336991, -179.648604329, 1],  # lift-up & rotate
                [3, 0.603413994, -0.340546236, 0.113155227, 179.091831882, -0.355685809, -0.515867006, 1],  # move & rotate
                [6, 0.603413994, -0.340546236, 0.113155227, 179.091831882, -0.355685809, -0.515867006, 0],  # ungrasp
                [7, 0.603413994, -0.340546236, 0.313155227, 179.091831882, -0.355685809, -0.515867006, 0],  # move back
            ]
        }
        if shovel_id != 1:  # the metal spoon
            if shovel_id == 2: length_offset = -0.050; height_offset = 0.010  # the plastic spoon
            if shovel_id == 3: length_offset = +0.040; height_offset = 0.000   # the plastic shovel
            if shovel_id == 4: length_offset = +0.030; height_offset = 0.000  # the metal shovel
            raw_eef_keypose_dict["R"][1][1] += length_offset  # enlarge x-value
            raw_eef_keypose_dict["R"][2][1] += length_offset  # enlarge x-value
            raw_eef_keypose_dict["R"][2][2] += height_offset  # adjust y-value
            raw_eef_keypose_dict["L"][1][1] -= length_offset*0.5  # shrink x-value
            raw_eef_keypose_dict["L"][2][1] -= length_offset*0.5  # shrink x-value
            raw_eef_keypose_dict["L"][2][3] -= height_offset  # adjust z-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "ppspoon" or task_name == "ppfork":
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.513413994, 0.344476218, -0.113155227, -89.610881514, 0.474229046, -0.517829792, 0],  # pre-grasp
                [2, 0.513413994, 0.444476218-0.003, -0.113155227, -89.610881514, 0.474229046, -0.517829792, 1],  # grasp (pick-up)
                [4, 0.513413994, 0.340476218, -0.113155227, -89.610881514, 0.474229046, -0.517829792, 1],  # lift-up
                [6, 0.760413994, 0.340476218, 0.013155227, -89.610881514, 0.474229046, -0.517829792, 1],  # move & rotate
                [8, 0.760413994, 0.434476218, 0.013155227, -89.610881514, 0.474229046, -0.517829792, 0],  # ungrasp (place-down)

            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [1, 0.513413994, -0.344476218, -0.113155227, -90.310744971, 180.516336991, -179.648604329, 0],  # pre-grasp
                [3, 0.513413994, -0.444476218, -0.113155227, -90.310744971, 180.516336991, -179.648604329, 1],  # grasp (pick-up)
                [5, 0.513413994, -0.340476218, -0.113155227, -90.310744971, 180.516336991, -179.648604329, 1],  # lift-up
                [7, 0.760413994, -0.340476218, 0.013155227, -90.310744971, 180.516336991, -179.648604329, 1],  # move & rotate
                [9, 0.760413994, -0.434476218, 0.013155227, -90.310744971, 180.516336991, -179.648604329, 0],  # ungrasp (place-down)
            ]
        }
        if task_name == "ppspoon":
            [spoon_id] = object_ids
            if spoon_id == 1: length_offset = +0.000  # the black plastic spoon
            if spoon_id == 2: length_offset = +0.030  # the metal large spoon
            if spoon_id == 3: length_offset = +0.000  # the metal small spoon
        if task_name == "ppfork":
            [fork_id] = object_ids
            if fork_id == 1: length_offset = +0.000  # the black plastic fork
            if fork_id == 2: length_offset = +0.030  # the metal large fork
            if fork_id == 3: length_offset = +0.000  # the metal small fork
        for step_id in [1, 2, 3]: raw_eef_keypose_dict["L"][step_id][1] += length_offset*0.5; raw_eef_keypose_dict["R"][step_id][1] += length_offset*0.5
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "ppspoon-ppfork" or task_name == "ppfork-ppspoon":
        [spoon_id, fork_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.513413994, 0.344476218, 0.047155227, -89.610881514, 0.474229046, -0.517829792, 0],  # pre-grasp
                [2, 0.513413994, 0.444476218-0.003, 0.047155227, -89.610881514, 0.474229046, -0.517829792, 1],  # grasp (pick-up)
                [4, 0.513413994, 0.340476218, 0.047155227, -89.610881514, 0.474229046, -0.517829792, 1],  # lift-up
                [6, 0.760413994, 0.340476218, 0.013155227, -89.610881514, 0.474229046, -0.517829792, 1],  # move & rotate
                [8, 0.760413994, 0.434476218, 0.013155227, -89.610881514, 0.474229046, -0.517829792, 0],  # ungrasp (place-down)

            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [1, 0.513413994, -0.344476218, 0.047155227, -90.310744971, 180.516336991, -179.648604329, 0],  # pre-grasp
                [3, 0.513413994, -0.444476218, 0.047155227, -90.310744971, 180.516336991, -179.648604329, 1],  # grasp (pick-up)
                [5, 0.513413994, -0.340476218, 0.047155227, -90.310744971, 180.516336991, -179.648604329, 1],  # lift-up
                [7, 0.760413994, -0.340476218, 0.013155227, -90.310744971, 180.516336991, -179.648604329, 1],  # move & rotate
                [9, 0.760413994, -0.434476218, 0.013155227, -90.310744971, 180.516336991, -179.648604329, 0],  # ungrasp (place-down)
            ]
        }
        if spoon_id == 1: length_offset = +0.000  # the black plastic spoon
        if spoon_id == 2: length_offset = +0.030  # the metal large spoon
        if spoon_id == 3: length_offset = +0.000  # the metal small spoon
        if fork_id == 1: length_offset = +0.000  # the black plastic fork
        if fork_id == 2: length_offset = +0.030  # the metal large fork
        if fork_id == 3: length_offset = +0.000  # the metal small fork
        assert spoon_id == fork_id, "[Sorry] we now only support to rearrange the spoon and fork with the same ids!"
        for step_id in [1, 2, 3]: raw_eef_keypose_dict["L"][step_id][1] += length_offset*0.5; raw_eef_keypose_dict["R"][step_id][1] += length_offset*0.5
        final_eef_keypose_dict = raw_eef_keypose_dict
    ################################################################################################


    ################################################################################################ 
    if task_name == "pivoting":
        [cirbowl_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0.75],  # init-pose
                [0, 0.493171478, 0.522483409, 0.233157832, -179.610881514, 25.474229046, -90.517829792, -1],  # pre-contact
                [2, 0.493171478, 0.562483409, 0.153157832, -179.610881514, 25.474229046, -90.517829792, -1],  # start contact
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0.75],  # init-pose
                [1, 0.493171478, -0.522483409, 0.233157832, 179.732957532, 25.68246746, 89.739817245, -1],  # pre-contact
                [3, 0.493171478, -0.562483409, 0.153157832, 179.732957532, 25.68246746, 89.739817245, -1]  # start contact
         ]
        }
        if cirbowl_id == 1: width_offset = -0.014  # the small white paper bowl
        if cirbowl_id == 2: width_offset = -0.013  # the small green plastic bowl 
        if cirbowl_id == 3: width_offset = -0.018  # the middle transparent plastic bowl
        if cirbowl_id == 4: width_offset = +0.018  # the large gray paper bowl  
        raw_eef_keypose_dict["L"][2][3] += (width_offset * 0.5)  # enlarge the z-value
        raw_eef_keypose_dict["R"][2][3] += (width_offset * 0.5)  # enlarge the z-value
        final_eef_keypose_dict = raw_eef_keypose_dict 
    #-------------------------------
    if task_name == "wrapping":
        [basket_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0.05],  # init-pose
                [0, 0.573171478, 0.382483409, 0.273157832, -179.610881514, 45.474229046, -90.517829792, -1],  # pre-contact
                [2, 0.573171478, 0.482483409, 0.173157832, -179.610881514, 45.474229046, -90.517829792, -1],  # start contact
                [4, 0.573171478, 0.332483409, 0.170157832, -179.610881514, 45.474229046, -90.517829792, -1]  # lift-up
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0.05],  # init-pose
                [1, 0.573171478, -0.382483409, 0.273157832, 179.732957532, 45.68246746, 89.739817245, -1],  # pre-contact
                [3, 0.573171478, -0.482483409, 0.173157832, 179.732957532, 45.68246746, 89.739817245, -1],  # start contact
                [5, 0.573171478, -0.332483409, 0.170157832, 179.732957532, 45.68246746, 89.739817245, -1]  # lift-up
            ]
        }
        if basket_id == 1: width_offset = -0.014; lifting_up_offset = 0.020  # the small pink mesh basket (left-most +0.004; right-most -0.004)
        if basket_id == 2: width_offset = +0.044; lifting_up_offset = 0.050  # the large green mesh basket (left-most +0.004; right-most -0.004)
        if basket_id == 3: width_offset = +0.074; lifting_up_offset = 0.060  # the large blue mesh basket
        raw_eef_keypose_dict["L"][2][3] += (width_offset * 0.5)  # enlarge the z-value
        raw_eef_keypose_dict["R"][2][3] += (width_offset * 0.5)  # enlarge the z-value
        raw_eef_keypose_dict["L"][3][2] -= lifting_up_offset  # shrink the y-value
        raw_eef_keypose_dict["R"][3][2] += lifting_up_offset  # enlarge the y-value
        raw_eef_keypose_dict["L"][3][3] += (width_offset * 0.5)  # enlarge the z-value
        raw_eef_keypose_dict["R"][3][3] += (width_offset * 0.5)  # enlarge the z-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    ################################  
    if task_name == "pivoting_rectbox":
        [rectbox_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.503171478, 0.368483409, 0.130157832, -179.610881514, 75.474229046, -90.517829792, 0],  # pre-contact
                [2, 0.503171478, 0.418483409, 0.030157832, -179.610881514, 75.474229046, -90.517829792, 0],  # start contact
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [1, 0.503171478, -0.512483409, 0.278157832, 179.732957532, 25.68246746, 89.739817245, 0],  # pre-contact
                [3, 0.503171478, -0.562483409, 0.178157832, 179.732957532, 25.68246746, 89.739817245, 0]  # start contact
         ]
        }
        if rectbox_id == 1: height_offset = 0.000; width_offset = 0.000  # the smallest gray rectbox
        if rectbox_id == 2: height_offset = 0.015; width_offset = 0.120  # the middle size gray rectbox
        if rectbox_id == 3: height_offset = 0.062; width_offset = 0.150  # the largest gray rectbox
        for step_id in [1, 2]:
            raw_eef_keypose_dict["L"][step_id][1] += width_offset*0.5; raw_eef_keypose_dict["R"][step_id][1] += width_offset*0.5
            raw_eef_keypose_dict["L"][step_id][3] += height_offset*0.5; raw_eef_keypose_dict["R"][step_id][3] += height_offset*0.5
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "pivoting_cirbowl":
        [cirbowl_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0.75],  # init-pose
                [0, 0.493171478, 0.522483409, 0.233157832, -179.610881514, 25.474229046, -90.517829792, -1],  # pre-contact
                [2, 0.493171478, 0.562483409, 0.153157832, -179.610881514, 25.474229046, -90.517829792, -1],  # start contact
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0.75],  # init-pose
                [1, 0.493171478, -0.522483409, 0.233157832, 179.732957532, 25.68246746, 89.739817245, -1],  # pre-contact
                [3, 0.493171478, -0.562483409, 0.153157832, 179.732957532, 25.68246746, 89.739817245, -1]  # start contact
         ]
        }
        if cirbowl_id == 1: width_offset = -0.014  # the small white paper bowl
        if cirbowl_id == 2: width_offset = -0.013  # the small green plastic bowl 
        if cirbowl_id == 3: width_offset = -0.018  # the middle transparent plastic bowl
        if cirbowl_id == 4: width_offset = +0.018  # the large gray paper bowl   
        raw_eef_keypose_dict["L"][2][3] += (width_offset * 0.5)  # enlarge the z-value
        raw_eef_keypose_dict["R"][2][3] += (width_offset * 0.5)  # enlarge the z-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "flipping_basket":
        [basket_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0.05],  # init-pose
                [0, 0.573171478, 0.382483409, 0.273157832, -179.610881514, 45.474229046, -90.517829792, -1],  # pre-contact
                [2, 0.573171478, 0.482483409, 0.173157832, -179.610881514, 45.474229046, -90.517829792, -1],  # start contact
                [4, 0.573171478, 0.332483409, 0.170157832, -179.610881514, 45.474229046, -90.517829792, -1]  # lift-up
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0.05],  # init-pose
                [1, 0.573171478, -0.382483409, 0.273157832, 179.732957532, 45.68246746, 89.739817245, -1],  # pre-contact
                [3, 0.573171478, -0.482483409, 0.173157832, 179.732957532, 45.68246746, 89.739817245, -1],  # start contact
                [5, 0.573171478, -0.332483409, 0.170157832, 179.732957532, 45.68246746, 89.739817245, -1]  # lift-up
            ]
        }
        if basket_id == 1: width_offset = -0.008; lifting_up_offset = 0.020  # the small pink mesh basket (left-most +0.004; right-most -0.004)
        if basket_id == 2: width_offset = +0.048; lifting_up_offset = 0.050  # the large green mesh basket (left-most +0.004; right-most -0.004)
        if basket_id == 3: width_offset = +0.074; lifting_up_offset = 0.060  # the large blue mesh basket
        raw_eef_keypose_dict["L"][2][3] += (width_offset * 0.5)  # enlarge the z-value
        raw_eef_keypose_dict["R"][2][3] += (width_offset * 0.5)  # enlarge the z-value
        raw_eef_keypose_dict["L"][3][2] -= lifting_up_offset  # shrink the y-value
        raw_eef_keypose_dict["R"][3][2] += lifting_up_offset  # enlarge the y-value
        raw_eef_keypose_dict["L"][3][3] += (width_offset * 0.5)  # enlarge the z-value
        raw_eef_keypose_dict["R"][3][3] += (width_offset * 0.5)  # enlarge the z-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "flipping_block":
        [block_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0.05],  # init-pose
                [0, 0.673171478, 0.35583409, 0.233157832, -179.610881514, 45.474229046, -90.517829792, -1],  # pre-contact
                [2, 0.673171478, 0.45583409, 0.133157832, -179.610881514, 45.474229046, -90.517829792, -1],  # start contact
                [4, 0.673171478, 0.30583409, 0.133157832, -179.610881514, 45.474229046, -90.517829792, -1]  # lift-up
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0.05],  # init-pose
                [1, 0.673171478, -0.35583409, 0.233157832, 179.732957532, 45.68246746, 89.739817245, -1],  # pre-contact
                [3, 0.673171478, -0.45583409, 0.133157832, 179.732957532, 45.68246746, 89.739817245, -1],  # start contact
                [5, 0.673171478, -0.30583409, 0.133157832, 179.732957532, 45.68246746, 89.739817245, -1]  # lift-up
            ]
        }
        if block_id == 1: width_offset = -0.008; lifting_up_offset = 0.100  # the large green plastic basket
        if block_id == 2: width_offset = -0.008; lifting_up_offset = 0.100  # the large purple plastic basket
        if block_id == 3: width_offset = -0.008; lifting_up_offset = 0.100  # the large orange plastic basket
        raw_eef_keypose_dict["L"][2][3] += (width_offset * 0.5)  # enlarge the z-value
        raw_eef_keypose_dict["R"][2][3] += (width_offset * 0.5)  # enlarge the z-value
        raw_eef_keypose_dict["L"][3][2] -= lifting_up_offset  # shrink the y-value
        raw_eef_keypose_dict["R"][3][2] += lifting_up_offset  # enlarge the y-value
        raw_eef_keypose_dict["L"][3][3] += (width_offset * 0.5)  # enlarge the z-value
        raw_eef_keypose_dict["R"][3][3] += (width_offset * 0.5)  # enlarge the z-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------  
    if task_name == "pivoting_bigjar":
        [bigjar_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0.5],  # init-pose
                [0, 0.473171478, 0.505483409, 0.265157832, -179.610881514, 25.474229046, -90.517829792, -1],  # pre-contact
                [2, 0.473171478, 0.555483409, 0.165157832, -179.610881514, 25.474229046, -90.517829792, -1],  # start contact
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0.5],  # init-pose
                [1, 0.473171478, -0.505483409, 0.265157832, 179.732957532, 25.68246746, 89.739817245, -1],  # pre-contact
                [3, 0.473171478, -0.555483409, 0.165157832, 179.732957532, 25.68246746, 89.739817245, -1]  # start contact
            ]
        }
        if bigjar_id == 1: width_offset_x = +0.000; width_offset_z = 0.020; height_offset = +0.000  # the big plastic jar / bottle (cylinder)
        if bigjar_id == 2: width_offset_x = -0.010; width_offset_z = 0.020; height_offset = +0.012  # the big plastic jar / bottle (cuboid)
        if bigjar_id == 3: width_offset_x = +0.055; width_offset_z = 0.025; height_offset = +0.010  # the big plastic jar / bottle (short cylinder)
        for step_id in [1, 2]:
            raw_eef_keypose_dict["L"][step_id][1] += width_offset_x*0.5; raw_eef_keypose_dict["R"][step_id][1] += width_offset_x*0.5
            raw_eef_keypose_dict["L"][step_id][2] -= width_offset_z*0.5; raw_eef_keypose_dict["R"][step_id][2] += width_offset_z*0.5
            raw_eef_keypose_dict["L"][step_id][3] += height_offset*0.5; raw_eef_keypose_dict["R"][step_id][3] += height_offset*0.5
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "pivoting_block":
        [block_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0.25],  # init-pose
                [0, 0.573171478, 0.545483409, 0.261157832, -179.610881514, 25.474229046, -90.517829792, -1],  # pre-contact
                [2, 0.573171478, 0.555483409, 0.181157832, -179.610881514, 25.474229046, -90.517829792, -1],  # start contact
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0.32],  # init-pose
                [1, 0.573171478, -0.545483409, 0.261157832, 179.732957532, 25.68246746, 89.739817245, -1],  # pre-contact
                [3, 0.573171478, -0.555483409, 0.181157832, 179.732957532, 25.68246746, 89.739817245, -1]  # start contact
            ]
        }
        if block_id == 1: width_offset_x = 0.000; width_offset_z = 0.000; height_offset = 0.000 # the large green plastic basket
        if block_id == 2: width_offset_x = 0.000; width_offset_z = 0.000; height_offset = 0.000  # the large purple plastic basket
        if block_id == 3: width_offset_x = 0.000; width_offset_z = 0.000; height_offset = 0.000  # the large orange plastic basket
        for step_id in [1, 2]:
            raw_eef_keypose_dict["L"][step_id][1] += width_offset_x*0.5; raw_eef_keypose_dict["R"][step_id][1] += width_offset_x*0.5
            raw_eef_keypose_dict["L"][step_id][2] -= width_offset_z*0.5; raw_eef_keypose_dict["R"][step_id][2] += width_offset_z*0.5
            raw_eef_keypose_dict["L"][step_id][3] += height_offset*0.5; raw_eef_keypose_dict["R"][step_id][3] += height_offset*0.5
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "toppling_holder":
        [holder_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0.5],  # init-pose
                [0, 0.475413994, 0.542483409, 0.220157832, -179.610881514, 25.474229046, -90.517829792, -1],  # pre-contact
                [2, 0.475413994, 0.562483409, 0.140157832, -179.610881514, 25.474229046, -90.517829792, -1],  # start contact
                [4, 0.475413994, 0.562483409, 0.140157832, -179.610881514, 25.474229046, -90.517829792, -1],  # keep static
                [6, 0.475513994, 0.562483409, 0.240157832, -179.610881514, 25.474229046, -90.517829792, -1],  # return back
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0.5],  # init-pose
                [1, 0.475413994, -0.320476218, 0.100155227, 179.732957532, 75.68246746, 89.739817245, -1],  # pre-contact
                [3, 0.475413994, -0.370476218, 0.000155227, 179.732957532, 75.68246746, 89.739817245, -1],  # start contact
                [5, 0.475413994, -0.330476218, -0.100155227, 179.732957532, 75.68246746, 89.739817245, -1],  # toppling
                [7, 0.475413994, -0.330476218, -0.000155227, 179.732957532, 75.68246746, 89.739817245, -1],  # return back
         ]
        }
        if holder_id == 1: width_offset = +0.000; height_offset = +0.000  # the black pen holder / cup (cylinder)  [height = 9.5 cm]
        if holder_id == 2: width_offset = -0.015; height_offset = -0.005  # the black pen holder / cup (cuboid)  [height = 9.0 cm]
        for step_id in [1, 2, 3, 4]:  # adjust the x-value / z-value
            raw_eef_keypose_dict["L"][step_id][1] += (width_offset * 0.5); raw_eef_keypose_dict["R"][step_id][1] += (width_offset * 0.5)
            raw_eef_keypose_dict["L"][step_id][3] += (width_offset * 0.5); raw_eef_keypose_dict["R"][step_id][3] += (width_offset * 0.5)
        for step_id in [1, 2, 3, 4]: raw_eef_keypose_dict["R"][step_id][2] += height_offset  # adjust the y-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "toppling_bigjar":
        [bigjar_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.475413994, 0.512483409, 0.213157832, -179.610881514, 25.474229046, -90.517829792, -1],  # pre-contact
                [2, 0.475413994, 0.552483409, 0.133157832, -179.610881514, 25.474229046, -90.517829792, -1],  # start contact
                [4, 0.475413994, 0.552483409, 0.133157832, -179.610881514, 25.474229046, -90.517829792, -1],  # keep static
                [6, 0.475413994, 0.552483409, 0.233157832, -179.610881514, 25.474229046, -90.517829792, -1],  # return back
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0.5],  # init-pose
                [1, 0.475413994, -0.300476218, 0.090155227, 179.732957532, 75.68246746, 89.739817245, -1],  # pre-contact
                [3, 0.475413994, -0.350476218, -0.010155227, 179.732957532, 75.68246746, 89.739817245, -1],  # start contact
                [5, 0.475413994, -0.320476218, -0.100155227, 179.732957532, 75.68246746, 89.739817245, -1],  # toppling
                [7, 0.475413994, -0.320476218, -0.000155227, 179.732957532, 75.68246746, 89.739817245, -1],  # return back
         ]
        }
        if bigjar_id == 1: width_offset = +0.000; height_offset = +0.000  # the big plastic jar / bottle (cylinder)
        if bigjar_id == 2: width_offset = -0.010; height_offset = +0.020  # the big plastic jar / bottle (cuboid)
        for step_id in [1, 2, 3, 4]:  # adjust the x-value / z-value
            raw_eef_keypose_dict["L"][step_id][1] += (width_offset * 0.5); raw_eef_keypose_dict["R"][step_id][1] += (width_offset * 0.5)
            raw_eef_keypose_dict["L"][step_id][3] += (width_offset * 0.5); raw_eef_keypose_dict["R"][step_id][3] += (width_offset * 0.5)
        for step_id in [1, 2, 3, 4]: raw_eef_keypose_dict["R"][step_id][2] += height_offset  # adjust the y-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "bilifting_bigjar":
        [bigjar_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.475413994, 0.342483409, 0.110157832, -179.610881514, 70.474229046, -90.517829792, -1],  # pre-contact
                [2, 0.475413994, 0.392483409, 0.010157832, -179.610881514, 70.474229046, -90.517829792, -1],  # start contact
                [4, 0.475413994, 0.292483409, 0.010157832, -179.610881514, 70.474229046, -90.517829792, -1],  # lift-up
                [6, 0.790413994, 0.292483409, 0.005157832, -179.610881514, 70.474229046, -90.517829792, -1],  # move & rotate
                [8, 0.790413994, 0.387483409, 0.005157832, -179.610881514, 70.474229046, -90.517829792, -1],  # place down
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [1, 0.475413994, -0.342483409, 0.110157832, 179.732957532, 70.68246746, 89.739817245, -1],  # pre-contact
                [3, 0.475413994, -0.392483409, 0.010157832, 179.732957532, 70.68246746, 89.739817245, -1],  # start contact
                [5, 0.475413994, -0.292483409, 0.010157832, 179.732957532, 70.68246746, 89.739817245, -1],  # lift-up
                [7, 0.790413994, -0.292483409, 0.005157832, 179.732957532, 70.68246746, 89.739817245, -1],  # move & rotate
                [9, 0.790413994, -0.387483409, 0.005157832, 179.732957532, 70.68246746, 89.739817245, -1],  # place down     
         ]
        }
        if bigjar_id == 1: width_offset_x = 0.000; width_offset_z = -0.005; height_offset = -0.010  # the big plastic jar / bottle (cylinder)
        if bigjar_id == 2: width_offset_x = 0.000; width_offset_z = -0.025; height_offset = +0.000  # the big plastic jar / bottle (cuboid)
        for step_id in [1, 2, 3, 4, 5]:
            raw_eef_keypose_dict["L"][step_id][1] += width_offset_x*0.5; raw_eef_keypose_dict["R"][step_id][1] += width_offset_x*0.5  # adjust the x-value 
            raw_eef_keypose_dict["L"][step_id][3] += width_offset_z*0.5; raw_eef_keypose_dict["R"][step_id][3] += width_offset_z*0.5  # adjust the z-value
            raw_eef_keypose_dict["L"][step_id][2] -= height_offset; raw_eef_keypose_dict["R"][step_id][2] += height_offset  # adjust the y-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "bilifting_block":
        [block_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.570413994, 0.317483409, 0.105157832, -179.610881514, 80.474229046, -90.517829792, -1],  # pre-contact
                [2, 0.570413994, 0.367483409, 0.005157832, -179.610881514, 80.474229046, -90.517829792, -1],  # start contact
                [4, 0.570413994, 0.267483409, 0.005157832, -179.610881514, 80.474229046, -90.517829792, -1],  # lift-up
                [6, 0.780413994, 0.267483409, 0.005157832, -179.610881514, 80.474229046, -90.517829792, -1],  # move & rotate
                [8, 0.780413994, 0.362483409, 0.005157832, -179.610881514, 80.474229046, -90.517829792, -1],  # place down
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [1, 0.570413994, -0.317483409, 0.105157832, 179.732957532, 80.68246746, 89.739817245, -1],  # pre-contact
                [3, 0.570413994, -0.367483409, 0.005157832, 179.732957532, 80.68246746, 89.739817245, -1],  # start contact
                [5, 0.570413994, -0.267483409, 0.005157832, 179.732957532, 80.68246746, 89.739817245, -1],  # lift-up
                [7, 0.780413994, -0.267483409, 0.005157832, 179.732957532, 80.68246746, 89.739817245, -1],  # move & rotate
                [9, 0.780413994, -0.362483409, 0.005157832, 179.732957532, 80.68246746, 89.739817245, -1],  # place down     
         ]
        }
        if block_id == 1: width_offset_x = 0.000; width_offset_z = 0.000; height_offset = 0.000 # the large green plastic basket
        if block_id == 2: width_offset_x = 0.000; width_offset_z = 0.000; height_offset = 0.000  # the large purple plastic basket
        if block_id == 3: width_offset_x = 0.000; width_offset_z = 0.000; height_offset = 0.000  # the large orange plastic basket
        for step_id in [1, 2, 3, 4, 5]:
            raw_eef_keypose_dict["L"][step_id][1] += width_offset_x*0.5; raw_eef_keypose_dict["R"][step_id][1] += width_offset_x*0.5  # adjust the x-value 
            raw_eef_keypose_dict["L"][step_id][3] += width_offset_z*0.5; raw_eef_keypose_dict["R"][step_id][3] += width_offset_z*0.5  # adjust the z-value
            raw_eef_keypose_dict["L"][step_id][2] -= height_offset; raw_eef_keypose_dict["R"][step_id][2] += height_offset  # adjust the y-value
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------

    ################################  

    ################################################################################################  
    if task_name == "penbagzip":
        [penbag_id, marker_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.530142021, 0.53552716, 0.222131294, 160.523111098, -0.31602756, 179.347483597, 0],  # pre-grasp

                [2, 0.530142021, 0.55552716, 0.122131294, 160.523111098, -0.31602756, 179.347483597, 1],  # grasp

                [4, 0.530142021, 0.25552716, 0.222131294, 120.523111098, -0.31602756, 179.347483597, 1],  # lift-up & rotate (for adjusting the arm-L)
                [5, 0.630142021, 0.25552716, 0.022131294, -110.610881514, 30.474229046, -0.517829792, 1],  # move & rotate (for catching marker pens)

                [10, 0.530142021, 0.25552716, 0.022131294, 179.610881514, 60.474229046, -100.517829792, 1],  # move & rotate (for adjusting the zipper)

                [12, 0.530142021, 0.25552716, 0.022131294, 179.610881514, 60.474229046, -100.517829792, 0],  # keep static (for open the gripper)
                [13, 0.510142021, 0.20552716, 0.102131294, 179.610881514, 60.474229046, -90.517829792, 0],  # move back (for adjusting the arm-L)

                [15, 0.555142021, 0.24052716, -0.112131294, -105.610881514, 30.474229046, -30.517829792, 1],  # move & rotate (for grasping the zipper)
                [16, 0.535142021, 0.22052716, -0.092131294, -105.610881514, 30.474229046, -30.517829792, 1],  # move back (for adjusting the arm-L)
                [17, 0.665142021, 0.19052716, -0.002131294, 179.610881514, 60.474229046, -90.517829792, 1],  # move & rotate (for adjusting the zipper)
                [18, 0.665142021, 0.23052716, 0.122131294, 179.610881514, 80.474229046, -90.517829792, 0],  # move back & zipped (for zipping penbag)
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [1, 0.508413994, -0.345476218, 0.050155227, -90.310744971, 180.516336991, -179.648604329, 0],  # pre-grasp

                [3, 0.508413994, -0.445476218, 0.050155227, -90.310744971, 180.516336991, -179.648604329, 1],  # grasp

                [6, 0.508413994, -0.345476218, 0.050155227, -90.310744971, 90.516336991, -179.648604329, 1],  # lift-up & rotate
                [7, 0.508413994, -0.345476218, 0.200155227, 170.310744971, -10.516336991, 89.648604329, 1],  # move & rotate
                [8, 0.523413994, -0.395476218, 0.065155227, 170.310744971, -10.516336991, 89.648604329, 0],  # place-down

                [9, 0.523171478, -0.252483409, 0.243157832, 179.732957532, 60.68246746, 99.739817245, 0],  # move back & pre-grasp

                [11, 0.503171478, -0.302483409, 0.143157832, 179.732957532, 30.68246746, 109.739817245, 1],  # grasp (for grasping the penbag)

                [14, 0.583171478, -0.382483409, 0.163157832, 159.732957532, 10.68246746, 84.739817245, 1],  # move & rotate (for adjusting the zipper)

                [19, 0.583171478, -0.582483409, 0.363157832, 159.732957532, 0.68246746, 54.739817245, 0],  # move back & place-down
            ]
        }
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    if task_name == "foldtowel":
        [towel_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [0, 0.448413994, 0.350476218, 0.170157832, -179.610881514, 75.474229046, -90.517829792, 0],  # pre-grasp
                [2, 0.448413994, 0.450476218, 0.070157832, -179.610881514, 75.474229046, -90.517829792, 1],  # start grasp
                [4, 0.648413994, 0.150476218, 0.060157832, -179.610881514, 75.474229046, -90.517829792, 1],  # lift-up higher
                [6, 0.348413994, 0.440476218, 0.060157832, -179.610881514, 75.474229046, -90.517829792, 1],  # put-down higher
                [8, 0.448413994, 0.350476218, 0.060157832, -179.610881514, 75.474229046, -90.517829792, 1],  # lift-up lower  
                [10, 0.548413994, 0.440476218, 0.060157832, -179.610881514, 75.474229046, -90.517829792, 0],  # put-down lower  
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [1, 0.448413994, -0.350476218, 0.170157832, 179.732957532, 75.68246746, 89.739817245, 0],  # pre-grasp
                [3, 0.448413994, -0.450476218, 0.070157832, 179.732957532, 75.68246746, 89.739817245, 1],  # start grasp
                [5, 0.648413994, -0.150476218, 0.060157832, 179.732957532, 75.68246746, 89.739817245, 1],  # lift-up higher
                [7, 0.348413994, -0.440476218, 0.060157832, 179.732957532, 75.68246746, 89.739817245, 1],  # put-down higher
                [9, 0.448413994, -0.350476218, 0.060157832, 179.732957532, 75.68246746, 89.739817245, 1],  # lift-up lower    
                [11, 0.548413994, -0.440476218, 0.060157832, 179.732957532, 75.68246746, 89.739817245, 0],  # put-down lower      
         ]
        }
        if towel_id == 1: width_offset = 0.000; length_offset = 0.000  # the smaller blue towel
        if towel_id == 2: width_offset = 0.000; length_offset = 0.000  # the smaller gray/yellow towel
        if towel_id == 3: width_offset = 0.000; length_offset = 0.400  # the larger blue towel
        if towel_id == 4: width_offset = 0.000; length_offset = 0.400  # the larger gray/yellow towel
        for step_id in [1, 2, 3, 4, 5, 6]:
            raw_eef_keypose_dict["L"][step_id][3] += width_offset*0.5; raw_eef_keypose_dict["R"][step_id][3] += width_offset*0.5  # adjust the z-value
            if step_id == 3:  # lift-up higher
                raw_eef_keypose_dict["L"][step_id][1] += length_offset*0.5; raw_eef_keypose_dict["R"][step_id][1] += length_offset*0.5  # adjust the x-value
                raw_eef_keypose_dict["L"][step_id][2] += length_offset; raw_eef_keypose_dict["R"][step_id][2] -= length_offset  # adjust the y-value
            if step_id in [5, 6]:  
                raw_eef_keypose_dict["L"][step_id][1] += length_offset*0.5; raw_eef_keypose_dict["R"][step_id][1] += length_offset*0.5  # adjust the x-value
                raw_eef_keypose_dict["L"][step_id][2] += length_offset*0.5; raw_eef_keypose_dict["R"][step_id][2] -= length_offset*0.5  # adjust the y-value
        final_eef_keypose_dict = raw_eef_keypose_dict
        
    #-------------------------------
    if task_name == "coilcable":
        [cable_id] = object_ids
        raw_eef_keypose_dict = {
            "L": [
                [-1, 0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792, 0],  # init-pose
                [4, 0.533413994, 0.204476218, 0.253155227, -179.610881514, 0.474229046, -90.517829792, 0],  # pre-grasp
                [5, 0.533413994, 0.184476218, 0.073155227, -179.466997012, 0.474229046, -90.517829792, 0.5],  # move above armR gripper
                [6, 0.533413994, 0.444476218, 0.103155227, -179.466997012, 0.474229046, -90.517829792, 1],  # move down along Z-line
                
                [9, 0.533413994, 0.149476218, -0.113155227, -89.610881514, 180.474229046, -0.517829792, 1],  # lift-up & rotate
                
                [13, 0.533413994, 0.149476218, -0.113155227, -89.610881514, 180.474229046, -0.517829792, 0],  # ungrasp
                [14, 0.533413994, 0.099476218, 0.113155227, -89.610881514, 90.474229046, -0.517829792, 0],  # move back
            ], 
            "R": [
                [-1, 0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329, 0],  # init-pose
                [0, 0.433413994, -0.349476218, -0.113155227, -90.310744971, 90.516336991, -179.648604329, 0],  # pre-grasp
                [1, 0.433413994, -0.449476218, -0.113155227, -90.310744971, 90.516336991, -179.648604329, 1],  # grasp
                [2, 0.433413994, -0.349476218, -0.113155227, -90.310744971, 90.516336991, -179.648604329, 1],  # lift-up
                [3, 0.533413994, -0.009476218, -0.113155227, -90.310744971, 180.516336991, -179.648604329, 1],  # lift-up & rotate
                
                [7, 0.533413994, -0.009476218, -0.113155227, -90.310744971, 180.516336991, -179.648604329, 0],  # ungrasp
                [8, 0.533413994, -0.244476218, 0.213155227, 179.732957532, 45.68246746, 89.739817245, 0],  # move back
                
                [10, 0.533413994, -0.344476218, 0.253155227, 179.732957532, 0.68246746, 89.739817245, 0],  # pre-grasp
                [11, 0.533413994, -0.324476218, 0.073155227, 179.732957532, 0.68246746, 89.739817245, 0.5],  # move above armL gripper
                [12, 0.533413994, -0.464476218, 0.103155227, 179.732957532, 0.68246746, 89.739817245, 1],  # move down along Z-line
                
                [15, 0.533413994, -0.364476218, 0.093155227, -90.310744971, 90.516336991, -179.648604329, 1],  # move & rotate
                [16, 0.533413994, -0.434476218, 0.093155227, -90.310744971, 90.516336991, -179.648604329, 0],  # place down
            ]
        }
        if cable_id == 1: length_offset = 0.000  # the short white cable
        if cable_id == 2: length_offset = 0.450  # the long black cable
        if cable_id == 3: length_offset = 0.320  # the super long red rope
        raw_eef_keypose_dict["R"][4][2] += length_offset * 0.250  # adjust the y-value for armR (with folding once)
        raw_eef_keypose_dict["R"][5][2] += length_offset * 0.250 
        raw_eef_keypose_dict["L"][1][2] -= length_offset * 0.250
        raw_eef_keypose_dict["L"][2][2] -= length_offset * 0.250
        raw_eef_keypose_dict["L"][4][2] -= length_offset * 0.125  # adjust the y-value for armL (with folding twice)
        raw_eef_keypose_dict["L"][5][2] -= length_offset * 0.125
        raw_eef_keypose_dict["R"][7][2] += length_offset * 0.125
        raw_eef_keypose_dict["R"][8][2] += length_offset * 0.125
        final_eef_keypose_dict = raw_eef_keypose_dict
    #-------------------------------
    
    ################################################################################################  

    return final_eef_keypose_dict

#################################################################
#################################################################
#################################################################
