
import os
import sys
import cv2
import time
import copy
import json
import numpy as np

sys.path.insert(0, os.getcwd())

from src.config import cfg_dict_init as cfg_dict
from src.vlms import conduct_object_detect_and_segment
from src.vlms import calInsideRectIOU
from src.pcd2sat import load_FS_model
from src.pcd2sat import check_state_via_obj_pcd
from src.pcd2sat import process_a_paired_binocular_images

fs_model = load_FS_model()  # load the Foundation-Stereo model

##################################################################################################################################
##################################################################################################################################

def processing_test_image_full_stage_urm_t1_box(left_image_test, cls_name, right_image_test):

    is_detected = False
    while not is_detected:
        # try:
        #     final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=cls_name, is_raw_result=True)
        # except:
        #     print("**********************************************Do Not Detect Anything!!!**********************************************")
        #     r_num = np.random.rand(); alpha = 1.0 * r_num; beta = 10*r_num
        #     left_image_test = cv2.convertScaleAbs(left_image_test.copy(), alpha=alpha, beta=beta)  # adjust the brightness and contrast
        #     time.sleep(1); continue
        final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=cls_name, is_raw_result=True)
        final_res_list_dict = {cls_name: []}
        for final_res_list in final_res_list_raw:
            [obj_name, bbox, obj_binary_mask, multi_masks] = final_res_list
            if len(final_res_list_dict[obj_name]) == 0:
                final_res_list_dict[obj_name] = final_res_list
        if len(final_res_list_dict[cls_name]) != 0:
            is_detected = True

    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']
    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    
    obj_binary_mask = final_res_list_raw[0][2]
    top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
    obj_binary_mask = cv2.copyMakeBorder(obj_binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    pcd_scene, pcd_obj, vis_res = process_a_paired_binocular_images(
        fs_model, left_image_test[:, :, ::-1], right_image_test[:, :, ::-1], obj_binary_mask)
    cur_state, pcd_obj_adjusted, cur_state_str = check_state_via_obj_pcd(pcd_obj, cls_name)
    
    # category_list = {"A": "standing", "B": "lying down"}
    assert cur_state in ["A", "B"], "the state of current detected urm_t1_box is not valid!!!"

    return final_res_list_dict, cur_state, img_vis_cv2
    
def processing_test_image_full_stage_urm_t2_bowl(left_image_test, cls_name, right_image_test):

    is_detected = False
    while not is_detected:
        # try:
        #     final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=cls_name, is_raw_result=True)
        # except:
        #     print("**********************************************Do Not Detect Anything!!!**********************************************")
        #     r_num = np.random.rand(); alpha = 1.0 * r_num; beta = 10*r_num
        #     left_image_test = cv2.convertScaleAbs(left_image_test.copy(), alpha=alpha, beta=beta)  # adjust the brightness and contrast
        #     time.sleep(1); continue
        final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=cls_name, is_raw_result=True)
        final_res_list_dict = {cls_name: []}
        max_y_value = 0 
        for final_res_list in final_res_list_raw:
            [obj_name, bbox, obj_binary_mask, multi_masks] = final_res_list
            # if len(final_res_list_dict[obj_name]) == 0:
            #     final_res_list_dict[obj_name] = final_res_list
            if obj_name == cls_name:
                if bbox[-1] > max_y_value:  # only fetch the lowest bowl
                    final_res_list_dict[obj_name] = final_res_list
                    max_y_value = bbox[-1]
        if len(final_res_list_dict[cls_name]) != 0:
            is_detected = True

    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']
    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    
    obj_binary_mask = final_res_list_raw[0][2]
    top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
    obj_binary_mask = cv2.copyMakeBorder(obj_binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    pcd_scene, pcd_obj, vis_res = process_a_paired_binocular_images(
        fs_model, left_image_test[:, :, ::-1], right_image_test[:, :, ::-1], obj_binary_mask)
    cur_state, pcd_obj_adjusted, cur_state_str = check_state_via_obj_pcd(pcd_obj, cls_name)
    
    # category_list = {"A": "upright", "B": "inverted"}
    assert cur_state in ["A", "B"], "the state of current detected urm_t2_bowl is not valid!!!"

    return final_res_list_dict, cur_state, img_vis_cv2

def processing_test_image_full_stage_urm_t3_basket(left_image_test, cls_name, right_image_test):

    is_detected = False
    while not is_detected:
        # try:
        #     final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=cls_name, is_raw_result=True)
        # except:
        #     print("**********************************************Do Not Detect Anything!!!**********************************************")
        #     r_num = np.random.rand(); alpha = 1.0 * r_num; beta = 10*r_num
        #     left_image_test = cv2.convertScaleAbs(left_image_test.copy(), alpha=alpha, beta=beta)  # adjust the brightness and contrast
        #     time.sleep(1); continue
        final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=cls_name, is_raw_result=True)
        final_res_list_dict = {cls_name: []}
        for final_res_list in final_res_list_raw:
            [obj_name, bbox, obj_binary_mask, multi_masks] = final_res_list
            if len(final_res_list_dict[obj_name]) == 0:
                final_res_list_dict[obj_name] = final_res_list
        if len(final_res_list_dict[cls_name]) != 0:
            is_detected = True

    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']
    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    
    obj_binary_mask = final_res_list_raw[0][2]
    top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
    obj_binary_mask = cv2.copyMakeBorder(obj_binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    pcd_scene, pcd_obj, vis_res = process_a_paired_binocular_images(
        fs_model, left_image_test[:, :, ::-1], right_image_test[:, :, ::-1], obj_binary_mask)
    cur_state, pcd_obj_adjusted, cur_state_str = check_state_via_obj_pcd(pcd_obj, cls_name)
    
    # category_list = {"A": "upright", "B": "inverted"}
    assert cur_state in ["A", "B"], "the state of current detected urm_t3_basket is not valid!!!"

    return final_res_list_dict, cur_state, img_vis_cv2

def processing_test_image_full_stage_urm_t4_holder(left_image_test, cls_name, right_image_test):
    is_detected = False
    while not is_detected:
        # try:
        #     final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=cls_name, is_raw_result=True)
        # except:
        #     print("**********************************************Do Not Detect Anything!!!**********************************************")
        #     r_num = np.random.rand(); alpha = 1.0 * r_num; beta = 10*r_num
        #     left_image_test = cv2.convertScaleAbs(left_image_test.copy(), alpha=alpha, beta=beta)  # adjust the brightness and contrast
        #     time.sleep(1); continue
        final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=cls_name, is_raw_result=True)
        final_res_list_dict = {cls_name: []}
        for final_res_list in final_res_list_raw:
            [obj_name, bbox, obj_binary_mask, multi_masks] = final_res_list
            if len(final_res_list_dict[obj_name]) == 0:
                final_res_list_dict[obj_name] = final_res_list
        if len(final_res_list_dict[cls_name]) != 0:
            is_detected = True

    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']
    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    
    obj_binary_mask = final_res_list_raw[0][2]
    top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
    obj_binary_mask = cv2.copyMakeBorder(obj_binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    '''
    pcd_scene, pcd_obj, vis_res = process_a_paired_binocular_images(
        fs_model, left_image_test[:, :, ::-1], right_image_test[:, :, ::-1], obj_binary_mask)
    cur_state, pcd_obj_adjusted, cur_state_str = check_state_via_obj_pcd(pcd_obj, cls_name)
    
    # category_list = {"A": "upright", "B": lying, "C": "inverted"}
    assert cur_state in ["A", "B", "C"], "the state of current detected urm_t4_holder is not valid!!!"
    '''
    cur_state = "A"
    time.sleep(1)

    return final_res_list_dict, cur_state, img_vis_cv2
  
def processing_test_image_full_stage_urm_t5_bigjar(left_image_test, cls_name, right_image_test):
    is_detected = False
    while not is_detected:
        # try:
        #     final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=cls_name, is_raw_result=True)
        # except:
        #     print("**********************************************Do Not Detect Anything!!!**********************************************")
        #     r_num = np.random.rand(); alpha = 1.0 * r_num; beta = 10*r_num
        #     left_image_test = cv2.convertScaleAbs(left_image_test.copy(), alpha=alpha, beta=beta)  # adjust the brightness and contrast
        #     time.sleep(1); continue
        final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=cls_name, is_raw_result=True)
        final_res_list_dict = {cls_name: []}
        for final_res_list in final_res_list_raw:
            [obj_name, bbox, obj_binary_mask, multi_masks] = final_res_list
            if len(final_res_list_dict[obj_name]) == 0:
                final_res_list_dict[obj_name] = final_res_list
        if len(final_res_list_dict[cls_name]) != 0:
            is_detected = True

    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']
    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    
    obj_binary_mask = final_res_list_raw[0][2]
    top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
    obj_binary_mask = cv2.copyMakeBorder(obj_binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    '''
    pcd_scene, pcd_obj, vis_res = process_a_paired_binocular_images(
        fs_model, left_image_test[:, :, ::-1], right_image_test[:, :, ::-1], obj_binary_mask)
    cur_state, pcd_obj_adjusted, cur_state_str = check_state_via_obj_pcd(pcd_obj, cls_name)
    
    # category_list = {"A": "upright", "B": lying, "C": "inverted"}
    assert cur_state in ["A", "B", "C"], "the state of current detected urm_t4_holder is not valid!!!"
    '''
    cur_state = "A"
    time.sleep(1)

    return final_res_list_dict, cur_state, img_vis_cv2

def processing_test_image_full_stage_urm_t6_block(left_image_test, cls_name, right_image_test):
    is_detected = False
    while not is_detected:
        # try:
        #     final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=cls_name, is_raw_result=True)
        # except:
        #     print("**********************************************Do Not Detect Anything!!!**********************************************")
        #     r_num = np.random.rand(); alpha = 1.0 * r_num; beta = 10*r_num
        #     left_image_test = cv2.convertScaleAbs(left_image_test.copy(), alpha=alpha, beta=beta)  # adjust the brightness and contrast
        #     time.sleep(1); continue
        final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=cls_name, is_raw_result=True)
        final_res_list_dict = {cls_name: []}
        for final_res_list in final_res_list_raw:
            [obj_name, bbox, obj_binary_mask, multi_masks] = final_res_list
            if len(final_res_list_dict[obj_name]) == 0:
                final_res_list_dict[obj_name] = final_res_list
        if len(final_res_list_dict[cls_name]) != 0:
            is_detected = True

    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']
    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    
    obj_binary_mask = final_res_list_raw[0][2]
    top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
    obj_binary_mask = cv2.copyMakeBorder(obj_binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    '''
    pcd_scene, pcd_obj, vis_res = process_a_paired_binocular_images(
        fs_model, left_image_test[:, :, ::-1], right_image_test[:, :, ::-1], obj_binary_mask)
    cur_state, pcd_obj_adjusted, cur_state_str = check_state_via_obj_pcd(pcd_obj, cls_name)
    
    # category_list = {"A": "upright", "B": lying, "C": "inverted"}
    assert cur_state in ["A", "B", "C"], "the state of current detected urm_t4_holder is not valid!!!"
    '''
    cur_state = "A"
    time.sleep(1)
    
    return final_res_list_dict, cur_state, img_vis_cv2
    
##################################################################################################################################
##################################################################################################################################

def processing_test_image_full_stage_rarg_t1_dining(left_image_test, cls_name1, cls_name2, right_image_test):

    is_detected = False
    degseg_pmts = f"{cls_name1},{cls_name2}"  # spoon,fork
    while not is_detected:
        final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=degseg_pmts, is_raw_result=True)
        final_res_list_dict = {cls_name1: [], cls_name2: []}  # spoon, fork
        for final_res_list in final_res_list_raw:
            [obj_name, bbox, obj_binary_mask, multi_masks] = final_res_list
            if len(final_res_list_dict[obj_name]) == 0:
                final_res_list_dict[obj_name] = final_res_list
        if len(final_res_list_dict[cls_name1]) != 0 or len(final_res_list_dict[cls_name2]) != 0:
            is_detected = True

    if len(final_res_list_dict[cls_name1]) != 0 and len(final_res_list_dict[cls_name2]) != 0:  # both spoon and fork are detected
        bbox1, bbox2 = final_res_list_dict[cls_name1][1], final_res_list_dict[cls_name2][1]
        if bbox1[0] < bbox2[0]: cur_state = "A"  # spoon in L; fork in R
        if bbox1[0] > bbox2[0]: cur_state = "B"  # spoon in R; fork in L
    elif len(final_res_list_dict[cls_name1]) != 0 and len(final_res_list_dict[cls_name2]) == 0:  # spoon is detected, fork is not detected
        bbox1 = final_res_list_dict[cls_name1][1]
        cur_state = "C"
    elif len(final_res_list_dict[cls_name1]) == 0 and len(final_res_list_dict[cls_name2]) != 0:  # spoon is not detected, fork is detected
        bbox2 = final_res_list_dict[cls_name2][1]
        cur_state = "D"
    else:
        cur_state = "E"  # the target object has already been rightly placed
    return final_res_list_dict, cur_state, img_vis_cv2

def processing_test_image_full_stage_rarg_t2_drinking(left_image_test, cls_name, cls_name1, right_image_test):

    is_detected = False
    degseg_pmts = f"{cls_name},{cls_name1}"  # bottle,cup
    while not is_detected:
        final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(left_image_test, prompts_str=degseg_pmts, is_raw_result=True)
        final_res_list_dict = {cls_name: [], cls_name1: []}  # bottle, cup
        for final_res_list in final_res_list_raw:
            [obj_name, bbox, obj_binary_mask, multi_masks] = final_res_list
            if len(final_res_list_dict[obj_name]) == 0:
                final_res_list_dict[obj_name] = final_res_list
        if len(final_res_list_dict[cls_name]) != 0 or len(final_res_list_dict[cls_name1]) != 0:
            is_detected = True

    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']
    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    
    assert (len(final_res_list_dict[cls_name]) != 0 and len(final_res_list_dict[cls_name1]) != 0), "we must have to find the bottle and mug cup!!!"

    # checking the state of bottle
    obj_binary_mask = final_res_list_dict[cls_name][2]  # <<---- note here 
    top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
    obj_binary_mask = cv2.copyMakeBorder(obj_binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    pcd_scene, pcd_obj, vis_res = process_a_paired_binocular_images(
        fs_model, left_image_test[:, :, ::-1], right_image_test[:, :, ::-1], obj_binary_mask)
    cur_state_1, pcd_obj_adjusted, cur_state_str = check_state_via_obj_pcd(pcd_obj, "bottle")  # <<---- note here 
    # cur_state_1 --> "B": "lying down", "A": "upright", "C": "upside down"
    if cur_state_1 == "C": bottle_state = "C"  # the flatting skill for a "upside down" bottle
    if cur_state_1 == "B": bottle_state = "B"  # the reorient skill for a "lying down" bottle
    if cur_state_1 == "A": bottle_state = "A"  # all unscrew skill for a "upright" bottle

    # checking the state of mug cup
    obj_binary_mask = final_res_list_dict[cls_name1][2]  # <<---- note here 
    top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
    obj_binary_mask = cv2.copyMakeBorder(obj_binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    pcd_scene, pcd_obj, vis_res = process_a_paired_binocular_images(
        fs_model, left_image_test[:, :, ::-1], right_image_test[:, :, ::-1], obj_binary_mask)
    cur_state_2, pcd_obj_adjusted, cur_state_str = check_state_via_obj_pcd(pcd_obj, "mug")  # <<---- note here 
    # cur_state_2 --> "B": "lying down", "A": "upright", "C": "upside down"
    if cur_state_2 == "C": mugcup_state = "C"  # the flipping skill for a "upside down" mug cup
    if cur_state_2 == "B": mugcup_state = "B"  # the grasping skill for a "lying down" mug cup
    if cur_state_2 == "A": mugcup_state = "A"  # all pre-manipulation steps are finished

    return final_res_list_dict, bottle_state, mugcup_state, img_vis_cv2


##################################################################################################################################
##################################################################################################################################

