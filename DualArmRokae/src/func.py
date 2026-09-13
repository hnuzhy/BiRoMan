
import os
import sys
import cv2
import time
import copy
import json
import numpy as np

sys.path.insert(0, os.getcwd())

from src.config import cfg_dict_init as cfg_dict
from src.util import compute_pts_abstract_xy_relative_rot
from src.vlms import conduct_object_detect_and_segment
from src.vlms import plot_processed_results_vis

from src.states_v1 import generate_prompt
from src.prompts import SYS_MESSAGE_BOTTLE_CAP_CHECK
from src.prompts import SYS_MESSAGE_BOTTLE_3_STATES as SYS_MESSAGE_BOTTLE_STATES
from src.prompts import SYS_MESSAGE_MUGCUP_3_STATES as SYS_MESSAGE_MUGCUP_STATES
from src.prompts import SYS_MESSAGE_BOWL_2_STATES as SYS_MESSAGE_BOWL_STATES
#from src.prompts import SYS_MESSAGE_POURING_V1 as SYS_MESSAGE_POURING
from src.prompts import SYS_MESSAGE_POURING_V2 as SYS_MESSAGE_POURING


#################################################################
def processing_seed_image(args, saved_seed_img_path, detseg_prompts, supp_prompts):
    rect_l, rect_w = cfg_dict['rect_l'], cfg_dict['rect_w']
    pts_polygon = cfg_dict['pts_polygon']; transform_mat_inv = cfg_dict['transform_mat_inv']
    [x1, y1, x2, y2] = cfg_dict['detection_roi_bbox']  # for fast and stable detection and segmentation
    cv2_font= cv2.FONT_HERSHEY_SIMPLEX

    ############################
    saved_seed_pts_path = saved_seed_img_path.replace(".jpg", ".json")
    saved_seed_vis_path = saved_seed_img_path.replace(".jpg", "_vis.jpg")
    saved_seed_prj_path = saved_seed_img_path.replace(".jpg", "_prj.jpg")

    if os.path.exists(saved_seed_pts_path):  # for saving time
        reprojected_seeding_pts = json.load(open(saved_seed_pts_path, "r"))
        if args.debug_close_loop_vis:
            left_image_vis = cv2.imread(saved_seed_vis_path)
            result_img_init = cv2.imread(saved_seed_prj_path)
            return reprojected_seeding_pts, left_image_vis, result_img_init
        else:
            return reprojected_seeding_pts, None, None
    else:
        left_image_seed = cv2.imread(saved_seed_img_path)
    ############################    

    '''VLMs: only use Florence2 + SAM2'''
    left_image_seed_sub = left_image_seed[y1:y2, x1:x2]
    new_detseg_prompts = detseg_prompts+","+supp_prompts if supp_prompts is not None else detseg_prompts
    final_res_list, img_vis_seed_cv2 = conduct_object_detect_and_segment(left_image_seed.copy(), prompts_str=new_detseg_prompts, is_raw_result=True)
    if args.debug_close_loop_vis:
        left_image_vis = left_image_seed.copy()
        left_image_vis[y1:y2, x1:x2] = plot_processed_results_vis(left_image_seed_sub.copy(), final_res_list, new_detseg_prompts.split(","))
    
    supp_mask = None if supp_prompts is None else [temp_res[-1][0] for temp_res in final_res_list if temp_res[0] == supp_prompts][0]
    reprojected_seeding_pts = []
    for [obj_name, bbox, obj_binary_mask, multi_masks] in final_res_list:
        if supp_mask is not None and obj_name == supp_prompts: continue
        return_list = compute_pts_abstract_xy_relative_rot(multi_masks[0], args.task_name, obj_name, supp_mask=supp_mask, binary_mask=obj_binary_mask)
        [pt_o_obj, pt_p_obj, rel_theta_deg, pt_o_supp, pt_p_supp] = return_list
        if type(pt_o_supp) != tuple: reprojected_seeding_pts.append([pt_p_obj[0], pt_p_obj[1], rel_theta_deg, bbox.tolist()])
        else: reprojected_seeding_pts.append([pt_p_obj[0], pt_p_obj[1], rel_theta_deg, bbox.tolist() + pt_o_supp[1] + pt_p_supp[1]])  # note this trick <<<---
        if args.debug_close_loop_vis:
            cv2.circle(left_image_vis, (int(pt_o_obj[0]), int(pt_o_obj[1])), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(left_image_vis, (int(pt_o_obj[0]), int(pt_o_obj[1])), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
            if supp_mask is not None:  # for task reorient of the lying down bottle
                cv2.circle(left_image_vis, (int(pt_o_supp[0]), int(pt_o_supp[1])), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(left_image_vis, (int(pt_o_supp[0]), int(pt_o_supp[1])), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
                cv2.arrowedLine(left_image_vis, (int(pt_o_obj[0]), int(pt_o_obj[1])), (int(pt_o_supp[0]), int(pt_o_supp[1])),
                    (0,255,255), thickness=3, tipLength=0.25, line_type=cv2.LINE_AA)
            elif pt_o_supp is not None and pt_p_supp is not None:  # for task inserting of the marker pen
                if type(pt_o_supp) == list:  # for tasks relating pen/spoon/fork
                    endpt_L, endpt_R = pt_o_supp, pt_p_supp
                    cv2.circle(left_image_vis, (int(endpt_L[0]), int(endpt_L[1])), radius=4, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(left_image_vis, (int(endpt_L[0]), int(endpt_L[1])), radius=2, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(left_image_vis, (int(endpt_R[0]), int(endpt_R[1])), radius=4, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(left_image_vis, (int(endpt_R[0]), int(endpt_R[1])), radius=2, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.arrowedLine(left_image_vis, (int(endpt_L[0]), int(endpt_L[1])), (int(endpt_R[0]), int(endpt_R[1])),
                        (0,255,255), thickness=3, tipLength=0.25, line_type=cv2.LINE_AA)
                if type(pt_o_supp) == tuple: # for tasks relating folding cloth
                    grasp_pt_L, grasp_pt_R = pt_o_supp[0], pt_p_supp[0]
                    cv2.circle(left_image_vis, (int(grasp_pt_L[0]), int(grasp_pt_L[1])), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(left_image_vis, (int(grasp_pt_R[0]), int(grasp_pt_R[1])), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.arrowedLine(left_image_vis, (int(grasp_pt_L[0]), int(grasp_pt_L[1])), (int(grasp_pt_R[0]), int(grasp_pt_R[1])),
                        (0,255,255), thickness=3, tipLength=0.25, line_type=cv2.LINE_AA)
                    cv2.arrowedLine(left_image_vis, (int(grasp_pt_R[0]), int(grasp_pt_R[1])), (int(grasp_pt_L[0]), int(grasp_pt_L[1])),
                        (0,255,255), thickness=3, tipLength=0.25, line_type=cv2.LINE_AA)
            # else: print("[Error] We dot not support other outputs now!!!"); os._exit(0)
    if args.debug_close_loop_vis:
        cv2.polylines(left_image_vis, [pts_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.arrowedLine(left_image_vis, (20, 20), (120, 20), (0,255,255), thickness=4, line_type=cv2.LINE_AA)
        cv2.arrowedLine(left_image_vis, (20, 20), (20, 120), (0,255,255), thickness=4, line_type=cv2.LINE_AA)
        cv2.putText(left_image_vis, "X", (125, 35), fontFace=cv2_font, fontScale=0.8, color=(0,255,255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(left_image_vis, "Y", (30, 120), fontFace=cv2_font, fontScale=0.8, color=(0,255,255), thickness=2, lineType=cv2.LINE_AA)

        result_img_init = cv2.warpPerspective(left_image_seed, transform_mat_inv, (rect_l, rect_w))
        for [new_x, new_y, _, _] in reprojected_seeding_pts:
            cv2.circle(result_img_init, (int(new_x), int(new_y)), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(result_img_init, (int(new_x), int(new_y)), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)

    with open(saved_seed_pts_path, "w") as json_file:
        json.dump(reprojected_seeding_pts, json_file)

    if args.debug_close_loop_vis:
        cv2.imwrite(saved_seed_vis_path, left_image_vis)
        cv2.imwrite(saved_seed_prj_path, result_img_init)
        return reprojected_seeding_pts, left_image_vis, result_img_init
    else:
        return reprojected_seeding_pts, None, None

#################################################################
def processing_test_image(args, left_image_test, detseg_prompts, supp_prompts, 
    left_image_vis, result_img_init, video_frame_index, kfr_height, kfr_width, reprojected_seeding_pts,
    final_res_list_pre=None, final_res_list_dict=None):

    rect_l, rect_w = cfg_dict['rect_l'], cfg_dict['rect_w']; colors_list = cfg_dict['colors_list']
    pts_polygon = cfg_dict['pts_polygon']; transform_mat_inv = cfg_dict['transform_mat_inv']
    [x1, y1, x2, y2] = cfg_dict['detection_roi_bbox']  # for fast and stable detection and segmentation
    cv2_font= cv2.FONT_HERSHEY_SIMPLEX

    ##########################
    '''VLMs: only use Florence2 + SAM2'''
    left_image_test_sub = left_image_test[y1:y2, x1:x2]
    try:  # at least one manipulated object is detected. we update the final_res_list and im_bgr
        new_detseg_prompts = detseg_prompts+","+supp_prompts if supp_prompts is not None else detseg_prompts
        if final_res_list_dict is None:
            final_res_list_raw, img_vis_seed_cv2 = conduct_object_detect_and_segment(left_image_test.copy(), prompts_str=new_detseg_prompts, is_raw_result=True)
        else:
            final_res_list_raw = [final_res_list_dict[detseg_prompt] for detseg_prompt in new_detseg_prompts.split(",")]
        for obj_info_list in final_res_list_raw:  # we need to check if the multi_masks are valid
            if type(obj_info_list[-1][0]) == np.int32:  # this invalid mask has only a 2D point
                raise ValueError("must be a list for a 2D obj_mask")  # not valid
        final_res_list_cur = final_res_list_raw  # valid, update the final_res_list
        final_res_list_temp = copy.deepcopy(final_res_list_cur)
    except:  # specifially for the task "inserting"
        img_vis_seed_cv2 = left_image_test_sub.copy()
        print("[Warning][Continue] No one manipulated object is detected. We still keep moving with using previous final_res_list!!!")
        final_res_list_temp = copy.deepcopy(final_res_list_pre)
    
    try:
        if args.debug_close_loop_vis:
            left_image_test_vis = left_image_test.copy()
            left_image_test_vis[y1:y2, x1:x2] = plot_processed_results_vis(left_image_test_sub.copy(), final_res_list_temp, new_detseg_prompts.split(","))

        supp_mask = None if supp_prompts is None else [temp_res[-1][0] for temp_res in final_res_list_temp if temp_res[0] == supp_prompts][0]
        reprojected_testing_pts = []
        for [obj_name, bbox, obj_binary_mask, multi_masks] in final_res_list_temp:
            if supp_mask is not None and obj_name == supp_prompts: continue
            return_list = compute_pts_abstract_xy_relative_rot(multi_masks[0], args.task_name, obj_name, supp_mask=supp_mask, binary_mask=obj_binary_mask)
            [pt_o_obj, pt_p_obj, rel_theta_deg, pt_o_supp, pt_p_supp] = return_list
            if type(pt_o_supp) != tuple: reprojected_testing_pts.append([pt_p_obj[0], pt_p_obj[1], rel_theta_deg, bbox])
            else: reprojected_testing_pts.append([pt_p_obj[0], pt_p_obj[1], rel_theta_deg, bbox.tolist() + pt_o_supp[1] + pt_p_supp[1]])  # note this trick <<<---
            if args.debug_close_loop_vis:
                cv2.circle(left_image_test_vis, (int(pt_o_obj[0]), int(pt_o_obj[1])), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(left_image_test_vis, (int(pt_o_obj[0]), int(pt_o_obj[1])), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
                if supp_mask is not None:
                    cv2.circle(left_image_test_vis, (int(pt_o_supp[0]), int(pt_o_supp[1])), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(left_image_test_vis, (int(pt_o_supp[0]), int(pt_o_supp[1])), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.arrowedLine(left_image_test_vis, (int(pt_o_obj[0]), int(pt_o_obj[1])), (int(pt_o_supp[0]), int(pt_o_supp[1])),
                    (0,255,255), thickness=3, tipLength=0.25, line_type=cv2.LINE_AA)
                elif pt_o_supp is not None and pt_p_supp is not None:  # for task inserting of the marker pen
                    if type(pt_o_supp) == list:  # for tasks relating pen/spoon/fork
                        endpt_L, endpt_R = pt_o_supp, pt_p_supp
                        cv2.circle(left_image_test_vis, (int(endpt_L[0]), int(endpt_L[1])), radius=4, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                        cv2.circle(left_image_test_vis, (int(endpt_L[0]), int(endpt_L[1])), radius=2, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
                        cv2.circle(left_image_test_vis, (int(endpt_R[0]), int(endpt_R[1])), radius=4, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                        cv2.circle(left_image_test_vis, (int(endpt_R[0]), int(endpt_R[1])), radius=2, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)
                        cv2.arrowedLine(left_image_test_vis, (int(endpt_L[0]), int(endpt_L[1])), (int(endpt_R[0]), int(endpt_R[1])),
                            (0,255,255), thickness=3, tipLength=0.25, line_type=cv2.LINE_AA)
                    if type(pt_o_supp) == tuple:  # for tasks relating folding cloth / coiling rope
                        grasp_pt_L, grasp_pt_R = pt_o_supp[0], pt_p_supp[0]  # in original camera image
                        cv2.circle(left_image_test_vis, (int(grasp_pt_L[0]), int(grasp_pt_L[1])), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                        cv2.circle(left_image_test_vis, (int(grasp_pt_R[0]), int(grasp_pt_R[1])), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                        cv2.arrowedLine(left_image_test_vis, (int(grasp_pt_L[0]), int(grasp_pt_L[1])), (int(grasp_pt_R[0]), int(grasp_pt_R[1])),
                            (0,255,255), thickness=3, tipLength=0.25, line_type=cv2.LINE_AA)
                        cv2.arrowedLine(left_image_test_vis, (int(grasp_pt_R[0]), int(grasp_pt_R[1])), (int(grasp_pt_L[0]), int(grasp_pt_L[1])),
                            (0,255,255), thickness=3, tipLength=0.25, line_type=cv2.LINE_AA)
                # else: print("[Error] We dot not support other outputs now!!!"); os._exit(0)
    except:
        print("[Warning][Continue] The detected object mask has errors!!! We need to retry the capture-detect-segment pipeline.")
        if args.debug_close_loop_vis: video_frame_index -= 1
        time.sleep(0.5)
        return None, video_frame_index, final_res_list_temp, None

    ##########################
    if args.debug_close_loop_vis:
        cv2.polylines(left_image_test_vis, [pts_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        result_img_test = cv2.warpPerspective(left_image_test, transform_mat_inv, (rect_l, rect_w))
        for [new_x, new_y, _, _] in reprojected_testing_pts:
            cv2.circle(result_img_test, (int(new_x), int(new_y)), radius=6, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(result_img_test, (int(new_x), int(new_y)), radius=3, color=(0,0,0), thickness=-1, lineType=cv2.LINE_AA)

        left_imgs = np.vstack((left_image_vis, left_image_test_vis))   # shape (1080, 960, 3)
        cv2.putText(left_imgs, f'Frame Index {video_frame_index}', (5, kfr_height+35), fontFace=cv2_font, fontScale=0.9, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(left_imgs, f'seeding (static)', (int(0.75*kfr_width), 35), fontFace=cv2_font, fontScale=0.9, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(left_imgs, f'testing (dynamic)', (int(0.72*kfr_width), kfr_height+35), fontFace=cv2_font, fontScale=0.9, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
        result_imgs = np.vstack((result_img_init, result_img_test))  # shape (rect_w*2, rect_l, 3) --> (1300, 1000, 3)
        
        left_cls_list = detseg_prompts.split(",")
        for idx, (pt_seeding, pt_testing) in enumerate(zip(reprojected_seeding_pts, reprojected_testing_pts)):
            delta_x, delta_y = pt_testing[0] - pt_seeding[0], pt_testing[1] - pt_seeding[1]
            print(video_frame_index, idx, "\n pt_seeding:", pt_seeding, "\n pt_testing:", pt_testing)
            print(f"Relative translation of two points (mm): (delta_x: {delta_x}, delta_y: {delta_y})")
            px1, py1, px2, py2 = int(pt_seeding[0]), int(pt_seeding[1]), int(pt_testing[0]), int(pt_testing[1])
            cv2.line(result_imgs, (px1, py1), (px2, py2+rect_w), color=colors_list[idx], thickness=2, lineType=cv2.LINE_AA)

            show_str, plot_xy = f'{left_cls_list[idx]}', (int(0.72*rect_l), int(0.2*rect_w+(4*idx-2)*30))
            cv2.putText(result_imgs, show_str, plot_xy, fontFace=cv2_font, fontScale=0.8, color=colors_list[idx], thickness=2, lineType=cv2.LINE_AA)
            show_str, plot_xy = f'Dx: {np.round(delta_x, 2)} mm', (int(0.72*rect_l), int(0.2*rect_w+(4*idx-1)*30))
            cv2.putText(result_imgs, show_str, plot_xy, fontFace=cv2_font, fontScale=0.8, color=colors_list[idx], thickness=2, lineType=cv2.LINE_AA)
            show_str, plot_xy = f'Dy: {np.round(delta_y, 2)} mm', (int(0.72*rect_l), int(0.2*rect_w+(4*idx)*30))
            cv2.putText(result_imgs, show_str, plot_xy, fontFace=cv2_font, fontScale=0.8, color=colors_list[idx], thickness=2, lineType=cv2.LINE_AA)

            rot_deg_init, rot_deg_test = pt_seeding[2], pt_testing[2]
            show_str, plot_xy = f'Rot: {np.round(rot_deg_test, 0)} deg', (int(0.72*rect_l), int(0.2*rect_w+(4*idx+1)*30))
            cv2.putText(result_imgs, show_str, plot_xy, fontFace=cv2_font, fontScale=0.85, color=colors_list[idx], thickness=3, lineType=cv2.LINE_AA)

            if abs(delta_x) > 10 or abs(delta_y) > 10:  # we think this object is moved
                plot_x1y1 = ( int(0.71*rect_l), int(0.2*rect_w+(4*idx-3)*30)+6 )
                plot_x2y2 = ( int(0.99*rect_l), int(0.2*rect_w+(4*idx+1)*30)+6 )
                cv2.rectangle(result_imgs, plot_x1y1, plot_x2y2, color=colors_list[idx], thickness=2)

        result_imgs = cv2.resize(result_imgs, (int(1000*1080/1300.0), 1080))  # shape (1080, 831, 3)
        cv2.putText(result_imgs, f'Top-view Reprojection', (5, 35), fontFace=cv2_font, fontScale=0.9, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
        final_imgs = np.hstack((left_imgs, result_imgs))  # shape (1080, 1790, 3)
        final_imgs = cv2.resize(final_imgs, (1253, 756))  # shape (1080, 1790, 3) --> (756, 1253, 0)   
        print("[*****Finished*****]", video_frame_index)
    ##########################

    if args.debug_close_loop_vis:
        return reprojected_testing_pts, video_frame_index, final_res_list_temp, final_imgs
    else:
        return reprojected_testing_pts, video_frame_index, final_res_list_temp, None

#################################################################
def processing_test_image_full_stage_pouring(left_image_test):
    final_res_list_raw, _ = conduct_object_detect_and_segment(left_image_test, prompts_str="bottle,cup,bottle cap", is_raw_result=True)

    final_res_list_dict = {"bottle": [], "cup": [], "bottle cap": []}
    for final_res_list in final_res_list_raw:
        [obj_name, bbox, obj_binary_mask, multi_masks] = final_res_list
        if len(final_res_list_dict[obj_name]) == 0:
            final_res_list_dict[obj_name] = final_res_list

    return final_res_list_dict


def llm_anchored_task_switching(args, llm_instance, left_image_check, object_type, final_res_list_dict=None):
    
    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation

    user_message = "please help to judge the object state."

    # 调用 LLM 生成任务计划
    if object_type == "bottle":
        SYS_MESSAGE = SYS_MESSAGE_BOTTLE_STATES; detseg_prompts = "bottle"
        category_list = {"A": "upright", "B": "lying down", "C": "upside down"}
    elif object_type == "cup":
        SYS_MESSAGE = SYS_MESSAGE_MUGCUP_STATES; detseg_prompts = "cup"
        category_list = {"A": "upright", "B": "lying down", "C": "upside down"}
    elif object_type == "bowl": 
        SYS_MESSAGE = SYS_MESSAGE_BOWL_STATES; detseg_prompts = "bowl"
        category_list = {"A": "upright", "B": "inverted"}

    left_img_test = left_image_check[y1:y2, x1:x2].copy()
    if final_res_list_dict is None:
        final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(
            left_img_test, prompts_str=detseg_prompts, is_raw_result=True, given_roi_bbox=[0, 0, x2-x1, y2-y1])
    else:
        final_res_list_raw = [final_res_list_dict[detseg_prompts]]

    enlarge_ratio = 2.0
    for [obj_name, bbox, obj_binary_mask, multi_masks] in final_res_list_raw:
        assert obj_name == detseg_prompts, "wrong detection result in the object type!!!"
        [bx1, by1, bx2, by2] = bbox
        ebx1 = max(0, int(bx1 - (enlarge_ratio - 1.0) * (bx2 - bx1) * 0.5))
        ebx2 = min(x2-x1-1, int(bx2 + (enlarge_ratio - 1.0) * (bx2 - bx1) * 0.5))
        eby1 = max(0, int(by1 - (enlarge_ratio - 1.0) * (by2 - by1) * 0.5))
        eby2 = min(y2-y1-1, int(by2 + (enlarge_ratio - 1.0) * (by2 - by1) * 0.5))
        break  # only care about one object every time

    image_name = f"test_img_LLM-{args.llm_type}_OBJ-{object_type}-{str(args.test_id).zfill(2)}.jpg" 
    image_path = os.path.join("/home/dex/zhouhuayi/rokaeDemo/results/tempLLM/", image_name)
    cv2.imwrite(image_path, left_image_check[y1:y2, x1:x2][eby1:eby2, ebx1:ebx2] )  # for LLM gpt-4o / doubao using

    formatted_messages = generate_prompt(SYS_MESSAGE, user_message, image_path=image_path)
    response = llm_instance.invoke(formatted_messages)
    print(f"LLM Response: {response.content}")

    response_dict = json.loads(response.content)
    state_res = response_dict["state"]  # state_res --> category_list[state_res] 

    if state_res == "A" and object_type == "bottle":
        formatted_messages = generate_prompt(SYS_MESSAGE_BOTTLE_CAP_CHECK, user_message, image_path=image_path)
        response = llm_instance.invoke(formatted_messages)
        print(f"LLM Response: {response.content}")

        response_dict = json.loads(response.content)
        state_res_supp = response_dict["bottle_cap"]  # bottle_cap --> YES or NO
        return state_res, state_res_supp

    else:
        return state_res, None


def llm_anchored_task_switching_for_pouring(args, llm_instance, left_image_check, final_res_list_dict=None, llm_invoke_count=1):

    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation

    detseg_prompts = "bottle,cup"

    user_message = "Please help to judge object states."

    SYS_MESSAGE = "As you can see, here is a RGB image with a bottle (leftside) and a mug cup (rightside) on the table. Please help me to judge their states. \
        (1) The bottle's state is categoried into three types: (A) standing upright on the table; (B) lying down on the table; (C) standing upside down on the table. \
        (2) The mug cup's state is categoried into three types: (A) standing upright on the table; (B) lying down on the table; (C) standing upside down on the table. \
        (3) Please also check whether there is a cap on this bottle or not. You can answer: (YES) having a cap on the bottle; (NO) not having a cap on the bottle. \
        The final output must be formatted in valid JSON, such as: {'bottle': 'A', 'cup': 'A', 'has_cap': 'NO'} or {'bottle': 'B', 'cup': 'A', 'has_cap': 'YES'} or \
        {'bottle': 'C', 'cup': 'B', 'has_cap': 'YES'} or {'bottle': 'A', 'cup': 'C', 'has_cap': 'NO'}. "

    left_img_test = left_image_check[y1:y2, x1:x2].copy()
    if final_res_list_dict is None:
        final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(
            left_img_test, prompts_str=detseg_prompts, is_raw_result=True, given_roi_bbox=[0, 0, x2-x1, y2-y1])
    else:
        final_res_list_raw = [final_res_list_dict[detseg_prompt] for detseg_prompt in detseg_prompts.split(",")]

    enlarge_ratio = 2.0
    obj_sub_img_list = []
    for detseg_prompt in detseg_prompts.split(","):
        for [obj_name, bbox, obj_binary_mask, multi_masks] in final_res_list_raw:
            if obj_name != detseg_prompt: continue
            [bx1, by1, bx2, by2] = bbox
            ebx1 = max(0, int(bx1 - (enlarge_ratio - 1.0) * (bx2 - bx1) * 0.5))
            ebx2 = min(x2-x1-1, int(bx2 + (enlarge_ratio - 1.0) * (bx2 - bx1) * 0.5))
            eby1 = max(0, int(by1 - (enlarge_ratio - 1.0) * (by2 - by1) * 0.5))
            eby2 = min(y2-y1-1, int(by2 + (enlarge_ratio - 1.0) * (by2 - by1) * 0.5))
            break  # only care about one object every time
        obj_sub_img = left_image_check[y1:y2, x1:x2][eby1:eby2, ebx1:ebx2]
        obj_sub_img_list.append([obj_sub_img, ebx2-ebx1, eby2-eby1])  # img_cv2, img_w, img_h
    
    new_img_h = max(obj_sub_img_list[0][-1], obj_sub_img_list[1][-1])
    new_img_w = obj_sub_img_list[0][1] + obj_sub_img_list[1][1] + 2
    new_img_canvas = np.full((new_img_h, new_img_w, 3), 0, dtype=np.uint8)  # 0 black. 255 white
    new_img_canvas[:obj_sub_img_list[0][-1], :obj_sub_img_list[0][1]] = obj_sub_img_list[0][0]  # bottle in the left side
    new_img_canvas[:obj_sub_img_list[1][-1], new_img_w-obj_sub_img_list[1][1]:] = obj_sub_img_list[1][0]  # cup in the right side

    image_name = f"LLM-{args.llm_type}_OBJ-{detseg_prompts}_TEST-{str(args.test_id).zfill(2)}_COUNT-{str(llm_invoke_count).zfill(2)}.jpg" 
    image_path = os.path.join("/home/dex/zhouhuayi/rokaeDemo/results/tempLLM/", image_name)
    cv2.imwrite(image_path, new_img_canvas)  # for LLM gpt-4o / doubao using

    count_contradiction = 0
    while True:
        if args.llm_type == 0 or args.llm_type == 1:
            formatted_messages = generate_prompt(SYS_MESSAGE_POURING, user_message, image_path=image_path)
            response = llm_instance.invoke(formatted_messages)
            print(f"LLM Response: {response.content}")
            response_dict = json.loads(response.content)
        if args.llm_type == 2:
            response = generate_prompt_and_result(llm_instance, SYS_MESSAGE_POURING, image_path)
            print(f"LLM Response: {response.text}")
            response_dict = json.loads(response.text)

        state_res1 = response_dict["bottle"]  # A or B or C
        state_res2 = response_dict["cup"]  # A or B or C
        state_res3 = response_dict["has_cap"]  # YES or NO

        if state_res1 == "B" and state_res3 == "NO":
            count_contradiction += 1
            SYS_MESSAGE_POURING += "[Hits] Please be careful about a possible contradiction that a lying down bottle without cap on it \
                will cause the water in the bottle to drip out. Thus, if you can make sure that there is no cap on the bottle, \
                the bottle's state might be standing upright. "
        else: count_contradiction -= 1
        if count_contradiction <= 0: break

    os.remove(image_path)
    image_name_new = image_name[:-4] + f"_{state_res1}-{state_res2}-{state_res3}.jpg"
    image_path_new = os.path.join("/home/dex/zhouhuayi/rokaeDemo/results/tempLLM/", image_name_new)
    cv2.imwrite(image_path_new, new_img_canvas)  # for LLM gpt-4o / doubao using

    return [state_res1, state_res2, state_res3]

#################################################################

