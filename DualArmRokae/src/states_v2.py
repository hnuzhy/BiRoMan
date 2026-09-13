

import os
import sys
import cv2
import json
import shutil
import argparse
import imageio
import open3d as o3d
import numpy as np
from copy import copy

import kingfisher
sys.path.insert(0, os.getcwd())
from src.config import cfg_dict_init as cfg_dict
from src.vlms import conduct_object_detect_and_segment
from src.pcd2sat import load_FS_model
from src.pcd2sat import check_state_via_obj_pcd
from src.pcd2sat import process_a_paired_binocular_images

#################################################################
def main_rigid_obj_states(args):  
    #####====================================================================
    save_res_path = "/home/dex/zhouhuayi/rokaeDemo/debug/test_pcdVLM_rigid_obj_states/"

    c = kingfisher.connect(cfg_dict["kfr_ip"])   # connect the camera
    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']  # 540, 960
    kingfisher.SetAUTO_EXPOSURE()

    time_interval, frame_count, enlarge_ratio = 1, 0, 1.8

    image_name = f"test_img_pcdVLM_OBJ-{args.obj_name}_ID-{str(args.test_id).zfill(2)}.jpg" 
    image_path = os.path.join(save_res_path, image_name)

    # image_path_dir = image_path.replace(".jpg", "")
    # if os.path.exists(image_path_dir): shutil.rmtree(image_path_dir)
    # os.mkdir(image_path_dir)

    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation

    video_output_path = image_path.replace(".jpg", ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v'); FPS = 5
    #vout = cv2.VideoWriter(video_output_path, fourcc, FPS, (x2-x1, (y2-y1)*2))  # Create VideoWriter object
    vout = cv2.VideoWriter(video_output_path, fourcc, FPS, (x2-x1, (y2-y1)))  # Create VideoWriter object
    #####====================================================================

    detseg_prompts = args.obj_name
    fs_model = load_FS_model()

    ############################################################# 

    while True:
        frame_count += 1
        imgL_ori, imgR_ori = kingfisher.captureQuarterSize()
        #left_img_test = imgL_ori[y1:y2, x1:x2].copy()

        #####====================================================================
        img_bgr = cv2.convertScaleAbs(imgL_ori.copy(), alpha=1.2, beta=10)  # adjust the brightness and contrast
        try:  # do detection and segmentation
            final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(img_bgr, prompts_str=detseg_prompts, is_raw_result=True)
            obj_binary_mask = final_res_list_raw[0][2]
        except:  # the object is not detected, try it again
            frame_count -= 1; continue
        top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
        obj_binary_mask = cv2.copyMakeBorder(obj_binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

        pcd_scene, pcd_obj, vis_res = process_a_paired_binocular_images(
            fs_model, imgL_ori.copy()[:, :, ::-1], imgR_ori.copy()[:, :, ::-1], obj_binary_mask)

        cur_state, pcd_obj_adjusted, cur_state_str = check_state_via_obj_pcd(pcd_obj, detseg_prompts)
        #####====================================================================

        img_vis_cv2 = img_vis_cv2.astype(np.uint8)
        show_str = str(frame_count).zfill(2) + " " + cur_state + " - " + cur_state_str
        (tw, th), _ = cv2.getTextSize(show_str, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.72, thickness=1)
        cv2.rectangle(img_vis_cv2, (10, 30-th-5), (10+tw, 30+5), color=(255,255,255), thickness=-1)
        cv2.putText(img_vis_cv2, show_str, (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.72, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
        pts_polygon = cfg_dict['pts_polygon']
        pts_polygon = np.array([[pxy[0][0]-x1, pxy[0][1]-y1] for pxy in pts_polygon], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_vis_cv2, [pts_polygon], isClosed=True, color=(255, 0, 0), thickness=2)

        #img_canvas = np.vstack((left_img_test, img_vis_cv2))
        img_canvas = img_vis_cv2
        cv2.namedWindow("MyDebugWindow", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("MyDebugWindow", x2-x1, (y2-y1) * 2)
        cv2.resizeWindow("MyDebugWindow", x2-x1, (y2-y1))
        cv2.imshow("MyDebugWindow", img_canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # or cv2.waitKey(0); wait until we close the plotted window (press Esc to continue)

        for _ in range(FPS): vout.write(img_canvas)

        # time.sleep(time_interval)
    
    print("Finished!!!")
    
    
def main_rect_container_obj_level_pcds(args):  

    #####====================================================================
    save_res_path = "/home/dex/zhouhuayi/rokaeDemo/debug/test_pcdVLM_rect_container/"

    c = kingfisher.connect(cfg_dict["kfr_ip"])   # connect the camera
    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']  # 540, 960
    kingfisher.SetAUTO_EXPOSURE()

    time_interval, frame_count, enlarge_ratio = 1, 0, 1.8

    image_name = f"test_OBJ-{args.obj_name}_ID-{str(args.test_id).zfill(2)}.jpg" 
    image_path = os.path.join(save_res_path, image_name)

    image_path_dir = image_path.replace(".jpg", "")
    if os.path.exists(image_path_dir): shutil.rmtree(image_path_dir)
    os.mkdir(image_path_dir)

    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation

    video_output_path = image_path.replace(".jpg", ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v'); FPS = 5
    #vout = cv2.VideoWriter(video_output_path, fourcc, FPS, (x2-x1, (y2-y1)*2))  # Create VideoWriter object
    vout = cv2.VideoWriter(video_output_path, fourcc, FPS, (x2-x1, (y2-y1)))  # Create VideoWriter object
    #####====================================================================

    detseg_prompts = args.obj_name
    fs_model = load_FS_model()

    ############################################################# 

    while True:
        frame_count += 1
        imgL_ori, imgR_ori = kingfisher.captureQuarterSize()
        #left_img_test = imgL_ori[y1:y2, x1:x2].copy()

        #####====================================================================
        img_bgr = cv2.convertScaleAbs(imgL_ori.copy(), alpha=1.2, beta=10)  # adjust the brightness and contrast
        try:  # do detection and segmentation
            final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(img_bgr, prompts_str=detseg_prompts, is_raw_result=True)
            obj_binary_mask = final_res_list_raw[0][2]
        except:  # the object is not detected, try it again
            frame_count -= 1; continue
        top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
        obj_binary_mask = cv2.copyMakeBorder(obj_binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

        pcd_scene, pcd_obj, vis_res = process_a_paired_binocular_images(
            fs_model, imgL_ori.copy()[:, :, ::-1], imgR_ori.copy()[:, :, ::-1], obj_binary_mask)

        cur_state, pcd_obj_adjusted, cur_state_str = check_state_via_obj_pcd(copy(pcd_obj), detseg_prompts)
        #####====================================================================
        
        #####====================================================================
        img_vis_cv2 = img_vis_cv2.astype(np.uint8)
        show_str = str(frame_count).zfill(2) + " " + cur_state + " - " + cur_state_str
        (tw, th), _ = cv2.getTextSize(show_str, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.72, thickness=1)
        cv2.rectangle(img_vis_cv2, (10, 30-th-5), (10+tw, 30+5), color=(255,255,255), thickness=-1)
        cv2.putText(img_vis_cv2, show_str, (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.72, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
        pts_polygon = cfg_dict['pts_polygon']
        pts_polygon = np.array([[pxy[0][0]-x1, pxy[0][1]-y1] for pxy in pts_polygon], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_vis_cv2, [pts_polygon], isClosed=True, color=(255, 0, 0), thickness=2)

        #img_canvas = np.vstack((left_img_test, img_vis_cv2))
        img_canvas = img_vis_cv2
        cv2.namedWindow("MyDebugWindow", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("MyDebugWindow", x2-x1, (y2-y1) * 2)
        cv2.resizeWindow("MyDebugWindow", x2-x1, (y2-y1))
        cv2.imshow("MyDebugWindow", img_canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # or cv2.waitKey(0); wait until we close the plotted window (press Esc to continue)

        for _ in range(FPS): vout.write(img_canvas)
        #####====================================================================
        
        if frame_count % 2 == 1:
            cv2.imwrite(os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_rgb_L.jpg"), imgL_ori)  # for further using
            cv2.imwrite(os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_rgb_R.jpg"), imgR_ori)  # for further using
            cv2.imwrite(os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_mask.jpg"), obj_binary_mask)
            with open(os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_mask.json"), "w") as json_file:
                bbox = final_res_list_raw[0][1]
                mask_arr = final_res_list_raw[0][3][0]
                json.dump({"bbox": bbox.tolist(), "mask": mask_arr.tolist()}, json_file)
            o3d.io.write_point_cloud(os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_pcd_scene.ply"), pcd_scene)
            o3d.io.write_point_cloud(os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_pcd_obj.ply"), pcd_obj)
            imageio.imwrite(os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_depth_vis.png"), vis_res)  # depth RGB image
            o3d.io.write_point_cloud(os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_pcd_obj(ad).ply"), pcd_obj_adjusted)
    
    print("Finished!!!")


def main_flattened_deformable_cloth(args):
    
    #####====================================================================
    save_res_path = "/home/dex/zhouhuayi/rokaeDemo/debug/test_pcdVLM_deformable_cloth/"

    c = kingfisher.connect(cfg_dict["kfr_ip"])   # connect the camera
    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']  # 540, 960
    kingfisher.SetAUTO_EXPOSURE()

    time_interval, frame_count, enlarge_ratio = 1, 0, 1.8

    image_name = f"test_OBJ-{args.obj_name}_ID-{str(args.test_id).zfill(2)}.jpg" 
    image_path = os.path.join(save_res_path, image_name)

    image_path_dir = image_path.replace(".jpg", "")
    if os.path.exists(image_path_dir): shutil.rmtree(image_path_dir)
    os.mkdir(image_path_dir)

    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation

    video_output_path = image_path.replace(".jpg", ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v'); FPS = 5
    #vout = cv2.VideoWriter(video_output_path, fourcc, FPS, (x2-x1, (y2-y1)*2))  # Create VideoWriter object
    vout = cv2.VideoWriter(video_output_path, fourcc, FPS, (x2-x1, (y2-y1)))  # Create VideoWriter object
    #####====================================================================

    detseg_prompts = args.obj_name
    fs_model = load_FS_model()

    ############################################################# 

    while True:
        frame_count += 1
        imgL_ori, imgR_ori = kingfisher.captureQuarterSize()
        #left_img_test = imgL_ori[y1:y2, x1:x2].copy()

        #####====================================================================
        img_bgr = cv2.convertScaleAbs(imgL_ori.copy(), alpha=1.2, beta=10)  # adjust the brightness and contrast
        try:  # do detection and segmentation
            final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(img_bgr, prompts_str=detseg_prompts, is_raw_result=True)
            obj_binary_mask = final_res_list_raw[0][2]
        except:  # the object is not detected, try it again
            frame_count -= 1; continue
        top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
        obj_binary_mask = cv2.copyMakeBorder(obj_binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

        #####====================================================================
        img_vis_cv2 = img_vis_cv2.astype(np.uint8)
        show_str = "Frame: " + str(frame_count).zfill(2)
        (tw, th), _ = cv2.getTextSize(show_str, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.72, thickness=1)
        cv2.rectangle(img_vis_cv2, (10, 30-th-5), (10+tw, 30+5), color=(255,255,255), thickness=-1)
        cv2.putText(img_vis_cv2, show_str, (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.72, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
        pts_polygon = cfg_dict['pts_polygon']
        pts_polygon = np.array([[pxy[0][0]-x1, pxy[0][1]-y1] for pxy in pts_polygon], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_vis_cv2, [pts_polygon], isClosed=True, color=(255, 0, 0), thickness=2)

        #img_canvas = np.vstack((left_img_test, img_vis_cv2))
        img_canvas = img_vis_cv2
        cv2.namedWindow("MyDebugWindow", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("MyDebugWindow", x2-x1, (y2-y1) * 2)
        cv2.resizeWindow("MyDebugWindow", x2-x1, (y2-y1))
        cv2.imshow("MyDebugWindow", img_canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # or cv2.waitKey(0); wait until we close the plotted window (press Esc to continue)

        for _ in range(FPS): vout.write(img_canvas)
        #####====================================================================
        
        if frame_count % 2 == 1:
            cv2.imwrite(os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_rgb_L.jpg"), imgL_ori)  # for further using
            cv2.imwrite(os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_rgb_R.jpg"), imgR_ori)  # for further using
            cv2.imwrite(os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_mask.jpg"), obj_binary_mask)
            with open(os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_mask.json"), "w") as json_file:
                bbox = final_res_list_raw[0][1]
                mask_arr = final_res_list_raw[0][3][0]
                json.dump({"bbox": bbox.tolist(), "mask": mask_arr.tolist()}, json_file)
                
    print("Finished!!!")
    

#################################################################
# python src/states_v2.py --obj_name bottle --test_id 1 
# python src/states_v2.py --obj_name mug --test_id 1 
# python src/states_v2.py --obj_name bowl --test_id 1 
# python src/states_v2.py --obj_name basket --test_id 1 

# python src/states_v2.py --obj_name basket --test_id 1 
# python src/states_v2.py --obj_name tray --test_id 1 
# python src/states_v2.py --obj_name box --test_id 1 

# python src/states_v2.py --obj_name T-shirt --test_id 1 
# python src/states_v2.py --obj_name towel --test_id 1 

#################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm_type", type=int, default=1, help="0: doubao; 1: gpt-4o")
    parser.add_argument("--obj_name", type=str, required=True, help="the object name for judging its state.")
    parser.add_argument("--test_id", type=int, default=1, help="the testing id for one specific object.")
    parser.add_argument("--online", action='store_true', help="default is False")
    args = parser.parse_args()
    
    # main_rigid_obj_states(args)
    # main_rect_container_obj_level_pcds(args)
    main_flattened_deformable_cloth(args)

