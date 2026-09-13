

import os
import sys
import cv2
import time
import shutil
import argparse


#################################################################
### This script can only be run in python3 for aubo/rokae robot's requirement
### conda acitvate py39
'''
########################################################################################################################
############################################
########## v1 for BiNoMaP (dual-arm rokae)
############################################

python src/video_gen.py --is_syn --cap_type L --task_name pouring_bottle_mugcup --bottle_id 1 --mugcup_id 1 --test_id 1
python src/video_gen.py --is_syn --cap_type L --task_name unscrew_bottle --bottle_id 1 --test_id 1
python src/video_gen.py --is_syn --cap_type L --task_name reorient_bottle --bottle_id 1 --test_id 1
python src/video_gen.py --is_syn --cap_type L --task_name grasping_mugcup --mugcup_id 1 --test_id 1
python src/video_gen.py --is_syn --cap_type L --task_name flatting_bottle --bottle_id 1 --test_id 1
python src/video_gen.py --is_syn --cap_type L --task_name flipping_mugcup --mugcup_id 1 --test_id 1

python src/video_gen.py --is_syn --cap_type L --task_name unscrew-pouring_bottle_mugcup --bottle_id 1 --mugcup_id 1 --test_id 1
python src/video_gen.py --is_syn --cap_type L --task_name flatting-reorient_bottle --bottle_id 1 --test_id 1

python src/video_gen.py --is_syn --cap_type L --task_name inserting_marker_ordcup --marker_id 1 --ordcup_id 1 --test_id 1

python src/video_gen.py --is_syn --cap_type L --task_name pivoting_cirbowl --cirbowl_id 1 --test_id 1
python src/video_gen.py --is_syn --cap_type L --task_name wrapping_basket --basket_id 1 --test_id 1
########################################################################################################################
'''
#################################################################

#################################################################
def enhance_image_lighting(img):

    # https://www.geeksforgeeks.org/machine-learning/image-enhancement-techniques-using-opencv-python/
    # Adjust the brightness and contrast 
    # g(i,j)=α⋅f(i,j)+β
    # control Contrast by 1.5 --> 1.2
    alpha = 1.2 
    # control brightness by 50 --> 20
    beta = 20  
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    return img
#################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_syn', action='store_true', help="is dual-arm running synchronously. default is False")
    parser.add_argument('--cap_type', default="LR", help="the capture camera left/right. It can be L, R or LR")
    parser.add_argument('--task_name', default="", help="string of bimanual task name.")
    parser.add_argument('--save_imgs_dir', default="D:/PostDocdor/MyWorkSpace/rokaeDemo/rollouts", 
                        help="path to save all captured images of dual-arm real robot execution")
    
    parser.add_argument('--bottle_id', type=int, default=1, help="bottle_id is selected from 1 ~ 4")
    parser.add_argument('--mugcup_id', type=int, default=1, help="mugcup_id is selected from 1 ~ 4")
    parser.add_argument('--marker_id', type=int, default=1, help="marker_id is selected from 1 ~ 4")
    parser.add_argument('--ordcup_id', type=int, default=1, help="ordcup_id is selected from 1 ~ 4")
    parser.add_argument('--cirbowl_id', type=int, default=1, help="cirbowl_id is selected from 1 ~ 8")  # for task pivoting_cirbowl
    parser.add_argument('--basket_id', type=int, default=1, help="basket_id is selected from 1 ~ 6")  # for task warpping_basket

    parser.add_argument('--is_capture', action='store_true', help="default is not the capture mode")
    parser.add_argument('--test_id', type=int, default=1, help="we may record many times of one specific task")
    parser.add_argument('--speed_up', type=int, default=2, help="default is 2 for most tasks")
    parser.add_argument('--start_idx', type=int, default=0, help="the start index of captured frames. default is 0")
    parser.add_argument('--end_idx', type=int, default=10000, help="the end index of captured frames. default is 10000")
    
    args = parser.parse_args()
    ############################################################################
    assert args.task_name in [ "pouring_bottle_mugcup", "unscrew_bottle",
        "reorient_bottle", "grasping_mugcup", "flatting_bottle", "flipping_mugcup",
        "unscrew-pouring_bottle_mugcup", "flatting-reorient_bottle",
        "inserting_marker_ordcup",
        "pivoting_cirbowl", "wrapping_basket"], "Please give a valid task name !!!"

    if args.task_name in ["pouring_bottle_mugcup", "unscrew-pouring_bottle_mugcup"]:
        obj_name_id_str = f"bottle{str(args.bottle_id).zfill(2)}-mugcup{str(args.mugcup_id).zfill(2)}"
    if args.task_name in ["unscrew_bottle", "reorient_bottle", "flatting_bottle", "flatting-reorient_bottle"]:
        obj_name_id_str = f"bottle{str(args.bottle_id).zfill(2)}"
    if args.task_name in ["grasping_mugcup", "flipping_mugcup"]:
        obj_name_id_str = f"mugcup{str(args.mugcup_id).zfill(2)}"
        
    if args.task_name == "inserting_marker_ordcup":
        obj_name_id_str = f"marker{str(args.marker_id).zfill(2)}-ordcup{str(args.ordcup_id).zfill(2)}"
        
    if args.task_name == "pivoting_cirbowl":
        obj_name_id_str = f"cirbowl{str(args.cirbowl_id).zfill(2)}"
    if args.task_name == "wrapping_basket":
        obj_name_id_str = f"basket{str(args.basket_id).zfill(2)}"

    syn_str = "syn" if args.is_syn else "asyn"
    prefix_str = f"{args.task_name}_{syn_str}_{args.cap_type}"
    sub_folder = f"{prefix_str}_{obj_name_id_str}_tid{str(args.test_id).zfill(2)}"
    ############################################################################
    
    task_level_save_path = os.path.join(args.save_imgs_dir, args.task_name.split("_")[0])
    if not os.path.exists(task_level_save_path): os.mkdir(task_level_save_path)
    final_save_path = os.path.join(task_level_save_path, sub_folder)
    if args.is_capture:  # we begin to capture dual cameras in new folders
        if os.path.exists(final_save_path): shutil.rmtree(final_save_path)
        os.mkdir(final_save_path)
    else:  # find already captured image folders
        if "L" in args.cap_type: video_save_path_L = final_save_path + f"_L_x{args.speed_up}.mp4"
        if "R" in args.cap_type: video_save_path_R = final_save_path + f"_R_x{args.speed_up}.mp4"

        
    ############################################################################

    width, height, scale_ratio = 960, 540, 0.8  # original image size
    # cx1, cx2, cy1, cy2 = 100, 900, 40, 540  # cropping bbox, 960*540 --> 800*500
    # cx1, cx2, cy1, cy2 = 20, 820, 40, 540  # cropping bbox, 960*540 --> 800*500 (for aubo)
    # cx1, cx2, cy1, cy2 = 150, 950, 40, 540  # cropping bbox, 960*540 --> 800*500 (for rokae + BiNoMaP)
    cx1, cx2, cy1, cy2 = 170, 910, 70, 520  # cropping bbox, 960*540 --> 720*450 (for rokae + VLBiMan)
    width, height = cx2 - cx1, cy2 - cy1  # cropped image size
    if scale_ratio != 1.0: width = int(width*scale_ratio); height = int(height*scale_ratio)
    if args.cap_type in ["L", "R"]: FPS = 13 * args.speed_up  # the value 13 is estimated manually
    if args.cap_type == "LR": FPS = 12 * args.speed_up  # the value 12 is estimated manually
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if "L" in args.cap_type:
        vout_L = cv2.VideoWriter(video_save_path_L, fourcc, FPS, (width, height))
        imgs_name_L = [ im_name for im_name in os.listdir(final_save_path) if "L_" in im_name ]
        imgs_name_L.sort()
    if "R" in args.cap_type:
        vout_R = cv2.VideoWriter(video_save_path_R, fourcc, FPS, (width, height))
        imgs_name_R = [ im_name for im_name in os.listdir(final_save_path) if "R_" in im_name ]
        imgs_name_R.sort()
        
    if args.cap_type == "LR":
        min_len = min(len(imgs_name_L), len(imgs_name_R))
        imgs_name_L = imgs_name_L[:min_len]; imgs_name_R = imgs_name_R[:min_len]
        
    if "L" in args.cap_type:
        for idx, img_name_L in enumerate(imgs_name_L):
            if idx%20==0: print(idx, "\t", img_name_L)
            if idx+1 < args.start_idx: continue  # we may give a start_idx after manually checking
            if idx+1 > args.end_idx: continue  # we may give a end_idx after manually checking
            left_image = cv2.imread(os.path.join(final_save_path, img_name_L))
            left_image = left_image[cy1:cy2, cx1:cx2]
            if scale_ratio != 1.0: left_image = cv2.resize(left_image, (0, 0), fx=scale_ratio, fy=scale_ratio)
            # vout_L.write(left_image)
            vout_L.write(enhance_image_lighting(left_image))
        vout_L.release()  # Release everything
        print("finished video of left camera")
    if "R" in args.cap_type:
        for idx, img_name_R in enumerate(imgs_name_R):
            if idx%20==0: print(idx, "\t", img_name_R)
            if idx+1 < args.start_idx: continue  # we may give a start_idx after manually checking
            if idx+1 > args.end_idx: continue  # we may give a end_idx after manually checking
            right_image = cv2.imread(os.path.join(final_save_path, img_name_R))
            right_image = right_image[cy1:cy2, cx1:cx2]
            if scale_ratio != 1.0: right_image = cv2.resize(right_image, (0, 0), fx=scale_ratio, fy=scale_ratio)
            # vout_R.write(right_image)
            vout_R.write(enhance_image_lighting(right_image))
        vout_R.release()  # Release everything
        print("finished video of right camera")
    
    print("finished all !!!")


    