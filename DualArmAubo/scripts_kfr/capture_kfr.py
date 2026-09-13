'''
sudo apt-get install libcharls2

vim ~/.bashrc
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
source ~/.bashrc
'''

import os
import cv2
import time
import shutil
import argparse
import numpy as np
import kingfisher


#################################################################
### This script can only be run in python3 for aubo robot's requirement
### conda acitvate embodychain
'''
##############################
########## v1 for sync test
##############################
python scripts_kfr/capture_kfr.py --cap_type L --task_name plugpen --marker_id 1 --is_syn --is_capture
python scripts_kfr/capture_kfr.py --cap_type L --task_name reorient --anyobj_id 1 --is_syn --is_capture
python scripts_kfr/capture_kfr.py --cap_type L --task_name unscrew --bottle_id 6 --is_syn --is_capture
python scripts_kfr/capture_kfr.py --cap_type L --task_name pouring --bottle_id 6 --mugcup_id 1 --is_syn --is_capture
python scripts_kfr/capture_kfr.py --cap_type L --task_name inserting --ordcup_id 1 --marker_id 1 --is_syn --is_capture
python scripts_kfr/capture_kfr.py --cap_type L --task_name pressing --ordcup_id 1 --nozzle_id 1 --is_syn --is_capture

python scripts_kfr/capture_kfr.py --cap_type L --task_name reorient_unscrew --bottle_id 6 --is_syn --is_capture
python scripts_kfr/capture_kfr.py --cap_type L --task_name unscrew_pouring --bottle_id 6 --mugcup_id 1 --is_syn --is_capture

python scripts_kfr/capture_kfr.py --cap_type L --task_name tool_spoon --is_syn --is_capture
python scripts_kfr/capture_kfr.py --cap_type L --task_name tool_funnel --is_syn --is_capture

##############################
########## v2 for BiDemoSyn
##############################
python scripts_kfr/capture_kfr.py --version v2 --cap_type L --task_name plugpen --marker_id 2 --test_id 1 --is_capture
python scripts_kfr/capture_kfr.py --version v2 --cap_type L --task_name reorient --anyobj_id 1 --test_id 1 --is_capture
python scripts_kfr/capture_kfr.py --version v2 --cap_type L --task_name unscrew --bottle_id 6 --test_id 1 --is_capture --speed_up 3
python scripts_kfr/capture_kfr.py --version v2 --cap_type L --task_name pouring --bottle_id 6 --mugcup_id 1 --test_id 1 --is_capture --speed_up 3
python scripts_kfr/capture_kfr.py --version v2 --cap_type L --task_name inserting --ordcup_id 1 --marker_id 2 --test_id 1 --is_capture --speed_up 3
python scripts_kfr/capture_kfr.py --version v2 --cap_type L --task_name pressing --ordcup_id 1 --nozzle_id 1 --test_id 1 --is_capture

##############################
########## v3 for VLBiMan
##############################
##### for external dynamic interference (--version v3)
python scripts_kfr/capture_kfr.py --version v3 --cap_type L --task_name plugpen --marker_id 1 --is_capture
python scripts_kfr/capture_kfr.py --version v3 --cap_type L --task_name reorient --anyobj_id 1 --is_capture
python scripts_kfr/capture_kfr.py --version v3 --cap_type L --task_name unscrew --bottle_id 6 --is_capture
python scripts_kfr/capture_kfr.py --version v3 --cap_type L --task_name pouring --bottle_id 6 --mugcup_id 1 --is_capture
python scripts_kfr/capture_kfr.py --version v3 --cap_type L --task_name inserting --ordcup_id 1 --marker_id 1 --is_capture
python scripts_kfr/capture_kfr.py --version v3 --cap_type L --task_name pressing --ordcup_id 1 --nozzle_id 1 --is_capture

python scripts_kfr/capture_kfr.py --version v3 --cap_type L --task_name reorient_unscrew --bottle_id 6 --is_capture
python scripts_kfr/capture_kfr.py --version v3 --cap_type L --task_name unscrew_pouring --bottle_id 6 --mugcup_id 1 --is_capture

python scripts_kfr/capture_kfr.py --version v3 --cap_type L --task_name tool_spoon --is_capture --test_id 1
python scripts_kfr/capture_kfr.py --version v3 --cap_type L --task_name tool_funnel --is_capture --test_id 1

##### for adding environmental changes (--version v4)
python scripts_kfr/capture_kfr.py --version v4 --cap_type L --task_name plugpen --marker_id 1 --is_capture
python scripts_kfr/capture_kfr.py --version v4 --cap_type L --task_name reorient --anyobj_id 1 --is_capture
python scripts_kfr/capture_kfr.py --version v4 --cap_type L --task_name unscrew --bottle_id 6 --is_capture
python scripts_kfr/capture_kfr.py --version v4 --cap_type L --task_name pouring --bottle_id 6 --mugcup_id 1 --is_capture
python scripts_kfr/capture_kfr.py --version v4 --cap_type L --task_name inserting --ordcup_id 1 --marker_id 1 --is_capture
python scripts_kfr/capture_kfr.py --version v4 --cap_type L --task_name pressing --ordcup_id 1 --nozzle_id 1 --is_capture

'''
#################################################################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfr_ip', default="192.168.4.83", help="the ip address of the kingfisher-R-6000")
    parser.add_argument('--is_syn', action='store_true', help="is dual-arm running synchronously. default is False")
    parser.add_argument('--cap_type', default="LR", help="the capture camera left/right. It can be L, R or LR")
    parser.add_argument('--task_name', default="", help="string of bimanual task name.")
    parser.add_argument('--save_imgs_dir', default="/home/dexforce/zhouhuayi/auboHandeyeCalib/scripts_kfr/realcap", 
                        help="path to save all captured images of dual-arm real robot execution")
    
    parser.add_argument('--marker_id', type=int, default=1, help="marker_id is selected from 1 ~ 8")  # for task plugpen
    parser.add_argument('--anyobj_id', type=int, default=1, help="anyobj_id is selected from 1 ~ 8")  # for task reorient
    parser.add_argument('--bottle_id', type=int, default=6, help="bottle_id is selected from 1 ~ 8")  # for task unscrew and pouring
    parser.add_argument('--mugcup_id', type=int, default=1, help="mugcup_id is selected from 1 ~ 4")  # for task pouring
    parser.add_argument('--ordcup_id', type=int, default=1, help="ordcup_id is selected from 1 ~ 4")  # for task inserting and pressing
    parser.add_argument('--nozzle_id', type=int, default=1, help="nozzle_id is selected from 1 ~ 4")  # for task pressing
    
    parser.add_argument('--is_capture', action='store_true', help="default is not the capture mode")
    parser.add_argument('--version', default="v1", help="string of version. v1 for sync test / v2 for BiDemoSyn / v3 for VLBiMan")
    parser.add_argument('--test_id', type=int, default=1, help="we may record many times of one specific task")
    parser.add_argument('--speed_up', type=int, default=2, help="default is 2 for most tasks")
    
    
    parser.add_argument('--start_idx', type=int, default=0, help="the start index of captured frames. default is 0")
    parser.add_argument('--end_idx', type=int, default=10000, help="the end index of captured frames. default is 10000")
    
    args = parser.parse_args()
    ############################################################################
    assert args.task_name in ["plugpen", "reorient", "unscrew", "pouring", "inserting", "pressing", "uncover", "openbox",
                         "reorient_unscrew", "unscrew_pouring", "tool_spoon", "tool_funnel"], "Please give a valid task name !!!"
    assert args.marker_id >=1 and args.marker_id <= 8, "Please note that marker_id is selected from 1 ~ 8 !!!"
    assert args.anyobj_id >=1 and args.anyobj_id <= 8, "Please note that anyobj_id is selected from 1 ~ 8 !!!"
    assert args.bottle_id >=1 and args.bottle_id <= 8, "Please note that bottle_id is selected from 1 ~ 8 !!!"
    assert args.mugcup_id >=1 and args.mugcup_id <= 4, "Please note that mugcup_id is selected from 1 ~ 4 !!!"
    assert args.ordcup_id >=1 and args.ordcup_id <= 4, "Please note that ordcup_id is selected from 1 ~ 4 !!!"
    assert args.nozzle_id >=1 and args.nozzle_id <= 4, "Please note that nozzle_id is selected from 1 ~ 4 !!!"
    
    syn_str = "syn" if args.is_syn else "asyn"
    prefix_str = f"{args.task_name}_{syn_str}_{args.cap_type}"
    if args.task_name == "plugpen": sub_folder = f"{prefix_str}_marker{str(args.marker_id).zfill(2)}"
    if args.task_name == "reorient": sub_folder = f"{prefix_str}_anyobj{str(args.anyobj_id).zfill(2)}"
    if args.task_name == "unscrew": sub_folder = f"{prefix_str}_bottle{str(args.bottle_id).zfill(2)}"
    if args.task_name == "pouring": sub_folder = f"{prefix_str}_bottle{str(args.bottle_id).zfill(2)}-mugcup{str(args.mugcup_id).zfill(2)}"
    if args.task_name == "inserting": sub_folder = f"{prefix_str}_ordcup{str(args.ordcup_id).zfill(2)}-marker{str(args.marker_id).zfill(2)}"
    if args.task_name == "pressing": sub_folder = f"{prefix_str}_ordcup{str(args.ordcup_id).zfill(2)}-nozzle{str(args.nozzle_id).zfill(2)}"
    if args.task_name == "reorient_unscrew": sub_folder = f"{prefix_str}_bottle{str(args.bottle_id).zfill(2)}"
    if args.task_name == "unscrew_pouring": sub_folder = f"{prefix_str}_bottle{str(args.bottle_id).zfill(2)}-mugcup{str(args.mugcup_id).zfill(2)}"
    if args.task_name == "tool_spoon": sub_folder = f"{prefix_str}_spoon-bowlL-bowlS"
    if args.task_name == "tool_funnel": sub_folder = f"{prefix_str}_funnel-bottle-cup"
    
    if args.version in ["v2", "v3", "v4"]: sub_folder += f"_tid{str(args.test_id).zfill(2)}"
    ############################################################################
    final_save_path = os.path.join(args.save_imgs_dir + "_" + args.version, sub_folder)
    if args.is_capture:  # we begin to capture dual cameras in new folders
        if os.path.exists(final_save_path):
            shutil.rmtree(final_save_path)
        os.mkdir(final_save_path)
    else:  # find already captured image folders
        if "L" in args.cap_type: video_save_path_L = final_save_path + f"_L_x{args.speed_up}.mp4"
        if "R" in args.cap_type: video_save_path_R = final_save_path + f"_R_x{args.speed_up}.mp4"
    
    kingfisher.connect(args.kfr_ip)   # connect the camera
    # kingfisher.SetExposure(20000)
    kingfisher.SetAUTO_EXPOSURE()
    
    # calib_file = kingfisher.getCalibData()
    # mac = kingfisher.getMac()
    # width, height = kingfisher.get_resolution()  # 960, 540
    # print(calib_file, "\n", mac, width, height)
        
    ############################################################################
    if args.is_capture:  # we begin to capture new dual cameras
        img_cnt = 0
        while True:
            img_cnt += 1
            img_cnt_str = str(img_cnt).zfill(6)
            left_img, right_img = kingfisher.captureQuarterSize()  # (960, 540)
            
            # combined = np.hstack((left, right))  # (1920, 540) <-- (960, 540) + (960, 540)
            # cv2.namedWindow("MyWindow", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("MyWindow", 1920, 540)
            # cv2.imshow("MyWindow", combined)
            
            if "L" in args.cap_type: cv2.imwrite(os.path.join(final_save_path, "L_"+img_cnt_str+".jpg"), left_img)
            if "R" in args.cap_type: cv2.imwrite(os.path.join(final_save_path, "R_"+img_cnt_str+".jpg"), right_img)

            print(img_cnt_str, "\t", time.time())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    ############################################################################
    else:  # finally, we can save captured images into videos
        width, height, scale_ratio = 960, 540, 0.8  # original image size
        cx1, cx2, cy1, cy2 = 40, 840, 0, 540  # cropping bbox, 960*540 --> 800*540
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
                if idx%10==0: print(idx, "\t", img_name_L)
                if idx+1 < args.start_idx: continue  # we may give a start_idx after manually checking
                if idx+1 > args.end_idx: continue  # we may give a end_idx after manually checking
                left_image = cv2.imread(os.path.join(final_save_path, img_name_L))
                left_image = left_image[cy1:cy2, cx1:cx2]
                if scale_ratio != 1.0: left_image = cv2.resize(left_image, (0, 0), fx=scale_ratio, fy=scale_ratio)
                vout_L.write(left_image)
            vout_L.release()  # Release everything
            print("finished video of left camera")
        if "R" in args.cap_type:
            for idx, img_name_R in enumerate(imgs_name_R):
                if idx%10==0: print(idx, "\t", img_name_R)
                if idx+1 < args.start_idx: continue  # we may give a start_idx after manually checking
                if idx+1 > args.end_idx: continue  # we may give a end_idx after manually checking
                right_image = cv2.imread(os.path.join(final_save_path, img_name_R))
                right_image = right_image[cy1:cy2, cx1:cx2]
                if scale_ratio != 1.0: right_image = cv2.resize(right_image, (0, 0), fx=scale_ratio, fy=scale_ratio)
                vout_R.write(right_image)
            vout_R.release()  # Release everything
            print("finished video of right camera")
        
        print("finished all !!!")