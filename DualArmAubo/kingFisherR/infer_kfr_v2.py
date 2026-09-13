import os
import sys
import torch
import argparse
import time
import open3d as o3d
from glob import glob
from datetime import datetime

import numpy as np
import cv2
import yaml
from multiprocessing import Process, Queue

import kingfisher
from utils import StereoRectifyUtil
from glia.dl.pipelines import StereoInterface


def generate_pcd(left_image, disp, fx, fy, cx, cy, baseline, 
                 object_mask_binary=None, roi_mask_binary=None):
    height, width, _ = left_image.shape
    depth = fx * baseline / disp
    
    left_img_o3d = o3d.geometry.Image(left_image.astype(np.uint8))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    depth_img_o3d = o3d.geometry.Image(depth.astype(np.float32))
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        left_img_o3d,
        depth_img_o3d,
        depth_scale=1.0,
        depth_trunc=depth.max()+1,  # do not remove 3d point in any pixel
        # depth_trunc=4.0,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic)
    print(len(pcd.points), np.array(pcd.points).shape, height*width)
    
    if object_mask_binary is not None:
        # https://www.cnblogs.com/massquantity/p/8908859.html
        left_pts_idx = np.where(object_mask_binary.reshape(-1) > 0)[0]
        print("left points:", len(left_pts_idx))
        pcd = pcd.select_by_index(left_pts_idx)
        
    if roi_mask_binary is not None:
        left_pts_idx = np.where(roi_mask_binary.reshape(-1) > 0)[0]
        print("left points:", len(left_pts_idx))
        pcd = pcd.select_by_index(left_pts_idx)
        
    return pcd, depth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="igev.trt", help="load the weights from a specific checkpoint")
    parser.add_argument('--kfr_ip', default="192.168.4.64", help="the ip address of the kingfisher-R-6000")
    parser.add_argument('--calib_file', default="calib_kfr_250221.yaml", help="camera params")
    parser.add_argument('--im_height', type=int, default=540, help="image height")
    parser.add_argument('--im_width', type=int, default=960, help="image width")
    parser.add_argument('--raw_imgs_dir', default="imgs_v2/", help="path to all left/right frames")
    parser.add_argument('--save_res_dir', default="res_v2/", help="directory of saved results")
    parser.add_argument('--interval', type=float, default=0.2, help="how many time (second) to sample a frame")
    parser.add_argument('--total_num', type=int, default=10, help="when to stop sampling after many frames")
    args = parser.parse_args()


    os.makedirs(args.raw_imgs_dir, exist_ok=True)
    os.makedirs(args.save_res_dir, exist_ok=True)
    model = StereoInterface(backend="trt", model_path=args.model_path)
    model.cuda()
    print("IGEV Tensorrt 模型初始化成功.")


    ####### load and initiate camera parameters
    c = kingfisher.connect(args.kfr_ip)   # connect the camera
    if not os.path.exists(args.calib_file):
        calib_file = kingfisher.getCalibData()
        print("[Error] You should manually copy all params into a yaml file!!!", args.calib_file)
        sys.exit()
    # exposure = 10000
    # kingfisher.SetExposure(exposure)
    kingfisher.SetAUTO_EXPOSURE()
    cam_mac = kingfisher.getMac()
    cam_width, cam_height = kingfisher.get_resolution()  # the default shape is (3840, 2160), not used
    print("camera info:", cam_mac, cam_width, cam_height)
    with open(args.calib_file, "r", encoding="utf-8") as f:
        stereo_res = yaml.load(stream=f, Loader=yaml.FullLoader)  # only for high res (3840, 2160)
        scale_ratio = 4.0  # for low res (960, 540), we should adjust the calib_file /4.0
        for para_key in ["cam1_k", "cam2_k"]:
            for [loc_i, loc_j] in [[0,0], [1,1], [0,2], [1,2]]:
                stereo_res[para_key][loc_i][loc_j] /= scale_ratio
    ori_k1, R1, rect_k1, baseline, map_x_1, map_y_1, map_x_2, map_y_2 = \
        StereoRectifyUtil.rectify_params(stereo_res, args.im_height, args.im_width)
    fx, fy, cx, cy = rect_k1[0, 0], rect_k1[1, 1], rect_k1[0, 2], rect_k1[1, 2]
    model.set_camera(0, args.im_height, args.im_width, fx=fx, fy=fy, cx=cx, cy=cy)
    model.set_camera(1, args.im_height, args.im_width, fx=fx, fy=fy, cx=cx, cy=cy)

    
    imgs_name_list = os.listdir(args.raw_imgs_dir)
    if len(imgs_name_list) == 0:
        img_num_count = 0
        def save_image_worker(q):
            while len(os.listdir(args.raw_imgs_dir)) < args.total_num * 2:
                time.sleep(0.1)  # to avoid bug
                [cur_time, left_image, right_image] = q.get()
                cv2.imwrite(os.path.join(args.raw_imgs_dir, f"{cur_time}_imgL.jpg") , left_image)
                cv2.imwrite(os.path.join(args.raw_imgs_dir, f"{cur_time}_imgR.jpg"), right_image)
                print(f"Save dual images of the time stamp: {cur_time}") 
        q = Queue()
        p = Process(target=save_image_worker, args=(q,))
        p.start()
        _, _ = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960), do not save. just for debugging
        _, _ = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960), do not save. just for debugging
        while True:
            ####### capture one or many left-right image pairs
            start_time = time.time()
            cur_time = str(datetime.now()).replace("-", "").replace(" ", "").replace(":", "").replace(".", "")
            left_image, right_image = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
            print(f"{img_num_count}. Capture dual images at the time stamp: {cur_time}")
            img_num_count += 1
            q.put([cur_time, left_image, right_image])
            while time.time() - start_time < args.interval:
                time.sleep(0.00001)  # waiting until it comes to the time interval
            # if 0xFF == ord('q'):  break
            if img_num_count == args.total_num:  break
        print("finished saving all captured images!!!")
        p.join()
        
    else:
        imgs_name_list.sort()
        for i in range(len(imgs_name_list)//2):
            cur_time = imgs_name_list[2*i].replace("_imgL.jpg", "")
            
            left_img_path = os.path.join(args.raw_imgs_dir, imgs_name_list[2*i]) 
            left_image = cv2.imread(left_img_path)
            right_img_path = os.path.join(args.raw_imgs_dir, imgs_name_list[2*i+1])
            right_image = cv2.imread(right_img_path)
            rect_left_img = StereoRectifyUtil.rectify_img(left_image, map_x_1, map_y_1)
            rect_right_img = StereoRectifyUtil.rectify_img(right_image, map_x_2, map_y_2)
            rect_left_img = cv2.cvtColor(rect_left_img, cv2.COLOR_BGR2RGB)
            rect_right_img = cv2.cvtColor(rect_right_img, cv2.COLOR_BGR2RGB)

            ##### set the left/right input for inferring
            model.update_camera_data(0, rect_left_img)
            model.update_camera_data(1, rect_right_img)
            start_time = time.time()
            disp_tensor = model.inference_disp([0, 1])
            end_time = time.time()
            print(f"Inference time: {end_time-start_time} s")
            disp_npy = disp_tensor.squeeze().cpu().numpy()

            ##### save the obtained point clouds
            [x1, y1, x2, y2] = [110, 90, 770, 530]  # the roi_bbox (960*540 --> 660*440; 16:9 --> 3:2)
            roi_mask_binary = np.zeros([args.im_height, args.im_width], dtype=np.uint8)
            roi_mask_binary[y1:y2, x1:x2] = 255
            pcd, depth = generate_pcd(rect_left_img, disp_npy, fx, fy, cx, cy, baseline, 
                                    object_mask_binary=None, roi_mask_binary=roi_mask_binary)
            pcd.rotate(R1)
            o3d.io.write_point_cloud(os.path.join(args.save_res_dir, f"{cur_time}.ply"), pcd) 
            print("processed a pair of frames", cur_time)

'''
# You can reconstruct the point clouds (pcd) of the entire scene at continuous specific time stamps automaticly.
$ python infer_kfr_v2.py --interval 0.2 --total_num 10

'''