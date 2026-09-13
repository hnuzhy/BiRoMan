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
    parser.add_argument('--is_high_res', action='store_true', help="default is low res (540, 960), high res (2160, 3840)")
    parser.add_argument('--is_resize_1k', action='store_true', help="default is low res (540, 960), high res (1080, 1920)")
    parser.add_argument('--img_time_stamp', default="", help="give the time stamp of one captured picture")
    parser.add_argument('--raw_imgs_dir', default="imgs_v1/", help="path to all left/right frames")
    parser.add_argument('--save_res_dir', default="res_v1/", help="directory of saved results")
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
    
    if args.is_high_res and (not args.is_resize_1k):  # for high res, we should adjust the size into (2160, 3840)
        args.im_height *= 4  # 540*4 = 2160
        args.im_width *= 4  # 960*4 = 3840
        scale_ratio = 1.0
        print("[image shape] high resoultion (2160, 3840)")
    elif (not args.is_high_res) and args.is_resize_1k:   # for resize 1k, we should adjust the size into (1080, 1920)
        args.im_height *= 2  # 540*2 = 1080
        args.im_width *= 2  # 960*2 = 1920
        scale_ratio = 2.0  # for resize 1k, we should adjust the calib_file /2.0
        print("[image shape] 1K resoultion (1080, 1920)")
    elif (not args.is_high_res) and (not args.is_resize_1k):
        scale_ratio = 4.0  # for low res, we should adjust the calib_file /4.0
        print("[image shape] low resoultion (540, 960)")
    else:
        print("[Error] You should only adjsut the size once in each time!!!")
        sys.exit()
    with open(args.calib_file, "r", encoding="utf-8") as f:
        stereo_res = yaml.load(stream=f, Loader=yaml.FullLoader)
        for para_key in ["cam1_k", "cam2_k"]:
            for [loc_i, loc_j] in [[0,0], [1,1], [0,2], [1,2]]:
                stereo_res[para_key][loc_i][loc_j] /= scale_ratio
    
    ori_k1, R1, rect_k1, baseline, map_x_1, map_y_1, map_x_2, map_y_2 = \
        StereoRectifyUtil.rectify_params(stereo_res, args.im_height, args.im_width)
    fx, fy, cx, cy = rect_k1[0, 0], rect_k1[1, 1], rect_k1[0, 2], rect_k1[1, 2]
    model.set_camera(0, args.im_height, args.im_width, fx=fx, fy=fy, cx=cx, cy=cy)
    model.set_camera(1, args.im_height, args.im_width, fx=fx, fy=fy, cx=cx, cy=cy)


    ####### capture one or many left-right image pairs
    if len(args.img_time_stamp) == 0:
        cur_time = str(datetime.now()).replace("-", "").replace(" ", "").replace(":", "").replace(".", "")
        if args.is_high_res or args.is_resize_1k:
            left_image, right_image = kingfisher.capture()  # hight resolution, shape is (3840, 2160)
            if args.is_resize_1k:  # (3840, 2160) --> (1920, 1080)
                left_image = cv2.resize(left_image, (0,0), fx=0.5, fy=0.5)  
                right_image = cv2.resize(right_image, (0,0), fx=0.5, fy=0.5)
        else:
            left_image, right_image = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)
        print(f"Capture dual images at the time stamp: {cur_time}")
        left_img_path = os.path.join(args.raw_imgs_dir, f"{cur_time}_imgL.jpg")  
        cv2.imwrite(left_img_path, left_image)
        right_img_path = os.path.join(args.raw_imgs_dir, f"{cur_time}_imgR.jpg")
        cv2.imwrite(right_img_path, right_image)
    else:
        cur_time = args.img_time_stamp
        print(f"Load dual images at the time stamp: {cur_time}")
        left_img_path = os.path.join(args.raw_imgs_dir, f"{cur_time}_imgL.jpg")  
        left_image = cv2.imread(left_img_path)
        right_img_path = os.path.join(args.raw_imgs_dir, f"{cur_time}_imgR.jpg")
        right_image = cv2.imread(right_img_path)
        
    rect_left_img = StereoRectifyUtil.rectify_img(left_image, map_x_1, map_y_1)
    rect_right_img = StereoRectifyUtil.rectify_img(right_image, map_x_2, map_y_2)
    StereoRectifyUtil.visualize_rect_imgs(rect_left_img, rect_right_img, 
        os.path.join(args.save_res_dir, f"{cur_time}_rect.jpg"))
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
    # import ipdb; ipdb.set_trace()


    ###### segment out the manipulated objects
    apply_obj_mask = True  # True or False
    object_mask_binary = None
    lp_rect_path = os.path.join(args.raw_imgs_dir, f"{cur_time}_imgL_rect.jpg")
    if not os.path.exists(lp_rect_path):
        cv2.imwrite(lp_rect_path, rect_left_img[:,:,::-1])
    else:
        lp_mask_path = lp_rect_path.replace("_rect.jpg", "_objmask.png")  # now, the mask is obtained manually
        if os.path.exists(lp_mask_path) and apply_obj_mask:
            object_mask_rgb = cv2.imread(lp_mask_path)
            object_mask_rgb = cv2.resize(object_mask_rgb[:,:,::-1], (args.im_width, args.im_height), cv2.INTER_NEAREST)
            object_mask_rgb = cv2.erode(object_mask_rgb, np.ones((8, 8), np.uint8))  # apply the erode operation
            object_mask_gray = cv2.cvtColor(object_mask_rgb, cv2.COLOR_BGR2GRAY)  # convert rgb img to gray img
            ret, object_mask_binary = cv2.threshold(object_mask_gray, 1, 255, cv2.THRESH_BINARY)
            lp_mask_binary_path = lp_mask_path.replace("_objmask.png", "_objmask_binary.png")
            cv2.imwrite(lp_mask_binary_path, object_mask_binary)

    ##### save the obtained point clouds
    pcd, depth = generate_pcd(rect_left_img, disp_npy, fx, fy, cx, cy, baseline, 
                              object_mask_binary=object_mask_binary, roi_mask_binary=None)
    pcd.rotate(R1)
    if object_mask_binary is not None and apply_obj_mask:
        o3d.io.write_point_cloud(os.path.join(args.save_res_dir, f"{cur_time}_obj.ply"), pcd)
    else:
        o3d.io.write_point_cloud(os.path.join(args.save_res_dir, f"{cur_time}.ply"), pcd) 
    # print(depth.shape, np.min(depth), np.max(depth))
    # cv2.imwrite(lp_rect_path.replace("_rect.jpg", "_rect_depth.jpg"), depth)
    # np.save(lp_rect_path.replace("_rect.jpg", "_rect_depth.npy"), depth)

'''
# You can first reconstruct the point clouds (pcd) of the entire scene at a specific time stamp automaticly.
$ python infer_kfr_v1.py --is_resize_1k 
# Then you give the 2D mask of manipulated objects in rectified image for reconstructing the object-level pcd.
$ python infer_kfr_v1.py --is_resize_1k --img_time_stamp 20250221183511335733


$ python infer_kfr_v1.py --img_time_stamp 20250814214912263888

'''