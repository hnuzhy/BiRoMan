
import os
import cv2
import sys
import torch
import time
import matplotlib
import numpy as np
import open3d as o3d
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from src.config import cfg_dict_init as cfg_dict
from src.vlms import conduct_object_detect_and_segment

from thirdlibs.FoundationStereo.core.utils.utils import InputPadder
from thirdlibs.FoundationStereo.core.foundation_stereo import *
from thirdlibs.FoundationStereo.Utils import *

'''Please refer https://github.com/NVlabs/FoundationStereo '''
code_dir = "/home/dex/zhouhuayi/rokaeDemo/thirdlibs/FoundationStereo/"

#################################################################
def load_FS_model():

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    # os.makedirs(args.out_dir, exist_ok=True)

    ckpt_dir = f'{code_dir}/pretrained_models/11-33-40/model_best_bp2.pth'
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'  # vits
    args = OmegaConf.create(cfg)
    logging.info(f"Using pretrained model from {ckpt_dir}")

    model = FoundationStereo(args)

    ckpt = torch.load(ckpt_dir, weights_only=False)  # for torch>=2.5
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])

    model.cuda()
    model.eval()

    return model

def process_a_paired_binocular_images(fs_model, img0, img1, mask=None):
    start_time = time.time()
    print("\n*******************start_time", time.time())

    args_scale = 1  # downsize the image by scale, must be <=1
    args_hiera = 0  # hierarchical inference (only needed for high-resolution images (>1K) )
    args_valid_iters = 16  # default is 32. number of flow-field updates during forward pass. 16 will make high inferring speed yet low pcd quality
    args_remove_invisible = 1  # remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable
    args_get_pc = 1  # save point cloud output
    args_intrinsic_file = f'{code_dir}/assets/KF.txt'  # camera intrinsic matrix and baseline file
    args_z_far = 10  # max depth to clip in point cloud
    args_clip_cloud = 0  # whether to clip the whole scene-level point cloud (we do not want to clip it defaultly)
    args_denoise_cloud = 0  # whether to denoise the point cloud. This will take about 5~8 seconds!!!
    args_denoise_nb_points = 30  # number of points to consider for radius outlier removal
    args_denoise_radius = 0.03  # radius to use for outlier removal

    scale = args_scale
    assert scale<=1, "scale must be <=1"
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H,W = img0.shape[:2]
    img0_ori = img0.copy()
    logging.info(f"img0: {img0.shape}")

    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    #with torch.cuda.amp.autocast(True):
    with torch.autocast(device_type="cuda"):
        if not args_hiera:
            disp = fs_model.forward(img0, img1, iters=args_valid_iters, test_mode=True)
        else:
            disp = fs_model.run_hierachical(img0, img1, iters=args_valid_iters, test_mode=True, small_ratio=0.5)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W)
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)
    # imageio.imwrite(f'{args.out_dir}/vis.png', vis)
    # logging.info(f"Output saved to {args.out_dir}")

    if args_remove_invisible:
        yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        us_right = xx-disp
        invalid = us_right<0
        disp[invalid] = np.inf

    if args_get_pc:
        with open(args_intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
            baseline = float(lines[1])
        K[:2] *= scale
        depth = K[0,0]*baseline/disp
        # np.save(f'{args.out_dir}/depth_meter.npy', depth)
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
        print(len(pcd.points), np.array(pcd.points).shape)

        if mask is not None:
            # https://www.cnblogs.com/massquantity/p/8908859.html
            left_pts_idx = np.where(mask.reshape(-1) > 0)[0]
            print("left points:", len(left_pts_idx))
            pcd_obj = pcd.select_by_index(left_pts_idx)
        
        if args_clip_cloud:
            keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args_z_far)
            keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
            pcd = pcd.select_by_index(keep_ids)
            # o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
            # logging.info(f"PCL saved to {args.out_dir}")
            print(len(pcd.points), np.array(pcd.points).shape)

        if args_denoise_cloud:
            logging.info("[Optional step] denoise point cloud...")
            cl, ind = pcd.remove_radius_outlier(nb_points=args_denoise_nb_points, radius=args_denoise_radius)
            inlier_cloud = pcd.select_by_index(ind)
            # o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
            pcd = inlier_cloud

    print("*******************end_time", time.time(), time.time() - start_time)
    return pcd, pcd_obj, vis


def check_state_via_obj_pcd(pcd_obj, cls_obj):

    if "box" in cls_obj:
        category_list = {"A": "standing", "B": "lying down"}
    if "bowl" in cls_obj or "basket" in cls_obj:
        category_list = {"A": "upright", "B": "inverted"}
    if "bottle" in cls_obj or "mug" in cls_obj:
        category_list = {"A": "upright", "B": "lying down", "C": "upside down"}

    cam2armL_mat = cfg_dict["arm1"]["handeye_para"]  # camera --> armL
    pcd_obj_arr = np.array(pcd_obj.points).transpose(1, 0)  # the shape is N*3 --> 3*N
    pcd_obj_arr_armL = cam2armL_mat[:3, :3] @ pcd_obj_arr + np.expand_dims(cam2armL_mat[:3, -1], axis=1)

    pt_arr_armL_adjusted = pcd_obj_arr_armL.copy()  # original XYZ --> X+ is front / Y+ is down / Z+ is left
    pt_arr_armL_adjusted[0, :] = pcd_obj_arr_armL[2, :] * (-1)  # X+ is right
    pt_arr_armL_adjusted[1, :] = pcd_obj_arr_armL[0, :] * (-1)  # Y+ is back
    pt_arr_armL_adjusted[2, :] = pcd_obj_arr_armL[1, :] * (-1)  # Z+ is up

    pcd_obj.points = o3d.utility.Vector3dVector(pt_arr_armL_adjusted.transpose(1, 0))  # adjusted obj pcd in the new world space

    #####===================================================================================
    # if "bowl" in cls_obj or "basket" in cls_obj:
    #     c_p_x = ( pt_arr_armL_adjusted[0, :].min() + pt_arr_armL_adjusted[0, :].max() ) / 2.0
    #     c_p_y = ( pt_arr_armL_adjusted[1, :].min() + pt_arr_armL_adjusted[1, :].max() ) / 2.0
    #     c_p_z = ( pt_arr_armL_adjusted[2, :].min() + pt_arr_armL_adjusted[2, :].max() ) / 2.0
    #     avg_height = pt_arr_armL_adjusted[2, :].mean()
    #     ref_height = c_p_z  # using the c_p_z instead of the avg_height is better for judging upright/inverted

    #     min_dis, sel_idx = 10000, -1
    #     for idx in range(pt_arr_armL_adjusted.shape[1]):
    #         temp_dis = (pt_arr_armL_adjusted[0, idx] - c_p_x)**2 + (pt_arr_armL_adjusted[1, idx] - c_p_y)**2
    #         if temp_dis < min_dis: min_dis = temp_dis; sel_idx = idx
    #     c_p_height = pt_arr_armL_adjusted[2, sel_idx]  # c_p_z

    #     if c_p_height < ref_height: cur_state = "A"  # "upright"
    #     if c_p_height > ref_height: cur_state = "B"  # "inverted"
    #     print(cls_obj, ref_height, c_p_height)

    upper_ratio_thre = 0.1  # 1/10 part of the whole pcd points
    z_height = pt_arr_armL_adjusted[2, :].max() - pt_arr_armL_adjusted[2, :].min()
    if "bottle" in cls_obj or "mug" in cls_obj or "box" in cls_obj:
        if "bottle" in cls_obj: height_thre = 0.120     # 12 cm
        if "mug" in cls_obj: height_thre = 0.080        # 8 cm
        if "box" in cls_obj: height_thre = 0.100        # 10 cm
        
        if z_height < height_thre:
            cur_state = "B"  # "lying down"
        else:
            if "bottle" in cls_obj:
                p_z_max = pt_arr_armL_adjusted[2, :].max()
                left_pts_idx = np.where(pt_arr_armL_adjusted[2, :] > p_z_max - 0.010)  # left points of the upper part
                left_ptx_arr, left_pty_arr = pt_arr_armL_adjusted[0, left_pts_idx], pt_arr_armL_adjusted[1, left_pts_idx]
                upper_r_x, upper_r_y = left_ptx_arr.max() - left_ptx_arr.min(), left_pty_arr.max() - left_pty_arr.min()
                upper_r = max(upper_r_x, upper_r_y)  # find the bigger
                whole_r_x = pt_arr_armL_adjusted[0, :].max() - pt_arr_armL_adjusted[0, :].min()
                whole_r_y = pt_arr_armL_adjusted[1, :].max() - pt_arr_armL_adjusted[1, :].min()
                whole_r = min(whole_r_x, whole_r_y)  # find the smaller
                if upper_r*1.0 / whole_r <= 0.5: cur_state = "A"  # "upright" 
                if upper_r*1.0 / whole_r > 0.5: cur_state = "C"  # "upside down"
                print(cls_obj, upper_r*1.0 / whole_r, upper_r, whole_r)

                # p_z_max = pt_arr_armL_adjusted[2, :].max(); ptz_arr = pt_arr_armL_adjusted[2, :]
                # left_pts_idx = np.where(ptz_arr > p_z_max - 0.010)  # left points of the upper part1
                # left_ptx_arr, left_pty_arr = pt_arr_armL_adjusted[0, left_pts_idx], pt_arr_armL_adjusted[1, left_pts_idx]
                # upper_r_x, upper_r_y = left_ptx_arr.max() - left_ptx_arr.min(), left_pty_arr.max() - left_pty_arr.min()
                # upper_r_part1 = (upper_r_x + upper_r_y) * 0.5
                # left_pts_idx = np.where((ptz_arr > p_z_max - 0.020) & (ptz_arr < p_z_max - 0.010) )  # left points of the upper part2
                # left_ptx_arr, left_pty_arr = pt_arr_armL_adjusted[0, left_pts_idx], pt_arr_armL_adjusted[1, left_pts_idx]
                # upper_r_x, upper_r_y = left_ptx_arr.max() - left_ptx_arr.min(), left_pty_arr.max() - left_pty_arr.min()
                # upper_r_part2 = (upper_r_x + upper_r_y) * 0.5
                # top_down_ratio = upper_r_part1*1.0 / upper_r_part2
                # if top_down_ratio <= 0.85: cur_state = "A"  # "upright" 
                # if top_down_ratio > 0.85: cur_state = "C"  # "upside down"
                # print(cls_obj, top_down_ratio, upper_r_part1, upper_r_part2)

            if "mug" in cls_obj:
                upper_thre = z_height * upper_ratio_thre  # choose the 1/10 upper part of the mug / bottle with a threshold
                p_z_max = pt_arr_armL_adjusted[2, :].max()
                left_pts_idx = np.where(pt_arr_armL_adjusted[2, :] > p_z_max - upper_thre)  # left points of the upper part
                pts_ratio = len(left_pts_idx[0])*1.0 / pt_arr_armL_adjusted.shape[1]
                if pts_ratio <= upper_ratio_thre: cur_state = "A"  # "upright" 
                if pts_ratio > upper_ratio_thre: cur_state = "C"  # "upside down"
                print(cls_obj, pts_ratio, len(left_pts_idx[0]), pt_arr_armL_adjusted.shape[1])
                
            if "box" in cls_obj:
                cur_state = "A"  # "standing", we now only consider two states of the rectangle box
            
    elif "bowl" in cls_obj or "basket" in cls_obj:
        if "gray bowl" == cls_obj: upper_ratio_thre = 0.25  # adjust this thre for this object
        if "basket" in cls_obj: upper_ratio_thre = 0.2
        upper_thre = z_height * upper_ratio_thre # choose the 1/10 upper part of the bowl / basket with a threshold
        p_z_max = pt_arr_armL_adjusted[2, :].max()
        left_pts_idx = np.where(pt_arr_armL_adjusted[2, :] > p_z_max - upper_thre)  # left points of the upper part
        pts_ratio = len(left_pts_idx[0])*1.0 / pt_arr_armL_adjusted.shape[1]
        if pts_ratio <= upper_ratio_thre: cur_state = "A"  # "upright" 
        if pts_ratio > upper_ratio_thre: cur_state = "B"  # "inverted"
        print(cls_obj, pts_ratio, len(left_pts_idx[0]), pt_arr_armL_adjusted.shape[1])

    else:  # the given object class name is not defined!!!
        return "A", pcd_obj, "standing"
        
    print("[state checking!!!]", cls_obj, cur_state, category_list[cur_state], z_height)
    #####===================================================================================

    return cur_state, pcd_obj, category_list[cur_state]

###########################################################################################
def test_paired_binocular_images(fs_model, imgL_path_file, imgR_path_file, detseg_prompts):

    test_img_path_L = imgL_path_file
    test_img_path_R = imgR_path_file

    out_scene_ply, out_object_ply = test_img_path_L[:-4]+"_scene.ply", test_img_path_L[:-4]+"_object.ply"

    img0 = imageio.imread(test_img_path_L)  # usually a rgb image with shape  (540, 960, 3)
    img1 = imageio.imread(test_img_path_R)  # usually a rgb image with shape  (540, 960, 3)

    img_bgr = cv2.convertScaleAbs(img0.copy()[:, :, ::-1], alpha=1.2, beta=10)  # adjust the brightness and contrast
    final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(img_bgr, prompts_str=detseg_prompts, is_raw_result=True)
    obj_binary_mask = final_res_list_raw[0][2]
    [h,w,c] = img_bgr.shape
    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    top, bottom, left, right = y1, h-y2, x1, w-x2
    obj_binary_mask = cv2.copyMakeBorder(obj_binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    cv2.imwrite(test_img_path_L[:-4]+f"_mask_{detseg_prompts}.png", obj_binary_mask)
    cv2.imwrite(test_img_path_L[:-4]+f"_mask_{detseg_prompts}_vis.png", img_vis_cv2.astype(np.uint8))

    pcd_scene, pcd_obj, vis_res = process_a_paired_binocular_images(fs_model, img0, img1, obj_binary_mask)

    o3d.io.write_point_cloud(test_img_path_L[:-4]+f"_pcd_scene.ply", pcd_scene)
    o3d.io.write_point_cloud(test_img_path_L[:-4]+f"_pcd_{detseg_prompts}.ply", pcd_obj)
    imageio.imwrite(test_img_path_L[:-4]+f"_depth_vis.png", vis_res)


    cur_state, pcd_obj_adjusted, cur_state_str = check_state_via_obj_pcd(pcd_obj, detseg_prompts)
    o3d.io.write_point_cloud(test_img_path_L[:-4]+f"_pcd_{detseg_prompts}(ad).ply", pcd_obj_adjusted)


#################################################################
if __name__ == "__main__":  # python src/pcd2sat.py

    print("[Testing] From depth to state...")

    fs_model = load_FS_model()

    test_dir = "/home/dex/zhouhuayi/rokaeDemo/debug/test_pcd2state/"
    test_imgpath_prompts_list = [
        # ["imgL_000000.jpg", "imgR_000000.jpg", "mug"],
        # ["imgL_000000.jpg", "imgR_000000.jpg", "bottle"],
        # ["imgL_000000.jpg", "imgR_000000.jpg", "white bowl"],
        # ["imgL_000000.jpg", "imgR_000000.jpg", "blue bowl"],

        # ["imgL_000001.jpg", "imgR_000001.jpg", "transparent bowl"],
        # ["imgL_000001.jpg", "imgR_000001.jpg", "white bowl"],
        # ["imgL_000001.jpg", "imgR_000001.jpg", "blue bowl"],
        # ["imgL_000002.jpg", "imgR_000002.jpg", "transparent bowl"],
        # ["imgL_000002.jpg", "imgR_000002.jpg", "white bowl"],
        # ["imgL_000002.jpg", "imgR_000002.jpg", "blue bowl"],
        # ["imgL_000003.jpg", "imgR_000003.jpg", "mug"],
        # ["imgL_000003.jpg", "imgR_000003.jpg", "bottle"],
        # ["imgL_000003.jpg", "imgR_000003.jpg", "white bowl"],
        # ["imgL_000003.jpg", "imgR_000003.jpg", "blue bowl"],
        # ["imgL_000004.jpg", "imgR_000004.jpg", "mug"],
        # ["imgL_000004.jpg", "imgR_000004.jpg", "bottle"],
        # ["imgL_000004.jpg", "imgR_000004.jpg", "basket"],


        # ["imgL_000005.jpg", "imgR_000005.jpg", "white bowl"],
        # ["imgL_000005.jpg", "imgR_000005.jpg", "blue bowl"],
        # ["imgL_000005.jpg", "imgR_000005.jpg", "brown bowl"],
        # ["imgL_000006.jpg", "imgR_000006.jpg", "white bowl"],
        # ["imgL_000006.jpg", "imgR_000006.jpg", "blue bowl"],
        # ["imgL_000006.jpg", "imgR_000006.jpg", "brown bowl"],

        ["imgL_000007.jpg", "imgR_000007.jpg", "box"],
        ["imgL_000007.jpg", "imgR_000007.jpg", "blue bowl"],
        ["imgL_000007.jpg", "imgR_000007.jpg", "pink basket"],
        ["imgL_000007.jpg", "imgR_000007.jpg", "black cup"],
        ["imgL_000007.jpg", "imgR_000007.jpg", "plastic jar"],  

    ]

    for test_id, [test_img_path_L, test_img_path_R, detseg_prompts] in enumerate(test_imgpath_prompts_list):
        imgL_path_file = os.path.join(test_dir, test_img_path_L) 
        imgR_path_file = os.path.join(test_dir, test_img_path_R)

        start_time = time.time()
        print("\n*******************************start_time", test_id, time.time())
        test_paired_binocular_images(fs_model, imgL_path_file, imgR_path_file, detseg_prompts)
        print("*******************************end_time", test_id, time.time(), time.time() - start_time)

