
import os
import cv2
import sys
import torch
import time
import matplotlib
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares

sys.path.insert(0, os.getcwd())
from src.config import cfg_dict_init as cfg_dict
from src.vlms import conduct_object_detect_and_segment

from thirdlibs.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

'''Please refer https://github.com/DepthAnything/Depth-Anything-V2 '''
dav2_path = "/home/dex/zhouhuayi/rokaeDemo/thirdlibs/DepthAnythingV2"

#################################################################

cmap = matplotlib.colormaps.get_cmap('Spectral_r')

#################################################################

def load_dav2_model(args_encoder='vits'):

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    weight_path = f'{dav2_path}/checkpoints/depth_anything_v2_{args_encoder}.pth'
    depth_anything = DepthAnythingV2(**model_configs[args_encoder])
    depth_anything.load_state_dict(torch.load(weight_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    return depth_anything


def process_one_monocular_image(depth_anything, raw_image, get_vis_img=False):
    args_input_size = 518  # By default, we use input size 518 for model inference. You can increase the size for even more fine-grained results.
    args_pred_only = False  # (optional): Only save the predicted depth map, without raw image.
    args_grayscale = False  # (optional): Save the grayscale depth map, without applying color palette.
    
    depth = depth_anything.infer_image(raw_image, args_input_size)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)  # 0 ~ 255, shape is (h, w)
    
    if args_grayscale:
        depth_show = np.repeat(depth[..., np.newaxis], 3, axis=-1)  # shape is (h, w, 3)
    else:
        depth_show = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)  # shape is (h, w, 3)
    
    if get_vis_img:
        return depth_show
    else:
        return depth

    '''
    if args_pred_only:
        cv2.imwrite(os.path.join(args_outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth_show)
    else:
        split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([raw_image, split_region, depth_show])
        cv2.imwrite(os.path.join(args_outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
    '''


def depth_to_pointcloud(depth_uint8, color_image=None, intrinsics=None, mask=None,
                        depth_scale=0.01, keep_zero=False, median_filter=False):
    """
    depth_uint8: HxW uint8 depth values (0-255)
    color_image: HxW x3 uint8 color image (RGB). 如果为 None，点云会使用灰度颜色
    intrinsics: 3x3 np.array or dict {'fx','fy','cx','cy'}
    mask: None 或 HxW bool/uint8 (1为目标)
    depth_scale: 每个深度单位映射到米（例如 0.01 -> 0..2.55m）
    keep_zero: 是否保留 depth==0 的点（通常要丢弃）
    median_filter: 是否先对 depth_uint8 做中值滤波以去噪
    返回: open3d.geometry.PointCloud
    """
    if intrinsics is None:
        raise ValueError("intrinsics required")

    [fx, fy, cx, cy] = intrinsics

    depth = depth_uint8.astype(np.float32)
    if median_filter:
        depth = cv2.medianBlur(depth, 3)

    H, W = depth.shape

    # mask valid depth
    if keep_zero:
        valid_mask = np.ones_like(depth, dtype=bool)
    else:
        valid_mask = depth > 0  # 排除 0 深度
    if mask is not None:
        mask_bool = (mask > 0)
        # 兼容 mask 与 depth 大小不同的情况
        if mask_bool.shape != depth.shape:
            raise ValueError("mask shape must equal depth shape")
        valid_mask = np.logical_and(valid_mask, mask_bool)

    # 生成像素网格
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    z = depth * depth_scale  # 将相对深度映射到米
    z_valid = z[valid_mask]
    if z_valid.size == 0:
        print("Warning: no valid depth pixels (after masking & threshold).")
        return o3d.geometry.PointCloud()

    uu_valid = uu[valid_mask].astype(np.float32)
    vv_valid = vv[valid_mask].astype(np.float32)

    # 从像素坐标反投影到相机坐标
    x = (uu_valid - cx) * z_valid / fx
    y = (vv_valid - cy) * z_valid / fy

    points = np.stack([x, y, z_valid], axis=1)

    # 颜色处理
    if color_image is None:
        # 使用灰度表示
        colors = np.repeat((depth[valid_mask:][:,None] / 255.0), 3, axis=1)
    else:
        # color_image 可能是 BGR（如果使用 cv2.imread），确保为 RGB
        if color_image.shape[:2] != depth.shape:
            raise ValueError("color image must have same H,W as depth")
        # 如果 color 是 uint8，归一化到 [0,1]
        if color_image.dtype == np.uint8:
            img_rgb = color_image.astype(np.float32) / 255.0
        else:
            img_rgb = color_image.astype(np.float32)
            # 如果已经是 0-255 but float，仍归一化
            if img_rgb.max() > 1.0:
                img_rgb = img_rgb / 255.0
        colors = img_rgb[valid_mask]

    # build open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def save_and_visualize_pcd(pcd, filename=None, show=True, downsample_voxel_size=None):
    """
    保存并可视化点云
    """
    if pcd.is_empty():
        print("PointCloud is empty, skipping save/visualize.")
        return

    pcd_to_show = pcd
    if downsample_voxel_size is not None and downsample_voxel_size > 0:
        pcd_to_show = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)

    if filename is not None:
        folder = os.path.dirname(filename)
        if folder != "" and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        o3d.io.write_point_cloud(filename, pcd_to_show)
        print(f"Saved point cloud to {filename}")

    if show:
        o3d.visualization.draw_geometries([pcd_to_show], window_name=os.path.basename(filename) if filename else "pcd")


# ---------------------------
# Main pipeline
# ---------------------------
def test_one_one_monocular_image():

    test_img_path, detseg_prompts = "/home/dex/zhouhuayi/rokaeDemo/debug/test_depth2state/imgL_000000.jpg", "mug"
    
    out_scene_ply, out_object_ply = test_img_path[:-4]+"_scene.ply", test_img_path[:-4]+"_object.ply"

    left_img_test_ori = cv2.imread(test_img_path)  # usually a rgb image with shape  (540, 960, 3)
    rgb = cv2.convertScaleAbs(left_img_test_ori, alpha=1.2, beta=10)  # Adjust the brightness and contrast 

    depth_anything = load_dav2_model()
    depth_rel = process_one_monocular_image(depth_anything, rgb.copy(), get_vis_img=False)

    final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(rgb.copy(), prompts_str=detseg_prompts, is_raw_result=True)
    mask = final_res_list_raw[0][2]  # the obj_binary_mask

    h,w = depth_rel.shape
    print(f"Image size: {w} x {h}")

    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    top, bottom, left, right = y1, h-y2, x1, w-x2
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    cv2.imwrite(test_img_path[:-4]+"_mask.png", mask)


    K = cfg_dict["cam1_K"]  # the shape is 3 * 3
    for [loc_i, loc_j] in [[0,0], [1,1], [0,2], [1,2]]: K[loc_i][loc_j] /= 2.0
    intr = [K[0,0], K[1,1], K[0,2], K[1,2]]  # fx, fy, cx, cy

    # depth_scale: 将 0-255 映射到米。根据你的单目深度估计器调整该值。
    depth_scale = 0.01  # 例如 1 单位 = 1cm -> 深度范围 0..2.55m

    # 生成整图点云
    pcd_scene = depth_to_pointcloud(depth_uint8=depth_rel, color_image=rgb, intrinsics=intr,
                                    mask=None, depth_scale=depth_scale, median_filter=True)
    save_and_visualize_pcd(pcd_scene, out_scene_ply, show=True, downsample_voxel_size=0.005)

    pcd_object = depth_to_pointcloud(depth_uint8=depth_rel, color_image=rgb, intrinsics=intr,
                                        mask=mask, depth_scale=depth_scale, median_filter=True)
    save_and_visualize_pcd(pcd_object, out_object_ply, show=True, downsample_voxel_size=0.002)



'''
if we do not have the right/accurate depth_scale, the final pcd results are totally shit!!!
'''

#################################################################

# python src/dpt2sat.py --llm_type 1 --object_type bottle --test_id 1 --online
# python src/dpt2sat.py --llm_type 1 --object_type mug --test_id 1 --online
# python src/dpt2sat.py --llm_type 1 --object_type bowl --test_id 1 --online

# python src/dpt2sat.py --llm_type 0 --object_type doubao_test1 --test_id 1 --online
# python src/dpt2sat.py --llm_type 0 --object_type doubao_test2 --test_id 1 --online

# def main_rigid_obj_states():  



#################################################################
if __name__ == "__main__":  # depth2state

    print("[Testing] From depth to state...")

    test_one_one_monocular_image()