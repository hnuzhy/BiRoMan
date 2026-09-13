
import os
import sys
import cv2
import json
import argparse
import numpy as np
import open3d as o3d

sys.path.insert(0, os.getcwd())
from src.config import cfg_dict_init as cfg_dict
from src.algs.slots_v1 import find_four_corners_of_a_rectangle_container
from src.algs.slots_v1 import find_dense_corners_of_a_deformable_cloth

#################################################################
def process_pcd_into_bev_results(pcd_scene, obj_mask, img_h, img_w, broken_mask=False):

    if broken_mask: # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        ###### Shape Completion / Inpainting
        add_size = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (add_size, add_size))
        closed_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_c = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_c)
        perimeter = cv2.arcLength(hull, True)
        epsilon = 0.00001 * perimeter
        polygon_arr = cv2.approxPolyDP(hull, epsilon, True)
        print("[0][broken_mask is True] polygon_arr length -->", len(polygon_arr))
        completed_mask = np.zeros_like(obj_mask)
        cv2.fillPoly(completed_mask, [polygon_arr], 255)
        obj_mask = completed_mask
        img_canvas_F = completed_mask.copy()
    else:
        img_canvas_F = None
        
        
    using_erode = True  # True or False, the default value is True
    if using_erode:  # https://blog.csdn.net/weixin_45939019/article/details/104391620
        obj_mask = cv2.erode(obj_mask, np.ones((4, 4), np.uint8))
    
    pcd_arr = np.array(pcd_scene.points)  # the shape is (N, 3)
    left_pts_idx = np.where(obj_mask.reshape(-1) > 0)[0]
    print("original / left points: ", len(pcd_arr), len(left_pts_idx))
    pcd_obj = pcd_scene.select_by_index(left_pts_idx)
    
    cam2armL_mat = cfg_dict["arm1"]["handeye_para"]  # camera --> armL
    camL_para = cfg_dict["cam1_K"]; camR_para = cfg_dict["cam2_K"]  # camera intrinsics
    pcd_obj_arr = np.array(pcd_obj.points).transpose(1, 0)  # the shape is N*3 --> 3*N
    pcd_obj_arr_armL = cam2armL_mat[:3, :3] @ pcd_obj_arr + np.expand_dims(cam2armL_mat[:3, -1], axis=1)
    
    # original XYZ --> X+ is front / Y+ is down / Z+ is left
    max_y_value = pcd_obj_arr_armL[1, :].max()
    pcd_obj_arr_armL[1, :] = max_y_value  # project all 3d points into the Z-plane (rectangle container's grounding)
    
    pcd_obj_arr_bev = cam2armL_mat[:3, :3].T @ (pcd_obj_arr_armL - np.expand_dims(cam2armL_mat[:3, -1], axis=1))
    img_canvas = np.zeros((img_h, img_w, 3), np.uint8)
    for pcd_3d_proj in pcd_obj_arr_bev.transpose(1, 0):  # 3*N --> N*3
        [pt3d_x, pt3d_y, pt3d_z] = pcd_3d_proj
        [p2d_x, p2d_y, temp_scale] = camL_para @ np.array([pt3d_x/pt3d_z, pt3d_y/pt3d_z, 1.0]).T  # temp_scale is always 1.0
        cv2.circle(img_canvas, (int(p2d_x), int(p2d_y)), radius=2, color=(255,255,255), thickness=-1, lineType=cv2.LINE_AA)
    
    
    if broken_mask:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (add_size, add_size))
        closed_mask = cv2.morphologyEx(img_canvas, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cv2.cvtColor(closed_mask, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_c = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_c)
        perimeter = cv2.arcLength(hull, True)
        epsilon = 0.00001 * perimeter
        polygon_arr = cv2.approxPolyDP(hull, epsilon, True)
        print("[1][broken_mask is True] polygon_arr length -->", len(polygon_arr))
        completed_mask = np.zeros_like(img_canvas)
        cv2.fillPoly(completed_mask, [polygon_arr], (255,255,255))
        img_canvas_P = cv2.erode(completed_mask, np.ones((4+add_size*2, 4+add_size*2), np.uint8))  # remove those noisy points or areas
        img_canvas_OP = img_canvas_P.copy()
    else:
        img_canvas_P = cv2.erode(img_canvas, np.ones((4, 4), np.uint8))  # remove those noisy points or areas
        mask_filling = np.zeros((img_h + 2, img_w + 2), np.uint8)
        img_canvas_OP = img_canvas_P.copy()
        # https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/  (note the args flags)
        cv2.floodFill(img_canvas_OP, mask_filling, (0,0), (255,255,255), flags=4)  
        img_canvas_OP = img_canvas_P | cv2.bitwise_not(img_canvas_OP)
        img_canvas_OP = cv2.erode(img_canvas_OP, np.ones((2, 2),np.uint8))  # to obtain a more smooth contour


    img_gray = cv2.cvtColor(img_canvas_OP.copy(), cv2.COLOR_BGR2GRAY)
    # https://learnopencv.com/contour-detection-using-opencv-python-c  (note the args cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_len_list = [len(contour) for contour in contours]; print("contour_len_list:", contour_len_list)
    contour_ideal = contours[contour_len_list.index(max(contour_len_list))].squeeze()  # (N, 1, 2) --> (N, 2)
    
    return contour_ideal, img_canvas_P, img_canvas_OP, img_canvas_F

#################################################################
def testing_rectangle_container_func(args):
    img_folder = os.path.join(args.root_dir, args.sub_folder)
    
    trans_mat = cfg_dict["transform_mat_inv"]; trans_mat_t = cfg_dict["transform_mat"]
    rect_l, rect_w = cfg_dict['rect_l'], cfg_dict['rect_w']; colors_list = cfg_dict['colors_list']
    roi_bbox = cfg_dict["detection_roi_bbox"]
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']  # 540, 960
    top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
    colors_list = [(0,255,128), (255,255,0), (128,0,255), (0,128,128), (0,0,255)]  # lawn green, cyan, light magenta, light yellow, red
    
    file_name_list = os.listdir(img_folder)
    img_names = [i for i in file_name_list if "_rgb_L.jpg" in i]; img_names.sort()
    mask_names = [i for i in file_name_list if "_mask.jpg" in i]; mask_names.sort()
    json_names = [i for i in file_name_list if "_mask.json" in i]; json_names.sort()
    # pcd_object_names = [i for i in file_name_list if "_pcd_obj.ply" in i]; pcd_object_names.sort()
    pcd_scene_names = [i for i in file_name_list if "_pcd_scene.ply" in i]; pcd_scene_names.sort()
    
    for tid, (img_name, mask_name, json_name, pcd_name) in enumerate(zip(img_names, mask_names, json_names, pcd_scene_names)):
        print("\n", tid, img_name)
        
        img_cv2_full = cv2.imread(os.path.join(img_folder, img_name))  # shape is (540, 960, 3)
        mask_cv2_full = cv2.cvtColor(cv2.imread(os.path.join(img_folder, mask_name)), cv2.COLOR_BGR2GRAY)  # shape is (540, 960, 1)
        json_dict = json.load(open(os.path.join(img_folder, json_name), "r"))  # {"bbox": ..., "mask": ...}
        pcd_scene = o3d.io.read_point_cloud(os.path.join(img_folder, pcd_name))  # 3D point clouds (scene level)
        
        img_cv2_plot = img_cv2_full[y1:y2, x1:x2]; color_mask_edge = (0,255,255)  # yellow
        rect_bbox, mask_arr = json_dict["bbox"], json_dict["mask"]
        if args.view_mode == "EDGE": 
            for [ptx, pty] in mask_arr: cv2.circle(img_cv2_plot, (int(ptx), int(pty)), 2, color_mask_edge, -1, cv2.LINE_AA)
            cpt_p, cpt_o, corners_p, corners_o, grid_nodes_p, grid_cells_p = find_four_corners_of_a_rectangle_container(mask_arr)
        if args.view_mode == "BEV":
            mask_arr_bev, maskP, maskOP, maskF = process_pcd_into_bev_results(pcd_scene, mask_cv2_full, 
                kfr_height, kfr_width, broken_mask=args.infill_mask)
            cv2.imwrite(os.path.join(img_folder, img_name[:-4]+f"_maskP.jpg"), maskP)
            cv2.imwrite(os.path.join(img_folder, img_name[:-4]+f"_maskOP.jpg"), maskOP)
            if maskF is not None: cv2.imwrite(os.path.join(img_folder, img_name[:-4]+f"_maskF.jpg"), maskF)
            mask_arr_bev[:, 0] -= x1; mask_arr_bev[:, 1] -= y1  # remember remove the offsets in x / y
            for [ptx, pty] in mask_arr_bev: cv2.circle(img_cv2_plot, (int(ptx), int(pty)), 2, color_mask_edge, -1, cv2.LINE_AA)
            cpt_p, cpt_o, corners_p, corners_o, grid_nodes_p, grid_cells_p = find_four_corners_of_a_rectangle_container(mask_arr_bev)
            
        # img_cv2_plot_pad = cv2.copyMakeBorder(img_cv2_plot.copy(), top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
        # img_cv2_project = cv2.warpPerspective(img_cv2_plot_pad, trans_mat, (rect_l, rect_w))  # (540, 960) --> sub_area (650, 1000)
        img_cv2_full[y1:y2, x1:x2] = img_cv2_plot.copy()  # sub image (500, 800) --> full image (540, 960)
        cv2.imwrite(os.path.join(img_folder, img_name[:-4]+f"_contour{args.view_mode}.jpg"), img_cv2_full)
        img_cv2_project = cv2.warpPerspective(img_cv2_full, trans_mat, (rect_l, rect_w))  # (540, 960) --> sub_area (650, 1000)
     
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
        labels = ["TL", "TR", "BR", "BL"]  # blue,green,red,purple --> TL TR BR BL
        
        img_cv2_p_vis = img_cv2_project.copy()
        for i, pt in enumerate(corners_p):
            cv2.line(img_cv2_p_vis, (int(cpt_p[0]), int(cpt_p[1])), (int(pt[0]), int(pt[1])), (192,192,192), 3)
            cv2.circle(img_cv2_p_vis, (int(pt[0]), int(pt[1])), 6, colors[i], -1)
            cv2.putText(img_cv2_p_vis, labels[i], (int(pt[0])+15, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.88, colors[i], 2)
        cv2.circle(img_cv2_p_vis, (int(cpt_p[0]), int(cpt_p[1])), 8, (255,255,255), -1)  # mask center point
        
        img_cv2_o_vis = img_cv2_plot.copy()
        for i, pt in enumerate(corners_o):
            cv2.line(img_cv2_o_vis, (int(cpt_o[0]), int(cpt_o[1])), (int(pt[0]), int(pt[1])), (192,192,192), 3)
            cv2.circle(img_cv2_o_vis, (int(pt[0]), int(pt[1])), 6, colors[i], -1)
            cv2.putText(img_cv2_o_vis, labels[i], (int(pt[0])+12, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.72, colors[i], 2)
        cv2.circle(img_cv2_o_vis, (int(cpt_o[0]), int(cpt_o[1])), 8, (255,255,255), -1)  # mask center point
        
        img_cv2_p_cell = img_cv2_project.copy()
        for i, cell in enumerate(grid_cells_p):
            # color = np.random.randint(0, 192, (3,)).tolist()  # a random color
            cv2.polylines(img_cv2_p_cell, [cell], isClosed=True, color=(192,192,192), thickness=2)  # plot polygon
            M = cv2.moments(cell)  # compute the center point for writing the id number
            assert M["m00"] != 0, "We do not accept a grid cell with zero area!!!"
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.putText(img_cv2_p_cell, str(i), (cX-10, cY+5), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2) 
        for r in range(grid_nodes_p.shape[0]):
            for c in range(grid_nodes_p.shape[1]):
                cv2.circle(img_cv2_p_cell, (int(grid_nodes_p[r,c,0]), int(grid_nodes_p[r,c,1])), 4, (255,255,255), -1)
                
        # img_cv2_project = cv2.resize(img_cv2_project, ( int(rect_l*(y2-y1)/rect_w), (y2-y1) ) )
        # img_cv2_plot_full1 = np.hstack((img_cv2_plot, img_cv2_project))
        
        img_cv2_p_cell = cv2.resize(img_cv2_p_cell, ( int(rect_l*(y2-y1)/rect_w), (y2-y1) ) )
        img_cv2_plot_full_1 = np.hstack((img_cv2_plot, img_cv2_p_cell))
        img_cv2_p_vis = cv2.resize(img_cv2_p_vis, ( int(rect_l*(y2-y1)/rect_w), (y2-y1) ) )
        img_cv2_plot_full_2 = np.hstack((img_cv2_o_vis, img_cv2_p_vis))
        img_cv2_plot_full = np.vstack((img_cv2_plot_full_1, img_cv2_plot_full_2))
        
        cv2.imwrite(os.path.join(img_folder, img_name[:-4]+f"_vis{args.view_mode}.jpg"), img_cv2_plot_full)
        
def testing_deformable_cloth_func(args):
    img_folder = os.path.join(args.root_dir, args.sub_folder)
    
    trans_mat = cfg_dict["transform_mat_inv"]; trans_mat_t = cfg_dict["transform_mat"]
    rect_l, rect_w = cfg_dict['rect_l'], cfg_dict['rect_w']; colors_list = cfg_dict['colors_list']
    roi_bbox = cfg_dict["detection_roi_bbox"]
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']  # 540, 960
    top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
    colors_list = [(0,255,128), (255,255,0), (128,0,255), (0,128,128), (0,0,255)]  # lawn green, cyan, light magenta, light yellow, red
    
    file_name_list = os.listdir(img_folder)
    img_names = [i for i in file_name_list if "_rgb_L.jpg" in i]; img_names.sort()
    mask_names = [i for i in file_name_list if "_mask.jpg" in i]; mask_names.sort()
    json_names = [i for i in file_name_list if "_mask.json" in i]; json_names.sort()
    
    if "towel" in args.sub_folder: object_name = "towel"
    if "pants" in args.sub_folder: object_name = "pants"
    if "T-shirt" in args.sub_folder: object_name = "T-shirt"
    
    for tid, (img_name, mask_name, json_name) in enumerate(zip(img_names, mask_names, json_names)):
        print("\n", tid, img_name)
        
        img_cv2_full = cv2.imread(os.path.join(img_folder, img_name))  # shape is (540, 960, 3)
        mask_cv2_full = cv2.cvtColor(cv2.imread(os.path.join(img_folder, mask_name)), cv2.COLOR_BGR2GRAY)  # shape is (540, 960, 1)
        json_dict = json.load(open(os.path.join(img_folder, json_name), "r"))  # {"bbox": ..., "mask": ...}
        
        img_cv2_plot = img_cv2_full[y1:y2, x1:x2]; color_mask_edge = (0,255,255)  # yellow
        rect_bbox, mask_arr = json_dict["bbox"], json_dict["mask"]
        for [ptx, pty] in mask_arr: cv2.circle(img_cv2_plot, (int(ptx), int(pty)), 2, color_mask_edge, -1, cv2.LINE_AA)
        cpt_p, cpt_o, corners_p, corners_o = find_dense_corners_of_a_deformable_cloth(mask_arr, args.kpts_num, object_name)

        img_cv2_full[y1:y2, x1:x2] = img_cv2_plot.copy()  # sub image (500, 800) --> full image (540, 960)
        cv2.imwrite(os.path.join(img_folder, img_name[:-4]+f"_visEDGE-{object_name}.jpg"), img_cv2_full)
        img_cv2_project = cv2.warpPerspective(img_cv2_full, trans_mat, (rect_l, rect_w))  # (540, 960) --> sub_area (650, 1000)
    
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255),  (127, 0, 127)]
        
        img_cv2_p_vis = img_cv2_project.copy()
        for i, pt in enumerate(corners_p):
            cv2.line(img_cv2_p_vis, (int(cpt_p[0]), int(cpt_p[1])), (int(pt[0]), int(pt[1])), (192,192,192), 3)
            cv2.circle(img_cv2_p_vis, (int(pt[0]), int(pt[1])), 6, colors[i], -1)
            (tw, th), _ = cv2.getTextSize(str(i+1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.88, thickness=2)
            cv2.circle(img_cv2_p_vis, (int(pt[0]+tw), int(pt[1]-th)), 20, (127,127,127), thickness=-1)
            cv2.circle(img_cv2_p_vis, (int(pt[0]+tw), int(pt[1]-th)), 20, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(img_cv2_p_vis, str(i+1), (int(pt[0]+tw*0.5), int(pt[1]-th*0.5)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.88, color=colors[i], thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img_cv2_p_vis, (int(cpt_p[0]), int(cpt_p[1])), 8, (255,255,255), -1)  # mask center point
        
        img_cv2_o_vis = img_cv2_plot.copy()
        for i, pt in enumerate(corners_o):
            pt = [pt[0] - x1, pt[1] - y1]
            cv2.line(img_cv2_o_vis, (int(cpt_o[0]-x1), int(cpt_o[1]-y1)), (int(pt[0]), int(pt[1])), (192,192,192), 3)
            cv2.circle(img_cv2_o_vis, (int(pt[0]), int(pt[1])), 6, colors[i], -1)
            (tw, th), _ = cv2.getTextSize(str(i+1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.72, thickness=2)
            cv2.circle(img_cv2_o_vis, (int(pt[0]+tw), int(pt[1]-th)), 16, (127,127,127), thickness=-1)
            cv2.circle(img_cv2_o_vis, (int(pt[0]+tw), int(pt[1]-th)), 16, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(img_cv2_o_vis, str(i+1), (int(pt[0]+tw*0.5), int(pt[1]-th*0.5)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.72, color=colors[i], thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img_cv2_o_vis, (int(cpt_o[0]-x1), int(cpt_o[1]-y1)), 8, (255,255,255), -1)  # mask center point
              
        img_cv2_project = cv2.resize(img_cv2_project, ( int(rect_l*(y2-y1)/rect_w), (y2-y1) ) )
        img_cv2_plot_full_1 = np.hstack((img_cv2_plot, img_cv2_project))
        img_cv2_p_vis = cv2.resize(img_cv2_p_vis, ( int(rect_l*(y2-y1)/rect_w), (y2-y1) ) )
        img_cv2_plot_full_2 = np.hstack((img_cv2_o_vis, img_cv2_p_vis))
        img_cv2_plot_full = np.vstack((img_cv2_plot_full_1, img_cv2_plot_full_2))
        
        cv2.imwrite(os.path.join(img_folder, img_name[:-4]+f"_visFULL-{object_name}.jpg"), img_cv2_plot_full)
        
def testing_deformable_rope_func(args):
    img_folder = os.path.join(args.root_dir, args.sub_folder)
    
    trans_mat = cfg_dict["transform_mat_inv"]; trans_mat_t = cfg_dict["transform_mat"]
    rect_l, rect_w = cfg_dict['rect_l'], cfg_dict['rect_w']; colors_list = cfg_dict['colors_list']
    roi_bbox = cfg_dict["detection_roi_bbox"]
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']  # 540, 960
    top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
    colors_list = [(0,255,128), (255,255,0), (128,0,255), (0,128,128), (0,0,255)]  # lawn green, cyan, light magenta, light yellow, red
    
    file_name_list = os.listdir(img_folder)
    img_names = [i for i in file_name_list if "_rgb_L.jpg" in i]; img_names.sort()
    mask_names = [i for i in file_name_list if "_mask.jpg" in i]; mask_names.sort()
    json_names = [i for i in file_name_list if "_mask.json" in i]; json_names.sort()
  
    
    return None
   
#################################################################

# python src/algs/slots_v2.py --root_dir ./debug/test_pcdVLM_rect_container/ --sub_folder test_OBJ-testing --view_mode BEV
# python src/algs/slots_v2.py --root_dir ./debug/test_pcdVLM_rect_container/ --sub_folder test_OBJ-basket_ID-01 --view_mode BEV
# python src/algs/slots_v2.py --root_dir ./debug/test_pcdVLM_rect_container/ --sub_folder test_OBJ-box_ID-01 --view_mode box --view_mode BEV
# python src/algs/slots_v2.py --root_dir ./debug/test_pcdVLM_rect_container/ --sub_folder test_OBJ-tray_ID-01 --view_mode tray --view_mode BEV

# python src/algs/slots_v2.py --root_dir ./debug/test_pcdVLM_rect_container/ --sub_folder test_OBJ-basket_ID-05 --view_mode BEV --infill_mask

# python src/algs/slots_v2.py --root_dir ./debug/test_pcdVLM_deformable_cloth/ --sub_folder test_OBJ-towel --kpts_num 4
# python src/algs/slots_v2.py --root_dir ./debug/test_pcdVLM_deformable_cloth/ --sub_folder test_OBJ-T-shirt --kpts_num 7
# python src/algs/slots_v2.py --root_dir ./debug/test_pcdVLM_deformable_cloth/ --sub_folder test_OBJ-towel_ID-01 --kpts_num 4
# python src/algs/slots_v2.py --root_dir ./debug/test_pcdVLM_deformable_cloth/ --sub_folder test_OBJ-T-shirt_ID-01 --kpts_num 7

# python src/algs/slots_v2.py --root_dir ./debug/test_pcdVLM_deformable_cloth/ --sub_folder test_OBJ-cable

#################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default="./debug/test_pcdVLM_rect_container/")
    parser.add_argument('--sub_folder', type=str, default="test_OBJ-basket_ID-01", help="sub folder name")
    parser.add_argument('--kpts_num', type=int, default=4, help="the default number of contour points")
    parser.add_argument('--view_mode', type=str, default="BEV", help="view via BEV or EDGE")
    parser.add_argument('--infill_mask', action='store_true', help="whether to infill BEV mask before processing") 
    args = parser.parse_args()   
    
    if "rect_container" in args.root_dir:  testing_rectangle_container_func(args)
    if "deformable_cloth" in args.root_dir:  testing_deformable_cloth_func(args)
    
    if "deformable_cloth" in args.root_dir and "cable" in args.sub_folder: testing_deformable_rope_func(args)

