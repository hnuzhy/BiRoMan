
import os
import sys
import math
import cv2
import json
import numpy as np

# pip install ultralytics
# pip install -U ultralytics
from ultralytics import YOLOWorld, SAM, FastSAM, YOLOE
from ultralytics.data.utils import polygon2mask


###################################################################
def ov_det_seg_yoloe_slim(model, img_path, imgsz=1080, is_save_plot=False, conf=0.5):
    
    ''' Execute inference with the YOLOE model on the specified image'''
    # results = model.predict(img_path)
    results = model.predict(source=img_path, show_labels=True, show_conf=True, device="cuda",
        conf=conf, iou=0.75, imgsz=imgsz)  # <<==== note the conf and iou
    
    ''' Show results '''
    # results[0].show()
    if is_save_plot: results[0].save(filename=img_path[:-4]+"_yoloe.jpg")  # "xxxx.jpg"
    im_bgr = results[0].plot()  # BGR-order numpy array

    # print(results[0])
    res_bboxes, res_names, res_masks = results[0].boxes, results[0].names, results[0].masks
    obj_cls_list = [int(i) for i in res_bboxes.cls.cpu().numpy().tolist()]  # integer
    obj_conf_list = res_bboxes.conf.cpu().numpy().tolist()  # float
    obj_bbox_list = res_bboxes.xyxy.cpu().numpy().tolist()  # bbox list
    obj_mask_list = res_masks.xy  # shape (N, 2), it indicates a sequence of 2D pixel coordinate
    final_res_list = []
    for cls, conf, bbox, mask in zip(obj_cls_list, obj_conf_list, obj_bbox_list, obj_mask_list):
        obj_name = res_names[cls]  # may be not correct
        if "table" in obj_name: continue
        print("open_vocabulary_detector (yoloe):", obj_name, conf, bbox, mask.shape)
        final_res_list.append([obj_name, conf, bbox, mask])  # bbox format [x1, y1, x2, y2]
    return final_res_list, im_bgr


def ov_det_seg_yoloe(img_path, imgsz=1080, is_save_plot=False, conf=0.5, prompts=None):
    # https://docs.ultralytics.com/models/yoloe/
    
    ''' Initialize a YOLOE model'''
    if prompts is None:  # (PF) Prompt-Free
        # model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11s-seg-pf.pt")
        model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11m-seg-pf.pt")
        # model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11l-seg-pf.pt")
    else:  # with Prompt (language/text or vision/image)
        # need to install https://github.com/ultralytics/CLIP and  https://github.com/apple/ml-mobileclip
        # model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11s-seg.pt")
        model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11m-seg.pt")
        # model = YOLOE("/home/dexforce/zhouhuayi/projects/ultralytics/yoloe-11l-seg.pt")
        model.set_classes(prompts, model.get_text_pe(prompts))  # Set text prompt. e.g., prompts = ["person", "bus"]

    ''' Execute inference with the YOLOE model on the specified image'''
    # results = model.predict(img_path)
    results = model.predict(source=img_path, show_labels=True, show_conf=True, device="cuda",
        conf=conf, iou=0.75, imgsz=imgsz)  # <<==== note the conf and iou
    
    ''' Show results '''
    # results[0].show()
    if is_save_plot: results[0].save(filename=img_path[:-4]+"_yoloe.jpg")  # "xxxx.jpg"
    im_bgr = results[0].plot()  # BGR-order numpy array

    # print(results[0])
    res_bboxes, res_names, res_masks = results[0].boxes, results[0].names, results[0].masks
    obj_cls_list = [int(i) for i in res_bboxes.cls.cpu().numpy().tolist()]  # integer
    obj_conf_list = res_bboxes.conf.cpu().numpy().tolist()  # float
    obj_bbox_list = res_bboxes.xyxy.cpu().numpy().tolist()  # bbox list
    obj_mask_list = res_masks.xy  # shape (N, 2), it indicates a sequence of 2D pixel coordinate
    final_res_list = []
    for cls, conf, bbox, mask in zip(obj_cls_list, obj_conf_list, obj_bbox_list, obj_mask_list):
        obj_name = res_names[cls]  # may be not correct
        if "table" in obj_name: continue
        print("open_vocabulary_detector (yoloe):", obj_name, conf, bbox, mask.shape)
        final_res_list.append([obj_name, conf, bbox, mask])  # bbox format [x1, y1, x2, y2]
    return final_res_list, im_bgr


###################################################################

def calInsideRectIOU(rectLarge, rectSmall):
    # calculate two rectangles IOU(intersection-over-union)
    [Ax0, Ay0, Ax1, Ay1] = rectLarge[0:4]
    [Bx0, By0, Bx1, By1] = rectSmall[0:4]
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        return crossArea/areaB

def calTwoRectIOU(rectA, rectB):
    # calculate two rectangles IOU(intersection-over-union)
    [Ax0, Ay0, Ax1, Ay1] = rectA[0:4]
    [Bx0, By0, Bx1, By1] = rectB[0:4]
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        return crossArea/(areaA + areaB - crossArea)

###################################################################

def post_processing_yoloe_results(res_list, task_name, roi_bbox, ref_areas=None):
    if task_name == "plugpen": obj_list = ["pen", "cap"]  # without prompts, need to split them
    elif task_name == "reorient": obj_list = ["spoon", "shovel"] # without prompts, spoon or shovel
    elif task_name == "unscrew": obj_list = ["bottle"]
    elif task_name == "pouring": obj_list = ["cup", "bottle"]
    elif task_name == "inserting": obj_list = ["cup", "pen"]
    elif task_name == "pressing": obj_list = ["cup", "bottle"]
    elif task_name == "reorient_unscrew": obj_list = ["bottle"]  # without prompts, lying down bottle
    elif task_name == "unscrew_pouring": obj_list = ["cup", "bottle"]
    elif task_name == "tool_spoon": obj_list = ["spoon", "bowlL", "bowlS"]
    elif task_name == "tool_funnel": obj_list = ["funnel", "bottle", "cup"]
    
    elif task_name == "cirbowl": obj_list = ["bowl"]
    elif task_name == "rectbox": obj_list = ["box"]
    elif task_name == "bottle": obj_list = ["bottle"]
    
    else: print("You must give a defined task name!!!"); sys.exit()
    
    [rx1, ry1, rx2, ry2] = roi_bbox
    
    final_res_list = []
    if task_name in ["plugpen"]:  # mugpen and mugcap can not be detected by yoloe with prompts
        if ref_areas is None:
            res_list_left, max_area = [], 0
            for [obj_name, conf, bbox, mask] in res_list:  # results of yoloe
                [x1, y1, x2, y2] = bbox  # bbox format [x1, y1, x2, y2]
                if x1 < 5 or y1 < 5 or x2 > rx2-rx1-5: continue  # not the ideal object (around the bounding)
                else:
                    temp_area = (y2-y1)*(x2-x1)
                    res_list_left.append([conf, bbox, mask, temp_area])
                    if temp_area > max_area: max_area = temp_area
            for [conf, bbox, mask, temp_area] in res_list_left:  # firstly, find the longer marker pen
                if temp_area == max_area: final_res_list.append([obj_list[0], conf, bbox, mask]); break
            for [conf, bbox, mask, temp_area] in res_list_left:  # then, find the shorter marker pencap
                temp_iou = calTwoRectIOU(bbox, final_res_list[0][-2])  # pencap should not be overlapped with pen
                if temp_iou == 0: final_res_list.append([obj_list[1], conf, bbox, mask]); break
        else:
            res_list_left = []
            for [obj_name, conf, bbox, mask] in res_list:  # results of yoloe
                [x1, y1, x2, y2] = bbox  # bbox format [x1, y1, x2, y2]
                if x1 < 5 or y1 < 5 or x2 > rx2-rx1-5: continue  # not the ideal object (around the bounding)
                else:
                    obj_area = cv2.countNonZero(polygon2mask((ry2-ry1, rx2-rx1), [mask], color=255, downsample_ratio=1))
                    res_list_left.append([conf, bbox, mask, obj_area])
            area_pen, pen_idx = ref_areas["pen"], -1
            for idx, [conf, bbox, mask, obj_area] in enumerate(res_list_left):  # firstly, find the most possible marker pen
                if obj_area > area_pen * 0.6 and obj_area < area_pen * 1.4:
                    final_res_list.append(["pen", conf, bbox, mask]); pen_idx = idx; break
            area_cap = ref_areas["cap"]
            for idx, [conf, bbox, mask, obj_area] in enumerate(res_list_left):  # then, find the most possible marker pencap
                if obj_area > area_cap * 0.6 and obj_area < area_cap * 1.4 and idx != pen_idx:
                    final_res_list.append(["cap", conf, bbox, mask]); break
    elif task_name in ["tool_spoon"]:  # two bowls (a large one and a small one) with one spoon
        for [obj_name, conf, bbox, mask] in res_list:  # results of yoloe
            if obj_name == "spoon": final_res_list.append([obj_name, conf, bbox, mask]); break
        res_bowls = []
        for [obj_name, conf, bbox, mask] in res_list:  # results of yoloe
            if obj_name == "bowl": 
                [x1, y1, x2, y2] = bbox
                res_bowls.append([obj_name, conf, bbox, mask, (y2-y1)*(x2-x1)])
            if len(res_bowls) == 2: break  # we only care about the first two bowls
        if res_bowls[0][-1] > res_bowls[1][-1]:
            final_res_list.append(["bowlL", res_bowls[0][1], res_bowls[0][2], res_bowls[0][3]])
            final_res_list.append(["bowlS", res_bowls[1][1], res_bowls[1][2], res_bowls[1][3]])
        else:
            final_res_list.append(["bowlS", res_bowls[0][1], res_bowls[0][2], res_bowls[0][3]])
            final_res_list.append(["bowlL", res_bowls[1][1], res_bowls[1][2], res_bowls[1][3]])
    elif task_name in ["tool_funnel"]:  # the funnel will always be recognized as bottle
        found_ids = []
        for idx, [obj_name, conf, bbox, mask] in enumerate(res_list):  # results of yoloe
            if obj_name not in obj_list:  continue  # we only care about manipulated objects
            [x1, y1, x2, y2] = bbox
            if x1 < 5 or y1 < 5 or x2 > rx2-rx1-5: continue  # not the ideal object (around the bounding)
            final_res_list.append([obj_name, conf, bbox, mask])  # find the real bottle and cup
            obj_list.remove(obj_name)  # treat the most confidented instance as the target object
            found_ids.append(idx)
        for idx, [obj_name, conf, bbox, mask] in enumerate(res_list):  # results of yoloe
            [x1, y1, x2, y2] = bbox
            if x1 < 5 or y1 < 5 or x2 > rx2-rx1-5: continue  # not the ideal object (around the bounding)
            if (idx not in found_ids) and (obj_name == "bottle") and ((y2-y1)/(x2-x1) < 2.0):  # find the possible funnel
                final_res_list.append(["funnel", conf, bbox, mask]); break
    else:       
        for [obj_name, conf, bbox, mask] in res_list:  # results of yoloe
            if obj_name not in obj_list:  continue  # we only care about manipulated objects
            [x1, y1, x2, y2] = bbox  # bbox format [x1, y1, x2, y2]
            if x1 < 5 or y1 < 5 or x2 > rx2-rx1-5: continue  # not the ideal object (around the bounding)
            final_res_list.append([obj_name, conf, bbox, mask])
            obj_list.remove(obj_name)  # treat the most confidented instance as the target object
        
    return final_res_list

###################################################################
def plot_processed_results_vis(sub_img, res_list, cls_list):
    colors_list = [(0,255,128), (255,255,0), (128,0,255), (0,128,128), (0,0,255)]  # lawn green, cyan, light magenta, light yellow, red
    
    for obj_idx, [obj_name, conf, bbox, mask] in enumerate(res_list):
        [img_h, img_w, img_c] = sub_img.shape
        [x1, y1, x2, y2] = bbox
        color = colors_list[cls_list.index(obj_name)]
        # obj_binary_mask = polygon2mask((img_h, img_w), [mask], color=128, downsample_ratio=1)
        # sub_img[:, :, 0] = sub_img[:, :, 0] + obj_binary_mask
        # sub_img[:, :, 1] = sub_img[:, :, 1] + obj_binary_mask
        # sub_img[:, :, 2] = sub_img[:, :, 2] + obj_binary_mask
        cv2.rectangle(sub_img, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=2)
        (tw, th), _ = cv2.getTextSize(obj_name, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.64, thickness=1)
        cv2.rectangle(sub_img, (int(x1), int(y1-th)-5), (int(x1+tw), int(y1)), color=color, thickness=-1)
        cv2.putText(sub_img, obj_name, (int(x1), int(y1)-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.64, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)

    return sub_img

###################################################################

if __name__ == "__main__": 
    
    
    ###########################################################################
    ''' testing the function compute_rotation_by_image_moments()
    # top-left, top-right, bottom-right, bottom-left --> [260, 86], [652, 84], [761, 535], [168, 538]. length*width --> 616mm * 675mm
    four_pts_list = [[260, 86], [652, 84], [761, 535], [168, 538]]  # (pts_tl, pts_tr, pts_br, pts_bl) in the affine transformed camera-view image 
    rect_l, rect_w = 616, 675  # the length / width of the top-viewd rectangle
    tgt_pts_list = [[0, 0], [rect_l-1, 0], [rect_l-1, rect_w-1], [0, rect_w-1]]  # (pts_tl, pts_tr, pts_br, pts_bl) in top-viewed rectangle platform
    
    selected_four_corners = np.array(four_pts_list, dtype=np.float32)
    show_window_corners = np.array(tgt_pts_list, dtype=np.float32)
    transform_mat = cv2.getPerspectiveTransform(show_window_corners, selected_four_corners)  # obtain the transform function
    print("top-view reporjection transform_mat:\n", transform_mat)
    transform_mat_inv = np.linalg.inv(transform_mat)

    ###########################################################################
    test_folder_dir = "/home/dexforce/zhouhuayi/projects/WiLoR_CL/backup/measuretrans/"
    task_folders = [
        "reorient/anyobj01_wooden_calib_board/",  # not suitable
        "reorient/anyobj02_plastic_yellow_blade/",  # not suitable
        "reorient/anyobj03_metal_long_spoon/",  # suitable
        "reorient/anyobj04_metal_long_blade/",  # suitable
        "reorient/anyobj05_plastic_pink_blade/",  # suitable
        "reorient/anyobj06_metal_short_spoon/",  # suitable
        "reorient/anyobj07_plastic_gray_spoon/",  # suitable
        "reorient/anyobj08_plastic_red_blade/"  # suitable
    ]

    for task_folder in task_folders:
        print("\n", task_folder)
        file_name_list = os.listdir(os.path.join(test_folder_dir, task_folder))
        json_names = [jn for jn in file_name_list if ".json" in jn]
        json_names.sort()
        
        for index, json_name in enumerate(json_names):
            json_file_path = os.path.join(test_folder_dir, task_folder, json_name)
            contour_points_list = json.load(open(json_file_path, "r"))
            img_file_path = json_file_path.replace("mask_unknown.json", "rect.jpg") 
            theta_deg, pts_raw, pts_trans = compute_rotation_by_image_moments(
                np.array(contour_points_list), trans_mat=transform_mat_inv, img_file_path=img_file_path, task_name="reorient" )
            print(index, json_name, theta_deg)
    '''
    ###########################################################################
    

    test_folder_dir = "/home/dexforce/zhouhuayi/auboHandeyeCalib/scripts_kfr/seedinit"
    task_folder_jpgs = [
        ["plugpen/marker02_seed.jpg", ["pen"]],  # prompt is ineffective
        ["plugpen/marker03_seed.jpg", ["pen"]],  # prompt is ineffective
        ["plugpen/marker06_seed.jpg", ["pen"]],  # prompt is ineffective
        
        # ["pouring/bottle01-mugcup03_seed.jpg", ["bottle", "cup"]],
        # ["pouring/bottle05-mugcup03_seed.jpg", ["bottle", "cup"]],
        # ["pouring/bottle06-mugcup01_seed.jpg", ["bottle", "cup"]],
        # ["pouring/bottle08-mugcup01_seed.jpg", ["bottle", "cup"]],
        # ["pouring/bottle08-mugcup03_seed.jpg", ["bottle", "cup"]],
        
        # ["unscrew/bottle01_seed.jpg", ["bottle"]],
        # ["unscrew/bottle02_seed.jpg", ["bottle"]],
        # ["unscrew/bottle03_seed.jpg", ["bottle"]],
        # ["unscrew/bottle04_seed.jpg", ["bottle"]],
        # ["unscrew/bottle05_seed.jpg", ["bottle"]],
        # ["unscrew/bottle06_seed.jpg", ["bottle"]],
        # ["unscrew/bottle07_seed.jpg", ["bottle"]],
        # ["unscrew/bottle08_seed.jpg", ["bottle"]],
        
        # ["reorient/anyobj01_seed.jpg", ["spoon"]],
        # ["reorient/anyobj02_seed.jpg", ["spoon"]],
        # ["reorient/anyobj03_seed.jpg", ["spoon"]],
        # ["reorient/anyobj04_seed.jpg", ["shovel"]],
        # ["reorient/anyobj05_seed.jpg", ["spoon"]],
        # ["reorient/anyobj06_seed.jpg", ["spoon"]],
        # ["reorient/anyobj07_seed.jpg", ["shovel"]],
        # ["reorient/anyobj08_seed.jpg", ["shovel"]],
        
        ["reorient_unscrew/bottle03_seed.jpg", ["bottle"]],  # prompt is ineffective 
        ["reorient_unscrew/bottle04_seed.jpg", ["bottle"]],  # prompt is ineffective
        ["reorient_unscrew/bottle06_seed.jpg", ["bottle"]],  # prompt is ineffective
        ["reorient_unscrew/bottle07_seed.jpg", ["bottle"]],  # prompt is ineffective
    ]
    
    for [task_folder_jpg, given_cls_names] in task_folder_jpgs:
        img_path = os.path.join(test_folder_dir, task_folder_jpg)
        final_res_list, im_bgr = ov_det_seg_yoloe(img_path, imgsz=640, conf=0.2, prompts=None)
        # final_res_list, im_bgr = ov_det_seg_yoloe(img_path, imgsz=640, conf=0.2, prompts=given_cls_names)
        
    ###########################################################################