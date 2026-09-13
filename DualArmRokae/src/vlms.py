
'''
############################################################################################################################
https://medium.com/@revonia64/%E5%9C%A8%E6%9C%AC%E5%9C%B0%E8%BF%90%E8%A1%8C-florence2-sam2-%E4%BD%BF%E7%94%A8-prompt-%E5%88%86%E5%89%B2%E7%9B%AE%E6%A0%87-58c130ae6726
https://huggingface.co/spaces/SkalskiP/florence-sam/tree/main
https://huggingface.co/microsoft/Florence-2-base
https://github.com/facebookresearch/segment-anything-2

$ conda activate base
$ pip install einops spaces timm transformers samv2 gradio supervision pytest

# [step 1] the flash_attn lib should be installed separately (this will takes several hours)
$ pip install flash_attn

# [step 2] the default version has bugs (https://github.com/FasterDecoding/Medusa/issues/98)
$ pip install transformers==4.34.1

# [step 3] still having bugs when using pypi (https://github.com/huggingface/transformers/issues/32129)
$ pip uninstall transformers
$ pip install git+https://github.com/huggingface/transformers

# [step 4] install from git which is unavailable in China (or too damn slow).
# try to install another old version released at [May 24, 2024], which is the same as florence-sam 
$ pip install transformers==v4.41.2

############################################################################################################################

# re-install these libs with using python 3.9 for keeping compatible with kingfisher-R-6000
$ conda create -n py39 python=3.9
$ conda activate py39
$ pip install kingfisher-0.1.1-cp39-cp39-linux_x86_64.whl

$ pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
$ pip install einops spaces timm transformers samv2 gradio supervision pytest
$ pip install flash_attn==2.7.4.post1
$ pip install transformers==v4.41.2

#############################################################

# Also for the rokaeDemo project archived in another computer in 2025-08-26
$ conda create -n py310 python=3.10
$ conda activate py310
$ pip install kingfisher-0.1.1-cp310-cp310-linux_x86_64.whl

$ pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
$ pip install einops spaces timm transformers samv2 gradio supervision pytest
$ pip install flash_attn==2.7.4.post1
$ pip install transformers==v4.41.2


############################################################################################################################
'''

import os
import cv2
import spaces
import numpy as np
import supervision as sv
import torch
from PIL import Image

try:
    from utils.sam import load_sam_image_model, run_sam_inference
    from utils.florence import load_florence_model, run_florence_inference
    from utils.florence import FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
    from config import cfg_dict_init
except:
    from .utils.sam import load_sam_image_model, run_sam_inference
    from .utils.florence import load_florence_model, run_florence_inference
    from .utils.florence import FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
    from .config import cfg_dict_init
    
#############################################################
DEVICE = torch.device("cuda")
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
#############################################################

FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)

#############################################################
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#000000"),
    border_radius=5)
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX)
#############################################################

def annotate_image(image, detections):
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image

def ov_detection_segmentation(image_input, text_input):
    # image_input is in PIL format; text_input is in text string format
    
    texts = [prompt.strip() for prompt in text_input.split(",")]
    detections_list = []
    for text in texts:
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
            text=text
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        detections_list.append(detections)

    detections = sv.Detections.merge(detections_list)
    detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
    img_vis = annotate_image(image_input, detections)
    
    det_results = []
    for det_idx, detection in enumerate(detections):  # detection is in tuple format
        cls_name = detection[-1]["class_name"]
        bbox = detection[0]  # in xyxy array format with shape (4,)
        mask = detection[1]  # in binary image format with shape (w, h)
        det_results.append([cls_name, bbox, mask])
        print("[Original Results]", det_idx, cls_name, bbox, mask.shape)
    return det_results, img_vis

#############################################################
def post_processing_det_seg_results(det_results, img_vis, roi_bbox, task_name_str=None):
    
    if task_name_str is not None:
        assert task_name_str in ["cirbowl", "rectbox", "ordcup", "mugcup", "bottle",
            "unscrew", "pouring", "twistpour", "sweeping" ], "An illegal task name!!!"

    [rx1, ry1, rx2, ry2] = roi_bbox
    img_vis_cv2 = np.array(img_vis)[:, :, ::-1]  # PIL -> cv2. np.array + RGB2BGR
    
    det_results_new = []
    for idx, det_list in enumerate(det_results):
        [cls_name, bbox, mask] = det_list  # Note: one category may have multiple instances
        [x1, y1, x2, y2] = bbox  # bbox format [x1, y1, x2, y2]
        if x1 < 5 or y1 < 5 or x2 > rx2-rx1-5: continue  # not the ideal object (around the bounding)
        binary_mask = np.array(mask*255, dtype=np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        polygon_mask = [np.array(polygon).squeeze() for polygon in contours]
        det_results_new.append([cls_name, bbox, binary_mask, polygon_mask])
    
    return det_results_new, img_vis_cv2

#############################################################   
def conduct_object_detect_and_segment(raw_img_input, prompts_str=None, is_raw_result=False, given_roi_bbox=None):
    if given_roi_bbox is None:
        roi_bbox, roi_bbox_2 = cfg_dict_init["detection_roi_bbox"], cfg_dict_init["detection_roi_bbox_2"]
        [x1, y1, x2, y2], [x21, y21, x22, y22] = roi_bbox, roi_bbox_2  # for fast and stable detection and segmentation
    else:
        roi_bbox = given_roi_bbox
        [x1, y1, x2, y2] = given_roi_bbox

    image_input_sub = raw_img_input[y1:y2, x1:x2]
    det_res_test, img_vis_test = ov_detection_segmentation(Image.fromarray(image_input_sub[:, :, ::-1]), prompts_str)  # CV2 --> PIL
    det_res_test_new, img_vis_test_cv2 = post_processing_det_seg_results(det_res_test, img_vis_test, roi_bbox)
    if len(det_res_test_new) == 0:  # re-detect and re-segment again with smaller ROI
        if given_roi_bbox is not None:
            print("*****[Do not detect any object in the given_roi_bbox !!!]*****")
            return det_res_test_new, img_vis_test_cv2
        print("*****[Re-Conduct the detection and segmentation again!!!]*****")
        image_input_sub2 = raw_img_input[y21:y22, x21:x22]
        det_res_test, img_vis_test = ov_detection_segmentation(Image.fromarray(image_input_sub2[:, :, ::-1]), prompts_str)  # CV2 --> PIL
        det_res_test_new, img_vis_test_cv2 = post_processing_det_seg_results(det_res_test, img_vis_test, roi_bbox_2)

        cur_obj_mask = det_res_test_new[0][-1][0]  # det_res_test_new is a list of [cls_name, bbox, binary_mask, polygon_mask]
        cur_obj_mask[:, 0] += (x21 - x1); cur_obj_mask[:, 1] += (y21 - y1)
        det_res_test_new[0][-1][0] = cur_obj_mask  # remember add-back and re-adjust the offsets in x / y
        cur_obj_bbox = det_res_test_new[0][1]
        cur_obj_bbox[0] += (x21 - x1); cur_obj_bbox[2] += (x21 - x1); cur_obj_bbox[1] += (y21 - y1); cur_obj_bbox[3] += (y21 - y1)
        det_res_test_new[0][1] = cur_obj_bbox  # remember add-back and re-adjust the offsets in x / y
        cur_obj_b_mask = det_res_test_new[0][2]
        top, bottom, left, right = y21 - y1, y2-y22, x21 - x1, x2 - x22
        cur_obj_b_mask = cv2.copyMakeBorder(cur_obj_b_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
        det_res_test_new[0][2] = cur_obj_b_mask  # remember add-back and re-adjust the offsets in x / y

        img_temp = image_input_sub.copy(); img_temp[y21-y1:y22-y1, x21-x1:x22-x1] = img_vis_test_cv2; img_vis_test_cv2 = img_temp
        cv2.rectangle(img_vis_test_cv2, (int(x21-x1), int(y21-y1)), (int(x22-x1), int(y22-y1)), color=(255,255,255), thickness=2)
        if len(det_res_test_new) > 1:  # there may be multiple target objects in the list
            for idx in range(len(det_res_test_new) - 1):
                cur_obj_mask_temp = det_res_test_new[1+idx][-1][0]
                cur_obj_mask_temp[:, 0] += (x21 - x1); cur_obj_mask_temp[:, 1] += (y21 - y1)
                det_res_test_new[1+idx][-1][0] = cur_obj_mask_temp  # remember add-back and re-adjust the offsets in x / y
                cur_obj_bbox_temp = det_res_test_new[1+idx][1]
                cur_obj_bbox_temp[0] += (x21 - x1); cur_obj_bbox_temp[2] += (x21 - x1); cur_obj_bbox_temp[1] += (y21 - y1); cur_obj_bbox_temp[3] += (y21 - y1)
                det_res_test_new[1+idx][1] = cur_obj_bbox_temp  # remember add-back and re-adjust the offsets in x / y
                cur_obj_b_mask_temp = det_res_test_new[1+idx][2]
                cur_obj_b_mask_temp = cv2.copyMakeBorder(cur_obj_b_mask_temp, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
                det_res_test_new[1+idx][2] = cur_obj_b_mask_temp  # remember add-back and re-adjust the offsets in x / y 
    else:
        # cur_obj_mask = det_res_test_new[0][-1][0]  # det_res_test_new is a list of [cls_name, bbox, binary_mask, polygon_mask]
        cur_obj_mask_list = det_res_test_new[0][-1]
        max_mask_idx, max_mask_pts_num = -1, 0
        for idx_temp, cur_obj_mask_temp in enumerate(cur_obj_mask_list):
            if len(cur_obj_mask_temp) > max_mask_pts_num:
                max_mask_pts_num = len(cur_obj_mask_temp)
                max_mask_idx = idx_temp
        cur_obj_mask = cur_obj_mask_list[max_mask_idx]
    
    if is_raw_result:
        return det_res_test_new, img_vis_test_cv2
    else:
        return cur_obj_mask, img_vis_test_cv2


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

def plot_processed_results_vis(sub_img, res_list, cls_list, is_plot_name=True):
    colors_list = [(0,255,128), (255,255,0), (128,0,255), (0,128,128), (0,0,255)]  # lawn green, cyan, light magenta, light yellow, red
    
    for obj_idx, [obj_name, bbox, obj_binary_mask, multi_masks] in enumerate(res_list):
        mask = multi_masks[0]
        [img_h, img_w, img_c] = sub_img.shape
        [x1, y1, x2, y2] = bbox
        color = colors_list[cls_list.index(obj_name)]

        # obj_binary_mask = polygon2mask((img_h, img_w), [mask], color=128, downsample_ratio=1)
        # sub_img[:, :, 0] = sub_img[:, :, 0] + obj_binary_mask
        # sub_img[:, :, 1] = sub_img[:, :, 1] + obj_binary_mask
        # sub_img[:, :, 2] = sub_img[:, :, 2] + obj_binary_mask

        # cv2.rectangle(sub_img, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=2)

        for [ptx, pty] in mask: cv2.circle(sub_img, (int(ptx), int(pty)), 2, color, -1, cv2.LINE_AA)

        if is_plot_name:
            # if obj_name == "pen": obj_name = "brush"  # spoon / brush / shovel / syringe / brush
            (tw, th), _ = cv2.getTextSize(obj_name, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.64, thickness=1)
            cv2.rectangle(sub_img, (int(x1), int(y1-th)-5), (int(x1+tw), int(y1)), color=color, thickness=-1)
            cv2.putText(sub_img, obj_name, (int(x1), int(y1)-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.64, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)

    return sub_img
#############################################################


'''
if __name__ == "__main__":

    test_folder_dir = "/home/dex/zhouhuayi/rokaeDemo/debug/test_ori/"
    task_folder_jpgs = [
        ["grasping_pencup_syn_L_pencup01_tid103-L_000310_enhanced-cut.jpg", "black cup"],
        ["grasping_pencup_syn_L_pencup01_tid103-L_000555_enhanced-cut.jpg", "black cup"],
        ["urm_t2_bowl_syn_L_cirbowl01_tid14-L_000270_enhanced-cut.jpg", "white bowl"],
        ["urm_t2_bowl_syn_L_cirbowl01_tid14-L_000450_enhanced-cut.jpg", "white bowl"],
        ["urm_t2_bowl_syn_L_cirbowl01_tid14-L_001520_enhanced-cut.jpg", "white bowl"],
        ["urm_t2_bowl_syn_L_cirbowl01_tid14-L_001665_enhanced-cut.jpg", "white bowl"],
    ]

    for jpg_idx, [task_folder_jpg, prompts] in enumerate(task_folder_jpgs):
        print("\n", jpg_idx, task_folder_jpg)
        img_path = os.path.join(test_folder_dir, task_folder_jpg)

        path_default = os.path.join(test_folder_dir, "../test_res", task_folder_jpg)
        save_vis_jpg_path = path_default.replace(".jpg", f"_DetSeg{str(jpg_idx+1).zfill(2)}.jpg")

        image_input_cv2 = cv2.imread(img_path)  #######################
        image_input = Image.fromarray(image_input_cv2[:, :, ::-1])  #######################
        
        text_input = prompts
        det_results, img_vis = ov_detection_segmentation(image_input, text_input)

        img_cv2 = np.array(img_vis)[:, :, ::-1]  # PIL -> cv2. np.array + RGB2BGR
        cv2.imwrite(save_vis_jpg_path, img_cv2)  

        res_list = []; cls_list = []; max_y_val = 0
        for det_list in det_results:
            [cls_name, bbox, mask] = det_list
            save_mask_path = save_vis_jpg_path[:-4] + f"_mask_{cls_name}.jpg"
            binary_mask = np.array(mask*255, dtype=np.uint8)

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            polygons = [np.array(polygon).squeeze() for polygon in contours]
            print(len(polygons[0]), polygons[0][:10])

            if max_y_val < bbox[-1]:
                max_y_val = bbox[-1]
                res_list = [[cls_name, bbox, mask, polygons]]
                cls_list = [cls_name]
                cv2.imwrite(save_mask_path, binary_mask)

        sub_img_vis = plot_processed_results_vis(image_input_cv2, res_list, cls_list, is_plot_name=False)
        cv2.imwrite(save_vis_jpg_path.replace(".jpg", "_single.jpg"), sub_img_vis)  

os._exit(0)
'''



#############################################################
if __name__ == "__main__":

    test_folder_dir = "/home/dex/zhouhuayi/rokaeDemo/debug/test_ori/"
    task_folder_jpgs = [

        # ["cirbowl_id01/img_seed.jpg", "blue bowl"],         # the large blue plastic bowl
        # ["cirbowl_id02/img_seed.jpg", "white bowl"],        # the large white paper bowl
        # ["cirbowl_id03/img_seed.jpg", "plastic bowl"],      # the middle transparent plastic bowl
        # ["cirbowl_id04/img_seed.jpg", "small bowl"],        # the small transparent plastic bowl
        # ["cirbowl_id05/img_seed.jpg", "green bowl"],        # the small green plastic bowl
        # ["cirbowl_id06/img_seed.jpg", "blue bowl"],         # the small blue plastic bowl
        # ["cirbowl_id07/img_seed.jpg", "brown bowl"],        # the small brown plastic bowl [not stable]
 
        # ["rectbox_id01/img_seed.jpg", "yellow box"],        # the yellow paper rectbox
        # ["rectbox_id02/img_seed.jpg", "blue box"],          # the blue paper rectbox
        # ["rectbox_id03/img_seed.jpg", "pink box"],          # the pink paper rectbox
        # ["rectbox_id04/img_seed.jpg", "paper box"],         # gray small-size paper rectbox
        # ["rectbox_id05/img_seed.jpg", "paper box"],         # gray middle-size paper rectbox
        # ["rectbox_id06/img_seed.jpg", "paper box"],         # gray lrage-size paper rectbox
        # ["rectbox_id07/img_seed.jpg", "brown paper box"],   # the middle paper lunch rectbox [not stable]
        # ["rectbox_id08/img_seed.jpg", "white lunchbox"],    # the middle plastic lunch rectbox [not stable]

        # ["ordcup_id01/img_seed.jpg", "yellow cup"],         # the yellow soft smooth plastic ordcup
        # ["ordcup_id02/img_seed.jpg", "brown cup"],          # the brown hard smooth plastic ordcup
        # ["ordcup_id03/img_seed.jpg", "blue cup"],           # the blue hard smooth plastic ordcup
        # ["ordcup_id04/img_seed.jpg", "gray cup"],           # the gray hard non-smooth plastic ordcup
        # ["ordcup_id05/img_seed.jpg", "blue cup"],           # the blue high plastic ordcup (slim container)
        # ["ordcup_id06/img_seed.jpg", "white cup"],          # the white high plastic ordcup (milk container)

        # ["mugcup_id01/img_seed.jpg", "blue cup,handle"],    # the middle blue plastic mugcup (used in LFHV) [handle][not stable]
        # ["mugcup_id02/img_seed.jpg", "yellow cup,handle"],  # the tallest porcelain mugcup (used in LFHV) 
        # ["mugcup_id03/img_seed.jpg", "green cup,handle"],   # the widest porcelain mugcup (used in LFHV)
        # ["mugcup_id04/img_seed.jpg", "yellow cup,handle"],  # the newly yellow plastic mugcup (used in LFHV)
        # ["mugcup_id05/img_seed.jpg", "white cup,handle"],   # the white transparent plastic mugcup [handle][not stable]
        # ["mugcup_id06/img_seed.jpg", "pink cup,handle"],    # the pink semi-transparent plastic mugcup [handle][not stable]

        # ["bottle_id01/img_seed.jpg", "bottle,cap"],         # the wide lvdou water bottle
        # ["bottle_id02/img_seed.jpg", "bottle,cap"],         # the high dongfang shuye bottle
        # ["bottle_id03/img_seed.jpg", "bottle,cap"],         # the dongfang shuye bottle (used in LFHV)
        # ["bottle_id04/img_seed.jpg", "bottle,cap"],         # the shuirongC bottle (used in LFHV)
        # ["bottle_id05/img_seed.jpg", "bottle,cap"],         # the dongningcha bottle (used in LFHV)
        # ["bottle_id06/img_seed.jpg", "bottle,cap"],         # the yanyulongjin bottle (used in LFHV)
        # ["bottle_id07/img_seed.jpg", "bottle,cap"],         # the jianjiao bottle (used in LFHV)
        # ["bottle_id08/img_seed.jpg", "bottle,cap"],         # the haizhiyan bottle (used in LFHV)
        
        
        # ["../src/test_img/img01_brush_dustpan.jpg", "brush"],          # A long-handled brush or broom which is used to sweep the floor.
        # ["../src/test_img/img01_brush_dustpan.jpg", "dustpan"],        # A flat container with a handle into which you brush dust and dirt.
        # ["../src/test_img/img02_brush_dustpan_bottle_mugcup_trash.jpg", "brush"],           # a cluster scene 1
        # ["../src/test_img/img02_brush_dustpan_bottle_mugcup_trash.jpg", "dustpan"],         # a cluster scene 1
        # ["../src/test_img/img02_brush_dustpan_bottle_mugcup_trash.jpg", "blue bottle"],     # a cluster scene 1
        # ["../src/test_img/img02_brush_dustpan_bottle_mugcup_trash.jpg", "green bottle"],    # a cluster scene 1
        # ["../src/test_img/img02_brush_dustpan_bottle_mugcup_trash.jpg", "blue cup"],        # a cluster scene 1
        # ["../src/test_img/img02_brush_dustpan_bottle_mugcup_trash.jpg", "green cup"],       # a cluster scene 1
        # ["../src/test_img/img02_brush_dustpan_bottle_mugcup_trash.jpg", "white paper"],     # a cluster scene 1
        
        # ["../src/test_img/img03_brush_dustpan_bottle_mugcup_trash.jpg", "brush"],           # a cluster scene 2
        # ["../src/test_img/img03_brush_dustpan_bottle_mugcup_trash.jpg", "dustpan"],         # a cluster scene 2
        # ["../src/test_img/img03_brush_dustpan_bottle_mugcup_trash.jpg", "blue bottle"],     # a cluster scene 2
        # ["../src/test_img/img03_brush_dustpan_bottle_mugcup_trash.jpg", "green bottle"],    # a cluster scene 2
        # ["../src/test_img/img03_brush_dustpan_bottle_mugcup_trash.jpg", "blue cup"],        # a cluster scene 2
        # ["../src/test_img/img03_brush_dustpan_bottle_mugcup_trash.jpg", "green cup"],       # a cluster scene 2
        # ["../src/test_img/img03_brush_dustpan_bottle_mugcup_trash.jpg", "white paper"],     # a cluster scene 2
        
        # ["../src/test_img/img04_brush_dustpan_bottle_mugcup_trash.jpg", "brush"],           # a cluster scene 3
        # ["../src/test_img/img04_brush_dustpan_bottle_mugcup_trash.jpg", "dustpan"],         # a cluster scene 3
        # ["../src/test_img/img04_brush_dustpan_bottle_mugcup_trash.jpg", "paper ball"],      # a cluster scene 3
        
        
        # ["wrapping_basket_id06_img_sL_test01_Wrapping.jpg", "blue basket,gray box"], 
        # ["wrapping_basket_id06_img_sL_test01_Grasping.jpg", "blue basket,gray box"], 
        
        # ["id02_img_test01_Grasping.jpg", "white bowl"],
        # ["id02_img_test01_Pivoting.jpg", "white bowl"],
        # ["id03_img_test01_Grasping.jpg", "transparent bowl"],
        # ["id03_img_test01_Pivoting.jpg", "transparent bowl"],
        # ["id07_img_test01_Grasping.jpg", "brown bowl"],
        # ["id07_img_test01_Pivoting.jpg", "brown bowl"],   
        
        # ["20250826_new_demo_test01.jpg", "mobile phone,marker pen,screw,computer"],

        # ["20251111_URM_imgL_000000.jpg", "box,blue bowl,pink basket,plastic jar"],

        ["005_testingFS_rarg-imgL_000001.jpg", "gray bag,zipper,pen"],
        ["005_testingFS_rarg-imgL_000002.jpg", "gray bag,zipper,pen"],
        ["005_testingFS_rarg-imgL_000003.jpg", "gray bag,zipper,pen"],

    ]
    
    kfr_height, kfr_width = cfg_dict_init["kfr_height"], cfg_dict_init["kfr_width"]  # 540, 960
    roi_bbox_0 = [0, 0, kfr_width, kfr_height]  # the orginal whole image as the input
    roi_bbox_1 = cfg_dict_init["detection_roi_bbox"]  # for fast and stable detection and segmentation
    roi_bbox_2 = cfg_dict_init["detection_roi_bbox_2"]  # for fast and stable detection and segmentation
    
    
    for jpg_idx, [task_folder_jpg, prompts] in enumerate(task_folder_jpgs):
        print("\n", jpg_idx, task_folder_jpg)
        img_path = os.path.join(test_folder_dir, task_folder_jpg)
        
        path_default = os.path.join(test_folder_dir, "../test_res", task_folder_jpg)
        if "cirbowl" in task_folder_jpg and "seed" in task_folder_jpg:  [x1, y1, x2, y2] = roi_bbox_1; save_vis_jpg_path = path_default
        elif "rectbox" in task_folder_jpg and "seed" in task_folder_jpg:  [x1, y1, x2, y2] = roi_bbox_1; save_vis_jpg_path = path_default
        elif "ordcup" in task_folder_jpg and "seed" in task_folder_jpg:  [x1, y1, x2, y2] = roi_bbox_0; save_vis_jpg_path = path_default
        elif "mugcup" in task_folder_jpg and "seed" in task_folder_jpg:  [x1, y1, x2, y2] = roi_bbox_2; save_vis_jpg_path = path_default
        elif "bottle" in task_folder_jpg and "seed" in task_folder_jpg:  [x1, y1, x2, y2] = roi_bbox_1; save_vis_jpg_path = path_default
        else: [x1, y1, x2, y2] = roi_bbox_1; save_vis_jpg_path = path_default.replace(".jpg", f"_DetSeg{str(jpg_idx+1).zfill(2)}.jpg")
        
        image_input_cv2 = cv2.imread(img_path)  #######################
        image_input_cv2 = cv2.convertScaleAbs(image_input_cv2, alpha=1.5, beta=20)  # adjust the brightness and contrast
        image_input = Image.fromarray(image_input_cv2[y1:y2, x1:x2][:, :, ::-1])  #######################
        
        text_input = prompts
        det_results, img_vis = ov_detection_segmentation(image_input, text_input)
        
        image_input_cv2[y1:y2, x1:x2] = np.array(img_vis)[:, :, ::-1]  #######################
        cv2.rectangle(image_input_cv2, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,0), thickness=2)  #######################
        img_vis = image_input_cv2[:, :, ::-1]  #######################

        img_cv2 = np.array(img_vis)[:, :, ::-1]  # PIL -> cv2. np.array + RGB2BGR
        cv2.imwrite(save_vis_jpg_path, img_cv2)  
        
        for det_list in det_results:
            [cls_name, bbox, mask] = det_list
            save_mask_path = save_vis_jpg_path[:-4] + f"_mask_{cls_name}.jpg"
            binary_mask = np.array(mask*255, dtype=np.uint8)
            
            top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2  #######################
            binary_mask = cv2.copyMakeBorder(binary_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))  #######################
            
            cv2.imwrite(save_mask_path, binary_mask)

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            polygons = [np.array(polygon).squeeze() for polygon in contours]
            print(len(polygons[0]), polygons[0][:10])
            
            '''
            image_input_cv2 = cv2.imread(img_path)
            for opt in polygons[0]: cv2.circle(image_input_cv2, (int(opt[0]), int(opt[1])), 2, (0,255,255), -1, cv2.LINE_AA)  # yellow
            cv2.imwrite(path_default.replace(".jpg", f"_VisMask.jpg"), image_input_cv2)
            '''
    
