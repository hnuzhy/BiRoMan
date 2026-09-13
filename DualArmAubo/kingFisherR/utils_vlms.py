
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
    from vlm_utils.sam import load_sam_image_model, run_sam_inference
    from vlm_utils.florence import load_florence_model, run_florence_inference
    from vlm_utils.florence import FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
except:
    from .vlm_utils.sam import load_sam_image_model, run_sam_inference
    from .vlm_utils.florence import load_florence_model, run_florence_inference
    from .vlm_utils.florence import FLORENCE_OPEN_VOCABULARY_DETECTION_TASK

    
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
        # if x1 < 5 or x2 > rx2-rx1-5: continue  # not the ideal object (around the bounding)
        binary_mask = np.array(mask*255, dtype=np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        polygon_mask = [np.array(polygon).squeeze() for polygon in contours]
        det_results_new.append([cls_name, bbox, binary_mask, polygon_mask])
    
    return det_results_new, img_vis_cv2

#############################################################   
def conduct_object_detect_and_segment(raw_img_input, prompts_str=None, is_raw_result=False, roi_bbox=None, roi_bbox_2=None):
    [x1, y1, x2, y2], [x21, y21, x22, y22] = roi_bbox, roi_bbox_2  # for fast and stable detection and segmentation
    
    image_input_sub = raw_img_input[y1:y2, x1:x2]
    det_res_test, img_vis_test = ov_detection_segmentation(Image.fromarray(image_input_sub[:, :, ::-1]), prompts_str)  # CV2 --> PIL
    det_res_test_new, img_vis_test_cv2 = post_processing_det_seg_results(det_res_test, img_vis_test, roi_bbox)
    if len(det_res_test_new) == 0:  # re-detect and re-segment again with smaller ROI
        print("*****[Re-Conduct the detection and segmentation again!!!]*****")
        image_input_sub2 = raw_img_input[y21:y22, x21:x22]
        det_res_test, img_vis_test = ov_detection_segmentation(Image.fromarray(image_input_sub2[:, :, ::-1]), prompts_str)  # CV2 --> PIL
        det_res_test_new, img_vis_test_cv2 = post_processing_det_seg_results(det_res_test, img_vis_test, roi_bbox_2)
        cur_obj_mask = det_res_test_new[0][-1][0]  # det_res_test_new is a list of [cls_name, bbox, binary_mask, polygon_mask]
        cur_obj_mask[:, 0] += (x21 - x1); cur_obj_mask[:, 1] += (y21 - y1)  # remember add-back and re-adjust the offsets in x / y
        det_res_test_new[0][-1][0] = cur_obj_mask
        img_temp = image_input_sub.copy(); img_temp[y21-y1:y22-y1, x21-x1:x22-x1] = img_vis_test_cv2; img_vis_test_cv2 = img_temp
        cv2.rectangle(img_vis_test_cv2, (int(x21-x1), int(y21-y1)), (int(x22-x1), int(y22-y1)), color=(255,255,255), thickness=2)
        if len(det_res_test_new) > 1:  # there may be multiple target objects in the list
            for idx in range(len(det_res_test_new) - 1):
                cur_obj_mask_temp = det_res_test_new[1+idx][-1][0]
                cur_obj_mask_temp[:, 0] += (x21 - x1); cur_obj_mask_temp[:, 1] += (y21 - y1)
                det_res_test_new[1+idx][-1][0] = cur_obj_mask_temp
    else:
        cur_obj_mask = det_res_test_new[0][-1][0]  # det_res_test_new is a list of [cls_name, bbox, binary_mask, polygon_mask]    
    
    if is_raw_result:
        return det_res_test_new, img_vis_test_cv2
    else:
        return cur_obj_mask, img_vis_test_cv2

#############################################################
def plot_processed_results_vis(sub_img, res_list, cls_list, trash_name=None):
    colors_list = [(0,255,128), (255,255,0), (128,0,255), (0,128,128), (0,0,255)]  # lawn green, cyan, light magenta, light yellow, red
    
    for obj_idx, [obj_name, bbox, obj_binary_mask, mask] in enumerate(res_list):
        [img_h, img_w, img_c] = sub_img.shape
        [x1, y1, x2, y2] = bbox
        if trash_name is not None:
            if obj_name == trash_name:
                color = colors_list[2]  # for sweeping task: "brush,dustpan,trash"
            else:
                color = colors_list[cls_list.index(obj_name)]
        else:
            color = colors_list[cls_list.index(obj_name)]
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



