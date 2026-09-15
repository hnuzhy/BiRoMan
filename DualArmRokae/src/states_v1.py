
import os
import sys
import json
import cv2
import copy
import time
import shutil
import subprocess
import argparse
import base64

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


'''
test_img_dir = "/home/dex/zhouhuayi/rokaeDemo/results/tempLLM/"
test_img_name = "LLM-1_OBJ-bottle,cup_TEST-101_COUNT-16_C-B-NO.jpg"
with open(os.path.join(test_img_dir, test_img_name), "rb") as img_reader:
    image_bytes_data = img_reader.read()

GEMINI_SYS_MESSAGE = "As you can see, here is a RGB image with a bottle (leftside) and a mug cup (rightside) on the table. Please help me to judge their states. \
    (1) The bottle's state is categoried into three types: (A) standing upright on the table; (B) lying down on the table; (C) standing upside down on the table. \
    (2) The mug cup's state is categoried into three types: (A) standing upright on the table; (B) lying down on the table; (C) standing upside down on the table. \
    (3) Please also check whether there is a cap on this bottle or not. You can answer: (YES) having a cap on the bottle; (NO) not having a cap on the bottle. \
    The final output must be formatted in valid JSON, such as: {'bottle': 'A', 'cup': 'A', 'has_cap': 'NO'} or {'bottle': 'B', 'cup': 'A', 'has_cap': 'YES'} or \
    {'bottle': 'C', 'cup': 'B', 'has_cap': 'YES'} or {'bottle': 'A', 'cup': 'C', 'has_cap': 'NO'}. "

# https://ai.google.dev/gemini-api/docs/image-understanding?hl=zh-cn
# https://blog.csdn.net/dafanpai/article/details/148479934
# pip install google-generativeai
import google.generativeai as genai
genai.configure(api_key="xxxxxx")


#model_gemini = genai.GenerativeModel('gemini-pro')  # for language only
#response = model_gemini.generate_content("帮我总结一下量子计算的核心概念")
#print(response.text)


#model_gemini = genai.GenerativeModel('gemini-pro-vision')  # abandoned officially
#model_gemini = genai.GenerativeModel('gemini-2.5-flash-image-preview')  # expensive
model_gemini = genai.GenerativeModel('gemini-2.5-flash')  # relatively fast and robust

for test_id in range(10):
    print(f"\n {test_id} Start Time:", time.time())
    response = model_gemini.generate_content(
        [GEMINI_SYS_MESSAGE, {"mime_type": "image/png", "data": image_bytes_data}]
    )
    print(f"{test_id} End Time:", time.time())
    print(response.text)

os._exit(0)
'''


# pip install langchain-core==0.3.69 langchain-openai==0.3.28
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

import google.generativeai as genai

os.environ["OPENAI_API_VERSION"] = "2024-02-01"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://dexforce-gpt.openai.azure.com/"

import kingfisher
sys.path.insert(0, os.getcwd())
from src.config import cfg_dict_init as cfg_dict
from src.vlms import conduct_object_detect_and_segment


from src.prompts import SYS_MESSAGE_BOTTLE_3_STATES as MSG_BOTTLE_3_STATES
from src.prompts import SYS_MESSAGE_BOTTLE_2_STATES as MSG_BOTTLE_2_STATES
from src.prompts import SYS_MESSAGE_MUGCUP_3_STATES as MSG_MUGCUP_3_STATES
from src.prompts import SYS_MESSAGE_MUGCUP_2_STATES as MSG_MUGCUP_2_STATES
from src.prompts import SYS_MESSAGE_BOTTLE_3_STATES_2_CAP as MSG_BOTTLE_3_STATES_2_CAP
from src.prompts import SYS_MESSAGE_BOTTLE_2_STATES_2_CAP as MSG_BOTTLE_2_STATES_2_CAP
from src.prompts import SYS_MESSAGE_BOWL_2_STATES as MSG_BOWL_2_STATES


#################################################################
def generate_prompt(
    sys_message: str, user_message: str, image_path: str
) -> ChatPromptTemplate:
    # 定义动态模板（支持变量 {observation} 和 {user_message}）
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=sys_message),  # 系统角色消息
            (
                "human",
                [  # 人类用户消息（多模态：图片 + 文本）
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpg;base64,{observation}"},
                    },
                    {
                        "type": "text",
                        "text": "{user_message}",
                    },
                ],
            ),
        ]
    )

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    inputs = {
        "observation": base64_image,  # 图片的Base64编码
        "user_message": user_message,
    }
    formatted_messages = prompt.format_messages(**inputs)
    return formatted_messages

def generate_prompt_and_result(model_gemini, sys_message, image_path):
    with open(image_path, "rb") as img_reader:
        image_bytes_data = img_reader.read()

    response = model_gemini.generate_content(
        [sys_message, {
            "mime_type": "image/png", 
            "data": image_bytes_data}
        ]
    )
    return response

def get_llm_instance(llm_type, temperature):
    if llm_type == 0:  # for the langchain-based doubao by bytedance
        api_key = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"  # please input your api_key here
        llm_instance = ChatOpenAI(
                temperature=temperature,
                base_url="https://ark.cn-beijing.volces.com/api/v3/",
                api_key=api_key,
                model_name="doubao-1.5-vision-pro-250328",
            )
        llm_instance = llm_instance.bind(response_format={"type": "json_object"})
        return llm_instance

    if llm_type == 1:  # for the langchain-based OpenAI by bytedance
        api_key = "xxxxxx"  # please input your api_key here
        llm_instance = AzureChatOpenAI(
                temperature=temperature,
                deployment_name="gpt-4o",
                openai_api_key=api_key,
            )
        llm_instance = llm_instance.bind(response_format={"type": "json_object"})
        return llm_instance

    ##########################################

    if llm_type == 2:  # for the genai-based Gemini by google
        genai.configure(api_key="xxxxxx")  # please input your api_key here
        #model_gemini = genai.GenerativeModel('gemini-pro')  # for language only
        #model_gemini = genai.GenerativeModel('gemini-pro-vision')  # abandoned officially
        #model_gemini = genai.GenerativeModel('gemini-2.5-flash-image-preview')  # expensive
        #model_gemini = genai.GenerativeModel('gemini-2.5-flash')  # relatively fast and robust

        model_gemini = genai.GenerativeModel('gemini-2.5-flash', 
            generation_config={"response_mime_type": "application/json"}) 
        return model_gemini

#################################################################


#################################################################

# python src/states_v1.py --object_type cloth --test_id 1
# python src/states_v1.py --object_type rope --test_id 1
# python src/states_v1.py --object_type towel --test_id 1

# python src/states_v1.py --object_type basket --test_id 1


def main_deformable_objs():
    args = get_args()

    save_res_path = "/home/dex/zhouhuayi/rokaeDemo/debug/test_mLLM_deformable_objs/"

    c = kingfisher.connect(cfg_dict["kfr_ip"])   # connect the camera
    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']  # 540, 960
    kingfisher.SetAUTO_EXPOSURE()

    time_interval, frame_count, enlarge_ratio = 1, 0, 1.25

    image_name = f"test_img_LLM-{args.llm_type}_OBJ-{args.object_type}_ID-{str(args.test_id).zfill(2)}.jpg" 
    image_path = os.path.join(save_res_path, image_name)
    roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
    roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation

    video_output_path = image_path.replace(".jpg", ".mp4")
    vout = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (x2-x1, y2-y1))  # Create VideoWriter object

    image_path_dir = image_path.replace(".jpg", "")
    if os.path.exists(image_path_dir): shutil.rmtree(image_path_dir)
    os.mkdir(image_path_dir)


    '''
    # 用户输入自然语言描述
    if len(args.user_message) == 0:
        user_message = input("请输入任务描述: ").strip()
    else:
        user_message = args.user_message.strip()

    # 调用 LLM 生成任务计划
    if args.object_type == "bottle":
        SYS_MESSAGE = SYS_MESSAGE_BOTTLE_STATES; detseg_prompts = "bottle"
        category_list = {"A": "upright", "B": "lying down", "C": "upside down"}
    elif args.object_type == "mug":
        SYS_MESSAGE = SYS_MESSAGE_MUGCUP_STATES; detseg_prompts = "mug"
        category_list = {"A": "upright", "B": "lying down", "C": "upside down"}
    elif args.object_type == "bowl": 
        SYS_MESSAGE = SYS_MESSAGE_BOWL_STATES; detseg_prompts = "bowl"
        category_list = {"A": "upright", "B": "inverted"}
    elif args.object_type == "phone": 
        SYS_MESSAGE = SYS_MESSAGE_PHONE_STATES; detseg_prompts = "phone"
        category_list = {"A": "upright", "B": "inverted"}
    '''

    if args.object_type == "cloth": detseg_prompts = "T-shirt,cuff,neckline"
    if args.object_type == "rope": detseg_prompts = "warped cable"
    if args.object_type == "towel": detseg_prompts = "blue towel"

    if args.object_type == "basket": detseg_prompts = "basket"

    while True:
        frame_count += 1
        left_img_test_ori, _ = kingfisher.captureQuarterSize()
        left_img_test = left_img_test_ori[y1:y2, x1:x2].copy()
        #cv2.imwrite(image_path, left_img_test)

        final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(
            left_img_test.copy(), prompts_str=detseg_prompts, is_raw_result=True, given_roi_bbox=[0, 0, x2-x1, y2-y1])

        if frame_count % 5 == 0:
            image_path_temp = os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_{detseg_prompts}.jpg")
            cv2.imwrite(image_path_temp, left_img_test)  # for further using

            if "basket" in detseg_prompts:
                mask_img = final_res_list_raw[0][2]
                cv2.imwrite(image_path_temp[:-4]+"_mask.jpg", mask_img)  # for further using
                bbox = final_res_list_raw[0][1]
                mask_arr = final_res_list_raw[0][3][0]
                with open(image_path_temp[:-4]+"_mask.json", "w") as json_file:
                    json.dump({"bbox": bbox.tolist(), "mask": mask_arr.tolist()}, json_file)

        '''
        for [obj_name, bbox, obj_binary_mask, multi_masks] in final_res_list_raw:
            assert obj_name == detseg_prompts, "wrong detection result in the object type!!!"
            [bx1, by1, bx2, by2] = bbox
            ebx1 = max(0, int(bx1 - (enlarge_ratio - 1.0) * (bx2 - bx1) * 0.5))
            ebx2 = min(x2-x1-1, int(bx2 + (enlarge_ratio - 1.0) * (bx2 - bx1) * 0.5))
            eby1 = max(0, int(by1 - (enlarge_ratio - 1.0) * (by2 - by1) * 0.5))
            eby2 = min(y2-y1-1, int(by2 + (enlarge_ratio - 1.0) * (by2 - by1) * 0.5))

            left_img_test[eby1:eby2, ebx1:ebx2] = img_vis_cv2[eby1:eby2, ebx1:ebx2] 
            cv2.rectangle(left_img_test, (ebx1, eby1), (ebx2, eby2), color=(0,0,0), thickness=2)  # for visualization
            cv2.putText(left_img_test, "RoI", (ebx1, eby1-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, 
                color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
            break  # only care about one object every time

        cv2.imwrite(image_path, left_img_test_ori[y1:y2, x1:x2][eby1:eby2, ebx1:ebx2] )  # for LLM gpt-4o / doubao using


        formatted_messages = generate_prompt(SYS_MESSAGE, user_message, image_path=image_path)
        response = llm_instance.invoke(formatted_messages)
        print(f"frame_count: {frame_count}. \t LLM Response: {response.content}")

        response_dict = json.loads(response.content)
        state_res = response_dict["state"]
        show_str = str(frame_count).zfill(2) + " " + state_res + ": " + category_list[state_res]
        cv2.putText(left_img_test, show_str, (10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, 
            color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)
        '''


        cv2.namedWindow("MyDebugWindow", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MyDebugWindow", x2-x1, y2-y1)
        cv2.imshow("MyDebugWindow", img_vis_cv2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # or cv2.waitKey(0); wait until we close the plotted window (press Esc to continue)

        for _ in range(5): vout.write(img_vis_cv2)

        # time.sleep(time_interval)



#################################################################

# python src/states_v1.py --llm_type 1 --object_type bottle --test_id 1 --online
# python src/states_v1.py --llm_type 1 --object_type mug --test_id 1 --online
# python src/states_v1.py --llm_type 1 --object_type bowl --test_id 1 --online

# python src/states_v1.py --llm_type 0 --object_type doubao_test1 --test_id 1 --online
# python src/states_v1.py --llm_type 0 --object_type doubao_test2 --test_id 1 --online

def main_rigid_obj_states():  
    args = get_args()

    save_res_path = "/home/dex/zhouhuayi/rokaeDemo/debug/test_mLLM_rigid_obj_states/"

    llm_instance = get_llm_instance(args.llm_type, args.temperature)

    c = kingfisher.connect(cfg_dict["kfr_ip"])   # connect the camera
    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']  # 540, 960
    kingfisher.SetAUTO_EXPOSURE()

    if not args.online:
        left_img_test_ori, _ = kingfisher.captureQuarterSize()
        image_name = f"test_img_LLM-{args.llm_type}_OBJ-{args.object_type}_ID-{str(args.test_id).zfill(2)}.jpg" 
        image_path = os.path.join(save_res_path, image_name)
        roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
        roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)

        [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
        left_img_test = left_img_test_ori[y1:y2, x1:x2]
        cv2.imwrite(image_path, left_img_test)

        # 用户输入自然语言描述
        if len(args.user_message) == 0:
            #user_message = input("请输入任务描述: ").strip()
            user_message = "please help to judge the object state." 
        else:
            user_message = args.user_message.strip()

        # 调用 LLM 生成任务计划
        if args.object_type == "bottle": SYS_MESSAGE = SYS_MESSAGE_BOTTLE_STATES
        elif args.object_type == "mug": SYS_MESSAGE = SYS_MESSAGE_MUGCUP_STATES
        elif args.object_type == "bowl": SYS_MESSAGE = SYS_MESSAGE_BOWL_STATES
        elif args.object_type == "phone": SYS_MESSAGE = SYS_MESSAGE_PHONE_STATES

        if args.llm_type == 0 or args.llm_type == 1:
            formatted_messages = generate_prompt(SYS_MESSAGE, user_message, image_path=image_path)
            response = llm_instance.invoke(formatted_messages)
            print(f"LLM Response: {response.content}")
        if args.llm_type == 2:
            response = generate_prompt_and_result(llm_instance, SYS_MESSAGE, image_path)
            print(f"LLM Response: {response.text}")
            
    else:
        time_interval, frame_count, enlarge_ratio = 1, 0, 1.8

        image_name = f"test_img_LLM-{args.llm_type}_OBJ-{args.object_type}_ID-{str(args.test_id).zfill(2)}.jpg" 
        image_path = os.path.join(save_res_path, image_name)

        image_path_dir = image_path.replace(".jpg", "")
        if os.path.exists(image_path_dir): shutil.rmtree(image_path_dir)
        os.mkdir(image_path_dir)

        roi_bbox = cfg_dict["detection_roi_bbox"]  # the [x1, y1, x2, y2] (960*540 --> 750*500; 16:9 --> 3:2)
        roi_bbox_2 = cfg_dict["detection_roi_bbox_2"]  # the [x1, y1, x2, y2] (960*540 --> 500*360; 16:9 --> 25:18)
        [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation

        video_output_path = image_path.replace(".jpg", ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v'); FPS = 5
        vout = cv2.VideoWriter(video_output_path, fourcc, FPS, (x2-x1, (y2-y1)*2))  # Create VideoWriter object
        
        # 用户输入自然语言描述
        if "doubao_test" in args.object_type:
            user_message = "check object's state"
        elif len(args.user_message) == 0:
            #user_message = input("请输入任务描述: ").strip()
            user_message = "please help to judge the object state." 
        else:
            user_message = args.user_message.strip()

        #############################################################
        # 调用 LLM 生成任务计划
        if args.object_type == "bottle":  # single VQA
            category_list = {"A": "upright", "B": "lying down", "C": "upside down"}
            SYS_MESSAGE_DICT = {"bottle": MSG_BOTTLE_3_STATES}; detseg_prompts = "bottle"
        elif args.object_type == "mug":  # single VQA
            category_list = {"A": "upright", "B": "lying down", "C": "upside down"}
            SYS_MESSAGE_DICT = {"mug": MSG_MUGCUP_3_STATES}; detseg_prompts = "mug"
        elif args.object_type == "bowl":  # single VQA
            SYS_MESSAGE_DICT = {"bowl": MSG_BOWL_2_STATES}; detseg_prompts = "bowl"
            category_list = {"A": "upright", "B": "inverted"}

        elif args.object_type == "doubao_test1":  # multiple VQA (using the for loop)
            category_list = {"A": "upright", "B": "lying down", "C": "upside down"}
            # SYS_MESSAGE_DICT = {"bottle": MSG_BOTTLE_3_STATES, "mug": MSG_MUGCUP_3_STATES}
            SYS_MESSAGE_DICT = {"bottle": MSG_BOTTLE_2_STATES_2_CAP, "mug": MSG_MUGCUP_2_STATES}
            detseg_prompts = "bottle,mug"
        elif args.object_type == "doubao_test2":  # multiple VQA (using the for loop)
            category_list = {"A": "upright", "B": "inverted"}
            SYS_MESSAGE_DICT = {"white bowl": MSG_BOWL_2_STATES, "green bowl": MSG_BOWL_2_STATES}
            detseg_prompts = "white bowl,green bowl"
            #SYS_MESSAGE_DICT = {"white bowl": MSG_BOWL_2_STATES, "green bowl": MSG_BOWL_2_STATES, "blue bowl": MSG_BOWL_2_STATES}
            #detseg_prompts = "white bowl,green bowl,transparent bowl"
        #############################################################

        while True:
            frame_count += 1
            left_img_test_ori, _ = kingfisher.captureQuarterSize()
            left_img_test_ori = cv2.convertScaleAbs(left_img_test_ori, alpha=1.2, beta=10)  # Adjust the brightness and contrast 
            left_img_test = left_img_test_ori[y1:y2, x1:x2].copy()
            #cv2.imwrite(image_path, left_img_test)

            final_res_list_raw, img_vis_cv2 = conduct_object_detect_and_segment(
                left_img_test, prompts_str=detseg_prompts, is_raw_result=True, given_roi_bbox=[0, 0, x2-x1, y2-y1])
            obj_sub_img_list = []
            for detseg_prompt in detseg_prompts.split(","):
                for [obj_name, bbox, obj_binary_mask, multi_masks] in final_res_list_raw:
                    if obj_name != detseg_prompt: continue
                    [bx1, by1, bx2, by2] = bbox
                    ebx1 = max(0, int(bx1 - (enlarge_ratio - 1.0) * (bx2 - bx1) * 0.5))
                    ebx2 = min(x2-x1-1, int(bx2 + (enlarge_ratio - 1.0) * (bx2 - bx1) * 0.5))
                    eby1 = max(0, int(by1 - (enlarge_ratio - 1.0) * (by2 - by1) * 0.5))
                    eby2 = min(y2-y1-1, int(by2 + (enlarge_ratio - 1.0) * (by2 - by1) * 0.5))
                    break  # only care about one object every time
                obj_sub_img = left_img_test_ori[y1:y2, x1:x2][eby1:eby2, ebx1:ebx2]
                obj_sub_img_list.append([obj_sub_img, ebx2-ebx1, eby2-eby1])  # img_cv2, img_w, img_h
            
            '''  # merge bottle and cup into one image (the VQA result will be unstable)
            new_img_h = max(obj_sub_img_list[0][-1], obj_sub_img_list[1][-1])
            new_img_w = obj_sub_img_list[0][1] + obj_sub_img_list[1][1] + 2
            new_img_canvas = np.full((new_img_h, new_img_w, 3), 0, dtype=np.uint8)  # 0 black. 255 white
            new_img_canvas[:obj_sub_img_list[0][-1], :obj_sub_img_list[0][1]] = obj_sub_img_list[0][0]  # bottle in the left side
            new_img_canvas[:obj_sub_img_list[1][-1], new_img_w-obj_sub_img_list[1][1]:] = obj_sub_img_list[1][0]  # cup in the right side            
            '''

            state_res_str, obj_state_str_list = "", []  # for checking multiple object in a once
            for iid, detseg_prompt in enumerate(detseg_prompts.split(",")):
                SYS_MESSAGE = SYS_MESSAGE_DICT[detseg_prompt]
                new_img_temp = obj_sub_img_list[iid][0]

                image_path_temp = os.path.join(image_path_dir, f"FRAME{str(frame_count).zfill(2)}_{detseg_prompt}.jpg")
                cv2.imwrite(image_path_temp, new_img_temp)  # for LLM gpt-4o / doubao using

                if args.llm_type == 0 or args.llm_type == 1:
                    formatted_messages = generate_prompt(SYS_MESSAGE, user_message, image_path=image_path_temp)
                    response = llm_instance.invoke(formatted_messages)
                    print(f"frame_count: {frame_count}. \t LLM Response: {response.content}")
                    response_dict = json.loads(response.content)
                if args.llm_type == 2:
                    response = generate_prompt_and_result(llm_instance, SYS_MESSAGE, image_path_temp)
                    print(f"frame_count: {frame_count}. \t LLM Response: {response.text}")
                    response_dict = json.loads(response.text)

                state_res_str += "( "
                for res_id, dict_key in enumerate(list(response_dict.keys())):
                    res_value = response_dict[dict_key]  # A or B or C / YES or NO
                    state_res_str += (res_value + " ")
                    if res_id == 0: obj_state_str_list.append(detseg_prompt + ": " + category_list[res_value])
                state_res_str += ") "

            img_vis_cv2 = img_vis_cv2.astype(np.uint8)
            show_str = str(frame_count).zfill(2) + " " + state_res_str
            (tw, th), _ = cv2.getTextSize(show_str, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.72, thickness=1)
            cv2.rectangle(img_vis_cv2, (10, 30-th-5), (10+tw, 30+5), color=(255,255,255), thickness=-1)
            cv2.putText(img_vis_cv2, show_str, (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.72, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
            for iid, obj_state_str in enumerate(obj_state_str_list):
                (tw, th), _ = cv2.getTextSize(obj_state_str, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.72, thickness=1)
                cv2.rectangle(img_vis_cv2, (10, 30*(iid+2)-th-5), (10+tw, 30*(iid+2)+5), color=(255,255,255), thickness=-1)
                cv2.putText(img_vis_cv2, obj_state_str, (10, 30*(iid+2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.72, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)

            img_canvas = np.vstack((left_img_test, img_vis_cv2))
            cv2.namedWindow("MyDebugWindow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("MyDebugWindow", x2-x1, (y2-y1) * 2)
            cv2.imshow("MyDebugWindow", img_canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # or cv2.waitKey(0); wait until we close the plotted window (press Esc to continue)

            for _ in range(FPS): vout.write(img_canvas)

            # time.sleep(time_interval)

#################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm_type", type=int, default=1, help="0: doubao; 1: gpt-4o")
    parser.add_argument("--object_type", type=str, required=True, help="the object name for judging its state.")
    parser.add_argument("--test_id", type=int, default=1, help="the testing id for one specific object.")
    parser.add_argument("--user_message", type=str, default="", help="the string message of user prompts.")
    parser.add_argument("--online", action='store_true', help="default is False")

    args = parser.parse_args()
    return args


#################################################################
if __name__ == "__main__":

    # main_rigid_obj_states()

    main_deformable_objs()

'''
[Hit 1]: single VQA (one single object in the image) is much better than multiple VQA (two or more objects in the image)
[Hit 2]: using less state categories is much better than adding even one more state (e.g., 2 states is better than 3 states)
'''
