
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

import argparse
import base64
import os, json
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import subprocess
import kingfisher

import time
import keyboard

os.environ["OPENAI_API_VERSION"] = "2024-02-01"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://dexforce-gpt.openai.azure.com/"
# model = input("请选择你的模型序号(1：gpt-4o/2：doubao-1.5-vision-pro): ")
# if model == "1":
#     if "AZURE_OPENAI_API_KEY" in os.environ:
#         api_key = os.getenv("AZURE_OPENAI_API_KEY")
#     else:
#         api_key = input("请输入您的 AZURE OpenAI API Key: ")
#     llm = AzureChatOpenAI(
#         temperature=0.0,
#         deployment_name="gpt-4o",
#         openai_api_key=api_key,
#     )
# elif model == "2":
#     if "DOUBAO_API_KEY" in os.environ:
#         api_key = os.getenv("DOUBAO_API_KEY")
#     else:
#         api_key = input("请输入您的豆包 API Key: ")
#     llm = ChatOpenAI(
#         temperature=0,
#         base_url="https://ark.cn-beijing.volces.com/api/v3/",
#         api_key=api_key,
#         model_name="doubao-1.5-vision-pro-250328",
#     )
# else:
#     raise ValueError("请选择现已集成的正确的模型")

DIR_PATH = "/home/wangrx/图片"




def encode_image_from_path(image_path: str):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

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
    base64_image = encode_image_from_path(image_path)
    inputs = {
        "observation": base64_image,  # 图片的Base64编码
        "user_message": user_message,
    }
    formatted_messages = prompt.format_messages(**inputs)
    return formatted_messages


def plot_task_plan(image_path, task_plan, save_path="./", show=False):

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    print(height, width)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")

    def draw_box(label, bbox, color="red"):
        bbox = [coordinate/10000.0 if coordinate > 1.0 else coordinate for coordinate in bbox]
        # print(bbox)
        min_x, max_x, min_y, max_y = bbox
        x = min_x * width
        y = min_y * height
        w = (max_x - min_x) * width
        h = (max_y - min_y) * height
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, label, color=color, fontsize=10, weight='bold', backgroundcolor="white")

    def draw_point(label, point, color="red"):
        x, y = point
        x = x * width
        y = y * height
        ax.plot(x, y, marker='o', color=color, markersize=6)

    for idx, step in enumerate(task_plan["steps"], 1):
        action = step["action"]
        color = "red" if idx == 1 else "blue"

        if "location" in list(step.keys()):
            label = f"Step {idx}: {step['object']} ({action})"
            if len(step["location"]) == 4:
                draw_box(label, step["location"], "blue")
            elif len(step["location"]) == 2:
                draw_point(label, step["location"], color)
        else:
            if "source_location" in list(step.keys()):
                label_src = f"Step {idx}: {step['source_object']} (source)"
                if len(step["source_location"]) == 4:
                    draw_box(label_src, step["source_location"], "green")
                elif len(step["source_location"]) == 2:
                    draw_point(label_src, step["source_location"], color)

            if "destination_location" in list(step.keys()):
                label_dst = f"Step {idx}: {step['destination_object']} (destination)"
                if len(step["destination_location"]) == 4:
                    draw_box(label_dst, step["destination_location"], "orange")
                elif len(step["destination_location"]) == 2:
                    draw_point(label_dst, step["destination_location"], color)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


BBOX_PROMPT = "The location should be expressed as a tuple: (min_x, max_x, min_y, max_y), representing the normalized bounding box coordinates of the target region, where each value is between 0 and 1 and rounded to 4 decimal places. Attach this tuple to the corresponding action step."
POINT_PROMPT = "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image. Attach this tuple to the corresponding action step."

OUTPUT_TEMPLATE_POURWATER = "{'steps': [\
                    {'action': '...', 'object': '...', 'location': [...]}, \
                    {'action': '...', 'source_object': '...', 'destination_object': '...', 'source_location': [...], 'destination_location': [...]},\
                    ]}"

SYS_MESSAGE_POURWATER = f"You are a reliable and capable robot task planning assistant. \
    You receive an image and a natural language instruction intended to guide a robot in completing a specified task in a real-world environment. \
    The task described by the image-language input may vary in complexity—it could be simple, involve articulated objects, or require long-horizon planning. \
    The robot is a dual-arm manipulator, and a primitive skill library is available, including: 'Unscrew bottle cap' and 'Pour water'. \
    Note that if task-related object is a bottle with a cap, please judge its cap state (open or close). If bottle don't have a cap then don't unscrew this bottle.\
    Your goal is to generate a clear and logically ordered task plan that the robot can execute, \
    Possible cups include 'purple mug' and 'green mug' \
    For each action step, locate and identify the spatial location of the relevant object in the input image. \
    {BBOX_PROMPT}\
    The final output must be formatted in valid JSON, like: {OUTPUT_TEMPLATE_POURWATER}"
# 'There might be  bottles without caps so you need to put back cap on those bottles'
    # and choose a task name from 'unscrew_pouring' and 'pouring'. \
OUTPUT_TEMPLATE_POURWATER_1 = "{'steps': [\
                    {'cap_existence': '...', 'is_cap_on_bottle': '...', 'action': '...', 'object_bottle': '...', 'object_cup': '...'}, \
                    ]}"
SYS_MESSAGE_POURWATER_1 = f"You are a reliable and capable robot task planning assistant. \
    You receive an image and a natural language instruction intended to guide a robot in completing a specified task in a real-world environment. \
    The task described by the image-language input may vary in complexity. The image depicts a tabletop with bottle, cup, brush, etc. \
    The robot is a dual-arm manipulator, and a primitive skill library is available, including: 'Unscrewing pour' and 'Pour water'. \
    User would give you a command related to pour water such as 'help me pour a cup of water', 'I want to drink', 'I am thirsty', 'pour water', 'pour me water' ect.\
    These commands and any of possible variants only represent pour water task.\
    Your first goal is to check if there exists bottle cap on table and the cap is on the bottle.\
    if cap is on the bottle, action is 'Unscrewing pour'. If cap is not on the bottle, action is 'Pour water'\
    For cap-related fields you only answer yes or no, for 'action' fields, you answer 'Unscrewing pour' or 'Pour water'.\
    For 'object_bottle' feild you fill in 'bottle' or 'plastic bottle' based on input image.\
    For 'object_cup' feild you fill in 'purple mug' or 'green mug' based on input image.\
    The final output must be formatted in valid JSON, like: {OUTPUT_TEMPLATE_POURWATER_1}\
    "
# USER_MESSAGE_POURWATER  = "Pour a half of blue plastic bottle into the green mug"
# USER_MESSAGE_POURWATER = "Help me pour a cup of coffee"
# USER_MESSAGE_POURWATER = "Help me put on the bottle cap"

mugcup_mapping = {
    "purple mug": 1,
    "green mug": 3,
}

bowl_mapping = {
    "white bowl": 2,
    "plastic bowl": 3,
    "transparent bowl": 3,
    "brown bowl": 7,
}


# bottle_mapping = {
#     "purple mug": 1,
#     "Scream drink": 7,
# }

OUTPUT_TEMPLATE_SWEEP = "{'steps': [\
                    {'action': '...', 'object': '...', 'location': [...]}, \
                    ]}"
SYS_MESSAGE_SWEEP = f"You are a reliable and capable robot task planning assistant. \
    You receive an image and a natural language instruction intended to guide a robot in completing a specified task in a real-world environment. \
    The task described by the image-language input may vary in complexity—it could be simple, involve articulated objects, or require long-horizon planning. \
    The robot is a dual-arm manipulator, and a primitive skill library is available, including: 'Sweep trash'.\
    Your goal is to detect an obvious trash related to sweep trash task. Possible trashs include paper ball, yellow pen, bottle cap etc.\
    For each object, you first recognize its class type then locate and identify the spatial location of the relevant object in the input image. \
    {BBOX_PROMPT}\
    The final output must be formatted in valid JSON, like: {OUTPUT_TEMPLATE_SWEEP}\
    Note that if trash like paper ball or cleaning tools like dustpan, bursh don't appear in given image, (min_x, max_x, min_y, max_y) will be set to (0, 0, 0, 0)"
# Your goal is to detect all objects related to sweep trash task. Possible objects include Brush, Dustpan and trash.\

# USER_MESSAGE_SWEEP = "Use brush to sweep the pen into the dustpan"
USER_MESSAGE_SWEEP = "Sweep trash on the table"


def vlm_pouring():
    image_name = "random_bottle.jpg"
    if not os.path.exists(os.path.join(DIR_PATH, image_name)):
        print(f"Skip missing: {image_name}")
    USER_MESSAGE_POURWATER = input("请输入倒水任务描述: ").strip()
    if not USER_MESSAGE_POURWATER:
        USER_MESSAGE_POURWATER = "Pour a half of green plastic bottle into the middle mug"
    formatted_messages = generate_prompt(
        SYS_MESSAGE_POURWATER, USER_MESSAGE_POURWATER, os.path.join(DIR_PATH, image_name)
    )
    response = llm.invoke(formatted_messages)
    print(f"Result for {image_name}: {response.content}")
    response_dict = json.loads(response.content)
    # return
    plot_task_plan(
        os.path.join(DIR_PATH, image_name), response_dict, save_path="./sweep_bbox.jpg", show=False
    )
    return response_dict


def vlm_sweeping():
    image_name = "random_bottle.jpg"
    if not os.path.exists(os.path.join(DIR_PATH, image_name)):
        print(f"Skip missing: {image_name}")
    USER_MESSAGE_SWEEP = input("请输入扫地任务描述: ").strip()
    if not USER_MESSAGE_SWEEP:
        USER_MESSAGE_SWEEP = "Sweep trash on the table, pen is trash"
    formatted_messages = generate_prompt(
        SYS_MESSAGE_SWEEP, USER_MESSAGE_SWEEP, os.path.join(DIR_PATH, image_name)
    )
    response = llm.invoke(formatted_messages)
    print(f"Result for {image_name}: {response.content}")
    response_dict = json.loads(response.content)
    # return
    plot_task_plan(
        os.path.join(DIR_PATH, image_name), response_dict, save_path="./sweep_bbox.jpg", show=False
    )
    return response_dict

def get_visual_prompts(task):
    """
    根据任务名称调用 vlm_sweeping 或 vlm_pouring，并将输出整合为指定的 visuals 格式(bounding box)

    Args:
        task (str): 任务名称，支持 "sweeping" 或 "pouring"

    Returns:
        dict: 包含 bboxes 的视觉提示字典
    """
    if task == "sweeping":
        response_dict = vlm_sweeping()  # 调用扫地任务的 VLM 函数
    elif task == "pouring":
        response_dict = vlm_pouring()  # 调用倒水任务的 VLM 函数
    else:
        raise ValueError("Unsupported task. Please use 'sweeping' or 'pouring'.")

    # 提取步骤中的 bounding box 信息
    bboxes = []
    for step in response_dict.get("steps", []):
        if "location" in step and len(step["location"]) == 4:
            bbox = step["location"]
            # 转换为像素坐标（假设输入是归一化坐标）
            bboxes.append(np.array(bbox))

    # 构造 visuals 字典
    visuals = dict(
        bboxes=bboxes
    )

    return visuals


OUTPUT_TEMPLATE_FLIPBOWL = "{'steps': [\
                    {'bowl_opening_state': '...', 'action': '...', 'object_bowl': '...', 'bowl_location': '...'},\
                    ]}"
SYS_MESSAGE_FLIPBOWL = f"You are a reliable and capable robot task planning and image understanding assistant. \
    You receive an image and a natural language instruction intended to guide a robot in completing a specified task in a real-world environment. \
    The task described by the image-language input may vary in complexity. The image depicts a tabletop with a couple of bowls, where some of them are facing upward and others are facing downward, etc. \
    The robot is a dual-arm manipulator, and a primitive skill library is available, including: 'Flipping bowl'. This action is to flip a bowl that is upside down so that its opening would faces upward.\
    User would give you a command related to flip a bowl such as 'flip the white bowl', 'flip the brown bowl', 'flip the plastic bowl', 'flip the bowl at the top', 'flip the bowl at the bottom', ect.\
    These commands and any of possible variants only represent flip bowl task.\
    The field 'object_bowl' expects the description of bowl like 'brown bowl', 'plastic bowl' rather than spatial location of the brow,\
    The field 'bowl_state' expects one of the 'upward' or 'downward',\
    If user prompt is like 'flip the bowl at the bottom', you have to judge what the bowl at the bottom is and then judge if it is upward or downward.\
    If user prompt points out to a specific bowl like 'flip the brown bowl' or 'flip the bowl at the top', your response should only be one entry related to that bowl.\
    If a 'bowl_opening_state' is upward, the field 'action' has to be 'None'.\
    If user prompt says like 'flip the bowl' or 'flip any of bowls', your response should include all of bowls in the image.\
    The final output must be formatted in valid JSON, like: {OUTPUT_TEMPLATE_FLIPBOWL} in a readable way.\
    "
    

SYS_MESSAGE_FINDBOWL = f"You are a reliable and capable robot task planning and image understanding assistant. \
    You receive an image and a natural language instruction intended to guide a robot in completing a specified task in a real-world environment. \
    The task described by the image-language input may vary in complexity. The image depicts a tabletop with a couple of bowls, where some of them are facing upward and others are facing downward, etc. \
    The robot is a dual-arm manipulator, and a primitive skill library is available, including: 'Flipping bowl'. This action is to flip a bowl that is upside down so its opening faces upward.\
    User would give you a command related to flip a bowl such as 'flip the white bowl', 'flip the brown bowl', 'flip the plastic bowl', 'flip the bowl at the top', 'flip the bowl at the bottom', ect.\
    These commands and any of possible variants only represent flip bowl task.\
    The field 'object_bowl' expects the description of bowl like 'brown bowl', 'plastic bowl' rather than spatial location of the brow,\
    The field 'bowl_state' expects one of the 'upward' or 'downward',\
    The field 'bowl_location' expects one of the 'bottom' or 'middle' or 'top',\
    If user prompt is like 'flip the bowl at the bottom', you have to judge what the bowl at the bottom is and then judge if it is upward or downward.\
    If user prompt points out to a specific, unique bowl like 'flip the brown bowl' or 'flip the bowl at the top', your response should only be one entry related to that bowl.\
    If user prompt says like 'flip the bowl' or 'flip any of bowls', your response should include all o bowls in the image.\
    The final output must be formatted in valid JSON, like: {OUTPUT_TEMPLATE_FLIPBOWL}\
    "


def main():  
    args = get_args()
    
    api_key = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"  # please input your api_key here
    llm = ChatOpenAI(
            temperature=args.temperature,
            base_url="https://ark.cn-beijing.volces.com/api/v3/",
            api_key=api_key,
            model_name="doubao-1.5-vision-pro-250328",
        )
    llm = llm.bind(response_format={"type": "json_object"})
    
    c = kingfisher.connect("192.168.4.64") 
    kingfisher.SetAUTO_EXPOSURE()
    image,_ = kingfisher.captureQuarterSize()
    

    image_name = "capture_init.jpg" 
    image_path = os.path.join(image_name)
    cv2.imwrite(image_path, image)
    
    # cv2.namedWindow("MyDebugWindow", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("MyDebugWindow", 1452, 1080)
    # cv2.imshow("MyDebugWindow", image)
    # # print("[*****Finished*****]", 0)
    # if not args.no_video_out:
    #     for _ in range(5): vout.write(final_imgs)
    #         if args.no_interaction:
    #             cv2.waitKey(0)  # wait until we close the plotted window (press Esc to continue)
    #         else:
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break

    # 用户输入自然语言描述
    user_message = input("请输入任务描述: ").strip()

    
    # 调用 LLM 生成任务计划
    if args.scene_id == 0: #pour
        SYS_MESSAGE = SYS_MESSAGE_POURWATER_1
    elif args.scene_id == 1: #sweep
        SYS_MESSAGE = SYS_MESSAGE_SWEEP
    elif args.scene_id == 2: #flip
        SYS_MESSAGE = SYS_MESSAGE_FLIPBOWL

    formatted_messages = generate_prompt(SYS_MESSAGE, user_message, image_path=image_path)
    response = llm.invoke(formatted_messages)
    print(f"LLM Response: {response.content}")
   

    # 解析任务计划
    try:
        response_dict = json.loads(response.content)
        if "unscrew" in response.content.lower():
            task_name = "unscrew_pouring"
        elif "sweep" in response.content.lower():
            task_name = "sweeping"
        elif "pour" in response.content.lower():
            task_name = "pouring"
        elif "bowl" in response.content.lower():
            task_name = "pivoting"
        else:
            print("Not supported this task plan !!!")
            return
        print(task_name)
        
        input("Please enter Enter key to continue... ")
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response: {e}")
        return

    if not task_name:
        print("Task name not found in LLM response!")
        return
    # 根据任务类型调用对应脚本
    if task_name == "pouring":
        mugcup = response_dict["steps"][0]["object_cup"]
        mugcup_id = mugcup_mapping[mugcup]
        # print(mugcup, mugcup_id)
        # 从任务计划中提取参数
        bottle_id = 6  # 示例值，实际应从任务计划中提取
        # mugcup_id = 1  # 示例值，实际应从任务计划中提取
        test_id = 40  # 示例值，实际应根据需求设置
        command = [
            "python", "scripts_kfr/runRealLFHV2.py",
            "--task_name", "pouring",
            "--test_id", str(test_id),
            "--no_interaction",
            "--bottle_id", str(bottle_id),
            "--mugcup_id", str(mugcup_id)
        ]
    elif task_name == "sweeping":
        # 从任务计划中提取参数
        
        # trash_name = "paper ball"  # 示例值，实际应从任务计划中提取
        trash_name = response_dict["steps"][0].get("object")
        test_id = 20  # 示例值，实际应根据需求设置
        command = [
            "python", "scripts_kfr/runRealLFHV4.py",
            "--task_name", "sweeping",
            "--no_interaction",
            "--test_id", str(test_id),
            "--trash_name", trash_name
        ]
    elif task_name == "unscrew_pouring":
        # 从任务计划中提取参数
        mugcup = response_dict["steps"][-1]["object_cup"]
        mugcup_id = mugcup_mapping[mugcup]
        bottle_id = 6  # 示例值，实际应从任务计划中提取
        # mugcup_id = 1  # 示例值，实际应从任务计划中提取
        test_id = 1  # 示例值，实际应根据需求设置
        command = [
            "python", "scripts_kfr/runRealLFHV2.py",
            "--task_name", "unscrew_pouring",
            "--no_interaction",
            "--test_id", str(test_id),
            "--bottle_id", str(bottle_id),
            "--mugcup_id", str(mugcup_id)        
        ]
    elif task_name == "pivoting":
        # 从任务计划中提取参数
        bowl = response_dict["steps"][-1]["object_bowl"]
        bowl_id = bowl_mapping[bowl]
        test_id = 1  # 示例值，实际应根据需求设置
        bowl_opening_state = response_dict["steps"][-1]["bowl_opening_state"]

        if bowl_opening_state=="upward":
            print(f"this bowl is {bowl_opening_state}, there is no need to flip the bowl")
            return
        else:
            command = [
                "python", "BiNoMaP/pivoting.py",
                "--task_name", "cirbowl",
                "--test_id", str(test_id),
                "--cirbowl_id", str(bowl_id),
                "--showing",
                "--gripper"     
            ]
    else:
        print(f"Unsupported task name: {task_name}")
        return

    # 执行命令
    try:
        print(f"Executing command: {' '.join(command)}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=int, required=False)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
