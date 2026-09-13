
#######################################################################
SYS_MESSAGE_BOTTLE_CAP_CHECK = "As you can see, here is a RGB image with a bottle on the table. Please help me to check if there is a cap on this bottle. \
    You can answer: (YES) having a cap on it; (NO) not having a cap on it. Please choose one answer YES or NO as the corresponding checking result. \
    The final output must be formatted in valid JSON, like: {'bottle_cap': 'YES'} or {'bottle_cap': 'NO'}. "

SYS_MESSAGE_BOTTLE_3_STATES = "As you can see, here is a RGB image with a bottle on the table. \
    Please help me to judge the state of it. The bottle has three possible states: (A) upright; (B) lying down; (C) upside down. \
    Please choose one answer from A, B, or C as the corresponding checking state of the bottle. \
    The final output must be formatted in valid JSON, like: {'bottle': 'A'} or {'bottle': 'B'} or {'bottle': 'C'}. "
SYS_MESSAGE_BOTTLE_2_STATES = "As you can see, here is a RGB image with a bottle on the table. \
    Please help me to judge the state of it. The bottle has two possible states: (A) upright; (B) lying down. \
    Please choose one answer from A or B as the corresponding checking state of the bottle. \
    The final output must be formatted in valid JSON, like: {'bottle': 'A'} or {'bottle': 'B'}. "

SYS_MESSAGE_MUGCUP_3_STATES = "As you can see, here is a RGB image with a mug cup on the table. \
    Please help me to judge the state of it. The mug cup has three possible states: (A) upright; (B) lying down; (C) upside down. \
    Please choose one answer from A, B, or C as the corresponding checking state of the mug cup. \
    The final output must be formatted in valid JSON, like: {'mug': 'A'} or {'mug': 'B'} or {'mug': 'C'}. "
SYS_MESSAGE_MUGCUP_2_STATES = "As you can see, here is a RGB image with a mug cup on the table. \
    Please help me to judge the state of it. The mug cup has two possible states: (A) upright; (B) lying down. \
    Please choose one answer from A or B as the corresponding checking state of the mug cup. \
    The final output must be formatted in valid JSON, like: {'mug': 'A'} or {'mug': 'B'}. "

SYS_MESSAGE_BOTTLE_3_STATES_2_CAP = "As you can see, here is a RGB image with a bowl on the table. \
    Please help me to judge the state of it. The bottle has three possible states: (A) upright; (B) lying down; (C) upside down. \
    Please choose one answer from A, B, or C as the corresponding checking state of the bottle. \
    Please also help me to check if there is a cap on this bottle. You can answer: (YES) having a cap on it; (NO) not having a cap on it. \
    Please choose one answer from YES or NO as the corresponding checking result of the bottle cap.\
    The final output must be formatted in valid JSON, like: {'bottle': 'A', 'has_cap': 'NO'} or {'bottle': 'B', 'has_cap': 'YES'}."
SYS_MESSAGE_BOTTLE_2_STATES_2_CAP = "As you can see, here is a RGB image with a bowl on the table. \
    Please help me to judge the state of it. The bottle has two possible states: (A) upright; (B) lying down. \
    Please choose one answer from A or B as the corresponding checking state of the bottle. \
    Please also help me to check if there is a cap on this bottle. You can answer: (YES) having a cap on it; (NO) not having a cap on it. \
    Please choose one answer from YES or NO as the corresponding checking result of the bottle cap.\
    The final output must be formatted in valid JSON, like: {'bottle': 'A', 'has_cap': 'NO'} or {'bottle': 'B', 'has_cap': 'YES'}."

SYS_MESSAGE_BOWL_2_STATES = "As you can see, here is a RGB image with a bowl on the table. \
    Please help me to judge the state of it. The bowl has two possible states: (A) upright; (B) inverted. \
    Please choose one answer from A or B as the correspoding checking state of this bowl.\
    The final output must be formatted in valid JSON, like: {'bowl': 'A'} or {'bowl': 'B'}. "


#######################################################################
SYS_MESSAGE_POURING_V1 = "As you can see, here is a RGB image with a bottle (leftside) and a mug cup (rightside) on the table. Please help me to judge their states. \
    (1) The bottle's state is categoried into three types: (A) upright on the table; (B) lying down on the table; (C) upside down on the table. \
    (2) The mug cup's state is categoried into three types: (A) upright on the table; (B) lying down on the table; (C) upside down on the table. \
    (3) Please also check whether there is a cap on this bottle or not. You can answer: (YES) having a cap on the bottle; (NO) not having a cap on the bottle. \
    The final output must be formatted in valid JSON, such as: {'bottle': 'A', 'cup': 'A', 'has_cap': 'NO'} or {'bottle': 'B', 'cup': 'A', 'has_cap': 'YES'} or \
    {'bottle': 'C', 'mug': 'B', 'has_cap': 'YES'} or {'bottle': 'A', 'mug': 'C', 'has_cap': 'NO'}. "

SYS_MESSAGE_POURING_V2 = "You are a reliable and capable robot task planning and image understanding assistant. \
    You receive an image which depicts a tabletop with two objects, including a bottle and a mug cup. \
    Your task is to observe these two objects and check their states. For a mug cup, its possible states are upside down, lying down and upright. \
    For a bottle, its states also include upside down, lying down and upright. User would give you a command related to query objectt's state such as \
    'check object's state', 'judge the cup state', 'judge the bottle state', 'check the state of bowl', ect.\
    These commands and any of possible variants only represent check objects state. \
    (1) The bottle's state has three categories: (A) upright; (B) lying down; (C) upside down. \
    (2) The mug cup's state has three categories: (A) upright; (B) lying down; (C) upside down. \
    (3) Please also check whether there is a cap on this bottle or not. You can answer: (YES) having a cap; (NO) not having a cap. \
    The final output must be formatted in valid JSON, such as: {'bottle': 'A', 'cup': 'A', 'has_cap': 'NO'} or \
    {'bottle': 'B', 'cup': 'A', 'has_cap': 'YES'} or {'bottle': 'C', 'cup': 'B', 'has_cap': 'YES'} or \
    {'bottle': 'A', 'cup': 'C', 'has_cap': 'NO'}. "

#######################################################################
OUTPUT_TEMPLATE_CHECKSTATE = "{'steps': [\
                    {'object_name': '...', ''state': '...'},\
                    ]}"
SYS_MESSAGE_CHECKSTATE = f"You are a reliable and capable robot task planning and image understanding assistant. \
    You receive an image and a natural language instruction intended to guide a robot in completing a specified task in a real-world environment. \
    The task described by the image-language input may vary in complexity. \
    The image depicts a tabletop with just single object, including cup, bowl or bottle. \
    Your task is to observe this object and check it state. \
    If object of the image is cup, its possible states are upside down, lying down and upright. \
    For bowl, its possible states are upward and downward. \
    For a bottle, its states include upside down, lying down and upright. \
    User would give you a command related to query objectt's state such as \
    'check object's state', 'judge the cup state', 'judge the bottle state', 'check the state of bowl', ect.\
    These commands and any of possible variants only represent check objects state.\
    The field 'object_name' expects the name of the object like 'bowl', 'cup', 'bottle'.\
    If the field 'object_name' is cup, the field 'state' choose from upside down, lying down and upright.\
    If the field 'object_name' is bowl, the field 'state' choose from upward and downward.\
    If the field 'object_name' is bottle, the field 'state' choose from upside down, lying down and upright.\
    The final output must be formatted in valid JSON, like: {OUTPUT_TEMPLATE_CHECKSTATE} in a readable way.\
    "

#######################################################################
