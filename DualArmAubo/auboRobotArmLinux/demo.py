
'''
# 初始化二指夹
The actual port of gripper
$ sudo chmod 666 /dev/ttyACM0

# 初始化机械臂  
ip = "192.168.3.160"  

'''

import os
import logging
import math
import time
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import serial  # pip install pyserial
try:  # for running demo.py alone
    import libpyauboi5
    from robotcontrol import Auboi5Robot
    from robotcontrol import RobotMoveTrackType
except:  # for callbacking demo.py
    from . import libpyauboi5
    from .robotcontrol import Auboi5Robot 
    from .robotcontrol import RobotMoveTrackType



def calculate_bcc(data):  
    """  
    计算BCC校验和  
    """  
    bcc = 0  
    for byte in data:  
        bcc ^= byte  
    return bcc  

def build_command(control_id, control_mode, direction, subdivision, angle, speed):  
    """  
    构建协议数据帧  
    """  
    # 帧头和帧尾  
    frame_head = 0x7B  
    frame_tail = 0x7D

    # 将角度和速度放大10倍  
    angle *= 10  
    speed *= 10  

    # 角度和速度的高八位和低八位  
    angle_high = (angle >> 8) & 0xFF  
    angle_low = angle & 0xFF  
    speed_high = (speed >> 8) & 0xFF  
    speed_low = speed & 0xFF  

    # 构建数据帧（不包括BCC和帧尾）  
    data = [  
        frame_head,  
        control_id,  
        control_mode,  
        direction,  
        subdivision,  
        angle_high,  
        angle_low,  
        speed_high,  
        speed_low  
    ]  

    # 计算BCC校验位  
    bcc = calculate_bcc(data)  

    # 添加BCC和帧尾  
    data.append(bcc)  
    data.append(frame_tail)  

    return bytearray(data) 


class RobotBase:
    def __init__(self, robot):
        self.robot = robot

    def get_current_pose(self):  
        """  
        获取当前机器人的位姿（位置和姿态）  
        返回：当前位姿（位置和姿态），其中位置以米为单位，姿态以角度为单位  
        """  
        current_pos = self.robot.get_current_waypoint()
        pos = current_pos['pos']
        rpy = self.robot.quaternion_to_rpy(current_pos['ori'])
        rx = math.degrees(rpy[0])  
        ry = math.degrees(rpy[1])  
        rz = math.degrees(rpy[2])  
        pose_rad = [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]
        pose_rad = [round(num, 9) for num in pose_rad]
        pose_deg = [pos[0], pos[1], pos[2], rx, ry, rz]
        pose_deg = [round(num, 9) for num in pose_deg]  
        logging.info('返回 6DoF Joint 为弧度单位,机器人可直接运动,当前位置{}'.format(pose_rad))  
        logging.info('返回XYZ以米为单位,角度为弧度,机器人可直接运动,当前位置{}'.format(pose_deg))
        return pose_deg, pose_rad 

    def get_current_joint(self): 
        joint_radian = self.robot.get_current_waypoint()
        return joint_radian['joint']
    
    def speed_init(self, line_acc=0.1, line_velc=0.1):
        self.robot.get_end_max_line_acc()
        self.robot.get_end_max_line_velc()
        self.robot.get_end_max_angle_acc()
        self.robot.get_end_max_angle_velc()
        # 设置机械臂末端最大线加速度(m/s)
        self.robot.set_end_max_line_acc(line_acc)
        # 获取机械臂末端最大线加速度(m/s)
        self.robot.set_end_max_line_velc(line_velc)


    def move_joint(self, joint_radian):
        result = self.robot.move_joint(joint_radian)
        return result
    
    def move_rotate_with_tool(self, z_len=0.161, rot_deg=15, rot_axis="X"):
        '''
        # 获取当前位置
        current_pos = self.robot.get_current_waypoint()
        # 工具转轴的向量（相对于法兰盘，这样需要测量得到x,y,z 本测试样例默认以x=0,y=0,ｚ轴为0.1米）
        tool_pos_on_end = (0, 0, z_len)
        # 工具姿态（w,x,y,z 相对于法兰盘，不知道的情况下，默认填写如下信息）
        tool_ori_on_end = (1, 0, 0, 0)
        tool_desc = {"pos": tool_pos_on_end, "ori": tool_ori_on_end}
        # 得到法兰盘工具末端点相对于基座坐标系中的位置
        tool_pos_on_base = self.robot.base_to_base_additional_tool(
            current_pos['pos'], current_pos['ori'], tool_desc)

        print("current_pos={0}".format(current_pos['pos']))
        print("tool_pos_on_base={0}".format(tool_pos_on_base['pos']))
        
        # 工具转轴向量平移到基座坐标系下(旋转方向符合右手准则)
        rotate_axis = map(lambda a, b: a - b, tool_pos_on_base['pos'], current_pos['pos'])
        rotate_axis = list(rotate_axis)
        print("rotate_axis={0}".format(rotate_axis))
        # sys.exit()
        '''
        
        # 坐标系默认使用基座坐标系（默认填写下面的值就可以了）
        user_coord = {'coord_type': 0, 
            'calibrate_method': 0,
            'calibrate_points':
                {"point1": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                "point2": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                "point3": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
            'tool_desc': {"pos": (0.0, 0.0, z_len), "ori": (1.0, 0.0, 0.0, 0.0)}
        }
        if rot_axis=="X": rotate_axis = (1,0,0)
        if rot_axis=="Y": rotate_axis = (0,1,0)
        if rot_axis=="Z": rotate_axis = (0,0,1)
        
        # 调用转轴旋转接口，最后一个参数为旋转角度（弧度）
        result = self.robot.move_rotate(user_coord, rotate_axis, np.pi*(rot_deg/180.0))
        return result
    
    
    def move_to_target_in_cartesian(self, target_pose):
        # position = target_pose[:3]  # xyz, in median
        # euler_pose = target_pose[3:]  # rpy_xyz, in degree
        # result = self.robot.move_to_target_in_cartesian(position, euler_pose)
        # print(target_pose)
        result = self.robot.move_to_target_in_cartesian(target_pose)
        return result

    async def move_to_target_in_cartesian_async(self, target_pose):
        await self.robot.move_to_target_in_cartesian_async(target_pose)
    
    def inverse_kin(self, joint_radian, target_pose):
        position = target_pose[:3]  # xyz, in median
        euler_pose = target_pose[3:]  # rpy_xyz, in degree
        euler_pose = [ei / 180 * np.pi for ei in euler_pose]  # in radius
        quat_pose = self.robot.rpy_to_quaternion(euler_pose)
        return self.robot.inverse_kin(joint_radian, position, quat_pose)

    def move_trajectory(self, joint_radian_list, is_cartesian=False, robot_move_type=3):
        if is_cartesian:  # need to convert 6-dof joint format into the cartesian pose format
            joint_radian_list_new = []
            for pose_temp in joint_radian_list:
                rpy_xyz = [i / 180.0 * np.pi for i in pose_temp[3:]]  # degree -> radian
                ori_quat = self.robot.rpy_to_quaternion(rpy_xyz)  # euler -> quaternion
                joint_radian = self.robot.get_current_waypoint()  # calculate the joint info
                ik_result = self.robot.inverse_kin(joint_radian['joint'], pose_temp[:3], ori_quat)
                joint_radian_list_new.append(ik_result["joint"])
            joint_radian_list = joint_radian_list_new

        self.robot.remove_all_waypoint()
        
        self.robot.move_joint(joint_radian_list[0])  # must run this for driving move_track() 
        
        for idx, joint_radian in enumerate(joint_radian_list):
            self.robot.add_waypoint(joint_radian)
        self.robot.set_circular_loop_times(0)

        if robot_move_type == 1: ret = self.robot.move_track(RobotMoveTrackType.ARC_CIR)  # relative slow (quite not stable)
        if robot_move_type == 2: ret = self.robot.move_track(RobotMoveTrackType.CARTESIAN_MOVEP)  # relative slow (not stable)
        if robot_move_type == 3: ret = self.robot.move_track(RobotMoveTrackType.CARTESIAN_CUBICSPLINE)  # relative normal (stable)
        if robot_move_type == 4: ret = self.robot.move_track(RobotMoveTrackType.JIONT_CUBICSPLINE)  # very fast speed (stable)

        self.robot.remove_all_waypoint()
        return ret

    @staticmethod 
    def usb_send_gripper(port, baudrate, control_id, control_mode, direction, subdivision, angle, speed):  
        """  
        通过串口发送协议数据帧  
        """  
        # if direction == 0x00 :
        #     print("open gripper",port, baudrate, control_id, control_mode, direction, subdivision, angle, speed)  
        # if direction == 0x01 :
        #     print("close gripper",port, baudrate, control_id, control_mode, direction, subdivision, angle, speed)  
        # 打开串口  
        ser = serial.Serial(port, baudrate, timeout=0.5)  
        # time.sleep(0.2)
        # 构建数据帧  
        command = build_command(control_id, control_mode, direction, subdivision, angle, speed)  
        # time.sleep(0.2)
        # 发送数据帧
        ser.write(command)  

    
        
if __name__ == '__main__':  
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s - Line: %(lineno)d - %(message)s')  
    robot = Auboi5Robot()
    RSHD = libpyauboi5.create_context()
    
    robot_ip = "192.168.4.109"  # ip address
    robot_port = 8899  # port number
    robot.connect(robot_ip, robot_port)
    
    robotBase = RobotBase(robot)
    robotBase.speed_init(line_acc=0.1, line_velc=0.1)  # seem to be useless
    
    
    #############################################################
    print(robotBase.get_current_pose()[0])

    ##### for the left robot arm
    #robotBase.move_to_target_in_cartesian([-0.108549306, -0.180096737, 0.364956553, -89.110557556, -0.773967803, 179.158752441])
    #robotBase.move_to_target_in_cartesian([-0.103300852, -0.270322391, 0.198726398, -91.685516357, -1.117003918, 135.645507812])
    #robotBase.move_to_target_in_cartesian([0.20024374, -0.367381272, 0.198726398, -90.310035706, -0.76980716, 135.316329956])
    #robotBase.move_to_target_in_cartesian([0.099277548, -0.448049293, 0.200160533, -89.451705933, -0.795468807, 136.76512146])
    
    ##### for the right robot arm
    #robotBase.move_to_target_in_cartesian([0.121810637, -0.122385385, 0.405342674, 178.836791992, -1.192816377, -88.780090332])
    #robotBase.move_to_target_in_cartesian([0.036135246, -0.446224408, 0.130150433, 179.012268066, 1.477859974, -179.721832275])
    #robotBase.move_to_target_in_cartesian([0.117455779, -0.372746367, 0.324294368, -179.445739746, -0.530248404, -90.059989929])
    #robotBase.move_to_target_in_cartesian([0.216238177, -0.305605026, 0.236702899, -38.282775879, -87.401573181, 114.295005798])

    #robotBase.move_to_target_in_cartesian([0.126238177, -0.425605026, 0.236702899, -38.282775879, -87.401573181, 114.295005798])
    #robotBase.move_to_target_in_cartesian([0.116238177, -0.435605026, 0.216702899, -38.282775879, -87.401573181, 114.295005798])
    #robotBase.move_to_target_in_cartesian([0.116238177, -0.435605026, 0.236702899, -38.282775879, -87.401573181, 114.295005798])

    
    sys.exit()

    #############################################################
    '''
    robotL = Auboi5Robot()
    robotL.connect("192.168.4.66", 8899)
    robotBaseL = RobotBase(robotL)
    robotBaseL.speed_init(line_acc=0.1, line_velc=0.1)
    target_pose_deg_L = [-0.105173219, -0.43848465, 0.021509144, -99.926361084, -0.649939656, -176.407028198]
    robotBaseL.move_to_target_in_cartesian(target_pose_deg_L)  # for keeping the object stable
    #sys.exit()
    
    current_pose = robotBase.get_current_pose()
    cur_pose_euler, cur_joint_radian = current_pose[0], current_pose[1]
    print(robotBase.get_current_joint())
    
    target_pose0 = [0.112415059, -0.39545583, 0.127148427, -179.830322266, -34.404933929, -87.141654968]
    
    target_pose1 = [0.112415059, -0.49545583, 0.107148427, -179.830322266, -34.404933929, -87.141654968]
    target_pose2 = [0.110072683, -0.50004591, 0.131148427, 179.457855225, -43.850227356, -85.510787964]
    target_pose3 = [0.108072683, -0.51304591, 0.151648427, 178.457855225, -53.850227356, -83.760787964]
    target_pose4 = [0.106072683, -0.53354591, 0.165148427, 177.457855225, -63.850227356, -82.010787964]
    target_pose5 = [0.104072683, -0.557545909, 0.171705941, 176.457855225, -73.850227356, -80.260787964]
    
    target_pose6 = [0.104072683, -0.657545909, 0.176705941, 176.457855225, -73.850227356, -80.260787964]
    target_pose7 = [0.104072683, -0.457545909, 0.376705941, 176.457855225, -73.850227356, -80.260787964]
    target_pose_list = [ target_pose0, target_pose1, target_pose2, target_pose3, target_pose4, target_pose5, target_pose6, target_pose7 ]
    joint_radian_list = []
    for idx, target_pose in enumerate(target_pose_list):
        robotBase.move_to_target_in_cartesian(target_pose)
        cur_joint_6dof = robotBase.get_current_joint()
        joint_radian_list.append(cur_joint_6dof)
        print(idx, cur_joint_6dof)
    
    robotBase.move_joint(joint_radian_list[0])
    robotBase.move_trajectory(joint_radian_list[1:7])
    robotBase.move_joint(joint_radian_list[7])
    
    sys.exit()
    '''
    #############################################################

    #############################################################
    '''
    current_pose = robotBase.get_current_pose()
    cur_pose_euler, cur_joint_radian = current_pose[0], current_pose[1]

    [cur_pose_x, cur_pose_y, cur_pose_z] = cur_pose_euler[:3]
    target_pose1 = [cur_pose_x, cur_pose_y-0.1, cur_pose_z] + cur_pose_euler[3:]
    target_pose2 = [cur_pose_x, cur_pose_y-0.1, cur_pose_z-0.1] + cur_pose_euler[3:]
    target_pose3 = [cur_pose_x, cur_pose_y, cur_pose_z-0.1] + cur_pose_euler[3:]
    target_pose4 = cur_pose_euler
    target_pose_list = [ target_pose1, target_pose2, target_pose3, target_pose4 ]
    joint_radian_list = [robotBase.get_current_joint()]
    for idx, target_pose in enumerate(target_pose_list):
        robotBase.move_to_target_in_cartesian(target_pose)
        cur_joint_6dof = robotBase.get_current_joint()
        joint_radian_list.append(cur_joint_6dof)
        print(idx, cur_joint_6dof)

    robotBase.move_trajectory(joint_radian_list)
    sys.exit()
    '''
    #############################################################
    
    #############################################################
    # joint_radian = (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000)
    # result = robotBase.move_joint(joint_radian)
    
    # current_joint = robotBase.get_current_joint()
    # target_joint = current_joint[:-1] + [current_joint[-1] + np.pi*(30/180.0)]
    # robotBase.move_joint(target_joint)  # Rotate Clockwise
    # time.sleep(1)
    # robotBase.move_joint(current_joint)  # Rotate Counter-Clockwise
    
    # robotBase.move_rotate_with_tool(z_len=0.161, rot_deg=-30, rot_axis="X")  # the eep is not right
    #############################################################
    
    #############################################################
    # pour_deg = 75, [-0.066905751, -0.310918512, 0.262361029, -89.101509094, 0.847340584, 178.873123169]
    # pour_deg = 80, [-0.06268972, -0.291133793, 0.227475017, -88.151908875, -0.363399923, 179.459655762]
    # pour_deg = 85, [-0.061167287, -0.284923031, 0.196612152, -90.7967453, -1.049981475, 179.264755249]
    from utils import apply_primitive_skill
    pour_deg = 80
    target_pose = robotBase.get_current_pose()[0]
    temp_eef_pose_deg = apply_primitive_skill(target_pose, pour_deg, "pour", rot_axis="X")
    robotBase.move_to_target_in_cartesian(temp_eef_pose_deg)  # moved&rotate
    time.sleep(3)
    robotBase.move_to_target_in_cartesian(target_pose)   # back-rotate
    #############################################################

    #############################################################
    # ''' gripper 串口配置 / 控制参数 '''
    # port = "/dev/ttyACM0"  # 修改为实际的串口号  
    # baudrate = 115200  
    # control_id = int("0x01", 16)  
    # control_mode = int("0x02", 16)   
    # direction = int("0x1", 16)       # 1合 0开
    # subdivision = int("0x20", 16)
    # angle = 400 # 1872°
    # speed = 20  # 20 Rad/s
    
    # ''' 发送 gripper 运动指令 '''
    # RobotBase.usb_send_gripper(port, baudrate, control_id, control_mode, direction, subdivision, angle, speed)
    #############################################################

