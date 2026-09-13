'''
# 初始化二指夹
The actual port of gripper
$ sudo chmod 666 /dev/ttyACM0
windows机器，请查找设备管理器
port = "COMxx"

# 初始化机械臂  
ip = "192.168.3.160"  
'''

######################
### 只能使用Python 3.7
######################


import logging
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import serial  # pip install pyserial
# import libpyauboi5
# from robotcontrol import Auboi5Robot
from . import libpyauboi5
from .robotcontrol import Auboi5Robot 

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
        logging.info('当前位置{},返回XYZ以米为单位,角度为弧度,机器人可直接运动'.format(pose_deg))  
        return pose_deg, pose_rad 

    def speed_init(self):
        self.robot.get_end_max_line_acc()
        self.robot.get_end_max_line_velc()
        self.robot.get_end_max_angle_acc()
        self.robot.get_end_max_angle_velc()
        # 设置机械臂末端最大线加速度(m/s)
        self.robot.set_end_max_line_acc(0.2)
        # 获取机械臂末端最大线加速度(m/s)
        self.robot.set_end_max_line_velc(0.1)
    
    def move_joint(self, joint_radian):
        result = self.robot.move_joint(joint_radian)
        return result
    
    def move_to_target_in_cartesian(self, target_pose):
        position = target_pose[:3]  # xyz, in median
        euler_pose = target_pose[3:]  # rpy_xyz, in degree
        result = self.robot.move_to_target_in_cartesian(
            position, euler_pose)

        return result

    def base_to_base_additional_tool(self, flange_pos, flange_ori, user_tool):
        ret = self.robot.base_to_base_additional_tool(flange_pos, flange_ori, user_tool)
        return ret
    
    def rpy_to_quaternion(self, rpy):
        result = self.robot.rpy_to_quaternion(rpy)
        return result
    
    def robot_offset(self, robot_pose:list, offset:list, euler="XYZ"):    
        '''
        robot_pose:机器人要走的坐标 [x,y,z,a,b,c]
        offset:工具坐标值[x_offest,y_offest,z_offest,a_offest,b_offest,c_offest]
        euler：机器人欧拉角
        '''
        if euler == "ZYX":  
            x, y, z, rz, ry, rx = robot_pose  
            rz_offset, ry_offset, rx_offset = offset[3:]  
        elif euler == "XYZ":  
            x, y, z, rx, ry, rz = robot_pose  
            rx_offset, ry_offset, rz_offset = offset[3:]  
        else:  
            raise ValueError("Unsupported Euler convention: {}".format(euler))   
        x_offset, y_offset, z_offset = offset[:3]  
        offset_vector = np.array([x_offset, y_offset, z_offset])  
        rotation = R.from_euler(euler, [rz, ry, rx], degrees=True)  # 假设输入的旋转角度是以度为单位  
        rotation_matrix = rotation.as_matrix()  
        local_offset = rotation_matrix.dot(offset_vector)  
        x_new = x + local_offset[0]  
        y_new = y + local_offset[1]  
        z_new = z - local_offset[2]  
        rz_new = rz + rz_offset  
        ry_new = ry + ry_offset  
        rx_new = rx + rx_offset  
        formatted_string = ",".join(["{:.2f}".format(x) for x in local_offset])  

        logging.info(f"沿{offset[:3]}方向, xyz各修正:{formatted_string},旋转偏移增加:{rx_offset}, {ry_offset}, {rz_offset}")  

        if euler == "ZYX":  
            return [x_new, y_new, z_new, rz_new, ry_new, rx_new]  
        elif euler == "XYZ":  
            return [x_new, y_new, z_new, rx_new, ry_new, rz_new]
       

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
    
    aubo_robot = Auboi5Robot()
    aubo_robot.initialize()
    aubo_robot.create_context()
    print("aubo_robot.rshd:", aubo_robot.rshd)

    robot_ip = "192.168.31.134"  # 服务器 IP 地址
    robot_port = 8899  # port number
    a = aubo_robot.connect(robot_ip, robot_port)

    robotBase = RobotBase(aubo_robot) 
    robotBase.speed_init()
    # current_pose, current_pose_rad = robotBase.get_current_pose()
    # current_pose[0]=current_pose[0]-0.05
    robotBase.move_to_target_in_cartesian(current_pose)

    ''' gripper 串口配置 / 控制参数 '''
    # port = "/dev/ttyACM0"  # 修改为实际的串口号, ubuntu 
    port = "COM3"  # 修改为实际的串口号, windows
    baudrate = 115200  
    control_id = int("0x01", 16)  
    control_mode = int("0x02", 16)   
    direction = int("0x1", 16)       # 1合 0开
    subdivision = int("0x20", 16)
    angle = 800 # 1872°
    speed = 40  # 20 Rad/s
    
    ''' 发送 gripper 运动指令 '''
    RobotBase.usb_send_gripper(port, baudrate, control_id, control_mode, direction, subdivision, angle, speed)
