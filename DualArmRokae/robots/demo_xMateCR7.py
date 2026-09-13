

import sys
import os
import math
import time
import threading
import numpy as np
from copy import copy
from scipy.spatial.transform import Rotation as R

current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)
example_directory = os.path.join(current_directory, 'example')
release_directory = os.path.join(current_directory, 'Release')
linux_directory = os.path.join(release_directory, 'linux')
pyi_l_directory = os.path.join(linux_directory, 'xCoreSDK_python')

sys.path.append(current_directory)
sys.path.append(example_directory)
sys.path.append(release_directory)
sys.path.append(linux_directory)
sys.path.append(pyi_l_directory)

from Release.linux import xCoreSDK_python
from example.log import print_log, print_separator
from example.move_example import pre_op, wait_robot, calcFk, calcIk

sys.path.insert(0, os.getcwd())
from gripper.demo_jodell_rg75 import JodellRG75
from src.util import interpolate_se3_bspline_startend_poses
from src.config import cfg_dict_init as cfg_dict
from src.BiNoMaP import pivot_around_spindle


#################################################################
class rokaeRobotBase():
    def __init__(self, robot_ip):
        self.ec = {}
        robot_instance = xCoreSDK_python.xMateRobot(robot_ip)
        robot_instance.connectToRobot(self.ec)
        robot_info = robot_instance.robotInfo(self.ec)
        print_log("\nrobotInfo (armL)", self.ec, f"{robot_info.id, robot_info.version, robot_info.type, robot_info.joint_num}")
        self.robot = robot_instance
        
    def get_current_pose(self):  
        """  
        获取当前机器人的位姿（位置和姿态）  
        返回：当前位姿（位置和姿态），其中位置以米为单位，姿态以角度为单位  
        """  
        cart_posture = self.robot.cartPosture(xCoreSDK_python.CoordinateType.endInRef, self.ec)
        [tran_x, tran_y, tran_z] = map(str, cart_posture.trans)
        [tran_x, tran_y, tran_z] = [float(tran_x), float(tran_y), float(tran_z)]
        [rad_r, rad_p, rad_y] = map(str, cart_posture.rpy)  # in rad
        [rad_r, rad_p, rad_y] = [float(rad_r), float(rad_p), float(rad_y)]
        [deg_r, deg_p, deg_y] = [rad_r * 180 / np.pi,  rad_p * 180 / np.pi, rad_y * 180 / np.pi]  # in deg
        print("[robot Current Pose](3 position + 3 orientation)\n", tran_x, tran_y, tran_z, deg_r, deg_p, deg_y)
        pose_rad = [tran_x, tran_y, tran_z, rad_r, rad_p, rad_y]
        pose_rad = [round(num, 9) for num in pose_rad]
        pose_deg = [tran_x, tran_y, tran_z, deg_r, deg_p, deg_y]
        pose_deg = [round(num, 9) for num in pose_deg]  
        return pose_deg, pose_rad 

    def set_motion_speed_ratio(self, given_speed=0.2):  # (0% ~ 100%) 20% is the default speed ratio
        self.robot.adjustSpeedOnline(given_speed, self.ec)

    def moving_pre_op(self):  
        pre_op(self.robot, self.ec)  # set automatic / zone / speed
        
    def move_to_a_waypoint(self, target_pose, is_waiting=True, move_type="L"):  # using MoveLCommand / MoveJCommand
        pose_deg = target_pose[3:]
        pose_rad = [pose_deg[0] * np.pi / 180.0, pose_deg[1] * np.pi / 180.0, pose_deg[2] * np.pi / 180.0]
        cart_pos = xCoreSDK_python.CartesianPosition(target_pose[:3] + pose_rad)
        if move_type == "L": movelcmd = xCoreSDK_python.MoveLCommand(cart_pos, 1000, 10)
        if move_type == "J": movelcmd = xCoreSDK_python.MoveJCommand(cart_pos, 1000, 10)
        cmdID = xCoreSDK_python.PyString()
        self.robot.moveAppend([movelcmd], cmdID, self.ec)  # [movelcmd] list, you can add multiple waypoints
        print("Command ID:", cmdID.content())
        print_log("moveAppend", self.ec)
        
        self.robot.moveStart(self.ec)
        print_log("moveStart", self.ec)
        if is_waiting:  # we can set this as False to make a flexible control
            wait_robot(self.robot, self.ec)  # this command can be removed for faster moving
    
    def move_by_trajectory(self, target_pose_list, is_waiting=True, move_type="L"):
        movelcmd_list = []
        for target_pose in target_pose_list:
            pose_deg = target_pose[3:]
            pose_rad = [pose_deg[0] * np.pi / 180.0, pose_deg[1] * np.pi / 180.0, pose_deg[2] * np.pi / 180.0]
            cart_pos = xCoreSDK_python.CartesianPosition(target_pose[:3] + pose_rad)
            if move_type == "L": movelcmd = xCoreSDK_python.MoveLCommand(cart_pos, 1000, 10)
            if move_type == "J": movelcmd = xCoreSDK_python.MoveJCommand(cart_pos, 1000, 10)
            movelcmd_list.append(movelcmd)
        cmdID = xCoreSDK_python.PyString()
        self.robot.moveAppend(movelcmd_list, cmdID, self.ec)  # [movelcmd] list, you can add multiple waypoints
        print("Command ID:", cmdID.content())
        print_log("moveAppend", self.ec)
        
        self.robot.moveStart(self.ec)
        print_log("moveStart", self.ec)
        if is_waiting:  # we can set this as False to make a flexible control
            wait_robot(self.robot, self.ec)  # this command can be removed for faster moving
        

#################################################################


if __name__ == "__main__":
    
    ip_armL = cfg_dict["arm1"]["robot_ip_add"]
    ip_armR = cfg_dict["arm2"]["robot_ip_add"]
    gripper_L = JodellRG75(give_torque=64, given_speed=255); gripper_R = JodellRG75(give_torque=64, given_speed=255)
    gripper_L.connect("/dev/ttyUSB1", 9); gripper_R.connect("/dev/ttyUSB0", 9)
    gripper_L.switch(0, True, given_val=32); gripper_R.switch(0, True, given_val=32)  # for pivoting cirbowl

    
    robotL = rokaeRobotBase(ip_armL)
    cur_pos_deg_L, cur_pos_rad_L = robotL.get_current_pose(); print("cur_pos_deg_L:", cur_pos_deg_L)
    new_pos_deg_L = [0.558413994, 0.185476218, 0.402155227, -89.610881514, 90.474229046, -0.517829792]
    new_pos_deg_L1 = [0.558413994, 0.185476218, 0.402155227, -89.610881514, 45.474229046, -0.517829792]  # ni-shi-zhen
    new_pos_deg_L2 = [0.558413994, 0.185476218, 0.402155227, -89.610881514, 135.474229046, -0.517829792]  # shun-shi-zhen
    #robotL.moving_pre_op(); robotL.set_motion_speed_ratio(0.2); gripper_L.switch(0, True); robotL.move_to_a_waypoint(new_pos_deg_L)

    #os._exit(0)

    new_pos_deg_L01 = [ 0.573171478, 0.382483409, 0.273157832, -179.610881514, 45.474229046, -90.517829792]  # pre-contact
    new_pos_deg_L02 = [ 0.573171478, 0.482483409, 0.173157832, -179.610881514, 45.474229046, -90.517829792]  # start contact

    robotL.moving_pre_op(); robotL.set_motion_speed_ratio(0.2)
    robotL.move_to_a_waypoint(new_pos_deg_L)
    #robotL.move_to_a_waypoint(new_pos_deg_L01)
    #robotL.move_to_a_waypoint(new_pos_deg_L02)
    #robotL.move_to_a_waypoint(new_pos_deg_L)

    #os._exit(0)

    robotR = rokaeRobotBase(ip_armR)
    cur_pos_deg_R, cur_pos_rad_R = robotR.get_current_pose(); print("cur_pos_deg_R:", cur_pos_deg_R)
    new_pos_deg_R = [0.558413994, -0.185476218, 0.402155227, -90.310744971, 90.516336991, -179.648604329]
    new_pos_deg_R1 = [0.558413994, -0.185476218, 0.402155227, -90.310744971, 45.516336991, -179.648604329]  # ni-shi-zhen
    new_pos_deg_R2 = [0.558413994, -0.185476218, 0.402155227, -90.310744971, 135.516336991, -179.648604329]  # shun-shi-zhen
    #robotR.moving_pre_op(); robotR.set_motion_speed_ratio(0.2); gripper_R.switch(0, True); robotR.move_to_a_waypoint(new_pos_deg_R)

    new_pos_deg_R01 = [0.573171478, -0.382483409, 0.273157832, 179.732957532, 45.68246746, 89.739817245]  # pre-contact
    new_pos_deg_R02 = [0.573171478, -0.482483409, 0.173157832, 179.732957532, 45.68246746, 89.739817245]  # start contact

    robotR.moving_pre_op(); robotR.set_motion_speed_ratio(0.2)
    robotR.move_to_a_waypoint(new_pos_deg_R)
    #robotR.move_to_a_waypoint(new_pos_deg_R01)
    #robotR.move_to_a_waypoint(new_pos_deg_R02)
    #robotR.move_to_a_waypoint(new_pos_deg_R)

    
    '''
    
    delta_x, delta_y, delta_z = 0.000, 0.000, 0.050
    
    ec = {}; ip_armL = "192.168.4.220"
    robotL = xCoreSDK_python.xMateRobot(ip_armL)
    robotL.connectToRobot(ec)
    robot_info_L = robotL.robotInfo(ec)
    print_log("\nrobotInfo (armL)", ec, f"{robot_info_L.id,robot_info_L.version,robot_info_L.type,robot_info_L.joint_num}")
    cart_posture_L = robotL.cartPosture(xCoreSDK_python.CoordinateType.endInRef, ec)
    print(f"elbow,{cart_posture_L.elbow}")
    print(f"hasElbow,{cart_posture_L.hasElbow}")
    print(f"confData,f{','.join(map(str,cart_posture_L.confData))}")
    print(f"external size,{len(cart_posture_L.external)}")
    print(f"trans,{','.join(map(str,cart_posture_L.trans))}")  # x, y, z
    print(f"rpy,{','.join(map(str,cart_posture_L.rpy))}")
    print(f"pos,{','.join(map(str,cart_posture_L.pos))}")
    
    pos = robotL.posture(xCoreSDK_python.CoordinateType.endInRef, ec)
    print_log("posture (armL)", ec, ', '.join(map(str, pos)))

    [tran_x, tran_y, tran_z] = map(str, cart_posture_L.trans)
    [tran_x, tran_y, tran_z] = [float(tran_x), float(tran_y), float(tran_z)]
    [pose_r, pose_p, pose_y] = map(str, cart_posture_L.rpy)  # in rad
    [pose_r, pose_p, pose_y] = [float(pose_r), float(pose_p), float(pose_y)]
    [deg_r, deg_p, deg_y] = [pose_r * 180 / np.pi,  pose_p * 180 / np.pi, pose_y * 180 / np.pi]  # in deg
    print("robotCurrentPose (armL)\n", tran_x, tran_y, tran_z, deg_r, deg_p, deg_y)

    cart_pos = xCoreSDK_python.CartesianPosition([tran_x+delta_x, tran_y-delta_y, tran_z+delta_z, pose_r, pose_p, pose_y])
    movelcmd = xCoreSDK_python.MoveLCommand(cart_pos, 1000, 10)
    cmdID = xCoreSDK_python.PyString()
    robotL.moveAppend([movelcmd], cmdID, ec)  # [movelcmd] list, you can add multiple waypoints
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    pre_op(robotL, ec); robotL.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robotL, ec)


    ##################################


    ec = {}; ip_armR = "192.168.4.221"
    robotR = xCoreSDK_python.xMateRobot(ip_armR)
    robotR.connectToRobot(ec)
    robot_info_R = robotR.robotInfo(ec)
    print_log("\nrobotInfo (armR)", ec, f"{robot_info_R.id,robot_info_R.version,robot_info_R.type,robot_info_R.joint_num}")
    cart_posture_R = robotR.cartPosture(xCoreSDK_python.CoordinateType.endInRef, ec)
    print(f"elbow,{cart_posture_R.elbow}")
    print(f"hasElbow,{cart_posture_R.hasElbow}")
    print(f"confData,f{','.join(map(str,cart_posture_R.confData))}")
    print(f"external size,{len(cart_posture_R.external)}")
    print(f"trans,{','.join(map(str,cart_posture_R.trans))}")  # x, y, z
    print(f"rpy,{','.join(map(str,cart_posture_R.rpy))}")
    print(f"pos,{','.join(map(str,cart_posture_R.pos))}")
    
    pos = robotR.posture(xCoreSDK_python.CoordinateType.endInRef, ec)
    print_log("posture (armR)", ec, ', '.join(map(str, pos)))
    
    [tran_x, tran_y, tran_z] = map(str, cart_posture_R.trans)
    [tran_x, tran_y, tran_z] = [float(tran_x), float(tran_y), float(tran_z)]
    [pose_r, pose_p, pose_y] = map(str, cart_posture_R.rpy)  # in rad
    [pose_r, pose_p, pose_y] = [float(pose_r), float(pose_p), float(pose_y)]
    [deg_r, deg_p, deg_y] = [pose_r * 180 / np.pi,  pose_p * 180 / np.pi, pose_y * 180 / np.pi]  # in deg
    print("robotCurrentPose (armR)\n", tran_x, tran_y, tran_z, deg_r, deg_p, deg_y)

    cart_pos = xCoreSDK_python.CartesianPosition([tran_x+delta_x, tran_y+delta_y, tran_z+delta_z, pose_r, pose_p, pose_y])
    movelcmd = xCoreSDK_python.MoveLCommand(cart_pos, 1000, 10)
    cmdID = xCoreSDK_python.PyString()
    robotR.moveAppend([movelcmd], cmdID, ec)  # [movelcmd] list, you can add multiple waypoints
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    pre_op(robotR, ec); robotR.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robotR, ec)

    '''
    
