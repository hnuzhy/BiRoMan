'''运动相关的示例'''
import time
import math
import sys
import setup_path
import platform
# 根据操作系统导入相应的模块
if platform.system() == 'Windows':
    from Release.windows import xCoreSDK_python
    from Release.windows.xCoreSDK_python.EventInfoKey import MoveExecution
elif platform.system() == 'Linux':
    from Release.linux import xCoreSDK_python
    from Release.linux.xCoreSDK_python.EventInfoKey import MoveExecution
else:
    raise ImportError("Unsupported operating system")
from log import print_log, print_separator
from datetime import timedelta

M_PI = math.pi


def move_op(robot, ec):
    print_separator("move_op", length=110)
    pre_op(robot, ec)
    calcFk(robot, ec)
    calcIk(robot, ec)
    move_absJ(robot, ec)
    moveL(robot, ec)
    moveJ(robot, ec)
    moveC(robot, ec)
    moveCF(robot, ec)
    moveSP(robot, ec)
    moveWait(robot, ec)
    pause_and_continue(robot, ec)
    print_move_info(get_move_info(robot, ec))
    query_controller_log(robot, ec)
    robot.setNoneEventWatcher(xCoreSDK_python.Event.moveExecution,
                              ec)  # 结束时需要关闭监视器


def pre_op(robot, ec):
    '''预操作'''
    print_separator("pre_op", length=80)
    # 切换到自动模式并上电
    #robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)
    print_log("setOperateMode", ec)
    robot.setPowerState(True, ec)
    print_log("setPowerState", ec)
    # 设置默认运动速度和转弯区
    robot.setMotionControlMode(
        xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    print_log("setMotionControlMode", ec)
    set_default_zone(robot, ec)
    set_default_speed(robot, ec)
    # 可选：设置运动指令执行完成和错误信息回调
    robot.setEventWatcher(xCoreSDK_python.Event.moveExecution, print_move_info,
                          ec)


def set_default_zone(robot, ec):
    '''设置默认转弯区'''
    print_separator("set_default_zone", length=80)
    robot.setDefaultZone(50, ec)
    # 可选：设置默认转弯区
    print_log("setDefaultZone", ec)


def set_default_speed(robot, ec):
    '''设置默认速度'''
    print_separator("set_default_speed", length=80)
    robot.setDefaultSpeed(200, ec)
    # 可选：设置默认速度
    print_log("setDefaultSpeed", ec)


def calcFk(robot, ec):
    '''计算正解，关节角度->笛卡尔坐标'''
    print_separator("calcFk", length=80)
    start_angle = [0, 0.557737, -1.5184888, 0, -1.3036738, 0]  # 单位弧度
    robot_model = robot.model()
    toolset = xCoreSDK_python.Toolset()  #新建toolset
    cart_pose = robot_model.calcFk(start_angle, toolset, ec)
    print_log("calcFk", ec)
    print(f"elbow,{cart_pose.elbow}")
    print(f"hasElbow,{cart_pose.hasElbow}")
    print(f"confData,f{','.join(map(str,cart_pose.confData))}")
    print(f"external size,{len(cart_pose.external)}")
    print(f"trans,{','.join(map(str,cart_pose.trans))}")
    print(f"rpy,{','.join(map(str,cart_pose.rpy))}")
    print(f"pos,{','.join(map(str,cart_pose.pos))}")


def calcIk(robot, ec):
    '''计算逆解，笛卡尔坐标 -> 关节角度'''
    print_separator("calcIk", length=80)
    cart_pos = xCoreSDK_python.CartesianPosition(
        [0.60, 0.13, 0.41, -3.14, -0.23, -3.14])  #s4点位
    robot_model = robot.model()
    toolset = xCoreSDK_python.Toolset()  #新建toolset
    joint_pos = robot_model.calcIk(cart_pos, toolset, ec)
    print_log("calcIk", ec, ','.join(map(str, joint_pos)))


def move_absJ(robot, ec):
    '''moveAbsJ运动'''
    print_separator("move_absJ", length=80)
    joint_pos = xCoreSDK_python.JointPosition([1, 1, 1, 1, 1, 1])
    absjcmd = xCoreSDK_python.MoveAbsJCommand(joint_pos, 1000, 10)
    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend([absjcmd], cmdID, ec)  # [absjcmd]指令列表，可以添加多条指令，须为同类型指令
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)


def moveL(robot, ec):
    '''moveL运动'''
    print_separator("moveL", length=80)
    cart_pos = xCoreSDK_python.CartesianPosition(
        [0.614711, 0.136, 0.416211, -1.57, 0, -1.57])  #s4点位
    movelcmd = xCoreSDK_python.MoveLCommand(cart_pos, 1000, 10)
    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend([movelcmd], cmdID, ec)  # [movelcmd]指令列表，可以添加多条指令，须为同类型指令
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)


def moveJ(robot, ec):
    '''moveJ运动'''
    print_separator("moveJ", length=80)
    cart_pos = xCoreSDK_python.CartesianPosition(
        [0.614711, 0.136, 0.416211, -M_PI, 0, -M_PI])  #s4点位
    movejcmd = xCoreSDK_python.MoveJCommand(cart_pos, 1000, 10)
    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend([movejcmd], cmdID, ec)  # [movejcmd]指令列表，可以添加多条指令，须为同类型指令
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)


def moveC(robot, ec):
    '''moveC运动'''
    print_separator("moveC", length=80)
    target = xCoreSDK_python.CartesianPosition(
        [0.214711, 0.236, 0.616211, -M_PI, 0, -M_PI])  #s4点位
    aux = xCoreSDK_python.CartesianPosition(
        [0.414711, 0.236, 0.416211, -M_PI, 0, -M_PI])  #s4点位
    moveccmd = xCoreSDK_python.MoveCCommand(target, aux, 1000, 10)
    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend([moveccmd], cmdID, ec)
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)


def moveCF(robot, ec):
    '''moveCF运动'''
    print_separator("moveCF", length=80)
    target = xCoreSDK_python.CartesianPosition(
        [0.614711, 0.136, 0.416211, -1.57, 0, -1.57])  #s4点位
    aux = xCoreSDK_python.CartesianPosition(
        [0.614711, 0.236, 0.416211, -1.57, 0, -1.57])  #s4点位
    movecf_cmd = xCoreSDK_python.MoveCFCommand(target, aux, 2, 1000, 10)
    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend([movecf_cmd], cmdID, ec)
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)


def moveSP(robot, ec):
    '''moveSP运动'''
    print_separator("moveSP", length=80)
    target = xCoreSDK_python.CartesianPosition(
        [0.214711, 0.136, 0.416211, -M_PI, 0, -M_PI])  #s4点位
    r0 = 0.1
    rstep = 0.05
    angle = 1
    dir = True
    movesp_cmd = xCoreSDK_python.MoveSPCommand(target, r0, rstep, angle, dir,
                                               1000)
    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend([movesp_cmd], cmdID, ec)
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)


def moveWait(robot, ec):
    '''moveWait等待'''
    print_separator("moveWait", length=80)
    robot.stop(ec)
    robot.moveReset(ec)
    joint_pos1 = xCoreSDK_python.JointPosition([0, 0, 0, 0, 0, 0])
    absjcmd1 = xCoreSDK_python.MoveAbsJCommand(joint_pos1, 1000, 10)
    joint_pos2 = xCoreSDK_python.JointPosition([1, 1, 1, 1, 1, 1])
    absjcmd2 = xCoreSDK_python.MoveAbsJCommand(joint_pos2, 1000, 10)

    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend([absjcmd1], cmdID, ec)
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    robot.moveStart(ec)
    print_log("moveStart", ec)

    # delay = timedelta(seconds=2)  # 延时两秒
    delay = timedelta(milliseconds=500)  # 延时500毫秒
    mwait = xCoreSDK_python.MoveWaitCommand(delay)
    robot.moveAppend(mwait, cmdID, ec)  # moveAppend使用moveWait指令时没有列表的形式
    print_log("moveAppend", ec)

    robot.moveAppend([absjcmd2], cmdID, ec)
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)


def wait_robot(robot, ec):
    '''等待运动结束 - 通过查询机械臂是否处于运动中的方式'''
    print_separator("wait_robot", length=80)
    running = True
    while (running):
        time.sleep(0.1)
        st = robot.operationState(ec)
        if (st == xCoreSDK_python.OperationState.idle
                or st == xCoreSDK_python.OperationState.unknown):
            running = False
    print("move finished")


def pause_and_continue(robot, ec):
    '''暂停和继续'''
    print_separator("pause and continue", length=80)
    joint_pos = xCoreSDK_python.JointPosition([1, 1, 1, 1, 1, 1])
    absjcmd = xCoreSDK_python.MoveAbsJCommand(joint_pos, 1000, 10)
    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend([absjcmd], cmdID, ec)  # [absjcmd]指令列表，可以添加多条指令，须为同类型指令
    print_log("moveAppend", ec)
    robot.moveStart(ec)
    print("start")
    time.sleep(2)  # 等待2秒后暂停
    robot.stop(ec)
    print("pause")
    time.sleep(2)  # 等待2秒后继续
    robot.moveStart(ec)
    print("continue")
    time.sleep(2)  # 等待2秒后结束
    robot.stop(ec)
    print("stop")


def query_controller_log(robot, ec):
    '''查询控制器日志'''
    print_separator("query controller log", length=80)
    # 查询最近5条错误级别控制器日志
    controller_logs = robot.queryControllerLog(
        5, {xCoreSDK_python.LogInfoLevel.error}, ec)
    print_log("queryControllerLog", ec)
    for log in controller_logs:
        print(log.content)


def get_move_info(robot, ec):
    '''获取运动信息'''
    print_separator("get move info", length=80)
    info = robot.queryEventInfo(xCoreSDK_python.Event.moveExecution, ec)
    print_log("queryEventInfo", ec)
    return info


def print_move_info(info: dict):
    '''打印运动执行信息'''
    print_separator("print move info", length=80)
    print(f"{MoveExecution.ID}:{info[MoveExecution.ID]}")
    print(f"{MoveExecution.ReachTarget}:{info[MoveExecution.ReachTarget]}")
    print(f"{MoveExecution.WaypointIndex}:{info[MoveExecution.WaypointIndex]}")
    print(f"{MoveExecution.Error}:{info[MoveExecution.Error]}")
    print(f"{MoveExecution.Remark}:{info[MoveExecution.Remark]}")


def main():
    try:
        ip = "192.168.0.160"
        robot = xCoreSDK_python.xMateRobot(ip)
        ec = {}
        move_op(robot, ec)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
