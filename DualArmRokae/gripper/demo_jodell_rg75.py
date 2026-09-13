import os
import sys
import time
from os.path import join, abspath, dirname

work_path = dirname(dirname(abspath(__file__)))
file_path = os.path.join(work_path, "gripper")
sys.path.append(work_path)
sys.path.append(file_path)
from jodell_hardware_interface import ClawRgTool


# class JodellRG75(BaseToolControl):
class JodellRG75():
    def __init__(self, give_torque=64, given_speed=128):
        super().__init__()
        self.speed = given_speed  # 0->255 slow->fast, default 155
        self.torque = give_torque  # 0->255 soft->hard, default 255, we set it to 64 for pouring; 128 for unscrew
        # TODO 初始化加入使能 sudo chmod a+rw /dev/ttyACM0  插拔需要重新使能
        # pos = 0 # 0->255 open->close
        # 实例化
        self.temp_tool = ClawRgTool()
        # 寻找端口
        # =========================
        # self.com = self.temp_tool.searchCom()[-1] # 请手动确认在pc上的端口
        # print((self.com))
        # # 连接
        # self.temp_tool.serialOperation(self.com, 115200, True)
        # =========================

    def connect(self, com:str, tool_id):
        self.com = com  # 请手动确认在pc上的端口
        # 连接
        self.temp_tool.serialOperation(self.com, 115200, True)
        self.temp_tool.clawEnable(tool_id, True)
        self.tool_id = tool_id

    def set_speed(self, speed: int) -> None:
        """
        设置夹爪速度。

        :param speed: 夹爪的速度值，必须是0到255之间的整数。 0->255 slow->fast
        :raises ValueError: 如果速度值不在0到255之间，则抛出异常。
        """
        if not 0 <= speed <= 255:
            raise ValueError("速度值必须在0到255之间")
        self.speed = speed

    def set_torque(self, torque: int) -> None:
        """
        设置夹爪扭矩。

        :param torque: 夹爪的扭矩值，必须是0到255之间的整数。 0->255 soft->hard
        :raises ValueError: 如果扭矩值不在0到255之间，则抛出异常。
        """
        if not 0 <= torque <= 255:
            raise ValueError("扭矩值必须在0到255之间")
        self.torque = torque

    def switch(self, value: bool, wait_finish: bool, given_val=None):
        """
        运行夹爪

        :param IO: 控制夹持器开闭的IO信号，'0'代表打开，'1'代表关闭
        :param is_wait: 是否等待夹持器操作完成后再继续执行后续程序

        返回值:
        无
        """

        open_val_uint = 255-given_val if given_val is not None else 0

        close_val_uint = given_val if given_val is not None else 255

        map = {
            False: open_val_uint,  # default is 0
            True: close_val_uint,  # default is 255
        }

        # 操作
        self.temp_tool.runWithParam(self.tool_id, pos=map[value], speed=self.speed, torque=self.torque)

        # {
        #     0: '未检测到物体',
        #     1: '手指在张开检测到物体',
        #     2: '手指在闭合检测到物体',
        #     3: '手指已到达指定的位置，没有检测到物体'
        # }

        # =========================================
        if wait_finish:
            # print('执行中...')

            # while True:
            #     res = self.temp_tool.getClawCurrentStatus(1)
            #     print(res)
            #     if res == 1:
            #         break
            while True:
            # while self.temp_tool.getClawCurrentStatus(1) == 0:  # BUG：想读取到状态，需要先修改手抓的savedID
                res = self.temp_tool.getClawCurrentStatus(self.tool_id)
                time.sleep(0.01)
                res2 = self.temp_tool.getClawCurrentStatus(self.tool_id)
                if res ==res2 and res != 0:
                    break
            # time.sleep(2)

            # sys.stdout.write('\x1b[1A')  # 光标上移一行
            # sys.stdout.write('\x1b[2K')  # 清除整行内容
            # sys.stdout.flush()           # 确保立即生效
        # print('Done')
        # ================================

        # res = temp_tool.getClawCurrentStatus(1)[0]
        # print(res)

         # 释放串口
        # self.temp_tool.serialOperation(self.com, 115200, False)

    def get_openness(self):  # int, 0 ~ 255 [ has bug !!! ]
        return self.temp_tool.getClawCurrentLocation(self.tool_id)


# sudo chmod 777 /dev/ttyUSB0
# sudo chmod 777 /dev/ttyUSB1

if __name__ == '__main__':
    # import time
    jc1 = JodellRG75()
    jc1.connect("/dev/ttyUSB0", 0)  # for the right-arm, set tool_id=0
    jc1.switch(0, True, given_val=255)  # open gripper, open_val is from 255 to 1
    time.sleep(1)  # sleep 1 seond
    #jc1.switch(1, True)  # close gripper
    
    jc2 = JodellRG75()
    jc2.connect("/dev/ttyUSB1", 0)  # for the left-arm, set tool_id=0
    jc2.switch(0, True)  # open gripper
    time.sleep(1)  # sleep 1 seond
    #jc2.switch(1, True)  # close gripper
    
    # #
    # p = True
    # while True:
    #     jc.switch(p, True)
    #     jc2.switch(p, True)
    #
    #     p = not p
    #     input('next')

