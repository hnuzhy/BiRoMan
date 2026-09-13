
import time
from jodellSdk.jodellSdkDemo import RgClawControl


class JodellRG75():

    def __init__(self, give_torque=64, given_speed=128, give_pos=255):
        super().__init__()
        self.pos = give_pos  # 0->255 close->open, default 255
        self.speed = given_speed  # 0->255 slow->fast, default 128
        self.torque = give_torque  # 0->255 soft->hard, default 64

        self.temp_tool = RgClawControl()
        # 寻找端口
        # =========================
        # self.com = self.temp_tool.searchCom()[-1] # 请手动确认在pc上的端口
        # print((self.com))
        # # 连接
        # self.temp_tool.serialOperation(self.com, 115200, True)
        # =========================

    def connect(self, com:str, tool_id):
        self.com = com  # 请手动确认在pc上的端口
        self.temp_tool.serialOperation(self.com, 115200, True)
        self.temp_tool.enableClamp(tool_id, True)
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
        """运行夹爪
        :param IO: 控制夹持器开闭的IO信号，'0'代表打开，'1'代表关闭
        :param is_wait: 是否等待夹持器操作完成后再继续执行后续程序
        返回值: 无
        """

        open_val_uint = 255 - given_val if given_val is not None else 0
        close_val_uint = 255 - given_val if given_val is not None else 255
        map = {
            False: open_val_uint,  # default is 0
            True: close_val_uint,  # default is 255
        }

        # 操作
        self.temp_tool.runWithParam(self.tool_id, pos=map[value], speed=self.speed, torque=self.torque)

        # =========================================
        if wait_finish:
            res1 = self.temp_tool.getClampCurrentState(self.tool_id)[0]
            print(res1)
            while True:
                res2 = self.temp_tool.getClampCurrentState(self.tool_id)[0]
                time.sleep(0.1)
                print(res2)
                if res1 != res2:
                    break
        print('Done', self.tool_id)
        # ================================

        # res = temp_tool.getClampCurrentState(1)[0]
        # print(res)

         # 释放串口
        # self.temp_tool.serialOperation(self.com, 115200, False)

    def get_pos(self):  # int, 0 ~ 255
        cur_pos_list = self.temp_tool.getClampCurrentLocation(self.tool_id)
        return cur_pos_list[0]

    def set_pos(self, give_pos):  # int, 0 ~ 255. 0-->fully open; 255-->fully closed
        self.temp_tool.runWithParam(self.tool_id, pos=give_pos, speed=self.speed, torque=self.torque)


# sudo chmod 777 /dev/ttyUSB0
# sudo chmod 777 /dev/ttyUSB1

if __name__ == '__main__':

    jc1 = JodellRG75()
    jc1.connect("/dev/ttyUSB0", 9)  # for the right-arm, set tool_id=9 (other id will not work?)
    #jc1.switch(0, True)  # open gripper, open_val is from 255 to 1
    time.sleep(1)  # sleep 1 seond
    #jc1.switch(1, True)  # close gripper
    
    jc2 = JodellRG75()
    jc2.connect("/dev/ttyUSB1", 9)  # for the left-arm, set tool_id=9 (other id will not work?)
    jc2.switch(0, True)  # open gripper
    time.sleep(1)  # sleep 1 seond
    jc2.switch(1, True)  # close gripper

    os._exit(0)

    cur_pos = jc2.get_pos()
    print(cur_pos)
    jc2.set_pos(cur_pos-3)  # delta = 3 is for reorienting bottle; delta = 5 is for inserting marker pen
    time.sleep(1)
    cur_pos = jc2.get_pos()
    print(cur_pos)
