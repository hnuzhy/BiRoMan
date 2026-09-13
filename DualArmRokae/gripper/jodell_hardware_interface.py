
# pip install modbus-tk
# pip install pyserial

import serial, time
import sys
import serial.tools.list_ports
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu


# SDK基础类
class JodellSDKDemo():
    def __init__(self):
        self.master = None
        self.clampStatus = {
            0: '未检测到物体',
            1: '手指在张开检测到物体',
            2: '手指在闭合检测到物体',
            3: '手指已到达指定的位置，没有检测到物体'
        }

    def getS16(self, val):
        if val < 0x8000:
            return val
        else:
            return (val - 0xffff)

    def getStatus(self, salveId, address, readMode, count):
        """
            状态查询
            参数说明：
                salveId：夹爪站点id
                address：modbus寄存器地址
                readMode：modbus读取状态指令编号03或者04
                count：读取寄存器的个数
        """
        if not self.master:
            return "通讯未连接"
        try:
            readBuf = self.master.execute(salveId, readMode, address, count)
            return readBuf
        except Exception as exc:
            return str(exc)

    def writeDataToRegister(self, salveId, address, sendCmdBuf):
        """
            将数据写入寄存器
            参数说明：
                salveId：夹爪站点id
                address：modbus寄存器起始地址
                sendCmdBuf：写入的数据（格式为列表，列表长度代表写入寄存器个数）
        """
        if not self.master:
            return "通讯未连接"
        try:
            self.master.execute(salveId, cst.WRITE_MULTIPLE_REGISTERS, address, output_value=sendCmdBuf)
            reVal = 1
        except Exception as exc:
            reVal = str(exc)
        return reVal

    def searchCom(self):
        """
            查询可用的串口
        """
        comList = []
        plist = list(serial.tools.list_ports.comports())  # 查询可用的串口
        for i in range(len(plist)):
            plist_0 = list(plist[i])
            comList.append(str(plist_0[0]))
        return comList

    def serialOperation(self, com, baudRate, status):
        """
            串口操作
            参数说明：
                com：串口号
                baudRate：波特率
                status：开关状态（打开：True，关闭：False）
        """
        try:
            if status:
                self.master = modbus_rtu.RtuMaster(
                    serial.Serial(port=com, baudrate=int(baudRate), bytesize=8, parity='N', stopbits=1))
                self.master.set_timeout(0.1)
                self.master.set_verbose(True)
            else:
                self.master._do_close()
            reVal = 1

        except Exception as exc:
            reVal = str(exc)

        return reVal

    def clawEnable(self, salveId, status):
        """
            夹爪使能
            参数说明：
                salveId：夹爪站点id
                status：是否使能（使能：True，去使能：False）
        """
        if not self.master:
            return "通讯未连接"
        try:
            if status:
                sendCmdBuf = [0x01]
                self.master.execute(salveId, cst.WRITE_MULTIPLE_REGISTERS, 1000, output_value=sendCmdBuf)
            else:
                sendCmdBuf = [0x00]
                self.master.execute(salveId, cst.WRITE_MULTIPLE_REGISTERS, 1000, output_value=sendCmdBuf)
            reVal = 1
        except Exception as exc:
            reVal = str(exc)
        return reVal

    def runWithoutParam(self, salveId, cmdId):
        """
            无参数运行指令
            参数说明：
                salveId：夹爪站点id
                cmdId: 无参数指令编号
        """
        if not self.master:
            return "通讯未连接"
        try:
            sendCmdBuf = [0x000B]
            sendCmdBuf[0] |= cmdId << 8
            self.master.execute(salveId, cst.WRITE_MULTIPLE_REGISTERS, 1000, output_value=sendCmdBuf)
            reVal = 1
        except Exception as exc:
            reVal = str(exc)
        return reVal

    def runWithParam(self, salveId, pos, speed, torque):
        """
            有参数运行指令
            参数说明：
                salveId：夹爪站点id
                pos：运行目标位置
                speed：运行中最大速度
                torque：抓取物体时的力矩
        """
        if not self.master:
            return "通讯未连接"
        try:
            sendCmdBuf = [0x0009, 0x0000, 0x000]
            sendCmdBuf[1] = pos << 8
            sendCmdBuf[2] = speed | torque << 8
            self.master.execute(salveId, cst.WRITE_MULTIPLE_REGISTERS, 1000, output_value=sendCmdBuf)
            reVal = 1
        except Exception as exc:
            reVal = str(exc)
        return reVal

    def getClawCurrentStatus(self, salveId):
        """
            获取夹爪当前状态
            参数说明：
                salveId：夹爪站点id
            返回值说明：
                0:未检测到物体
                1:手指在张开检测到物体
                2:手指在闭合检测到物体
                3:手指已到达指定的位置，没有检测到物体
        """
        if not self.master:
            return "通讯未连接"
        try:
            readBuf = self.master.execute(salveId, cst.READ_INPUT_REGISTERS, 2000, 1)
            # return [self.clampStatus[(readBuf[0] >> 6) & 0x03]]
            return (readBuf[0] >> 6) & 0x03
        except Exception as exc:
            return str(exc)

    def getClawCurrentLocation(self, salveId):
        """
            获取夹爪当前位置
            参数说明：
                salveId：夹爪站点id
        """
        if not self.master:
            return "通讯未连接"
        try:
            readBuf = self.master.execute(salveId, cst.READ_INPUT_REGISTERS, 2001, 1)
            return [readBuf[0] >> 8]
        except Exception as exc:
            return str(exc)

    def getClawCurrentSpeed(self, salveId):
        """
            获取夹爪当前速度
            参数说明：
                salveId：夹爪站点id
        """
        if not self.master:
            return "通讯未连接"
        try:
            readBuf = self.master.execute(salveId, cst.READ_INPUT_REGISTERS, 2002, 1)
            return [readBuf[0] & 0xff]
        except Exception as exc:
            return str(exc)

    def getClawCurrentTorque(self, salveId):
        """
            获取夹爪当前力矩
            参数说明：
                salveId：夹爪站点id
        """
        if not self.master:
            return "通讯未连接"
        try:
            readBuf = self.master.execute(salveId, cst.READ_INPUT_REGISTERS, 2002, 1)
            return [readBuf[0] >> 8]
        except Exception as exc:
            return str(exc)

    def getClawCurrentTemperature(self, salveId):
        """
            获取夹爪当前温度
            参数说明：
                salveId：夹爪站点id
        """
        if not self.master:
            return "通讯未连接"
        try:
            readBuf = self.master.execute(salveId, cst.READ_INPUT_REGISTERS, 2003, 1)
            return [readBuf[0] >> 8]
        except Exception as exc:
            return str(exc)

    def getClawCurrentVoltage(self, salveId):
        """
            获取夹爪当前电压
            参数说明：
                salveId：夹爪站点id
        """
        if not self.master:
            return "通讯未连接"
        try:
            readBuf = self.master.execute(salveId, cst.READ_INPUT_REGISTERS, 2003, 1)
            return [readBuf[0] & 0xff]
        except Exception as exc:
            return str(exc)

    def querySoftwareVersion(self, salveId):
        """
            查询软件版本
            参数说明：
                salveId：夹爪站点id
        """
        if not self.master:
            return "通讯未连接"
        try:
            readBuf = self.master.execute(salveId, cst.READ_HOLDING_REGISTERS, 5004, 1)
            return ["V" + str(readBuf[0] >> 8) + "." + str(readBuf[0] & 0xff)]
        except Exception as exc:
            return str(exc)

    def changeSalveId(self, oldId, newId):
        """
            修改夹爪站点ID
            参数说明：
                oldId：当前的id
                newId：修改后的id
        """
        if not self.master:
            return "通讯未连接"
        try:
            sendCmdBuf = [newId]
            self.master.execute(oldId, cst.WRITE_MULTIPLE_REGISTERS, 5005, output_value=sendCmdBuf)
            reVal = 1
        except Exception as exc:
            reVal = str(exc)
        return reVal

    def scanSalveId(self, startId, stopId):
        """
            扫描连接的夹爪ID
            参数说明：
                startId：起始ID
                stopId：终止ID
        """
        if not self.master:
            return "通讯未连接"
        try:
            idList = []
            for myId in range(startId, stopId + 1):
                myId += 1
                self.master.execute(myId, cst.READ_INPUT_REGISTERS, 2000, 4)
                idList.append(myId)
                time.sleep(0.02)
        except Exception as exc:
            print(str(exc))
        finally:
            return idList

    def changeBaudRate(self, salveId, baudRate):
        """
            修改夹爪波特率
            参数说明：
                salveId：夹爪站点ID
                baudRate：目标波特率编号（0:115200, 1：57600, 2：38400, 3：19200, 4：9600, 5：4800）
        """
        if not self.master:
            return "通讯未连接"
        try:
            sendCmdBuf = [baudRate]
            self.master.execute(salveId, cst.WRITE_MULTIPLE_REGISTERS, 5006, output_value=sendCmdBuf)
            reVal = 1
        except Exception as exc:
            reVal = str(exc)
        return reVal

    def clawEncoderZero(self, salveId):
        """
            夹持端编码器对零
            参数说明：
                salveId：夹爪站点ID
        """
        if not self.master:
            return "通讯未连接"
        try:
            sendCmdBuf = [0x01]
            self.master.execute(salveId, cst.WRITE_MULTIPLE_REGISTERS, 1019, output_value=sendCmdBuf)
            reVal = 1
        except Exception as exc:
            reVal = str(exc)
        return reVal

    def switchAutoPatrolInspection(self, salveId, status):
        """
            开关自动巡检
            参数说明：
                salveId：夹爪站点ID
                status：开关状态（开：True，关：False）
        """
        if not self.master:
            return "通讯未连接"
        if status:
            sendCmdBuf = [0x0010]
        else:
            sendCmdBuf = [0x0030]
        try:
            self.master.execute(salveId, cst.WRITE_MULTIPLE_REGISTERS, 1000, output_value=sendCmdBuf)
            reVal = 1
        except Exception as exc:
            reVal = str(exc)
        return reVal


class ClawRgTool(JodellSDKDemo):
    def __init__(self):
        super().__init__()
        self.master = None
        self.clampStatus = {
            0: '未检测到物体',
            1: '手指在张开检测到物体',
            2: '手指在闭合检测到物体',
            3: '手指已到达指定的位置，没有检测到物体'
        }

    def switchMode(self, salveId, modeIndex):
        """
            选择模式
            参数说明：
                salveId：夹爪站点ID
                modeIndex：模式编号（0：串口通讯模式，1：IO模式）
        """
        if not self.master:
            return "通讯未连接"
        sendCmdBuf = [0x00]
        if modeIndex == 0:
            sendCmdBuf = [0x00]
        elif modeIndex == 1:
            sendCmdBuf = [0x55]
        try:
            self.master.execute(salveId, cst.WRITE_MULTIPLE_REGISTERS, 1021, output_value=sendCmdBuf)
            reVal = 1
        except Exception as exc:
            reVal = str(exc)
        return reVal
