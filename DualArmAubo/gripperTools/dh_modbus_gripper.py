
import serial
serialPort = serial.Serial()

class dh_device(object) :
    def __init__(self, portname):
        self.portname = portname
        
    def connect_device(self, Baudrate) :
        ret = -1
        serialPort.port = self.portname
        serialPort.baudrate = Baudrate
        serialPort.bytesize = 8
        serialPort.parity = 'N'
        serialPort.stopbits = 1
        serialPort.set_output_flow_control = 'N'
        serialPort.set_input_flow_control = 'N'
        
        print('portname: ', self.portname, serialPort.isOpen())
        
        # serialPort.open()
        if(serialPort.isOpen()) :
            print('Serial Open Success')
            ret = 0
        else:
            serialPort.open()
            if(serialPort.isOpen()) :
                print('Serial Open Success')
                ret = 0
            else:
                print('Serial Open Error')
                ret = -1
        return ret

    def disconnect_device() :
        if(serialPort.isOpen()) :
            serialPort.close()
        else :
            return

    def device_wrire(self, write_data) :
        write_lenght = 0
        if(serialPort.isOpen()) :
            write_lenght = serialPort.write(write_data)
            if(write_lenght == len(write_data)) :
                return write_lenght
            else :
                print('write error ! send_buff :',write_data)
                return 0;
        else :
            return -1

    def device_read(self, wlen) :
        responseData = [0,0,0,0,0,0,0,0]
        if(serialPort.isOpen()) :
            responseData = serialPort.readline(wlen)
            #print('read_buff: ',responseData.hex())
            return responseData
        else :
            return -1
        

    """description of class"""

    
class dh_modbus_gripper(object):
    def __init__(self, m_device, gripper_ID):
        self.m_device = m_device  # default is dh_device()
        self.gripper_ID = gripper_ID  # default is 0x01

    def CRC16(self,nData, wLength) :
        if nData==0x00:
            return 0x0000
        wCRCWord=0xFFFF
        poly=0xA001
        for num in range(wLength):
            date = nData[num]
            wCRCWord = (date & 0xFF)^ wCRCWord
            for bit in range(8) : 
                if(wCRCWord&0x01)!=0:
                    wCRCWord>>=1
                    wCRCWord^= poly
                else:
                    wCRCWord>>=1
        return wCRCWord

    def open(self, BaudRate) :
        ret = 0
        ret = self.m_device.connect_device(BaudRate)
        if(ret < 0) :
            print('open failed!!!')
            return ret
        else:
            print('open successful!')
            return ret

    def close() :
        self.m_device.disconnect_device()

    def WriteRegisterFunc(self,index, value) :
        send_buf = [0,0,0,0,0,0,0,0]
        send_buf[0] = self.gripper_ID
        send_buf[1] = 0x06
        send_buf[2] = (index >> 8) & 0xFF
        send_buf[3] = index & 0xFF
        send_buf[4] = (value >> 8) & 0xFF
        send_buf[5] = value & 0xFF

        crc = self.CRC16(send_buf,len(send_buf)-2)
        send_buf[6] = crc & 0xFF
        send_buf[7] = (crc >> 8) & 0xFF

        send_temp = send_buf
        ret = False
        retrycount = 3

        while ( ret == False ):
            ret = False

            if(retrycount < 0) :
                break
            retrycount = retrycount - 1

            wdlen = self.m_device.device_wrire(send_temp)
            if(len(send_temp) != wdlen) :
                print('write error ! write : ', send_temp)
                continue

            rev_buf = self.m_device.device_read(8)
            if(len(rev_buf) == wdlen) :
                ret = True
        return ret

    def ReadRegisterFunc(self,index) :
        send_buf = [0,0,0,0,0,0,0,0]
        send_buf[0] = self.gripper_ID
        send_buf[1] = 0x03
        send_buf[2] = (index >> 8) & 0xFF
        send_buf[3] = index & 0xFF
        send_buf[4] = 0x00
        send_buf[5] = 0x01

        crc = self.CRC16(send_buf,len(send_buf)-2)
        send_buf[6] = crc & 0xFF
        send_buf[7] = (crc >> 8) & 0xFF

        send_temp = send_buf
        ret = False
        retrycount = 3

        while ( ret == False ):
            ret = False

            if(retrycount < 0) :
                break
            retrycount = retrycount - 1

            wdlen = self.m_device.device_wrire(send_temp)
            if(len(send_temp) != wdlen) :
                print('write error ! write : ', send_temp)
                continue

            rev_buf = self.m_device.device_read(7)
            if(len(rev_buf) == 7) :
                value = ((rev_buf[4]&0xFF)|(rev_buf[3] << 8))
                ret = True
            #('read value : ', value)
        return value

    def Initialization(self) :
        self.WriteRegisterFunc(0x0100,0xA5)
        
    def SetTargetPosition(self,refpos) :
        self.WriteRegisterFunc(0x0103,refpos);

    def SetTargetForce(self,force) :
        self.WriteRegisterFunc(0x0101,force);
        
    def SetTargetSpeed(self,speed) :
        self.WriteRegisterFunc(0x0104,speed);

    def GetCurrentPosition(self) :
        return self.ReadRegisterFunc(0x0202);

    def GetCurrentTargetForce(self) :
        return self.ReadRegisterFunc(0x0101);

    def GetCurrentTargetSpeed(self) :
        return self.ReadRegisterFunc(0x0104);

    def GetInitState(self) :
        return self.ReadRegisterFunc(0x0200);

    def GetGripState(self) :
        return self.ReadRegisterFunc(0x0201);

    """description of class"""



