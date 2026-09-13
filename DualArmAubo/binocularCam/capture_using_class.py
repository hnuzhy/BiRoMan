# -- coding: utf-8 --
import time
import sys
import threading
import getch
import numpy as np
import cv2
import os
import platform
from ctypes import *

# sys.path.append("../MvImport")
# sys.path.append("./")


if platform.system() == "Linux":
    try:
        from .MvImport_linux.MvCameraControl_class import *
    except:
        from MvImport_linux.MvCameraControl_class import *
if platform.system() == "Windows":
    import msvcrt  # for windows only
    from .MvImport_win.MvCameraControl_class import *

g_bExit = False

############################配置##################################
file_save_path = './'
image_num_count = 0
camera_nums = 2
image_num_count_lock = threading.Lock()

# 保存图片信息的全局变量
saved_files = {
    'L': [],
    'R': []
}

############################配置##################################

# Mono图像转为python数组
def Mono_numpy(data, nWidth, nHeight):
    data_ = np.frombuffer(data, count=int(nWidth * nHeight), dtype=np.uint8, offset=0)
    data_mono_arr = data_.reshape(nHeight, nWidth)
    numArray = np.zeros([nHeight, nWidth, 1], "uint8")
    numArray[:, :, 0] = data_mono_arr
    data_out = cv2.cvtColor(numArray, cv2.COLOR_BAYER_GB2RGB)
    return data_out


# 彩色图像转为python数组
def Color_numpy(data, nWidth, nHeight):
    data_ = np.frombuffer(data, count=int(nWidth * nHeight * 3), dtype=np.uint8, offset=0)
    data_r = data_[0:nWidth * nHeight * 3:3]
    data_g = data_[1:nWidth * nHeight * 3:3]
    data_b = data_[2:nWidth * nHeight * 3:3]

    data_r_arr = data_r.reshape(nHeight, nWidth)
    data_g_arr = data_g.reshape(nHeight, nWidth)
    data_b_arr = data_b.reshape(nHeight, nWidth)
    numArray = np.zeros([nHeight, nWidth, 3], "uint8")

    numArray[:, :, 0] = data_r_arr
    numArray[:, :, 1] = data_g_arr
    numArray[:, :, 2] = data_b_arr
    return numArray


# 为线程定义一个函数
def work_thread(cam=0, pData=0, nDataSize=0, win_name = 0):
    stOutFrame = MV_FRAME_OUT()
    saved_nums = 0

    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    while True:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if None != stOutFrame.pBufAddr and 0 == ret:
            image_data = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)()
            if platform.system() == "Windows":
                cdll.msvcrt.memcpy(byref(image_data), stOutFrame.pBufAddr, 
                    stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
            if platform.system() == "Linux":
                memmove(byref(image_data), stOutFrame.pBufAddr,
                    stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
            captured_image = Mono_numpy(image_data, stOutFrame.stFrameInfo.nWidth,
                                        stOutFrame.stFrameInfo.nHeight)
            show_image = cv2.resize(captured_image, (
                int(stOutFrame.stFrameInfo.nWidth / 2), int(stOutFrame.stFrameInfo.nHeight / 2)))


            cv2.imshow(win_name, show_image)


            key = cv2.waitKey(100)
            saved_nums = handle_key(key, captured_image, win_name, saved_nums)
            nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
        else:
            print("no data[0x%x]" % ret)
        if g_bExit == True:
            break

def handle_key(key, image_input, image_name, saved_nums):
    global g_bExit, image_num_count, saved_files
    if key == 113: # q
        g_bExit = True
        time.sleep(1)
    if key == 99: # c
        image_num_count_lock.acquire()
        image_num_count = image_num_count + 1
        image_num_count_lock.release()
        file_name = file_save_path + image_name + str(int(saved_nums)) + '.bmp'
        cv2.imwrite(file_name, image_input)
        saved_nums = saved_nums + 1
        saved_files[image_name].append(file_name)
        print("image saved: " + file_name)

    elif saved_nums < int(image_num_count):
        file_name = file_save_path + image_name + str(int(saved_nums)) + '.bmp'
        saved_nums = saved_nums + 1
        cv2.imwrite(file_name, image_input)
        saved_files[image_name].append(file_name)
        print("image saved: " + file_name)

    if key == 100: # d
        if saved_files['L'] and saved_files['R']:
            last_saved_file_L = saved_files['L'][-1]
            last_saved_file_R = saved_files['R'][-1]
            if os.path.exists(last_saved_file_L) and os.path.exists(last_saved_file_R):
                os.remove(last_saved_file_L)
                os.remove(last_saved_file_R)
                print("images deleted: " + last_saved_file_L + " and " + last_saved_file_R)
                saved_files['L'].pop()
                saved_files['R'].pop()
                saved_nums = saved_nums - 1
                image_num_count_lock.acquire()
                image_num_count = image_num_count - 1
                image_num_count_lock.release()

    return saved_nums


def open_camera(nConnectionNum, win_name, deviceList):
    global g_bExit
    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄 | en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    stBool = c_bool(False)
    ret = cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
    if ret != 0:
        print("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()

    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    try:
        hThreadHandle = threading.Thread(target=work_thread, args=(cam, None, None, win_name))
        hThreadHandle.start()
    except:
        print("error: unable to start thread")


### newly defined by zhouhuayi@cuhk.edu.cn

def open_cam_func(nConnectionNum, deviceList):

    
    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄 | en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    stBool = c_bool(False)
    ret = cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
    if ret != 0:
        print("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()

    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()
        
    return cam


#################################################################
# for Windows / Linux platform (origanized by zhouhuayi)
#################################################################
def cam_snapshot_func(cam):
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
    if None != stOutFrame.pBufAddr and 0 == ret:
        image_data = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)()
        if platform.system() == "Windows":
            cdll.msvcrt.memcpy(byref(image_data), stOutFrame.pBufAddr, 
                stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
        if platform.system() == "Linux":
            memmove(byref(image_data), stOutFrame.pBufAddr,
                stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
        captured_image = Mono_numpy(image_data, stOutFrame.stFrameInfo.nWidth,
                                    stOutFrame.stFrameInfo.nHeight)
        nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
        return captured_image
    else:
        print("no data[0x%x]" % ret)
        return None

def stop_relase_cam_func(cam):

	# ch:停止取流 | en:Stop grab image
	ret = cam.MV_CC_StopGrabbing()
	if ret != 0:
		print ("stop grabbing fail! ret[0x%x]" % ret)
		del data_buf
		sys.exit()

	# ch:关闭设备 | Close device
	ret = cam.MV_CC_CloseDevice()
	if ret != 0:
		print ("close deivce fail! ret[0x%x]" % ret)
		del data_buf
		sys.exit()

	# ch:销毁句柄 | Destroy handle
	ret = cam.MV_CC_DestroyHandle()
	if ret != 0:
		print ("destroy handle fail! ret[0x%x]" % ret)
		del data_buf
		sys.exit()


def find_camera_func(sn_left, sn_right):
    left_num = -1
    right_num = -1

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    print("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)


            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chSerialNumber:
                if per == 0:
                    break

                strSerialNumber = strSerialNumber + chr(per)
            print("device model name: %s" % strSerialNumber)

            if strSerialNumber == sn_left:
                print("找到左相机")
                left_num = i
            if strSerialNumber == sn_right:
                print("找到右相机")
                right_num = i

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)
            if strSerialNumber == sn_right:
                print("找到右相机：[%d]" % i)
                right_num = i
            if strSerialNumber == sn_left:
                print("找到左相机：[%d]" % i)
                left_num = i

    if left_num < 0 or right_num < 0:
        print("未找到指定的相机")
        exit(0)
    
    return left_num, right_num, deviceList



if __name__ == "__main__":
    # sn_left = "DA3249444"
    # sn_right = "DA3245647"
    sn_left = "DA3245572"
    sn_right = "DA3245534"
    
    left_num, right_num, deviceList = find_camera_func(sn_left, sn_right)

    nConnectionNum = input("按 Enter 继续...")

    open_camera(right_num, "R", deviceList)
    open_camera(left_num, "L", deviceList)
