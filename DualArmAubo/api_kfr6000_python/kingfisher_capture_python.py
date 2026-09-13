'''
sudo apt-get install libcharls2

vim ~/.bashrc
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
source ~/.bashrc
'''

import cv2
import numpy as np
import kingfisher
import time

c=kingfisher.connect("192.168.4.94")   # connect the camera

# exposure = 15000
# kingfisher.SetExposure(exposure)

kingfisher.SetAUTO_EXPOSURE()

calib_file = ""
mac = ""
height=0
width = 0

calib_file=kingfisher.getCalibData()
mac=kingfisher.getMac()
width,height=kingfisher.get_resolution()

print(mac, width, height)
print(calib_file)


j=0
while True:

    j += 1
    
    left, right = kingfisher.captureQuarterSize()
    combined = np.hstack((left, right))  # (1920, 540) <-- (960, 540) + (960, 540)
    # left, right = kingfisher.capture()
    # combined = np.hstack((left, right))  # (7680, 540) <-- (3840, 2160) + (3840, 2160)
    
    
    # if j == 105: cv2.imwrite(f"rearrange/id{j}_imgL.jpg", left); cv2.imwrite(f"rearrange/id{j}_imgR.jpg", right)
    
    
    cv2.namedWindow("MyWindow", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MyWindow", 1920, 540)
    # cv2.resizeWindow("MyWindow", 7680, 2160)
    cv2.imshow("MyWindow", combined)
    
    print(j, time.time())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # for i in range(500):
    #     j += 1
    #     left, right=kingfisher.capture()
    #     combined = np.hstack((left, right))
    #     cv2.namedWindow("MyWindow", cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow("MyWindow", 1920, 540)
    #     cv2.imshow("MyWindow", combined)
    #     cv2.waitKey(10)
    #     print(j)
    #
    # for i in range(100):
    #     j += 1
    #     exposure_num = [20000, 150000]
    #     left, right = kingfisher.Get_HDR_Data(0, exposure_num)
    #     combined = np.hstack((left, right))
    #     cv2.namedWindow("MyWindow", cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow("MyWindow", 1920, 540)
    #     cv2.imshow("MyWindow", combined)
    #     cv2.waitKey(10)
    #     print(j)





