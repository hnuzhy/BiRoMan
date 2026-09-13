'''
sudo apt-get install libcharls2

vim ~/.bashrc
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
source ~/.bashrc
'''

import os
import cv2
import time
import argparse
import numpy as np
import kingfisher

# c=kingfisher.connect("192.168.4.64")   # connect the camera (for dual-arm aubo-i5)
c=kingfisher.connect("192.168.4.93")   # connect the camera (for dual-arm rokae-CR7)
# c=kingfisher.connect("192.168.4.127")   # connect the camera (for dual-arm SongLing)

# exposure = 15000
# kingfisher.SetExposure(exposure)
kingfisher.SetAUTO_EXPOSURE()

width, height=kingfisher.get_resolution()  # 960, 540

##################################################################################################################################  
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', default="", help="the string of newly saved folder name.")
    args = parser.parse_args()
    
    save_dir = f"./api_kfr6000_python/{args.folder_name}"
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    
    img_count = 0
    while True:
        imgL, imgR = kingfisher.captureQuarterSize()
        
        cv2.imwrite(os.path.join(save_dir, f"imgL_{str(img_count).zfill(6)}.jpg"), imgL)
        cv2.imwrite(os.path.join(save_dir, f"imgR_{str(img_count).zfill(6)}.jpg"), imgR)
        
        print(img_count, time.time())
        img_count += 1
        if 0xFF == ord('q'): break

'''
$ python api_kfr6000_python/kingfisher_capture_saving.py --folder_name 001_pushing_basket
$ python api_kfr6000_python/kingfisher_capture_saving.py --folder_name 002_picking_placing
'''
    




