
'''
conda activate py39
python src/cap_img.py
'''

import cv2
import kingfisher

# c=kingfisher.connect("192.168.4.64")   # connect the camera (for dual-arm aubo-i5)
c=kingfisher.connect("192.168.4.93")   # connect the camera (for dual-arm rokae-CR7)

kingfisher.SetAUTO_EXPOSURE()
kfr_height, kfr_width = 540, 960

image_L, image_R = kingfisher.captureQuarterSize()  # low resolution, shape is (540, 960)

# saved_img_path = "./src/test_img/img01_brush_dustpan.jpg"
# saved_img_path = "./src/test_img/img02_brush_dustpan_bottle_mugcup_trash.jpg"
# saved_img_path = "./src/test_img/img03_brush_dustpan_bottle_mugcup_trash.jpg"
# saved_img_path = "./src/test_img/img04_brush_dustpan_bottle_mugcup_trash.jpg"

# saved_img_path = "./applications/vlms_test_ori/wrapping_basket_id06_img_sL_test01_Wrapping.jpg"
# saved_img_path = "./applications/vlms_test_ori/wrapping_basket_id06_img_sL_test01_Grasping.jpg"

saved_img_path = "../debug/test_ori/20250826_new_demo_test01.jpg"
# saved_img_path = "../debug/test_ori/20250826_new_demo_test02.jpg"

print("Saving image ...", saved_img_path)
cv2.imwrite(saved_img_path, image_L)
