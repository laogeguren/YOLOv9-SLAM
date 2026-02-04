from Yolov9Detector import *
import cv2
import os
import numpy as np
import shutil


detector =Yolov9Detector()
image_dir = r'/home/zs/YOLOV8_ORB_SLAM/tum/rgbd_dataset_freiburg3_walking_xyz_validation/rgb'
end_dir = r'/home/zs/YOLOV8_ORB_SLAM/tum/rgbd_dataset_freiburg3_walking_xyz_validation/result01'

shutil.rmtree( end_dir )
os.mkdir(end_dir)

for file in sorted(os.listdir(image_dir)):  
    img_file = os.path.join(image_dir, file)
    img = cv2.imread(img_file)
    results = detector.inference_image(img)	
    image_name = os.path.splitext(file)[0]
    txt_file = os.path.join(end_dir, image_name+".txt")
    s = ''
    for result in results:
    	#res_dict = {}
    	if result[0] == 'person' or result[0] == 'tv' or result[0] == 'refrigerator' or result[0] == 'chair':
	    	s+=f'left:{result[2]}'
	    	s+=f' top:{result[3]}'
	    	s+=f' right:{result[4]}'
	    	s+=f' bottom:{result[5]}'
	    	s+=f' class:{result[0]} {result[1]}\n'
	    	print(txt_file)
    with open(txt_file,'w') as f:
    	f.write(s)
