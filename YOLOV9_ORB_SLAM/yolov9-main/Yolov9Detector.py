import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

class Yolov9Detector(object):
	def __init__(self,weights='yolov9-c-converted.pt',device='',imgsz=640):
	    self.confidence =0.25
	    self.iou =0.45
	    self.max_det =1000
	    self.auto=True
	    self.classes=None
	    self.agnostic_nms=False 
	    self.device = select_device(device)
	    self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=None, fp16=False)
	    self.stride = self.model.stride
	    self.names = self.model.names
	    self.imgsz =[imgsz,imgsz] # check image size
	    self.model.warmup(imgsz=(1, 3, *self.imgsz))  
	    self.name_id= {}
	    for k,v in self.names.items():
	        self.name_id[v] = k
            
	def inference_image(self, opencv_img):
	    im = letterbox(opencv_img, self.imgsz, stride=self.stride, auto=self.auto)[0]  # padded resize
	    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
	    im = np.ascontiguousarray(im)  # contiguous
	    im = torch.from_numpy(im).to(self.model.device)
	    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
	    im /= 255  # 0 - 255 to 0.0 - 1.0
	    if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
	    pred = self.model(im, augment=False, visualize=False)
	    pred = non_max_suppression(pred, self.confidence, self.iou, self.classes, self.agnostic_nms, max_det=self.max_det)
	    result_list=[]
             # Process predictions
	    for i, det in enumerate(pred):  # per image
                if len(det):
                   det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], opencv_img.shape).round()
                   for *xyxy, conf, cls in reversed(det):
                       result_list.append([self.names[int(cls)],round(float(conf),2),int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])])
	    return result_list
	
	
	def imshow(self, result_list,opencv_img):
	    result_img = self.draw_image(result_list, opencv_img)
	    cv2.imshow('result',result_img)
	    cv2.waitKey(0)
	    cv2.destroyAllWindows()
	    
	def draw_image(self,result_list,opencv_img):
	    ann = Annotator(opencv_img)
	    for result in result_list:
	        label = result[0] + ',' + str(result[1])
	        ann.box_label(result[2:6],label, color = colors(self.name_id[result[0]],True))
	    return ann.result()
	    
	def start_video(self,video_path):
	    pass
	
	def start_camera(self,camera_index=0):
	    pass
	    
if __name__ == '__main__':
   detector = Yolov9Detector()
   img=cv2.imread('data/zidane.jpg')
   result_list = detector.inference_image(img)
   print(result_list)
   detector.imshow(result_list,opencv_img=img)  
   
