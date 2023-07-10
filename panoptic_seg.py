
import cv2 as cv
import argparse
import matplotlib.pyplot as plt
import math
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.projects.deeplab import add_deeplab_config

import warnings 
warnings.filterwarnings('ignore')
import torch
import imutils

parser = argparse.ArgumentParser('run video')
parser.add_argument('-v', '--video_dir', type = str, help ='video directory')
parser.add_argument('-yaml', '--config_yaml', type = str, help ='config yaml file')
args = parser.parse_args()

    
    
def run_video(video_dir, config_yaml):
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_yaml))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.INPUT.CROP.ENABLED = False
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_yaml)
    cfg.MODEL.DEVICE = 'cuda'
    predictor = DefaultPredictor(cfg)
    
    
    # run video
    cap = cv.VideoCapture(video_dir)   #for webcam: 0 if single, if multiple: choose for e.g 1; if no rto -> video_dir

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    size =(width, height)
    fourcc= cv.VideoWriter_fourcc(*"MP4V")

    # VideoWriter for output video
    video_writer = cv.VideoWriter(video_dir[:-4] +'_output.mp4', fourcc, 20.0, size)

    while True:
        isTrue, frame = cap.read()
        if isTrue:    
#             frame = imutils.resize(frame, width=640, height=480)
            panoptic_seg, segments_info = predictor(frame)["panoptic_seg"]

            v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
            output = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
            
            
            # get labels for panoptic seg
            thing_classes = [i['category_id'] for i in segments_info if i['isthing'] == True]
            stuff_classes = [i['category_id'] for i in segments_info if i['isthing'] == False]

            thing_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            stuff_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes
            pred_class_names = list(map(lambda x: thing_names[x], thing_classes)) + list(map(lambda x: stuff_names[x], stuff_classes))  
            
            
            # display and save result
            cv.cvtColor(output.get_image()[:, :, ::-1], cv.COLOR_BGR2RGB)
            cv.imshow('result', output.get_image()[:, :, ::-1])
            video_writer.write(output.get_image()[:, :, ::-1])
            cv.waitKey(1)
            
            
        else:
            break
                    
    cap.release()
    video_writer.release()
    cv.destroyAllWindows()
    
            
        

if __name__=='__main__':
    run_video(args.video_dir, args.config_yaml)
