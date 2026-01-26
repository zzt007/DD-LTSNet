from ultralytics import YOLO
import argparse
import os

if __name__ == '__main__':
    model = YOLO('/home/2024-Zhong.zt/HHD/detection/yolo-v11/yolov11_baseline_gtsdb_4090.pt')
    model.val(
        data='/home/2024-Zhong.zt/HHD/detection/yolo-v11/cntsss.yaml',
              split='val',  
              imgsz=640,
              batch=1,
              workers=0,
              save_json=True, # if you need to cal coco metrice
              project='runs/gtsdb_yolov11',
              name='test',
              device='0',
              )
