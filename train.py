import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
  model = YOLO('/home/2024-Zhong.zt/HHD/detection/DD-LTSNet/DD-LTSNet.yaml')
  results = model.train(
    data='/home/2024-Zhong.zt/HHD/detection/DD-LTSNet/cntsss.yaml',  #数据集配置文件的路径
    epochs=300, 
    imgsz=640, 
    workers=4,  
    batch=4,
    device=[0], 
    amp=True,  
    cache=False, 
    project='runs/',
    name='train',
    # resume=True, 
)