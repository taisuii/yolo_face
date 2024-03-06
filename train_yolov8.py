from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(torch.__version__)
    model = YOLO("yolov8n.pt").cuda()
    results = model.train(data="config/coco8.yaml", epochs=10, batch=8, workers=1)
    result_val = model.val()
    success = model.export()
