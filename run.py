import torch
from torchvision.transforms import transforms
from ultralytics import YOLO
from PIL import ImageDraw

import config.config
from resnet18.model import Model

import cv2
import os
import numpy as np
from PIL import Image
from config import config

# 加载训练好的模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./opencv_faceClassify/model.yml")


def name_pre_opencv(image):
    open_cv_image = np.array(image)
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    # 进行人脸识别
    label, confidence = recognizer.predict(gray_image)
    predicted_celebrity = config.NAME_LIST[label]
    return predicted_celebrity, confidence


def img():
    # model = YOLO("yolov8n.pt")
    model = YOLO("runs/detect/train2/weights/best.pt")
    results = model("./datasets/1.jpg")
    print(results)
    pil_image = Image.fromarray(results[0].orig_img)
    print(results[0].boxes)
    face_boxes = [box.xyxy for box in results[0].boxes]
    if face_boxes:
        box = face_boxes[0][0].cpu().numpy()
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=2)

    cv2.imshow('Object Detection', np.array(pil_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cam():
    cap = cv2.VideoCapture(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("runs/detect/train2/weights/best.pt")
    model = model.to(device)
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()

        # 将图像转换为 PyTorch Tensor，并移到 GPU 上
        tensor_img = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # 对图像进行推理
        with torch.no_grad():
            results = model(tensor_img)

        face_boxes = [box.xyxy for box in results[0].boxes]
        conf = [box.conf for box in results[0].boxes]

        for i in range(face_boxes.__len__()):
            box = face_boxes[i][0].cpu().numpy()
            if conf[i].cpu().numpy() >= 0.6:
                draw = ImageDraw.Draw(pil_image)

                expanded_box = [int(box[0] - (box[2] - box[0]) / 5),
                                int(box[1] - (box[3] - box[1]) / 5),
                                int(box[2] + (box[2] - box[0]) / 5),
                                int(box[3] + (box[3] - box[1]) / 5)]
                draw.rectangle([(expanded_box[0], expanded_box[1]), (expanded_box[2], expanded_box[3])],
                               outline="red", width=2)
                face_image = pil_image.crop((expanded_box[0], expanded_box[1], expanded_box[2], expanded_box[3]))
                face_image = face_image.convert("RGB")
                name = ""
                # name, confidence = name_pre_opencv(face_image)
                print(name)
                draw.text((expanded_box[0], expanded_box[1] - 10), name, fill="white")
        # 显示结果图像
        cv2.imshow('Face Detection', cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))

        # if cv2.waitKey(1) & 0xFF == ord('s'):
        #     face_image.save(
        #         r"d:\Users\charm\Documents\User\Document\数据集\明星图片\images\face\R\{}.jpg".format(time.time()))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()


cam()
