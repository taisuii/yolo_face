import cv2
import os
import numpy as np
from PIL import Image
from config import config


def load_image_cv2(path):
    pil_image = Image.open(path)
    pil_image = pil_image.convert('RGB')  # 转换为RGB模式
    open_cv_image = np.array(pil_image)
    # 将RGB图像转换为BGR格式（OpenCV默认格式）
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    return gray_image


def pre(test_image_path):
    # 加载训练好的模型
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("model.yml")
    # 准备测试图像
    test_image = load_image_cv2(test_image_path)
    # 进行人脸识别
    label, confidence = recognizer.predict(test_image)
    predicted_celebrity = config.NAME_LIST[label]
    print("Predicted celebrity:", predicted_celebrity)
    print("Confidence:", confidence)


pre("../datasets/FACE_train/冯提莫_12.jpg")
