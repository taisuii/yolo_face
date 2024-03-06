from PIL import Image
import cv2
import os
import numpy as np
from config import config


# 读取图像并转换为OpenCV格式
def load_image_cv2(path):
    pil_image = Image.open(path)
    pil_image = pil_image.convert('RGB')  # 转换为RGB模式
    open_cv_image = np.array(pil_image)
    # 将RGB图像转换为BGR格式（OpenCV默认格式）
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    return gray_image


def filter(name, files):
    arr = []
    for i in files:
        if name in i:
            arr.append(i)
    return arr


def train(path_img):
    # 准备数据集
    face_images = []
    labels = []
    files = os.listdir(path_img)

    for idx, name in enumerate(config.NAME_LIST):
        arr = filter(name, files)
        for i in arr:
            img_path = os.path.join(path_img, i)
            img = load_image_cv2(img_path)
            face_images.append(img)
            labels.append(idx)
    # 创建LBPH人脸识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 训练模型
    recognizer.train(face_images, np.array(labels))
    # 保存模型
    recognizer.save("model.yml")


train(r"../datasets/FACE_train/")
