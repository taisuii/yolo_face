# 输出所有的人脸类别
import os


def name_list(path_img):
    files = os.listdir(path_img)
    names = []
    for i in files:
        i = i.split('_')[0]
        if i not in names:
            names.append(i)
    # 所有人名
    print(names)


name_list(r"datasets/FACE_train/")
