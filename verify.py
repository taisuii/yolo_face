import os


def verify(m):
    # 检查数据集是否完整
    path_img = "datasets/coco8/images/" + m + "/"
    path_label = "datasets/coco8/labels/" + m + "/"
    paths_img = os.listdir(path_img)
    paths_label = os.listdir(path_label)
    for i in paths_label:
        img_name = i.split('.')[0]  # 获取图片文件名（不含扩展名）
        label_name = img_name + ".jpg"  # 对应的标签文件名
        if label_name not in paths_img:
            print(f"{label_name} does not exist")


if __name__ == '__main__':
    verify("train")
