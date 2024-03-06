import cv2
from PIL import Image
import shutil


def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height


def save(boxs, path2):
    boxs_with_newline = [box + '\n' for box in boxs]
    with open(path2, "w") as w:
        w.writelines(boxs_with_newline)


def draw(x_min, y_min, x_max, y_max, img_path):
    if not all(isinstance(val, int) for val in [x_min, y_min, x_max, y_max]):
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)
    img = cv2.imread(img_path)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    # 显示图像
    cv2.imshow("Image with Bounding Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# YOLO标签文件和图像路径
def draw_yolo(label_path, img_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    img = cv2.imread(img_path)
    # 解析标签并绘制边界框
    for line in lines:
        class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
        # 将相对坐标转换为绝对坐标
        img_h, img_w, _ = img.shape
        x_center_abs = int(x_center * img_w)
        y_center_abs = int(y_center * img_h)
        box_width_abs = int(box_width * img_w)
        box_height_abs = int(box_width * img_h)
        # 中心坐标
        print("center_abs", x_center_abs, y_center_abs, box_width_abs, box_height_abs)
        # 计算边界框的左上角和右下角坐标
        x_min = int(x_center_abs - box_width_abs / 2)
        y_min = int(y_center_abs - box_height_abs / 2)
        x_max = int(x_center_abs + box_width_abs / 2)
        y_max = int(y_center_abs + box_height_abs / 2)

        # 绘制边界框
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    # 显示图像
    cv2.imshow("Image with Bounding Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# xmin,ymin 为左上角坐标   xmax,ymax为右上角坐标    img_width为图片宽 img_height为图片高
def convert(xmin, ymin, xmax, ymax, img_width, img_height):
    box_width = (xmax - xmin) / img_width
    box_height = (ymax - ymin) / img_height
    x_center = (xmax - (xmax - xmin) / 2) / img_width
    y_center = (ymax - (ymax - ymin) / 2) / img_height
    box_width = round(box_width, 6)
    box_height = round(box_height, 6)
    x_center = round(x_center, 6)
    y_center = round(y_center, 6)
    return x_center, y_center, box_width, box_height


def handel():
    rootpath = "WIDER_FACE/WIDER_val/images/"
    with open("WIDER_FACE/wider_face_split/wider_face_val_bbx_gt.txt") as r:
        lines = r.readlines()
    for i in range(lines.__len__()):
        line = lines[i].replace("\n", "")
        if "/" in line:
            boxs = []
            img_width, img_height = get_image_size(rootpath + line)
            face_sum = int(lines[i + 1])
            for j in range(face_sum):
                face_line = lines[i + 2 + j]
                xys = face_line.split(" ")
                x_center = (int(xys[0]) + int(xys[2]) / 2) / img_width
                y_center = (int(xys[1]) + int(xys[3]) / 2) / img_height
                box_width = int(xys[2]) / img_width
                box_height = int(xys[3]) / img_height

                box_width = round(box_width, 6)
                box_height = round(box_height, 6)
                x_center = round(x_center, 6)
                y_center = round(y_center, 6)
                boxs.append("0 {} {} {} {}".format(x_center, y_center, box_width, box_height))
            name = line.split("/")[-1]
            # 保存标签且复制图像
            save(boxs, f"datasets/coco8/labels/val/{name.split('.')[0]}.txt")
            shutil.copy(rootpath + line, f"datasets/coco8/images/val/{name}")
            # draw_yolo(f"datasets/coco8/labels/train/{name.split('.')[0]}.txt", f"datasets/coco8/images/train/{name}")


if __name__ == '__main__':
    handel()
