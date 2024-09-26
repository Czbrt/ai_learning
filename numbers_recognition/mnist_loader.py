import os
import random
from torch.utils.data import Dataset
import cv2


def image_list(imageRoot: str, txt: str = 'list.txt') -> None:
    with open(txt, 'wt') as f:  # 使用 with 语句管理文件上下文
        for filename in sorted(os.listdir(imageRoot), reverse=False):
            # os.listdir(imageRoot) 列出根目录下的所有文件和文件夹。
            # sorted(..., reverse=False) 将这些文件和文件夹按字母顺序排序。
            # enumerate(...) 用于给每个文件夹分配一个标签 label（从 0 开始）。运行错误 - 从1开始
            if os.path.isdir(os.path.join(imageRoot, filename)):
                # 使用 os.path.isdir(...) 检查当前项 filename 是否为一个目录。
                label = int(filename)
                # 文件夹名作为label
                for image_name in os.listdir(os.path.join(imageRoot, filename)):
                    # 如果是目录，则列出该目录下的所有文件（图片）。
                    name, ext = os.path.splitext(image_name)
                    ext = ext[1:]
                    # 使用 os.path.splitext(...) 将文件名分为名称 (name) 和扩展名 (ext)。
                    # ext = ext[1:] 去掉扩展名前的点（.），只保留扩展名的字母。
                    if ext in ['jpg', 'png', 'bmp']:
                        # 检查文件扩展名是否为支持的图片格式（jpg, png, bmp）。
                        f.write(f'{os.path.join(imageRoot, filename, image_name)} {label}\n')
                        # 将图片的完整路径和对应的标签写入文本文件，格式为 图片路径 标签，并换行。


def shuffle_split(listFile, trainFile, valFile):
    with open(listFile, 'r') as f:
        records = f.readlines()
    # read file's content

    random.shuffle(records)
    # shuffle the data

    num = len(records)
    trainNum = int(num * 0.8)
    # trainNum = 100
    # divided into train set and val set

    with open(trainFile, 'w') as f:
        f.writelines(records[0:trainNum])
    with open(valFile, 'w') as f1:
        f1.writelines(records[trainNum:])
        # f1.writelines(records[trainNum:trainNum + 100])
    # write the data into two files


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            # 移除行尾的换行符
            line = line.rstrip()
            # 去除右侧的多余空白字符
            words = line.split()
            # 将字符串按空格分割为列表，假设列表第一个元素是图像路径，第二个元素是标签
            imgs.append((words[0], int(words[1])))
            # 将图像路径和标签（转换为整数）以二元组形式加入 imgs 列表
        self.imgs = imgs
        # 将 imgs 列表保存为实例变量
        self.transform = transform
        self.target_transform = target_transform
        # 这两个变量用于存储图像和标签的变换操作。它们可以接受传入的数据变换函数

    def __getitem__(self, index):
        # __getitem__() 的调用是在每次从 DataLoader 迭代获取新批次时自动发生的
        fn, label = self.imgs[index]
        # 从 self.imgs 中提取第 index 个图像文件路径和对应的标签
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        # 使用 OpenCV 的 cv2.imread 方法读取图像文件
        # cv2.IMREAD_COLOR 参数表示读取彩色图像。fn 是图像的文件路径
        # cv2.imread 会返回一个 Numpy 数组，表示图像的像素矩阵
        # 图像的每个通道 (RGB) 的像素值都存在这个矩阵中
        # img = cv2.resize(img, (28, 28))
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    # 判断是否传入了 transform 变换函数，将数据归一化
    # 如果有变换，则将变换函数应用到图像上

    def __len__(self):
        return len(self.imgs)
    # 这个方法返回数据集的长度，即包含多少张图像
    # len(self.imgs) 计算 self.imgs 列表中的元素数量，即图像和标签对的数量
