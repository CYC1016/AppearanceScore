import os
import paddle
import numpy as np
from PIL import Image
from PIL.Image import Resampling

# 数据集类
class MyDataset(paddle.io.Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """

    def __init__(self, mode='train'):
        """
        步骤二：实现构造函数，定义数据集大小
        """
        super(MyDataset, self).__init__()
        self.data = []
        self.label = []
        train_list_path = "./face_data_5/face_image_train"
        eval_list_path = "./face_data_5/face_image_test"
        # 训练数据集
        if mode == 'train':
            for image in os.listdir(train_list_path):
                # 解析文件名 取第一位数为评分
                label = int(image.split('-')[0]) - 1
                img_path = os.path.join(train_list_path + '/' + image)
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((100, 100), Resampling.BILINEAR)
                img = np.array(img).astype('float32')
                img = img.transpose((2, 0, 1))  # HWC to CHW
                img = img / 255  # 像素值归一化
                # 图片加入到数据集 评分加入到标签集
                self.data.append(img)
                self.label.append(int(label))
        # 测试数据集 同上操作 不同路径
        else:
            for image in os.listdir(eval_list_path):
                label = int(image.split('-')[0]) - 1
                img_path = os.path.join(eval_list_path + '/' + image)
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((100, 100), Resampling.BILINEAR)
                img = np.array(img).astype('float32')
                img = img.transpose((2, 0, 1))  # HWC to CHW
                img = img / 255  # 像素值归一化
                self.data.append(img)
                self.label.append(int(label))
    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        #返回单一数据和标签
        data = self.data[index]
        label = self.label[index]
        #注：返回标签数据时必须是int64
        return data, np.array(label).astype('int64')
    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        #返回数据总数
        return len(self.data)
#图片预处理
def load_image(img_path):
    # 打开图片
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((100, 100), Resampling.BILINEAR)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = img / 255  # 像素值归一化
    image = np.expand_dims(img, axis=0)
    # 保持和之前输入image维度一致
    print('图片的维度：', image.shape)
    return image


