from data import MyDataset
from model import MyModel

# 导入必要的包
import os
import paddle
import numpy as np
import zipfile



#参数配置
train_parameters = {
    "input_size": [3, 100, 100],  # 输入图片的shape
    "class_dim": 5,  # 分类数
    "src_path": "./data/data17941/face_data_5.zip",  # 原始数据集路径
    "target_path": "./face_data_5",  # 要解压的路径
    "train_list_path": "./face_data_5/face_image_train",  # train.txt路径
    "eval_list_path": "./face_data_5/face_image_test",  # eval.txt路径
    "label_dict": {},  # 标签字典
    "num_epochs": 20,  # 训练轮数
    "train_batch_size": 64,  # 训练时每个批次的大小
    "learning_strategy": {  # 优化函数相关的配置
        "lr": 0.001  # 超参数学习率
    }
}

def train():
    # 解压数据集
    src_path = train_parameters["src_path"]
    target_path = train_parameters["target_path"]
    if (not os.path.isdir(target_path)):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()

    # 测试定义的数据集
    train2 = MyDataset(mode='train')
    test2 = MyDataset(mode='val')

    # 数据预处理 list转为数组
    train_data = np.array(train2.data)
    train_label = np.array(train2.label)
    test_data = np.array(test2.data)
    test_label = np.array(test2.label)

    #创建模型
    model = MyModel().model
    # 配置模型
    optim = paddle.optimizer.Adam(
        learning_rate=train_parameters["learning_strategy"]["lr"], parameters=model.parameters())
    model.prepare(
        optim,
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 2)))
    # 训练模型
    history = model.fit(train2,
                        test2,
                        epochs=train_parameters["num_epochs"],
                        batch_size=train_parameters["train_batch_size"],
                        verbose=1,
                        shuffle=True,
                        save_dir='./chk_points/')
if __name__ == '__main__':
    train()

