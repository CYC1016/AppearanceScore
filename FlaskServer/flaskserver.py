import os
import uuid
from wsgiref.simple_server import make_server
import numpy as np
import paddle
from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from data import load_image
from model import MyModel
from train import train_parameters

app = Flask(__name__)
CORS(app)
# 人脸颜值打分接口
@app.route('/pred', methods=['POST'])
def predict():
    # 获取图片
    f = request.files['img']
    # 保存图片
    savePath = 'images'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    imgPath = os.path.join(savePath, str(uuid.uuid1()) + secure_filename(f.filename).split('.')[-1])
    f.save(imgPath)
    label_list = ["1", "2", "3", "4", "5"]
    # 预处理图片
    img = load_image(imgPath)
    # 将图片变为数组
    img = np.array(img).astype('float32')
    # 加载模型
    model=MyModel().model
    model.load("./chk_points/97")
    # 配置模型
    optim = paddle.optimizer.Adam(
        learning_rate=train_parameters["learning_strategy"]["lr"], parameters=model.parameters())
    model.prepare(
        optim,
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 2)))
    # 预测
    result = model.predict(np.expand_dims(img, axis=0))
    print('results', result)
    print('预测得分',label_list[np.argmax(result)])
    return label_list[np.argmax(result)]

@app.route('/test')
def test():
    return "接口测试成功"

if __name__ == '__main__':
    print("后台服务启动")
    server = make_server('127.0.0.1', 5000, app)
    server.serve_forever()
    app.run(port=5000)
