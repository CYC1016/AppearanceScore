import paddle
import paddle.nn as nn
# 检查paddle是否可用
# paddle.fluid.install_check.run_check()

# 构建CNN模型
class MyModel():
    def __init__(self):
        net = nn.Sequential(
    nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Conv2D(in_channels=16, out_channels=36, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Dropout(0.25),
    nn.Flatten(),
    nn.Linear(in_features=22500, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=5),
        )
        # paddle.summary(net, (450, 3, 100, 100))
        self.model=paddle.Model(net)


