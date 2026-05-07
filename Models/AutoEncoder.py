import torch   # pytorch 基础库
import torch.nn as nn   # 用于构建卷积层、全连接层等基础层
import torch.nn.functional as F   # 用于构建激活函数、进行卷积池化等操作
from torch.nn.utils import weight_norm   # 用于权重归一化，加速模型训练

# =================== 自动编码器 ==================
def create_layer(in_channels, out_channels, kernel_size, wn=True, bn=True,activation=nn.ReLU, convolution=nn.Conv2d):
    assert kernel_size % 2 == 1   # 卷积核大小为奇数
    layer = []   # 创建一个列表，用于储存神经网络各层
    conv = convolution(in_channels, out_channels, kernel_size,padding=kernel_size//2)   # 给卷积层构造函数传入形参
    if wn:
        conv = weight_norm(conv)   # 权重归一化，对卷积层进行归一化
    layer.append(conv)   # 将归一化后的卷积层加入layer
    if activation is not None:
        layer.append(activation())   # 默认激活函数是ReLU
    if bn:
        layer.append(nn.BatchNorm2d(out_channels))   #批归一化，对网络中间特征进行归一化（激活值/输出值）
    return nn.Sequential(*layer)

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64],
                 weight_norm=True, batch_norm=True, activation=nn.ReLU, final_activation=None):
        super().__init__()
        assert len(filters) > 0
        encoder = []
        decoder = []
        for i in range(len(filters)):
            if i == 0:
                encoder_layer = create_layer(in_channels, filters[i], kernel_size, weight_norm, batch_norm, activation, nn.Conv2d)
                decoder_layer = create_layer(filters[i], out_channels, kernel_size, weight_norm, False, final_activation, nn.ConvTranspose2d)
            else:
                encoder_layer = create_layer(filters[i-1], filters[i], kernel_size, weight_norm, batch_norm, activation, nn.Conv2d)
                decoder_layer = create_layer(filters[i], filters[i-1], kernel_size, weight_norm, batch_norm, activation, nn.ConvTranspose2d)
            encoder = encoder + [encoder_layer]
            decoder = [decoder_layer] + decoder
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(self.encoder(x))