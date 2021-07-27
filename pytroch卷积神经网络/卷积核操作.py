import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open("cat.png").convert("L")
im = np.array(im, np.float32)

print(im)
plt.imshow(im)
plt.show()

im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))

# 定义一个卷积核对图像进行边缘提取的操作
"""
        in_channels: int, 输入通道数
        out_channels: int, 输出通道数
        kernel_size: _size_2_t, 卷积核的大小
        stride: _size_2_t = 1, 步长
        padding: Union[str, _size_2_t] = 0, 填充
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True, 是否加上偏置项
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
"""
def conv1_():
    conv1 = nn.Conv2d(1, 1, (3, 3), bias=False)
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    conv1.weight.data = torch.from_numpy(sobel_kernel)
    edge1 = conv1(im)
    edge1 = edge1.data.squeeze().numpy()
    return edge1
plt.imshow(conv1_())
plt.show()
def max_pool():
    """
    kernel_size: _size_2_t 池化核的大小
    stride: _size_2_t 池化核移动的步长
    padding: _size_2_t 填充的大小
    dilation: _size_2_t
    :return:
    """
    pool1 = nn.MaxPool2d(kernel_size=(2, 2),  stride=2)
    small_im = pool1(im)
    small_im = small_im.data.squeeze().numpy()
    return small_im



plt.imshow(max_pool())
plt.show()