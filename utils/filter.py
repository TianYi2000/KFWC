import numpy as np
from PIL import Image
import scipy.signal as signal


im=Image.open('../test.jpg')   # 读入图片并建立Image对象im
# im=im.convert('L')          # 转为灰度图
data=[]                     # 存储图像中所有像素值的list(二维)
width,height=im.size        # 图片尺寸,长宽，或者说宽高-->横竖
print(width, height)

# 读取图像像素的值
for w in range(width):      # 对每个行号h
    row=[]                  # 记录每一行像素
    for h in range(height): # 对每行的每个像素列位置w
        value=im.getpixel((w, h))    # 用getpixel读取这一点像素值
        row.append(value)            # 把它加到这一行的list中去
    data.append(row)        # 把记录好的每一行加到data的子list中去，就建立了模拟的二维list
# print(data)
### 彩色图滤波
data=signal.medfilt(data,kernel_size=3)           # 二维中值滤波
# data=signal.medfilt(data,kernel_size=5)
# data=signal.medfilt(data,kernel_size=(3, 3, 3))

### 灰度图滤波
##1
# data=signal.medfilt(data,kernel_size=(3, 3))
##2
# data = np.uint8(data)
# data=signal.medfilt2d(data,kernel_size=(11, 11))

#
# data=np.int32(data)                         # 转换为int类型，以使用快速二维滤波
# data=np.int8(data)
# print(data)

# 创建并保存结果
for w in range(width):       # 对每一行
    for h in range(height):  # 对该行的每一个列号
        im.putpixel((w,h),tuple(data[w][h])) # 将data中该位置的值存进图像,要求参数为tuple
        # im.putpixel((w,h),data[w][h])       # 灰度图的处理方式

im.save('result2.jpg')       # 存储