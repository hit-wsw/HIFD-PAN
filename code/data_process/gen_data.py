import numpy as np
import cv2 
from PIL import Image

def average_pooling_kernel(image, scale_factor):
    # Define the average pooling kernel
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    
    # Perform convolution on the image
    output_image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    
    # Keep the dimensions of the scaled down image
    output_image = output_image[::scale_factor, ::scale_factor]
    
    return output_image

# 读取原始图像
MS = cv2.imread('./data/P+XS/data.jpg')
IR = cv2.imread('./data/P+XS/IR.jpg')


resized_image = average_pooling_kernel(IR, 4)

cv2.imwrite("./data/IR_ini.png",resized_image)

# 使用最近邻插值进行上采样
upsampled_image = cv2.resize(resized_image, (IR.shape[1],IR.shape[0]), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./data/IR_resize.png",upsampled_image)

gray_image = cv2.cvtColor(MS, cv2.COLOR_BGR2GRAY)
# 将彩色图像转换为灰度图像 
cv2.imwrite("./data/pan.png",gray_image)

resized_image = average_pooling_kernel(MS, 4)

cv2.imwrite("./data/MS_ini.png",resized_image)

# 使用最近邻插值进行上采样
upsampled_image = cv2.resize(resized_image, (MS.shape[1],MS.shape[0]), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./data/MS_resize.png",upsampled_image)