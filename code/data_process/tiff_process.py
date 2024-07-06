import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
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

input_folder = './data/tiff/'

tiff_file = 'MS.tiff'
tiff_data = tifffile.imread(os.path.join(input_folder, tiff_file))
tiff_data_uint8 = ((tiff_data - np.min(tiff_data)) / (np.max(tiff_data) - np.min(tiff_data)) * 255).astype(np.uint8)
save_channels = [2, 1, 0]  # 选择前三个通道
tiff_data_uint8 = tiff_data_uint8[:, :, save_channels]
last_channel = tiff_data_uint8[:, :, -1] 
    

#plt.imshow(tiff_data_uint8)  # 不使用 cmap='gray' 参数
plt.imsave(os.path.join(input_folder, tiff_file.replace('.tiff', '.png')), tiff_data_uint8)
plt.imsave(os.path.join(input_folder, tiff_file.replace('MS.tiff', 'ir.png')), last_channel, cmap='gray')

image = cv2.imread('./data/tiff/MS.png')

'''gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 将彩色图像转换为灰度图像 
cv2.imwrite("./data/pan.jpg",gray_image)'''

resized_image = average_pooling_kernel(image, 4)

cv2.imwrite("./data/MS_ini.png",resized_image)

# 使用最近邻插值进行上采样
upsampled_image = cv2.resize(resized_image, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./data/MS_resize.png",upsampled_image)

image = cv2.imread('./data/tiff/ir.png')


resized_image = average_pooling_kernel(image, 4)

cv2.imwrite("./data/IR_ini.png",resized_image)

# 使用最近邻插值进行上采样
upsampled_image = cv2.resize(resized_image, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./data/IR_resize.png",upsampled_image)

tiff_data = tifffile.imread("./data//tiff/P.tiff")
tiff_data_uint8 = ((tiff_data - np.min(tiff_data)) / (np.max(tiff_data) - np.min(tiff_data)) * 255).astype(np.uint8)

# 转换为灰度图像
gray_image = Image.fromarray(tiff_data_uint8).convert('L')

# 保存单通道的灰度图像为PNG格式
gray_image.save("./data/pan.png")