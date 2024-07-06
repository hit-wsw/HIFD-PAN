import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

def average_pooling_kernel(image, scale_factor):
    # Define the average pooling kernel
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    
    # Perform convolution on the image
    output_image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    
    # Keep the dimensions of the scaled down image
    output_image = output_image[::scale_factor, ::scale_factor]
    
    return output_image



def SSIM(image1, image2):

    # Calculate structural similarity
    SSIM = ssim(image1, image2)

    return SSIM

def crop_img(img):
    height, width = img.shape[:2]
    top_left = (2, 2)  # 左上角坐标
    bottom_right = (width-2, height-2)  # 右下角坐标
    cropped_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return cropped_img


def main():
    # 读入两张彩色图像
    pan = cv2.imread('./data/pan.png')
    MS = cv2.imread('./data/MS_ini.png')
    MS_resize = cv2.imread('./data/MS_resize.png')
    data = cv2.imread('./highest/true_color_result.png')
    data_resize = average_pooling_kernel(data,4)

    pan = crop_img(pan)
    MS = crop_img(MS)
    MS_resize = crop_img(MS_resize)
    data = crop_img(data)
    data_resize = crop_img(data_resize)
    
    

    # 计算每个通道的RMSE和UIQI
    DL = []
    DS = []
    weights = [1/3, 1/3, 1/3]  # 三个通道的权值

    for channel in range(3):  # 三个通道：蓝、绿、红
        pan_c = pan[:,:,channel]
        MS_c = MS[:,:,channel]
        MS_resize_c = MS_resize[:,:,channel]
        data_c = data[:,:,channel]
        data_resize_c = data_resize[:,:,channel]
        
        s3 = SSIM(MS_resize_c, data_c)
        s4 = SSIM(MS_c,data_resize_c)
        DL.append(np.abs(s3-s4))

        s1 = SSIM(pan_c, MS_resize_c)
        s2 = SSIM(pan_c, data_c)
        DS.append(np.abs(s1-s2))

    # 使用权值加权计算最终结果
    DS_value = np.average(DS, weights=weights)
    
    DL_value = np.average(DL, weights=weights)
    return DL_value, DS_value

if __name__ == "__main__":
    DL_value, DS_value = main()
    print("DL:", DL_value)
    print("DS:", DS_value)
    QNR = (1-DS_value)*(1-DL_value)
    print("QNR:", QNR)
