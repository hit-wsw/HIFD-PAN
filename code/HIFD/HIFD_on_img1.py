from skimage.transform import rescale
import numpy as np
from numpy import pi, exp, sqrt
import matplotlib.pyplot as plt
from helpers import load_image, save_image, my_imfilter
from skimage import color
import cv2
from scipy.special import gamma

def parse_args(v,n):
    result = (-v**4/64 + 3*v**2/16 -v/4)*(gamma(n-1-v)/(gamma(n)*gamma(-v))) + (v**4/32 - 3*v**2/8 + 1)*(gamma(n-v)/(gamma(n+1)*gamma(-v))) + (-v**4/64 + 3*v**2/16 + v/4)*(gamma(n+1-v)/(gamma(n+2)*gamma(-v)))
    return result

def cal_args(v):#计算系数
    args = [0,0,0,0,0]
    args[0] = -v**4/64 + 3*v**2/16 + v/4
    args[1] = v**5/64 + v**4/32 -3*v**3/16 -5*v**2/8 + 1
    args[2] = parse_args(v,1)
    args[3] = parse_args(v,2)
    args[4] = parse_args(v,3)
    return args

def NIFD(v):#计算卷积核
    para = cal_args(v)
    NIDF_conv = []
    NIDF0 = np.asarray([[0, 0, para[0], 0, 0], [0, 0, para[1], 0, 0], [0, 0, para[2], 0, 0], [0, 0, para[3], 0, 0], [0, 0, para[4], 0, 0]], dtype=np.float32)
    NIDF1 = np.asarray([[0, 0, para[4], 0, 0], [0, 0, para[3], 0, 0], [0, 0, para[2], 0, 0], [0, 0, para[1], 0, 0], [0, 0, para[0], 0, 0]], dtype=np.float32)
    NIDF2 = np.asarray([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [para[4], para[3], para[2], para[1], para[0]], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    NIDF3 = np.asarray([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [para[0], para[1], para[2], para[3], para[4]], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    NIDF4 = np.asarray([[para[0], 0, 0, 0, 0], [0, para[1], 0, 0, 0], [0, 0, para[2], 0, 0], [0, 0, 0, para[3], 0], [0, 0, 0, 0, para[4]]], dtype=np.float32)
    NIDF5 = np.asarray([[0, 0, 0, 0, para[0]], [0, 0, 0, para[1], 0], [0, 0, para[2], 0, 0], [0, para[3], 0, 0, 0], [para[4], 0, 0, 0, 0]], dtype=np.float32)
    NIDF6 = np.asarray([[para[4], 0, 0, 0, 0], [0, para[3], 0, 0, 0], [0, 0, para[2], 0, 0], [0, 0, 0, para[1], 0], [0, 0, 0, 0, para[0]]], dtype=np.float32)
    NIDF7 = np.asarray([[0, 0, 0, 0, para[4]], [0, 0, 0, para[3], 0], [0, 0, para[2], 0, 0], [0, para[1], 0, 0, 0], [para[0], 0, 0, 0, 0]], dtype=np.float32)
    NIDF_conv.append(NIDF0)
    NIDF_conv.append(NIDF1)
    NIDF_conv.append(NIDF2)
    NIDF_conv.append(NIDF3)
    NIDF_conv.append(NIDF4)
    NIDF_conv.append(NIDF5)
    NIDF_conv.append(NIDF6)
    NIDF_conv.append(NIDF7)
    return NIDF_conv

def NIFD_filter(v):
    NIDF_conv = NIFD(v)
    for i in range(8):
        NIDF_image = my_imfilter(test_image, NIDF_conv[i],fft=True)
        plt.imshow(NIDF_image)
        #反色
        NIDF_image = 255 - NIDF_image
        cv2.imwrite('./results_H/mask/NIDF_image' + str(i) + '.jpg', NIDF_image)

def NIFD_gradxy(v):
    NIDF_conv = NIFD(v)
    pic0 = my_imfilter(test_image, NIDF_conv[0],fft=True)
    pic1 = my_imfilter(test_image, NIDF_conv[1],fft=True)
    pic2 = my_imfilter(test_image, NIDF_conv[2],fft=True)
    pic3 = my_imfilter(test_image, NIDF_conv[3],fft=True)   
    pic4 = my_imfilter(test_image, NIDF_conv[4],fft=True)
    pic5 = my_imfilter(test_image, NIDF_conv[5],fft=True)
    pic6 = my_imfilter(test_image, NIDF_conv[6],fft=True)
    pic7 = my_imfilter(test_image, NIDF_conv[7],fft=True)
    # 定义系数
    coefficients = [1, -1, sqrt(2)/2, -sqrt(2)/2, sqrt(2)/2, -sqrt(2)/2]
    imagesX = [pic2, pic3, pic4, pic5, pic6, pic7]
    resultX = np.zeros_like(imagesX[0], dtype=np.float32)
    for i in range(len(imagesX)):
        resultX += coefficients[i] * imagesX[i]

    #resultX = non_maximum_suppression(resultX)
    resultX = cv2.normalize(resultX, None, -1, 1, cv2.NORM_MINMAX)

    imagesY = [pic0, pic1, pic4, pic6, pic5, pic7]
    resultY = np.zeros_like(imagesY[0], dtype=np.float32)
    for i in range(len(imagesY)):
        resultY += coefficients[i] * imagesY[i]
    #resultY = non_maximum_suppression(resultY)
    resultY = cv2.normalize(resultY, None, -1, 1, cv2.NORM_MINMAX)

    height, width = resultY.shape[:2]
    resultY[:2, :]  = 0
    resultX[:, -2:] = 0
    save_image('./results_H/NIDF_gradx.jpg', resultX)
    save_image('./results_H/NIDF_grady.jpg', resultY)
    return resultX, resultY

def NIFD_grad():
    resultX = cv2.imread('./results_H/NIDF_gradx.jpg')
    resultY = cv2.imread('./results_H/NIDF_grady.jpg')
    squared_sum = np.square(resultX.astype(np.float32)) + np.square(resultY.astype(np.float32))

    # 求和
    sum_result = np.sum(squared_sum, axis=-1)

    # 开根号
    sqrt_result = np.sqrt(sum_result)

    # 将结果缩放到0到255之间
    scaled_result = cv2.normalize(sqrt_result, None, 0, 255, cv2.NORM_MINMAX)
    scaled_result = np.uint8(scaled_result)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    scaled_result = clahe.apply(scaled_result)
    #scaled_result[scaled_result > 50] = 255
    done = save_image('HIDF.jpg', scaled_result)

image = cv2.imread('./data/true/data.png', cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32) / 255.0
test_image = (image * 255).astype(np.uint8)

NIFD_filter(0.5)
#NIFD_gradxy(0.1)
#NIFD_grad()

#NIFD_filter(0.5)
