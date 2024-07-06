from skimage.transform import rescale
import numpy as np
from numpy import pi, exp, sqrt
import cv2
from scipy.special import gamma


def process_image(image):#预处理
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = image.astype(np.float32) / 255.0
    image = (image * 255).astype(np.uint8)
    kernel_size = (5, 5)
    sigma = 0.5
    image = cv2.GaussianBlur(image, kernel_size, sigma)
    return image

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

def HIFD_conv(v):#计算卷积核
    para = cal_args(v)
    HIDF_conv = []
    HIDF0 = np.asarray([[0, 0, para[0], 0, 0], [0, 0, para[1], 0, 0], [0, 0, para[2], 0, 0], [0, 0, para[3], 0, 0], [0, 0, para[4], 0, 0]], dtype=np.float32)
    HIDF1 = np.asarray([[0, 0, para[4], 0, 0], [0, 0, para[3], 0, 0], [0, 0, para[2], 0, 0], [0, 0, para[1], 0, 0], [0, 0, para[0], 0, 0]], dtype=np.float32)
    HIDF2 = np.asarray([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [para[4], para[3], para[2], para[1], para[0]], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    HIDF3 = np.asarray([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [para[0], para[1], para[2], para[3], para[4]], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    HIDF4 = np.asarray([[para[0], 0, 0, 0, 0], [0, para[1], 0, 0, 0], [0, 0, para[2], 0, 0], [0, 0, 0, para[3], 0], [0, 0, 0, 0, para[4]]], dtype=np.float32)
    HIDF5 = np.asarray([[0, 0, 0, 0, para[0]], [0, 0, 0, para[1], 0], [0, 0, para[2], 0, 0], [0, para[3], 0, 0, 0], [para[4], 0, 0, 0, 0]], dtype=np.float32)
    HIDF6 = np.asarray([[para[4], 0, 0, 0, 0], [0, para[3], 0, 0, 0], [0, 0, para[2], 0, 0], [0, 0, 0, para[1], 0], [0, 0, 0, 0, para[0]]], dtype=np.float32)
    HIDF7 = np.asarray([[0, 0, 0, 0, para[4]], [0, 0, 0, para[3], 0], [0, 0, para[2], 0, 0], [0, para[1], 0, 0, 0], [para[0], 0, 0, 0, 0]], dtype=np.float32)
    HIDF_conv.append(HIDF0)
    HIDF_conv.append(HIDF1)
    HIDF_conv.append(HIDF2)
    HIDF_conv.append(HIDF3)
    HIDF_conv.append(HIDF4)
    HIDF_conv.append(HIDF5)
    HIDF_conv.append(HIDF6)
    HIDF_conv.append(HIDF7)
    return HIDF_conv

def HIDF_filter(v):#8个卷积核分别作用
    HIDF_conv = HIFD_conv(v)
    padding_size = 2
    padded_image = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT)
    for i in range(8):
        output_image = cv2.filter2D(padded_image, -1, HIDF_conv[i])
        output_image = output_image[padding_size*2:-padding_size*2, padding_size*2:-padding_size*2]
        cv2.imwrite('./results_H/mask/HIDF_image' + str(i) + '.jpg', output_image)
    return output_image

def HIDF_gradxy(v):#x,y方向上的梯度
    padding_size = 2
    HIDF_conv = HIFD_conv(v)
    padded_image = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT)
    pic0 = cv2.filter2D(padded_image, -1, HIDF_conv[0])
    pic0 = pic0[padding_size*2:-padding_size*2, padding_size*2:-padding_size*2]
    pic1 = cv2.filter2D(padded_image, -1, HIDF_conv[1])
    pic1 = pic1[padding_size*2:-padding_size*2, padding_size*2:-padding_size*2]
    pic2 = cv2.filter2D(padded_image, -1, HIDF_conv[2])
    pic2 = pic2[padding_size*2:-padding_size*2, padding_size*2:-padding_size*2]
    pic3 = cv2.filter2D(padded_image, -1, HIDF_conv[3])
    pic3 = pic3[padding_size*2:-padding_size*2, padding_size*2:-padding_size*2]
    pic4 = cv2.filter2D(padded_image, -1, HIDF_conv[4])
    pic4 = pic4[padding_size*2:-padding_size*2, padding_size*2:-padding_size*2]
    pic5 = cv2.filter2D(padded_image, -1, HIDF_conv[5])
    pic5 = pic5[padding_size*2:-padding_size*2, padding_size*2:-padding_size*2]
    pic6 = cv2.filter2D(padded_image, -1, HIDF_conv[6])
    pic6 = pic6[padding_size*2:-padding_size*2, padding_size*2:-padding_size*2]
    pic7 = cv2.filter2D(padded_image, -1, HIDF_conv[7])
    pic7 = pic7[padding_size*2:-padding_size*2, padding_size*2:-padding_size*2]

    coefficients = [1, -1, sqrt(2)/2, -sqrt(2)/2, sqrt(2)/2, -sqrt(2)/2]
    imagesX = [pic2, pic3, pic4, pic5, pic6, pic7]
    resultX = np.zeros_like(imagesX[0], dtype=np.float32)
    for i in range(len(imagesX)):
        resultX += coefficients[i] * imagesX[i]  

    imagesY = [pic0, pic1, pic4, pic6, pic5, pic7]
    resultY = np.zeros_like(imagesY[0], dtype=np.float32)
    for i in range(len(imagesY)):
        resultY += coefficients[i] * imagesY[i]

    cv2.imwrite('./results_H/gradX.jpg', resultX)
    cv2.imwrite('./results_H/gradY.jpg', resultY)
    return resultX, resultY

def non_maximum_suppression(gradient_magnitude, gradient_direction):#非极大值抑制
    rows, cols = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            direction = gradient_direction[i, j]
            neighbor_coords = []
            # 获取当前像素的两个相邻像素的坐标
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                neighbor_coords = [(i, j + 1), (i, j - 1)]
            elif 22.5 <= direction < 67.5:
                neighbor_coords = [(i - 1, j + 1), (i + 1, j - 1)]
            elif 67.5 <= direction < 112.5:
                neighbor_coords = [(i - 1, j), (i + 1, j)]
            elif 112.5 <= direction < 157.5:
                neighbor_coords = [(i - 1, j - 1), (i + 1, j + 1)]

            # 获取相邻像素的梯度幅值
            neighbor_mags = [gradient_magnitude[x, y] for x, y in neighbor_coords]

            # 如果当前像素的梯度幅值大于相邻像素，则将其保留为局部极大值
            if neighbor_coords:
                if gradient_magnitude[i, j] >= max(neighbor_mags):
                    suppressed[i, j] = gradient_magnitude[i, j]

    return suppressed

def apply_double_threshold(image, low_threshold, high_threshold):#双阈值检验
    strong_edges = np.where(image >= high_threshold, image, 0)
    weak_edges = np.where((image >= low_threshold) & (image < high_threshold), image, 0)
    return strong_edges, weak_edges

def edge_linking(strong_edges, weak_edges):#边缘连接
    rows, cols = strong_edges.shape
    linked_edges = np.copy(strong_edges)

    # 检查弱边缘像素的8邻域是否有强边缘像素
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j] == 255:
                if (strong_edges[i - 1:i + 2, j - 1:j + 2] == 255).any():
                    linked_edges[i, j] = 255

    return linked_edges

def enhance(image):#增强图像表现形式
    return np.where(image > 20, 255, image)

def HIDF1(v):
    gradx ,grady= HIDF_gradxy(v)
    grad_magnitude = cv2.sqrt(gradx**2 + grady**2)
    grad_direction = cv2.phase(gradx, grady, angleInDegrees=True)
    suppressed = non_maximum_suppression(grad_magnitude, grad_direction)
    strong_edges, weak_edges = apply_double_threshold(suppressed,0,30)
    #cv2.imwrite('./results_H/strong_edges.jpg', strong_edges)
    #cv2.imwrite('./results_H/weak_edges.jpg', weak_edges)
    HIDF_image = edge_linking(strong_edges, weak_edges)
    HIDF_image = enhance(HIDF_image)
    
    cv2.imwrite('./results_H/HIDF5g.jpg', HIDF_image)
    return suppressed

def HIDF(v):
    gradx ,grady= HIDF_gradxy(v)
    grad_magnitude = cv2.sqrt(gradx**2 + grady**2)
    #grad_direction = cv2.phase(gradx, grady, angleInDegrees=True)
    grad_magnitude = np.uint8(grad_magnitude)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grad_magnitude = clahe.apply(grad_magnitude)
    cv2.imwrite('./results_H/gauss/HIDF1.jpg', grad_magnitude)

image = cv2.imread("./data/ms_gauss.png", cv2.IMREAD_GRAYSCALE)
image = process_image(image)
#HIDF_filter(0.5)
#HIDF_gradxy(0.3)
HIDF1(0.5)