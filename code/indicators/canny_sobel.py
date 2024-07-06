import cv2
import numpy as np
from skimage.transform import rescale
# 读取灰度图像
image = cv2.imread("./data/ms_gauss.png", cv2.IMREAD_GRAYSCALE)

# Normalize the image to the range [0, 1]
image = image.astype(np.float32) / 255.0
image = (image * 255).astype(np.uint8)

blurred = cv2.GaussianBlur(image, (0, 0), 2)
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
#反色
#laplacian = 255-laplacian
cv2.imwrite('./results_H/LoG.jpg', laplacian)


#Sobel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度幅值和方向
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_direction = np.arctan2(sobel_y, sobel_x)

# 将梯度幅值映射到0-255范围
magnitude_scaled = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
#magnitude_scaled = 255 - magnitude_scaled
cv2.imwrite('./results_H/sobel.jpg', magnitude_scaled)

# 使用Canny边缘检测
edges = cv2.Canny(image, 100, 200)  # 参数50和150是低阈值和高阈值

# 显示原始图像和Canny边缘检测结果
edges = 255-edges
cv2.imwrite('./results_H/canny.jpg', edges)

# 定义Laplacian算子
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# 对图像进行卷积操作
def convolution(image, kernel):
    height, width = image.shape
    k_size = len(kernel)
    result = np.zeros((height - k_size + 1, width - k_size + 1))
    for i in range(height - k_size + 1):
        for j in range(width - k_size + 1):
            result[i, j] = np.sum(image[i:i+k_size, j:j+k_size] * kernel)
    return result

# 对图像应用Laplacian算子
laplacian_output = convolution(image, laplacian_kernel)

# 将结果映射到0-255范围内
laplacian_output = cv2.normalize(laplacian_output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imwrite('./results_H/laplacian.jpg', laplacian_output)

