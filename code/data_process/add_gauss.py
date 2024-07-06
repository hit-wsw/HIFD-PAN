import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=15):
    """
    添加高斯噪声到图像
    :param image: 输入图像
    :param mean: 噪声的均值
    :param sigma: 噪声的标准差
    :return: 添加噪声后的图像
    """
    h, w, c = image.shape
    gauss = np.random.normal(mean, sigma, (h, w, c))
    noisy_image = np.clip(image + gauss, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


# Load grayscale image
image1 = cv2.imread('./data/true/irdata.png')
image = cv2.imread('./data/true/data.png')

# Add Gaussian noise with standard deviation of 0.05
noisy_image1 = add_gaussian_noise(image1, 2)
noisy_image = add_gaussian_noise(image, 2)

# Display the original and noisy images
cv2.imwrite('data/ir_gauss.png', noisy_image1)
cv2.imwrite("data/ms_gauss.png",noisy_image)
