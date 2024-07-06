import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

def RMSE(image1, image2):
    # 计算均方根误差（RMSE）
    mse = np.mean((image1 - image2) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def SSIM(image1, image2):

    # Calculate structural similarity
    SSIM = ssim(image1, image2)

    return SSIM

def entropy(image):

    # Compute histogram of grayscale image
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Normalize histogram
    hist_norm = hist.ravel() / hist.sum()

    # Compute entropy
    entropy_value = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))

    return entropy_value

def FCC(image1, image2, sigma=1.5):
    # Apply Gaussian filter to both images
    filtered_image1 = gaussian_filter(image1, sigma=sigma)
    filtered_image2 = gaussian_filter(image2, sigma=sigma)

    # Compute correlation coefficient between filtered images
    correlation_coefficient, _ = pearsonr(filtered_image1.flatten(), filtered_image2.flatten())

    return correlation_coefficient

def spatial_frequency(image):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Compute 2D Fourier Transform
    f_transform = np.fft.fft2(gray_image)

    # Shift the zero frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)

    # Compute spatial frequency
    spatial_freq = np.mean(magnitude_spectrum)

    return spatial_freq

def crop_img(img):
    height, width = img.shape[:2]
    top_left = (2, 2)  # 左上角坐标
    bottom_right = (width-2, height-2)  # 右下角坐标
    cropped_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return cropped_img

# 读取灰度图像和彩色图像
gray_image = cv2.imread('./data/P+XS/pan.png', cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread('./P+XS/true_color_result.jpg')

gray_image = crop_img(gray_image)
color_image = crop_img(color_image)
# 将彩色图像转换为灰度图像
gray_color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# 计算均方根误差（RMSE）和结构相似性指标（UIQI）
rmse = RMSE(gray_image, gray_color_image)
Ssim = SSIM(gray_image, gray_color_image)
Entropy = entropy(gray_color_image)
FCC_value = FCC(gray_image, gray_color_image)
SF_value = spatial_frequency(gray_color_image)

print("RMSE:", rmse)
print("SSIM:", Ssim)
print("Entropy:", Entropy)
print("FCC:", FCC_value)
print("SF:", SF_value)