import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

def calculate_rmse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
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

def crop_img(img):
    height, width = img.shape[:2]
    top_left = (2, 2)  # 左上角坐标
    bottom_right = (width-2, height-2)  # 右下角坐标
    cropped_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return cropped_img


def main():
    # 读入两张彩色图像
    data1 = cv2.imread('./gauss/ms_gauss.png')
    data2 = cv2.imread('./gauss/true_color_result_gauss_normal.png')
    data1 = crop_img(data1)
    data2 = crop_img(data2)

    # 计算每个通道的RMSE和UIQI
    rmse_channels = []
    ssim_channels = []
    entropy_channels = []
    fcc_channels = []
    weights = [1/3, 1/3, 1/3]  # 三个通道的权值

    for channel in range(3):  # 三个通道：蓝、绿、红
        data1_channel = data1[:,:,channel]
        data2_channel = data2[:,:,channel]
        rmse_channels.append(calculate_rmse(data1_channel, data2_channel))
        ssim_channels.append(SSIM(data1_channel, data2_channel))
        entropy_channels.append(entropy(data2_channel))
        fcc_channels.append(FCC(data1_channel, data2_channel))

    # 使用权值加权计算最终结果
    rmse_value = np.average(rmse_channels, weights=weights)
    ssim_value = np.average(ssim_channels, weights=weights)
    entropy_value = np.average(entropy_channels, weights=weights)
    fcc_value = np.average(fcc_channels, weights=weights)

    return rmse_value, ssim_value,entropy_value,fcc_value

if __name__ == "__main__":
    rmse, ssim1, entropy_value,fcc_value = main()
    print("RMSE:", rmse)
    print("ssim:", ssim1)
    print("FCC:", fcc_value)
    print("Entropy:", entropy_value)
    
