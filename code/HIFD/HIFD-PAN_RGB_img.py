import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np
import torchvision.utils as vutils
import torch.nn.functional as F
from scipy.special import gamma
from torch.utils.tensorboard import SummaryWriter


import os
import glob

def process_img(MS,s):

    gray_image = cv2.cvtColor(MS, cv2.COLOR_BGR2GRAY)
    # 将彩色图像转换为灰度图像 
    cv2.imwrite("./data/pan.png",gray_image)

    resized_image = average_pooling_kernel(MS, s)

    cv2.imwrite("./data/MS_ini.png",resized_image)

    # 使用最近邻插值进行上采样
    upsampled_image = cv2.resize(resized_image, (MS.shape[1],MS.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("./data/MS_resize.png",upsampled_image)

def pic_to_tensor(pan,ms,img2):#将图片转换为tensor张量
    pan = torch.from_numpy(pan).float().unsqueeze(0).unsqueeze(0) / 255.0
    ms = torch.from_numpy(ms).float().unsqueeze(0).unsqueeze(0) / 255.0
    color_image = cv2.cvtColor(img2, cv2.IMREAD_COLOR)  # 如果图像是BGR格式，请转换为RGB格式
    img2 = torch.from_numpy(color_image).float().permute(2, 0, 1) / 255.0  # 调整通道顺序并归一化
    img2 = img2.unsqueeze(0)  # 添加批次维度
    return pan,ms,img2

def average_pooling_kernel(image, scale_factor):#平均池化（kn*Xn)
    # Define the average pooling kernel as a tensor
    kernel = torch.tensor([[[[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]]]]) / 9.0
    
    # Move the kernel to the same device as the input image
    kernel = kernel.to(image.device)
    
    # Perform convolution on the image, using cv2.BORDER_REPLICATE for border filling
    output_image = F.conv2d(image, kernel, stride=1, padding=1)
    
    # Keep the dimensions of the scaled down image
    output_image = output_image[:, :, ::scale_factor, ::scale_factor]
    
    return output_image

def parse_args(v,n):#计算HIDF算子参数
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
    HIDF0 = torch.tensor([
        [
            [[0, 0, para[0], 0, 0], 
             [0, 0, para[1], 0, 0], 
             [0, 0, para[2], 0, 0], 
             [0, 0, para[3], 0, 0], 
             [0, 0, para[4], 0, 0]]
        ]
    ], dtype=torch.float32)

    HIDF1 = torch.tensor([
        [
            [[0, 0, para[4], 0, 0], 
             [0, 0, para[3], 0, 0], 
             [0, 0, para[2], 0, 0], 
             [0, 0, para[1], 0, 0], 
             [0, 0, para[0], 0, 0]]
        ]
    ], dtype=torch.float32)

    HIDF2 = torch.tensor([
        [
            [[0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0], 
             [para[4], para[3], para[2], para[1], para[0]], 
             [0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0]]
        ]
    ], dtype=torch.float32)

    HIDF3 = torch.tensor([
        [
            [[0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0], 
             [para[0], para[1], para[2], para[3], para[4]], 
             [0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0]]
        ]
    ], dtype=torch.float32)

    HIDF4 = torch.tensor([
        [
            [[para[0], 0, 0, 0, 0], 
             [0, para[1], 0, 0, 0], 
             [0, 0, para[2], 0, 0], 
             [0, 0, 0, para[3], 0], 
             [0, 0, 0, 0, para[4]]]
        ]
    ], dtype=torch.float32)

    HIDF5 = torch.tensor([
        [
            [[0, 0, 0, 0, para[0]], 
             [0, 0, 0, para[1], 0], 
             [0, 0, para[2], 0, 0], 
             [0, para[3], 0, 0, 0], 
             [para[4], 0, 0, 0, 0]]
        ]
    ], dtype=torch.float32)

    HIDF6 = torch.tensor([
        [
            [[para[4], 0, 0, 0, 0], 
             [0, para[3], 0, 0, 0], 
             [0, 0, para[2], 0, 0], 
             [0, 0, 0, para[1], 0], 
             [0, 0, 0, 0, para[0]]]
        ]
    ], dtype=torch.float32)

    HIDF7 = torch.tensor([
        [
            [[0, 0, 0, 0, para[4]], 
             [0, 0, 0, para[3], 0], 
             [0, 0, para[2], 0, 0], 
             [0, para[1], 0, 0, 0], 
             [para[0], 0, 0, 0, 0]]
        ]
    ], dtype=torch.float32)

    HIDF_conv.append(HIDF0)
    HIDF_conv.append(HIDF1)
    HIDF_conv.append(HIDF2)
    HIDF_conv.append(HIDF3)
    HIDF_conv.append(HIDF4)
    HIDF_conv.append(HIDF5)
    HIDF_conv.append(HIDF6)
    HIDF_conv.append(HIDF7)
    return HIDF_conv

def torch_conv(image,filter):#将filter转移到GPU上并做卷积运算
    kernel = filter.to(image.device)
    output_image = F.conv2d(image, kernel, stride=1, padding=1)
    return output_image

def gradxy1(image,alpha,beta):#HIDF分数阶梯度
    H =HIFD_conv(0.5)
    pic0 = torch_conv(image, H[0])
    pic1 = torch_conv(image, H[1])
    pic2 = torch_conv(image, H[2])
    pic3 = torch_conv(image, H[3])
    pic4 = torch_conv(image, H[4])
    pic5 = torch_conv(image, H[5])
    pic6 = torch_conv(image, H[6])
    pic7 = torch_conv(image, H[7])
    coefficients = torch.tensor([1, -1, np.sqrt(2)/2, -np.sqrt(2)/2, np.sqrt(2)/2, -np.sqrt(2)/2], dtype=torch.float32)
    imagesX = [pic2, pic3, pic4, pic5, pic6, pic7]
    grad_x = torch.zeros_like(imagesX[0], dtype=torch.float32)
    for i in range(len(imagesX)):
        grad_x += coefficients[i] * imagesX[i]
    grad_x[..., :2] = 0.0
    imagesY = [pic0, pic1, pic4, pic6, pic5, pic7]
    grad_y = torch.zeros_like(imagesY[0], dtype=torch.float32)
    for i in range(len(imagesY)):
        grad_y += coefficients[i] * imagesY[i]
    grad_y[..., -2:] = 0.0
    #print("grad_x shape:", grad_x.shape)
    #print("grad_y shape:", grad_y.shape)

    gradient_vectors = torch.zeros(grad_x.shape + (2,), dtype=torch.float32)
    if alpha == "+":
        gradient_vectors[..., 0] = grad_x
    elif alpha == "-":
        gradient_vectors[..., 0] = -grad_x
    if beta == "+":
        gradient_vectors[..., 1] = grad_y
    elif beta == "-":
        gradient_vectors[..., 1] = -grad_y
    return gradient_vectors

def gradxy(image,alpha,beta):#Sobel梯度
    # Define Sobel operator kernels for gradient calculation
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)

    # Apply Sobel operator to compute gradients in x and y directions
    grad_x = torch_conv(image, sobel_x)
    grad_y = torch_conv(image, sobel_y)

    gradient_vectors = torch.zeros(grad_x.shape + (2,), dtype=torch.float32)
    if alpha == "+":
        gradient_vectors[..., 0] = grad_x
    elif alpha == "-":
        gradient_vectors[..., 0] = -grad_x
    if beta == "+":
        gradient_vectors[..., 1] = grad_y
    elif beta == "-":
        gradient_vectors[..., 1] = -grad_y

    return gradient_vectors

def normalize_vector(vector):#正则化梯度向量
    # 计算向量长度
    length = torch.norm(vector, dim=-1, keepdim=True)
    # 避免除以零
    length[length == 0] = 1
    # 单位化向量
    normalized_vector = vector / length
    return normalized_vector

def perpendicular_vector(vector):# 将向量旋转90度（交换坐标并取负值）
    rotated_vector = torch.stack([-vector[..., 1], vector[..., 0]], dim=-1)
    # 单位化向量
    perpendicular_unit_vector = normalize_vector(rotated_vector)
    return perpendicular_unit_vector

def dot_product_gradients(gradient_vectors1, gradient_vectors2):#向量点乘
    # 确保梯度向量的形状相同
    assert gradient_vectors1.shape == gradient_vectors2.shape, "Gradient vector shapes do not match"

    dot_products = torch.sum(gradient_vectors1 * gradient_vectors2, dim=-1)
    abs = torch.abs(dot_products)
    sum = torch.sum(abs)
    return sum

def loss1(img2,pan):#保真度假设1
    # 定义每个通道的系数
    R = img2[:, 0, :, :]
    G = img2[:, 1, :, :]
    B = img2[:, 2, :, :]
    mean_rgb = (R * coefficients[0] + G * coefficients[1] + B * coefficients[2])
    loss1 = torch.sum((mean_rgb - pan) ** 2)
    return loss1

def loss2(img2, ms):#保真度假设2
    R = img2[:, 0, :, :]
    G = img2[:, 1, :, :]
    B = img2[:, 2, :, :]
    x1 = torch.sum((average_pooling_kernel(R.unsqueeze(1), 4)-ms[:, :, :, :, 0])**2)
    x2 = torch.sum((average_pooling_kernel(G.unsqueeze(1), 4)-ms[:, :, :, :, 1])**2)
    x3 = torch.sum((average_pooling_kernel(B.unsqueeze(1), 4)-ms[:, :, :, :, 2])**2)
    loss2 = x1 + x2 + x3
    return loss2

def loss3(img2,pan):#几何形状假设
    R = img2[:, 0:1, :, :]
    G = img2[:, 1:2, :, :]
    B = img2[:, 2:3, :, :]

    pan_vec1 = perpendicular_vector(gradxy(pan,"+","+"))
    R_vec1 = gradxy(R,"+","+")
    G_vec1 = gradxy(G,"+","+")
    B_vec1 = gradxy(B,"+","+")

    pan_vec2 = perpendicular_vector(gradxy(pan,"+","-"))
    R_vec2 = gradxy(R,"+","-")
    G_vec2 = gradxy(G,"+","-")
    B_vec2 = gradxy(B,"+","-")

    pan_vec3 = perpendicular_vector(gradxy(pan,"-","+"))
    R_vec3 = gradxy(R,"-","+")
    G_vec3 = gradxy(G,"-","+")
    B_vec3 = gradxy(B,"-","+")

    pan_vec4 = perpendicular_vector(gradxy(pan,"-","-"))
    R_vec4 = gradxy(R,"-","-")
    G_vec4 = gradxy(G,"-","-")
    B_vec4 = gradxy(B,"-","-")

    loss3 = (1/3)*(
    dot_product_gradients(pan_vec1, R_vec1) +
    dot_product_gradients(pan_vec2, R_vec2) +
    dot_product_gradients(pan_vec3, R_vec3) +
    dot_product_gradients(pan_vec4, R_vec4)
    )
    loss3 += (1/3)*(
    dot_product_gradients(pan_vec1, G_vec1) +
    dot_product_gradients(pan_vec2, G_vec2) +
    dot_product_gradients(pan_vec3, G_vec3) +
    dot_product_gradients(pan_vec4, G_vec4)
    )
    loss3 += (1/3)*(
    dot_product_gradients(pan_vec1, B_vec1) +
    dot_product_gradients(pan_vec2, B_vec2) +
    dot_product_gradients(pan_vec3, B_vec3) +
    dot_product_gradients(pan_vec4, B_vec4)
    )

    return loss3

def train(img2, pan, ms, max_iter):
    loss = loss1(img2, pan) + loss2(img2, ms) + loss3(img2, pan)
    img2 = Variable(img2, requires_grad=True)
    optimizer = optim.Adam([img2], lr=0.01)  # 使用SGD优化器
    epoch = 1
    prev_loss = loss.item() + 0.5
    while epoch <= max_iter:
        optimizer.zero_grad()
        loss = loss1(img2, pan) + loss2(img2, ms) + loss3(img2, pan)
        print("Epoch [{}/{}], loss: {:.4f}".format(epoch, max_iter, loss.item()))
        #loss_diff = (prev_loss - loss.item())
        prev_loss = loss.item()
        loss.backward()
        optimizer.step()
        epoch += 1
    return img2



if __name__ == "__main__":

    png_files = []
    png_files.extend(glob.glob(os.path.join("./data/RGB", '*.png')))
    for image in png_files:
        process_img(cv2.imread(image),4)
        print("Processing image: ", image)
        pan = cv2.imread("./data/pan.png", cv2.IMREAD_GRAYSCALE)
        ms = cv2.imread("./data/MS_ini.png",cv2.IMREAD_COLOR)
        img2 = cv2.imread("./data/MS_resize.png", cv2.IMREAD_COLOR)

        '''random_pixels = np.random.randint(0, 256, size=img1.shape, dtype=np.uint8)
        img2 = np.stack([random_pixels[:,:,0], random_pixels[:,:,1], random_pixels[:,:,2]], axis=-1)'''

        pan,ms,img2 = pic_to_tensor(pan,ms,img2)

        # 检查是否有 GPU 可用
        if torch.cuda.is_available():
            img2 = img2.cuda()
            pan = pan.cuda()
            ms = ms.cuda()

        coefficients = torch.tensor([1/3, 1/3, 1/3])
        img2 = train(img2, pan, ms, 100)
        result_img = img2.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        file_name = os.path.basename(image)
        new_file_name = file_name.split('.')[0] + " s=4 n.png"
        cv2.imwrite(new_file_name, result_img * 255)

