import cv2

# 读取灰度图像
gray_image = cv2.imread('./data/ir.bmp', cv2.IMREAD_GRAYSCALE)

# 复制灰度图像的通道三次
bgr_image = cv2.merge([gray_image, gray_image, gray_image])

# 显示彩色图像
cv2.imshow('Color Image', bgr_image)
cv2.waitKey(0)
cv2.imwrite('ir.bmp', bgr_image)
