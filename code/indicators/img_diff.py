import cv2

# 读入两张彩色图片
image1 = cv2.imread('./data/RGB/field.png')
image2 = cv2.imread('./RGB/field s=4.png')

# 将图片转换为RGB通道
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# 分离RGB通道
r1, g1, b1 = cv2.split(image1_rgb)
r2, g2, b2 = cv2.split(image2_rgb)

# 计算通道差异
diff_r = cv2.absdiff(r1, r2)
diff_g = cv2.absdiff(g1, g2)
diff_b = cv2.absdiff(b1, b2)

# 合并差异图像
diff_image = cv2.merge((diff_r, diff_g, diff_b))

# 反色
diff_image = 255 - diff_image

# 显示或保存结果图像
cv2.imshow('Inverted Difference Image', diff_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果需要保存差异图像
cv2.imwrite('difference_image1.jpg', diff_image)
