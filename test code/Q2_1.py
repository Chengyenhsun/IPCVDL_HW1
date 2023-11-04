import cv2
import numpy as np


Q2_image_path = "Dataset_OpenCvDl_Hw1/Q2_image/"

# 讀取彩色圖片
image = cv2.imread(Q2_image_path + "image1.jpg")


# 创建一个函数，用于更新图像
def update_image(*args):
    # 获取trackbar的当前值
    radius = cv2.getTrackbarPos("Radius", "Gaussian Blur")

    # 计算核的大小
    kernel_size = (2 * radius + 1, 2 * radius + 1)

    # 应用高斯滤波
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

    # 在弹出窗口中显示结果
    cv2.imshow("Gaussian Blur", blurred_image)


# 创建弹出窗口
cv2.namedWindow("Gaussian Blur")

# 创建trackbar
cv2.createTrackbar("Radius", "Gaussian Blur", 1, 5, update_image)

# 初始显示原始图像
cv2.imshow("Gaussian Blur", image)

while True:
    key = cv2.waitKey(1)

    # 按ESC键退出
    if key == 27:
        break

cv2.destroyAllWindows()
