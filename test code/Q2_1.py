import cv2
import numpy as np


Q2_image_path = "Dataset_OpenCvDl_Hw1/Q2_image/"

# 讀取彩色圖片
image = cv2.imread(Q2_image_path + "image1.jpg")

# 設定初始的半徑大小
radius = 0


# 創建一個函數，用於更新圖像
def update_image(value):
    global radius
    radius = value + 1
    # 獲取trackbar的當前值
    radius = cv2.getTrackbarPos("Radius", "Gaussian Blur")

    # 計算kernel的大小
    kernel_size = (2 * radius + 1, 2 * radius + 1)

    # 運用高斯濾波
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

    # 顯示結果
    cv2.imshow("Gaussian Blur", blurred_image)


cv2.namedWindow("Gaussian Blur")

cv2.createTrackbar("Radius", "Gaussian Blur", 0, 5, update_image)

cv2.imshow("Gaussian Blur", image)

# 等待用戶按下任意按鍵
cv2.waitKey(0)

# 釋放資源
cv2.destroyAllWindows()
