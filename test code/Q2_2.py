import cv2
import numpy as np

Q2_image_path = "Dataset_OpenCvDl_Hw1/Q2_image/"

# 讀取彩色圖片
image = cv2.imread(Q2_image_path + "image1.jpg")

# 初始化窗口半徑
radius = 0

# 設定sigmaColor和sigmaSpace的值
sigmaColor = 90
sigmaSpace = 90

# 創建一個空視窗
cv2.namedWindow("Bilateral Filter")


# 創建一個回呼函數，當軌跡條值改變時調用
def update_radius(value):
    global radius
    radius = value + 1
    # 使用Bilateral Filter處理圖片
    filtered_image = cv2.bilateralFilter(
        image, (2 * radius + 1), sigmaColor, sigmaSpace
    )
    # 顯示處理後的圖片
    cv2.imshow("Bilateral Filter", filtered_image)


# 創建一個軌跡條，用於調整半徑大小
cv2.createTrackbar("Radius", "Bilateral Filter", 0, 5, update_radius)

# 等待用戶按下任意按鍵
cv2.waitKey(0)


cv2.destroyAllWindows()
