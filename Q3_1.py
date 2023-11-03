import cv2
import numpy as np


Q3_image_path = "Dataset_OpenCvDl_Hw1/Q3_image/"

# 讀取彩色圖片
image = cv2.imread(Q3_image_path + "building.jpg")

# 步驟1：將RGB圖像轉換為灰度圖像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 調整平滑程度的核大小
def apply_gaussian_blur(image, kernel_size):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


# 步驟2：使用高斯平滑濾波器對灰度圖像進行平滑處理
kernel_size = 5
smoothed_image = apply_gaussian_blur(gray, kernel_size)


# 步驟3：使用Sobel x運算子進行邊緣檢測
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_x_image = cv2.filter2D(smoothed_image, -1, sobel_x)


# 步驟4：顯示結果
cv2.imshow("sobel x", sobel_x_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
