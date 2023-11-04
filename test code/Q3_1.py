import cv2
import numpy as np

Q3_image_path = "Dataset_OpenCvDl_Hw1/Q3_image/"

# 讀取彩色圖片
image = cv2.imread(Q3_image_path + "building.jpg")

# 步驟1：將RGB圖像轉換為灰度圖像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 步骤2：使用高斯平滑滤波器对灰度图像进行平滑处理
kernel_size = 5
smoothed_image = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# Sobel x運算子進行邊緣檢測
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

rows, cols = smoothed_image.shape
sobel_x_image = np.zeros_like(smoothed_image, dtype=np.float32)

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        sobel_x_image[i, j] = np.sum(
            smoothed_image[i - 1 : i + 2, j - 1 : j + 2] * sobel_x
        )

# 步驟4：顯示結果
sobel_x_image = np.uint8(np.abs(sobel_x_image))
cv2.imshow("Sobel x", sobel_x_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
