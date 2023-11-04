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
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

rows, cols = smoothed_image.shape
sobel_y_image = np.zeros_like(smoothed_image, dtype=np.float32)

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        sobel_y_image[i, j] = np.sum(
            smoothed_image[i - 1 : i + 2, j - 1 : j + 2] * sobel_y
        )

sobel_y_image = np.uint8(np.abs(sobel_y_image))
cv2.imwrite(Q3_image_path + "sobel_x_image.jpg", sobel_y_image)

# 步驟4：顯示結果
cv2.imshow("sobel y", sobel_y_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
