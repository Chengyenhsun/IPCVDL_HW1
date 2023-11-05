import cv2
import numpy as np


Q1_image_path = "Dataset_OpenCvDl_Hw1/Q1_image/"

# 讀取彩色圖片
image = cv2.imread(Q1_image_path + "rgb.jpg")

# 步驟1: 轉換圖像為HSV格式
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 步驟2: 提取黃色和綠色的遮罩，生成I1
lower_yellow = np.array([12, 43, 43])  # HSV中黄色的下限值
upper_yellow = np.array([35, 255, 255])  # HSV中黄色的上限值
lower_green = np.array([35, 43, 46])  # HSV中绿色的下限值
upper_green = np.array([77, 255, 255])  # HSV中绿色的上限值
mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
mask_green = cv2.inRange(hsv_img, lower_green, upper_green)
mask_i1 = mask_yellow + mask_green

# 步驟3: 將黃色和綠色的遮罩轉成BGR格式
mask_i1_bgr = cv2.cvtColor(mask_i1, cv2.COLOR_GRAY2BGR)
mask = cv2.bitwise_not(mask_i1_bgr)

# 步驟4: 從圖像中移除黃色和綠色，生成I2
i2 = cv2.bitwise_and(mask, image)

# 顯示 I1 及 I2
cv2.imshow("I1", mask_i1)
cv2.imshow("I2", i2)
cv2.waitKey(0)
cv2.destroyAllWindows()
