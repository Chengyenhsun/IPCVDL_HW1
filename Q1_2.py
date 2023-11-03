import cv2
import numpy as np

Q1_image_path = "Dataset_OpenCvDl_Hw1/Q1_image/"

# 讀取彩色圖片
image = cv2.imread(Q1_image_path + "rgb.jpg")


# 將彩色圖片轉成灰階圖片
I1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 分離通道
b, g, r = cv2.split(image)


# 創建與原始圖像相同大小的全黑圖像
black_image = np.zeros_like(b)


# 分離通道
b, g, r = cv2.split(image)


# 創建三張單通道圖片
blue_channel = cv2.merge([b, black_image, black_image])
green_channel = cv2.merge([black_image, g, black_image])
red_channel = cv2.merge([black_image, black_image, r])


# 計算I2 = (R + G + B) / 3
I2 = (r + g + b) / 3

# 將I2轉換為灰階圖片
I2 = I2.astype(np.uint8)

# 顯示灰階圖片
cv2.imshow("I1", I1)
cv2.imshow("I2", I2)
cv2.waitKey(0)
cv2.destroyAllWindows()
