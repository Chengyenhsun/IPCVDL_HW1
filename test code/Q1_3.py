import cv2
import numpy as np


Q1_image_path = "Dataset_OpenCvDl_Hw1/Q1_image/"

# 讀取彩色圖片
image = cv2.imread(Q1_image_path + "rgb.jpg")

# 步驟1：將RGB圖像轉換為灰度圖像
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 步骤2: 提取黄色和绿色的遮罩，生成I1
lower_bound = np.array([25, 25, 25])  # 下限值 (Hue, Saturation, Value)
upper_bound = np.array([35, 255, 255])  # 上限值 (Hue, Saturation, Value)
mask_i1 = cv2.inRange(hsv_img, lower_bound, upper_bound)

# 步骤3: 将黄色和绿色遮罩转换为BGR格式
mask_i1_bgr = cv2.cvtColor(mask_i1, cv2.COLOR_GRAY2BGR)

# 步骤4: 从图像中移除黄色和绿色，生成I2
i2 = cv2.bitwise_not(mask_i1_bgr, image, mask=mask_i1_bgr)

# 显示I1和I2图像
cv2.imshow("I1 - 黄色和绿色掩码", mask_i1)
cv2.imshow("I2 - 移除黄色和绿色后的图像", i2)
cv2.waitKey(0)
cv2.destroyAllWindows()
