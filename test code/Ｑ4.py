import cv2
import numpy as np

Q4_image_path = "Dataset_OpenCvDl_Hw1/Q4_image/"
# 讀取彩色圖片
image = cv2.imread(Q4_image_path + "burger.png")

# 旋轉角度、縮放比例和平移距離
angle = 30
scale = 0.9
tx = 535
ty = 335

# 圖像中心
center_x = 240
center_y = 200

# 構建旋轉矩陣
rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)

# 執行選轉操作
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

# 執行平移操作
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated_image = cv2.warpAffine(
    rotated_image, translation_matrix, (image.shape[1], image.shape[0])
)

# 顯示結果圖像
cv2.imshow("Transformed Image", translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
