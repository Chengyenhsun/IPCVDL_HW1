import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖片
Q3_image_path = "Dataset_OpenCvDl_Hw1/Q3_image/"

# 讀取彩色圖片
image = cv2.imread(Q3_image_path + "building.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Smooth the grayscale image with Gaussian smoothing
smoothed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Define the Sobel x operator
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# Define the Sobel y operator
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Initialize empty output images
sobel_x_image = np.zeros_like(smoothed_image, dtype=np.float32)
sobel_y_image = np.zeros_like(smoothed_image, dtype=np.float32)

# Apply the Sobel x and Sobel y operators to the smoothed image
for y in range(1, smoothed_image.shape[0] - 1):
    for x in range(1, smoothed_image.shape[1] - 1):
        sobel_x_value = np.sum(smoothed_image[y - 1 : y + 2, x - 1 : x + 2] * sobel_x)
        sobel_y_value = np.sum(smoothed_image[y - 1 : y + 2, x - 1 : x + 2] * sobel_y)
        sobel_x_image[y, x] = sobel_x_value
        sobel_y_image[y, x] = sobel_y_value

# Clip pixel values to the range [0, 255]
sobel_x_image = np.clip(sobel_x_image, 0, 255).astype(np.uint8)
sobel_y_image = np.clip(sobel_y_image, 0, 255).astype(np.uint8)

gx = np.zeros(sobel_x_image.shape, dtype=np.uint8)
gy = np.zeros(sobel_y_image.shape, dtype=np.uint8)
gxy = np.zeros(sobel_x_image.shape, dtype=np.uint8)

for h in range(1, sobel_x_image.shape[0] - 1):
    for w in range(1, sobel_x_image.shape[1] - 1):
        sx = sobel_x_image[h, w]
        sy = sobel_y_image[h, w]

        sxy = int(np.round(np.sqrt(sx**2 + sy**2)))

        gx[h, w] = np.clip(sx, 0, 255)
        gy[h, w] = np.clip(sy, 0, 255)
        gxy[h, w] = np.clip(sxy, 0, 255)


# 計算梯度角度
gradient_angle = np.arctan2(sobel_y_image, sobel_x_image) * 180 / np.pi
# Convert gradient_angle to 8-bit integer
gradient_angle = ((gradient_angle + 180) * 255 / 360).astype(np.uint8)
print("Gradient Angle:\n", gradient_angle)

# # 歸一化梯度角度到0~360度範圍
# lower_angle = 120
# upper_angle = 180

# # 生成指定角度範圍的掩碼
# angle_range_mask = np.where(
#     (gradient_angle >= lower_angle) & (gradient_angle <= upper_angle), 255, 0
# ).astype(np.uint8)

# Show the mask for the specified angle range
# cv2.imshow("Mask (120-180 degrees)", angle_range_mask)

# 生成兩個不同範圍的角度掩碼
# mask1 = ((gradient_angle >= 120) & (gradient_angle <= 180)).astype(np.uint8) * 255
# mask2 = ((gradient_angle >= 210) & (gradient_angle <= 330)).astype(np.uint8) * 255
mask1 = cv2.inRange(gradient_angle, 120, 180)
mask2 = cv2.inRange(gradient_angle, 210, 330)


result1 = cv2.bitwise_and(gxy, gxy, mask=mask1)
result2 = cv2.bitwise_and(gxy, gxy, mask=mask2)
# fig = plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1), plt.title("120~180")
# plt.imshow(result1, cmap=plt.get_cmap("gray")), plt.axis("off")
# plt.subplot(1, 2, 2), plt.title("210~330")
# plt.imshow(result2, cmap=plt.get_cmap("gray")), plt.axis("off")


cv2.imshow("result1", result1)
cv2.imshow("result2", result2)
cv2.waitKey(0)
cv2.destroyAllWindows()
