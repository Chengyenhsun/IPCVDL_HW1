import cv2
import numpy as np

Q3_image_path = "Dataset_OpenCvDl_Hw1/Q3_image/"

# 讀取彩色圖片
image = cv2.imread(Q3_image_path + "building.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Smooth the grayscale image with Gaussian smoothing
smoothed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Define the Sobel x operator
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# Initialize an empty output image
sobel_x_image = np.zeros_like(smoothed_image, dtype=np.float32)

# Apply the Sobel x operator to the smoothed image
for y in range(1, smoothed_image.shape[0] - 1):
    for x in range(1, smoothed_image.shape[1] - 1):
        sobel_x_value = np.sum(smoothed_image[y - 1 : y + 2, x - 1 : x + 2] * sobel_x)
        sobel_x_image[y, x] = sobel_x_value

# Clip pixel values to the range [0, 255]
sobel_x_image = np.clip(sobel_x_image, 0, 255).astype(np.uint8)

# Show the Sobel x image
cv2.imshow("Sobel X Image", sobel_x_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
