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

img_magnitude = np.sqrt(np.square(sobel_x_image) + np.square(sobel_y_image))
img_magnitude *= 255.0 / np.max(img_magnitude)  # Normalize the result to 0~255.

# gx = np.zeros(sobel_x_image.shape, dtype=np.uint8)
# gy = np.zeros(sobel_y_image.shape, dtype=np.uint8)
# gxy = np.zeros(sobel_x_image.shape, dtype=np.uint8)

# for h in range(1, sobel_x_image.shape[0] - 1):
#     for w in range(1, sobel_x_image.shape[1] - 1):
#         sx = sobel_x_image[h, w]
#         sy = sobel_y_image[h, w]

#         sxy = int(np.round(np.sqrt(sx**2 + sy**2)))

#         gx[h, w] = np.clip(sx, 0, 255)
#         gy[h, w] = np.clip(sy, 0, 255)
#         gxy[h, w] = np.clip(sxy, 0, 255)

# threshold = 128
# ret, thresholded_image = cv2.threshold(gxy, threshold, 255, cv2.THRESH_BINARY)

# Show the thresholded image
# cv2.imshow("Thresholded Image", thresholded_image)
cv2.imshow("Gradient Magnitude", img_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()
