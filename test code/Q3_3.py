import cv2
import numpy as np

Q3_image_path = "Dataset_OpenCvDl_Hw1/Q3_image/"

# 讀取彩色圖片
image = cv2.imread(Q3_image_path + "building.jpg")

# 將RGB圖像轉換為灰度圖像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 調整平滑程度的核大小
def apply_gaussian_blur(image, kernel_size):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


# 步驟2：使用高斯平滑濾波器對灰度圖像進行平滑處理
kernel_size = 5
smoothed_image = apply_gaussian_blur(gray, kernel_size)

# 讀取Sobel x和Sobel y圖像，這些圖像已經經過相同的前處理步驟
sobel_x_image = cv2.imread("sobel_x.jpg", cv2.IMREAD_GRAYSCALE)
sobel_y_image = cv2.imread("sobel_y.jpg", cv2.IMREAD_GRAYSCALE)

# 步驟3：使用Sobel運算子進行邊緣檢測
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_x_image = cv2.filter2D(smoothed_image, -1, sobel_x)

sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
sobel_y_image = cv2.filter2D(smoothed_image, -1, sobel_y)

# 步驟1：結合Sobel x和Sobel y圖像
combined_image = np.sqrt(sobel_x_image**2 + sobel_y_image**2)

# 步驟2：正規化結合的圖像
combined_image = (
    (combined_image - np.min(combined_image))
    / (np.max(combined_image) - np.min(combined_image))
    * 255
)
combined_image = combined_image.astype(np.uint8)

# 步驟3：應用閾值處理
threshold_value = 128
thresholded_image = np.where(combined_image >= threshold_value, 255, 0)

# 顯示結果
# cv2.imshow("Combined Image", combined_image)
cv2.imshow("Thresholded Image", thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
