import cv2

Q2_image_path = "Dataset_OpenCvDl_Hw1/Q2_image/"

# 讀取彩色圖片
image = cv2.imread(Q2_image_path + "image2.jpg")

# 初始化窗口半徑
radius = 0

# 創建一個空視窗
cv2.namedWindow("Median Filter")


# 回呼函數，當軌跡條值改變時調用
def update_radius(value):
    global radius
    radius = value + 1
    # 計算kernel大小
    kernel_size = 2 * radius + 1
    # 使用Median Filter處理圖片
    filtered_image = cv2.medianBlur(image, kernel_size)
    # 顯示處理後的圖片
    cv2.imshow("Median Filter", filtered_image)


# 創建軌跡條，用於調整半徑大小
cv2.createTrackbar("半徑", "Median Filter", 0, 5, update_radius)

# 顯示初始處理後的圖片
filtered_image = cv2.medianBlur(image, 2 * radius + 1)
cv2.imshow("Median Filter", filtered_image)

# 等待用戶按下任意按鍵
cv2.waitKey(0)

# 釋放資源
cv2.destroyAllWindows()
