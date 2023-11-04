import cv2
import numpy as np

Q1_image_path = "Dataset_OpenCvDl_Hw1/Q1_image/"
Q3_image_path = "Dataset_OpenCvDl_Hw1/Q3_image/"


def Q1_1():
    # 讀取彩色圖片
    image = cv2.imread(Q1_image_path + "rgb.jpg")

    # 分離通道
    b, g, r = cv2.split(image)

    # 創建與原始圖像相同大小的全黑圖像
    black_image = np.zeros_like(b)

    # 創建三張單通道圖片
    blue_channel = cv2.merge([b, black_image, black_image])
    green_channel = cv2.merge([black_image, g, black_image])
    red_channel = cv2.merge([black_image, black_image, r])

    # 顯示分離的通道圖片（可選）
    cv2.imshow("Blue Channel", blue_channel)
    cv2.imshow("Green Channel", green_channel)
    cv2.imshow("Red Channel", red_channel)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q1_2():
    # 讀取彩色圖片
    image = cv2.imread(Q1_image_path + "rgb.jpg")

    # 將彩色圖片轉成灰階圖片
    I1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 分離通道
    b, g, r = cv2.split(image)

    # 計算I2 = (R + G + B) / 3
    I2 = (r + g + b) / 3

    # 將I2轉換為灰階圖片
    I2 = I2.astype(np.uint8)

    # 顯示灰階圖片
    cv2.imshow("I1", I1)
    cv2.imshow("I2", I2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q3_1():
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
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_x_image = cv2.filter2D(smoothed_image, -1, sobel_x)

    # 步驟4：顯示結果
    cv2.imshow("sobel x", sobel_x_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q3_2():
    # 讀取彩色圖片
    imageq32 = cv2.imread(Q3_image_path + "building.jpg")

    # 步驟1：將RGB圖像轉換為灰度圖像
    grayq32 = cv2.cvtColor(imageq32, cv2.COLOR_BGR2GRAY)

    # 調整平滑程度的核大小
    def apply_gaussian_blur(image, kernel_size):
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred

    # 步驟2：使用高斯平滑濾波器對灰度圖像進行平滑處理
    kernel_size = 5
    smoothed_image2 = apply_gaussian_blur(grayq32, kernel_size)

    # 步驟3：使用Sobel x運算子進行邊緣檢測
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    sobel_y_image = cv2.filter2D(smoothed_image2, -1, sobel_y)

    # 步驟4：顯示結果
    cv2.imshow("sobel x", sobel_y_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


Q3_1()
