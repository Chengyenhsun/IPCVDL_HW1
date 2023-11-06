import cv2
import numpy as np
import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from HW1UI_ui import Ui_MainWindow


def load_image():
    global filePath
    filePath = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", ".")[0]
    print(filePath)


def load_image5():
    global filePath2
    filePath2 = QtWidgets.QFileDialog.getOpenFileNames(None, "Open File", ".")[0]
    print(filePath2)


def Q1_1():
    # 讀取彩色圖片
    image = cv2.imread(filePath)

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
    image = cv2.imread(filePath)

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


def Q1_3():
    # 讀取彩色圖片
    image = cv2.imread(filePath)

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


def Q2_1():
    # 讀取彩色圖片
    image = cv2.imread(filePath)

    # 創建一個函數，用於更新圖像
    def update_Q21(value):
        # 設定初始的半徑大小
        radius = 0
        radius = value + 1
        # 獲取trackbar的當前值
        radius = cv2.getTrackbarPos("Radius", "Gaussian Blur")

        # 計算kernel的大小
        kernel_size = (2 * radius + 1, 2 * radius + 1)

        # 運用高斯濾波
        blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

        # 顯示結果
        cv2.imshow("Gaussian Blur", blurred_image)

    cv2.namedWindow("Gaussian Blur")
    cv2.createTrackbar("Radius", "Gaussian Blur", 0, 5, update_Q21)
    cv2.imshow("Gaussian Blur", image)

    # 等待用戶按下任意按鍵
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q2_2():
    # 讀取彩色圖片
    image = cv2.imread(filePath)

    # 設定sigmaColor和sigmaSpace的值
    sigmaColor = 90
    sigmaSpace = 90

    # 創建一個回呼函數，當軌跡條值改變時調用
    def update_Q22(value):
        # 初始化窗口半徑
        radius = 0
        radius = value + 1
        # 使用Bilateral Filter處理圖片
        filtered_image = cv2.bilateralFilter(
            image, (2 * radius + 1), sigmaColor, sigmaSpace
        )
        # 顯示處理後的圖片
        cv2.imshow("Bilateral Filter", filtered_image)

    # 創建一個空視窗
    cv2.namedWindow("Bilateral Filter")
    # 創建一個軌跡條，用於調整半徑大小
    cv2.createTrackbar("Radius", "Bilateral Filter", 0, 5, update_Q22)
    cv2.imshow("Bilateral Filter", image)

    # 等待用戶按下任意按鍵
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q2_3():
    # 讀取彩色圖片
    image = cv2.imread(filePath)

    # 回呼函數，當軌跡條值改變時調用
    def update_Q23(value):
        # 初始化窗口半徑
        radius = 0
        radius = value + 1
        # 計算kernel大小
        kernel_size = 2 * radius + 1
        # 使用Median Filter處理圖片
        filtered_image = cv2.medianBlur(image, kernel_size)
        # 顯示處理後的圖片
        cv2.imshow("Median Filter", filtered_image)

    # 創建一個空視窗
    cv2.namedWindow("Median Filter")
    # 創建軌跡條，用於調整半徑大小
    cv2.createTrackbar("Radius", "Median Filter", 0, 5, update_Q23)
    cv2.imshow("Median Filter", image)

    # 等待用戶按下任意按鍵
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q3_1():
    # 讀取彩色圖片
    image = cv2.imread(filePath)

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
    cv2.imshow("Sobel X", sobel_x_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q3_2():
    # 讀取彩色圖片
    imageq32 = cv2.imread(filePath)

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
    cv2.imshow("Sobel Y", sobel_y_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q4():
    # 讀取彩色圖片
    image = cv2.imread(filePath)

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
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (image.shape[1], image.shape[0])
    )

    # 執行平移操作
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(
        rotated_image, translation_matrix, (image.shape[1], image.shape[0])
    )

    # 顯示結果圖像
    cv2.imshow("Transformed Image", translated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)

ui.LoadImage1_Button.clicked.connect(load_image)
ui.Q1_1_Button.clicked.connect(Q1_1)
ui.Q1_2_Button.clicked.connect(Q1_2)
ui.Q1_3_Button.clicked.connect(Q1_3)
ui.Q2_1_Button.clicked.connect(Q2_1)
ui.Q2_2_Button.clicked.connect(Q2_2)
ui.Q2_3_Button.clicked.connect(Q2_3)
ui.Q3_1_Button.clicked.connect(Q3_1)
ui.Q3_2_Button.clicked.connect(Q3_2)
# ui.Q3_3_Button.clicked.connect()
# ui.Q3_4_Button.clicked.connect()
# ui.Q4_Button.clicked.connect()
ui.Q5_Load_Button.clicked.connect(load_image5)
# ui.Q5_1_Button.clicked.connect()
# ui.Q5_2_Button.clicked.connect()
# ui.Q5_3_Button.clicked.connect()
# ui.Q5_4_Button.clicked.connect()

MainWindow.show()
app.exec_()
