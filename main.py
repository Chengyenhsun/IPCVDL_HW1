import cv2
import numpy as np
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap
from HW1UI_ui import Ui_MainWindow
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torchsummary import summary
from torchvision.transforms import Compose, ToTensor, Normalize


def load_image():
    global filePath
    try:
        abs_path = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", ".")[0]
        # 获取当前工作目录
        current_dir = os.getcwd()

        # 计算相对路径
        filePath = os.path.relpath(abs_path, current_dir)
    except:
        pass
    # print(abs_path)
    # print(filePath)


def load_image5():
    global filePath
    try:
        abs_path = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", ".")[0]
        # 获取当前工作目录
        current_dir = os.getcwd()

        # 计算相对路径
        filePath = os.path.relpath(abs_path, current_dir)
        # print(filePath)

        image = QPixmap(filePath).scaled(148, 148)
        scene = QtWidgets.QGraphicsScene()  # 加入圖片
        scene.addPixmap(image)  # 將圖片加入 scene
        ui.InferenceImage_View.setScene(scene)
    except:
        pass


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


def Q3_3():
    # 讀取彩色圖片
    image = cv2.imread(filePath)

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
            sobel_x_value = np.sum(
                smoothed_image[y - 1 : y + 2, x - 1 : x + 2] * sobel_x
            )
            sobel_y_value = np.sum(
                smoothed_image[y - 1 : y + 2, x - 1 : x + 2] * sobel_y
            )
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

    threshold = 128
    ret, thresholded_image = cv2.threshold(gxy, threshold, 255, cv2.THRESH_BINARY)

    # Show the gradient magnitude
    cv2.imshow("Gradient Magnitude", gxy)
    # Show the thresholded image
    cv2.imshow("Thresholded Image", thresholded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q4():
    # 讀取彩色圖片
    image = cv2.imread(filePath)

    # 旋轉角度、縮放比例和平移距離
    try:
        angle = int(ui.Rotation_Input.text())
        scale = float(ui.Scaling_Input.text())
        tx = int(ui.Tx_Input.text())
        ty = int(ui.Ty_Input.text())
    except:
        angle = 0
        scale = 1
        tx = 0
        ty = 0

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


def Q5_1():
    # images = [Image.open(image_file) for image_file in filePath2]
    image_folder = "Dataset_OpenCvDl_Hw1/Q5_image/Q5_1"
    # 載入圖像
    image_files = [
        os.path.join(image_folder, filename) for filename in os.listdir(image_folder)
    ]
    images = [Image.open(image_file) for image_file in image_files]

    # 資料增強
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
            transforms.RandomVerticalFlip(),  # 隨機垂直翻轉
            transforms.RandomRotation(30),  # 隨機旋轉 (-30 到 30 度之間)
        ]
    )

    augmented_images = [transform(image) for image in images]

    # 提取檔名（不包含格式）作為標籤
    labels = [os.path.splitext(os.path.basename(file))[0] for file in image_files]

    # 顯示增強後的圖像和標籤在一個新視窗中
    _, axes = plt.subplots(3, 3, figsize=(6, 6))

    for i, (original, augmented, label) in enumerate(
        zip(images, augmented_images, labels)
    ):
        ax = axes[i // 3, i % 3]
        ax.set_title(label)
        ax.imshow(original if i % 3 == 0 else augmented)
        ax.axis("off")

    plt.show()


def Q5_2():
    # 建立一個帶有批量歸一化的 VGG19 模型
    vgg19_bn = models.vgg19_bn(num_classes=10)

    # 使用 torchsummary.summary 在終端中顯示模型結構
    summary(vgg19_bn, (3, 224, 224))  # 輸入圖像維度 (3, 224, 224)


def Q5_3():
    image = cv2.imread("learning_history.jpg")
    # 顯示結果圖像
    cv2.imshow("Training/Validating Accuracy and Loss ", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q5_4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = [x / 255 for x in [125.3, 23.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # 創建一個一般的 VGG19 模型
    vgg19 = models.vgg19_bn(num_classes=10)

    # 載入你的訓練好的權重
    vgg19.load_state_dict(torch.load("vgg19_final.pt", map_location=device))  # 請確保路徑正確
    vgg19.to(device)

    # 設置模型為評估模式
    vgg19.eval()

    # 載入圖片並進行預處理
    transform = Compose([ToTensor(), Normalize(mean, std)])

    image = Image.open(filePath)
    image = transform(image).unsqueeze(0).to(device)  # 添加一個批次維度並移到GPU（如果可用）

    # 使用模型進行推論
    with torch.no_grad():
        outputs = vgg19(image)

    # 取得類別機率
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # 取得預測的類別索引
    predicted_class = torch.argmax(probabilities).item()

    # 載入CIFAR-10類別名稱對照表
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # 輸出結果
    # print(
    #     "Predicted class: {} ({})".format(class_names[predicted_class], predicted_class)
    # )
    # print("Class probabilities:")
    # for i, prob in enumerate(probabilities):
    #     print("{}: {:.2f}%".format(class_names[i], prob * 100))

    ui.predict_label.setText("Predicted = " + class_names[predicted_class])

    probs = [prob.item() for prob in probabilities]

    # 創建一個長條圖
    plt.figure(figsize=(6, 6))
    plt.bar(class_names, probs, alpha=0.7)

    # 設置圖表標題和軸標籤
    plt.title("Probability of each class")
    plt.xlabel("Class Name")
    plt.ylabel("Probability")

    # 顯示機率值在長條上
    for i, prob in enumerate(probs):
        plt.text(i, prob, f"{prob:.2f}", ha="center", va="bottom")

    # 顯示長條圖
    plt.xticks(rotation=45)  # 使x軸標籤更易讀
    plt.tight_layout()
    plt.show()


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
ui.Q3_3_Button.clicked.connect(Q3_3)
# ui.Q3_4_Button.clicked.connect()
ui.Q4_Button.clicked.connect(Q4)
ui.Q5_Load_Button.clicked.connect(load_image5)
ui.Q5_1_Button.clicked.connect(Q5_1)
ui.Q5_2_Button.clicked.connect(Q5_2)
ui.Q5_3_Button.clicked.connect(Q5_3)
ui.Q5_4_Button.clicked.connect(Q5_4)

MainWindow.show()
app.exec_()
