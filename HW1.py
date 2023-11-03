import cv2
import numpy as np

Q1_image_path = "Dataset_OpenCvDl_Hw1/Q1_image/"


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


Q1_1()
