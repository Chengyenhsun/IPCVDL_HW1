import os
from PIL import Image
import torchvision.transforms as transforms

# 定義圖像路徑
image_folder = "Dataset_OpenCvDl_Hw1/Q5_image/Q5_1/"

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

# 顯示增強後的圖像
for i, augmented_image in enumerate(augmented_images):
    augmented_image.show()
