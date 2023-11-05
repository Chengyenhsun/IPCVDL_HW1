import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

Q5_image_path = "Dataset_OpenCvDl_Hw1/Q5_image/"

# 定義圖像路徑
image_folder = Q5_image_path + "Q5_1"

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
fig, axes = plt.subplots(3, 3, figsize=(12, 4))

for i, (original, augmented, label) in enumerate(zip(images, augmented_images, labels)):
    ax = axes[i // 3, i % 3]
    ax.set_title(label)
    ax.imshow(original if i % 3 == 0 else augmented)
    ax.axis("off")

plt.show()
