import torch
import torchvision.models as models
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt

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

image = Image.open("automobile.png")  # 替換為你的圖片路徑
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
print("Predicted class: {} ({})".format(class_names[predicted_class], predicted_class))
print("Class probabilities:")
for i, prob in enumerate(probabilities):
    print("{}: {:.2f}%".format(class_names[i], prob * 100))


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
