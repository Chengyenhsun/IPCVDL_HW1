import torch
import torchvision.models as models
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 創建一個一般的 VGG19 模型
vgg19 = models.vgg19_bn(num_classes=10)
# 載入你的訓練好的權重
vgg19.load_state_dict(torch.load("vgg19_final.pt", map_location=device))  # 請確保路徑正確
vgg19.to(device)

# 設置模型為評估模式
vgg19.eval()

# 載入圖片並進行預處理
transform = Compose(
    [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

image = Image.open("airplane.png")  # 替換為你的圖片路徑
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
