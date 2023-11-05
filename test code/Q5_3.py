import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 步驟 1：載入 CIFAR-10 數據集
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
        transforms.RandomCrop(32, padding=4),  # 隨機裁剪
        transforms.ToTensor(),  # 轉換為張量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)  # 正規化

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2
)


# 步驟 2：定義並訓練 VGG19 模型與批標準化
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init()
        # 定義你的 VGG19 模型，包括批標準化層
        super(VGG19, self).__init()
        # 定義 VGG19 模型的卷積層部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10),  # 假設有 10 個類別
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 正向傳遞實現
model = VGG19()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train(model, trainloader, criterion, optimizer, epochs):
    train_loss_values = []
    train_acc_values = []
    test_loss_values = []
    test_acc_values = []
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        train_loss = running_loss / len(trainloader)

        model.eval()
        running_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test
        test_loss = running_loss / len(testloader)

        print(
            f"第 {epoch+1}/{epochs} 輪 - 訓練損失：{train_loss:.4f} - 訓練準確度：{train_accuracy:.2f}% - 測試損失：{test_loss:.4f} - 測試準確度：{test_accuracy:.2f}%"
        )

        train_loss_values.append(train_loss)
        train_acc_values.append(train_accuracy)
        test_loss_values.append(test_loss)
        test_acc_values.append(test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), "best_model.pth")

    return train_loss_values, train_acc_values, test_loss_values, test_acc_values


# 步驟 3：資料擴增技術已經在轉換中應用

# 步驟 4：保存具有最高測試準確度的權重文件已在訓練迴圈中實現


# 步驟 5：為訓練和驗證損失和準確度值創建折線圖
def plot_metrics(train_loss, train_acc, test_loss, test_acc):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, "b", label="訓練損失")
    plt.plot(epochs, test_loss, "g", label="驗證損失")
    plt.title("訓練和驗證損失")
    plt.xlabel("輪數")
    plt.ylabel("損失")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, "b", label="訓練準確度")
    plt.plot(epochs, test_acc, "g", label="驗證準確度")
    plt.title("訓練和驗證準確度")
    plt.xlabel("輪數")
    plt.ylabel("準確度")
    plt.legend()

    plt.tight_layout()
    plt.show()


# 步驟 6：保存圖形
train_loss_values, train_acc_values, test_loss_values, test_acc_values = train(
    model, trainloader, criterion, optimizer, epochs=40
)
plot_metrics(train_loss_values, train_acc_values, test_loss_values, test_acc_values)
