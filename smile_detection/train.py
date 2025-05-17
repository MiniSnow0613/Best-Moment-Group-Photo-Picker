import os
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Dataset
class SmileDataset(Dataset):
    def __init__(self, image_paths, labels, train=True):
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = PIL.Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

# 載入圖片路徑與標籤
def load_smile_dataset(root):
    image_paths = []
    labels = []
    class_map = {'0': 0, '1': 1, '2': 2}

    for class_name, label in class_map.items():
        class_dir = os.path.join(root, class_name)
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('jpg', 'png', 'jpeg')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(label)
    return image_paths, labels

# 模型定義
class SmileClassifier(nn.Module):
    def __init__(self):
        super(SmileClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 訓練流程
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(__file__)
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    image_paths, labels = load_smile_dataset(DATASET_DIR)

    if len(image_paths) == 0:
        print("資料集為空")
        exit(1)

    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42)

    train_dataset = SmileDataset(train_imgs, train_labels, train=True)
    val_dataset = SmileDataset(val_imgs, val_labels, train=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmileClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc=f"Train {epoch+1:02}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        scheduler.step()
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        acc_train = correct_train / total_train
        acc_val = correct_val / total_val
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        print(f"[Epoch {epoch+1}] "
              f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} "
              f"| Train Acc: {acc_train:.4f} | Val Acc: {acc_val:.4f}")

        # 早停
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(BASE_DIR, 'smile_model.pth'))  # 儲存最佳模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("提早結束！")
                break
 
    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'model.pth'))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(BASE_DIR, "loss.png"))
    print("訓練完成，最佳模型(smile_model.pth)與loss圖(loss.png)已儲存")
