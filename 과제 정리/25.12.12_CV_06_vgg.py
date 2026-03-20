import os
import random
import warnings
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models  # models 추가

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 기본 설정
data_root = './dataset'
train_dir = os.path.join(data_root, 'seg_train')
val_dir   = os.path.join(data_root, 'seg_test')
pred_dir  = os.path.join(data_root, 'seg_pred')

# 기존 단순 CNN용
model_save_path = 'cnn_model.pth'
history_save_path = 'train_history.pkl'

# VGG16용 추가 경로
vgg_input_size = 224
vgg_feature_model_path = 'vgg16_feature.pth'
vgg_feature_history_path = 'vgg16_feature_history.pkl'
vgg_finetune_model_path = 'vgg16_finetune.pth'
vgg_finetune_history_path = 'vgg16_finetune_history.pkl'

batch_size = 32
img_height = 180
img_width  = 180
num_epochs = 2
learning_rate = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")


# 시드 고정
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 클래스 이름 자동 가져오기
def get_class_names():
    dirs = [d.name for d in os.scandir(train_dir) if d.is_dir()]
    class_names = sorted(dirs)
    return class_names


# =====================
# 1) 사용자 정의 CNN
# =====================
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 180 -> 90

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 90 -> 45

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 45 -> 22(버림)

        # 128 x 22 x 22 기준 (입력 180x180 전제)
        self.fc1 = nn.Linear(128 * 22 * 22, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# DataLoader 구성 (사용자 CNN용: 180x180)
def get_dataloaders():
    train_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    print("클래스 목록:", class_names)

    return train_loader, val_loader, class_names


# 학습 함수 (사용자 CNN용)
def train_model():
    train_loader, val_loader, class_names = get_dataloaders()

    model = ConvNet(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'val_accuracy': []
    }

    print("모델 학습 시작...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            train_loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                val_loop.set_postfix(loss=loss.item())

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)

        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        print(f"  Val   Loss: {epoch_val_loss:.4f}, Val   Acc: {epoch_val_acc:.2f}%")

    print("학습 완료!")

    torch.save(model.state_dict(), model_save_path)
    print(f"모델 가중치 저장: {model_save_path}")

    with open(history_save_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"학습 히스토리 저장: {history_save_path}")

    return model, history, class_names


# 모델 로드 (사용자 CNN용)
def load_model():
    class_names = get_class_names()
    model = ConvNet(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"저장된 모델 로드 완료: {model_save_path}")
    return model, class_names


# 히스토리 그래프
def plot_history():
    if not os.path.exists(history_save_path):
        print("히스토리 파일이 없습니다.")
        return

    with open(history_save_path, 'rb') as f:
        history = pickle.load(f)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()


# 검증 데이터 평가 (사용자 CNN용)
def evaluate_on_val():
    model, class_names = load_model()

    val_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating [val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100.0 * correct / total
    print(f"검증 데이터 기준 정확도: {acc:.2f}%")

    return acc


# 단일 이미지 예측 (사용자 CNN용)
def predict_single_image(image_path):
    model, class_names = load_model()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_width, img_height))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        max_prob, pred_idx = torch.max(probs, 0)

    pred_class = class_names[pred_idx.item()]
    print("\n=== 단일 이미지 예측 결과 ===")
    print(f"파일: {image_path}")
    print(f"예측 클래스: {pred_class}")
    print(f"확률: {max_prob.item():.4f}")

    cv2.putText(img_bgr, f"{pred_class} ({max_prob.item():.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow("Prediction", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# seg_pred 폴더 전체 예측 (사용자 CNN용)
def predict_on_seg_pred():
    model, class_names = load_model()

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image_files = [f for f in os.listdir(pred_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if not image_files:
        print(f"{pred_dir} 안에 예측할 이미지가 없습니다.")
        return

    print(f"\n=== seg_pred 폴더 예측 결과 ===")
    for fname in image_files:
        path = os.path.join(pred_dir, fname)
        img = Image.open(path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)[0]
            max_prob, pred_idx = torch.max(probs, 0)

        pred_class = class_names[pred_idx.item()]
        print(f"{fname}  ->  {pred_class}  ({max_prob.item():.4f})")


# =====================
# 2) VGG16 전이학습
# =====================

# VGG16용 DataLoader (224x224)
def get_vgg_dataloaders():
    train_transforms = transforms.Compose([
        transforms.Resize((vgg_input_size, vgg_input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((vgg_input_size, vgg_input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    print("VGG16용 클래스 목록:", class_names)

    return train_loader, val_loader, class_names


# 사전학습된 VGG16 기반 모델 생성 (특징 추출 모드)
def create_vgg16_feature_extractor(num_classes):
    vgg = models.vgg16(pretrained=True)
    # 합성곱 기반 층(feature extractor)은 동결
    for param in vgg.features.parameters():
        param.requires_grad = False

    # 마지막 분류기 레이어만 우리 데이터셋 클래스 수에 맞게 교체
    in_features = vgg.classifier[6].in_features
    vgg.classifier[6] = nn.Linear(in_features, num_classes)

    return vgg


# VGG16 학습 공통 함수
def train_vgg16_model(model, train_loader, val_loader,
                      num_epochs, learning_rate,
                      save_path, history_path=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'val_accuracy': []
    }

    print("VGG16 학습 시작...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loop = tqdm(train_loader, desc=f"VGG16 Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            train_loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_loop = tqdm(val_loader, desc=f"VGG16 Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                val_loop.set_postfix(loss=loss.item())

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)

        print(f"\n[VGG16 Epoch {epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        print(f"  Val   Loss: {epoch_val_loss:.4f}, Val   Acc: {epoch_val_acc:.2f}%")

    print("VGG16 학습 완료!")

    torch.save(model.state_dict(), save_path)
    print(f"VGG16 모델 가중치 저장: {save_path}")

    if history_path is not None:
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        print(f"VGG16 학습 히스토리 저장: {history_path}")

    return model, history


# 2-1) 특징 추출 + 분류기 학습
def train_vgg16_feature():
    train_loader, val_loader, class_names = get_vgg_dataloaders()
    model = create_vgg16_feature_extractor(num_classes=len(class_names))
    model, history = train_vgg16_model(
        model,
        train_loader,
        val_loader,
        num_epochs=2,           # 필요에 따라 조정
        learning_rate=1e-3,     # 분류기 학습용 lr
        save_path=vgg_feature_model_path,
        history_path=vgg_feature_history_path
    )
    return model, history, class_names


# 2-2) 파인튜닝
def finetune_vgg16():
    train_loader, val_loader, class_names = get_vgg_dataloaders()
    model = create_vgg16_feature_extractor(num_classes=len(class_names))

    if os.path.exists(vgg_feature_model_path):
        model.load_state_dict(torch.load(vgg_feature_model_path, map_location=device))
        print("기존 VGG16 특징 추출 모델 가중치를 로드했습니다. 이 상태에서 파인튜닝을 진행합니다.")
    else:
        print("특징 추출용 가중치를 찾지 못했습니다. ImageNet 가중치에서 바로 파인튜닝을 진행합니다.")

    # VGG16 마지막 합성곱 블록(conv5_1~3)만 동결 해제
    # features[24] ~ features[30] 범위가 conv5_x 블록에 해당
    for idx in range(24, len(model.features)):
        for param in model.features[idx].parameters():
            param.requires_grad = True

    model, history = train_vgg16_model(
        model,
        train_loader,
        val_loader,
        num_epochs=2,           # 필요에 따라 조정
        learning_rate=1e-4,     # 파인튜닝은 더 작은 lr
        save_path=vgg_finetune_model_path,
        history_path=vgg_finetune_history_path
    )
    return model, history, class_names


# 메인 메뉴
if __name__ == "__main__":
    set_seed(42)

    while True:
        print("\n--- 이미지 분류 CNN 메뉴 ---")
        print("1. (직접 정의한 CNN) seg_train으로 학습")
        print("2. (직접 정의한 CNN) seg_test(검증) 정확도 평가")
        print("3. (직접 정의한 CNN) 학습 히스토리 그래프 보기")
        print("4. (직접 정의한 CNN) seg_pred 폴더 전체 예측")
        print("5. (직접 정의한 CNN) 이미지 경로 직접 입력하여 예측")
        print("6. VGG16(사전학습) - 특징 추출 + 분류기 학습")
        print("7. VGG16(사전학습) - 파인튜닝")
        print("0. 종료")

        sel = input("선택: ")

        if sel == "1":
            train_model()
        elif sel == "2":
            evaluate_on_val()
        elif sel == "3":
            plot_history()
        elif sel == "4":
            predict_on_seg_pred()
        elif sel == "5":
            path = input("이미지 파일 경로를 입력하세요: ")
            predict_single_image(path)
        elif sel == "6":
            train_vgg16_feature()
        elif sel == "7":
            finetune_vgg16()
        elif sel == "0":
            print("종료합니다.")
            break
        else:
            print("잘못된 입력입니다.")
