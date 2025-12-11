import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json
import timm  # timm 라이브러리 사용 (pip install timm)

# 경로 설정
train_dir = "./data/train"  # 학습 데이터 경로
test_dir = "./data/test"  # 테스트 데이터 경로

# 데이터 전처리 (EfficientNet은 224x224 입력 사용)
transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# 데이터셋 및 데이터로더
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"훈련셋 크기: {len(train_dataset)}")
print(f"테스트셋 크기: {len(test_dataset)}")
print(f"클래스 수: {len(train_dataset.classes)}")
print(f"클래스: {train_dataset.classes}")

# EfficientNet-B0 모델 로드 (timm 라이브러리 사용)
# timm.create_model을 사용하면 사전학습된 가중치를 쉽게 로드 가능
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# EfficientNet-B0 모델 생성 (pretrained=True)
model = timm.create_model(
    "efficientnet_b0", pretrained=True, num_classes=len(train_dataset.classes)
)
model = model.to(device)

print(f"\nEfficientNet-B0 모델 구조:")
print(f"총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
print(
    f"학습 가능한 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()


def train_and_test(
    model,
    train_loader,
    test_loader,
    criterion,
    num_epochs=30,
    log_file="efficientnet_training_log.txt",
):
    """
    EfficientNet 모델을 학습하고 평가하는 함수

    Args:
        model: EfficientNet 모델
        train_loader: 훈련 데이터 로더
        test_loader: 테스트 데이터 로더
        criterion: 손실 함수
        num_epochs: 학습 에포크 수
        log_file: 로그 파일 경로
    """
    best_test_acc = 0.0

    # 로그 파일 헤더 작성
    os.makedirs("./log", exist_ok=True)
    log_file = f"./log/{log_file}"

    with open(log_file, mode="w") as file:
        file.write("Epoch\tTrain Loss\tTrain Acc\tTest Loss\tTest Acc\tTest F1\n")

    print("\n" + "=" * 70)
    print("EfficientNet-B0 학습 시작")
    print("=" * 70)

    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # 첫 번째 epoch: Feature extraction (classifier만 학습)
        # 두 번째 epoch부터: Fine-tuning (전체 모델 학습)
        if epoch == 0:
            # 첫 번째 epoch: classifier만 학습
            for name, param in model.named_parameters():
                if "classifier" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
            )
            print("첫 번째 epoch: Classifier만 학습 (Feature Extraction)")
        elif epoch == 1:
            # 두 번째 epoch부터 전체 모델을 미세 조정
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            print("두 번째 epoch부터 전체 모델을 미세 조정 (Fine-tuning)")

        # 학습 단계
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")

        # 평가 단계
        model.eval()
        test_loss = 0.0
        test_corrects = 0
        all_test_preds = []
        all_test_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                test_loss += loss.item() * inputs.size(0)
                test_corrects += torch.sum(preds == labels.data)
                all_test_preds.extend(preds.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())

        epoch_test_loss = test_loss / len(test_loader.dataset)
        epoch_test_acc = test_corrects.double() / len(test_loader.dataset)
        epoch_test_f1 = f1_score(all_test_labels, all_test_preds, average="macro")
        print(
            f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.4f} | Test F1: {epoch_test_f1:.4f}"
        )

        # 로그 저장
        with open(log_file, mode="a") as file:
            file.write(
                f"{epoch+1}\t{epoch_train_loss:.4f}\t{epoch_train_acc:.4f}\t{epoch_test_loss:.4f}\t{epoch_test_acc:.4f}\t{epoch_test_f1:.4f}\n"
            )

        # 최고 테스트 정확도 모델 저장
        if epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc
            torch.save(model.state_dict(), "best_finetuned_efficientnet.pth")
            print("✓ Best model saved.")

    print("\n" + "=" * 70)
    print(f"학습 완료! Best Test Acc: {best_test_acc:.4f}")
    print("=" * 70)

    # 최종 평가 및 메트릭 저장
    print("\n최종 평가 및 메트릭 저장 중...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Classification Report 저장
    class_names = train_loader.dataset.classes
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )

    with open("efficientnet_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("✓ Classification Report saved to efficientnet_classification_report.json")

    # Confusion Matrix 저장
    cm = confusion_matrix(all_labels, all_preds)
    np.save("efficientnet_confusion_matrix.npy", cm)
    print("✓ Confusion Matrix saved to efficientnet_confusion_matrix.npy")

    # 예측 확률 저장 (ROC curve 분석용)
    np.save("efficientnet_test_predictions.npy", all_preds)
    np.save("efficientnet_test_labels.npy", all_labels)
    np.save("efficientnet_test_probs.npy", all_probs)
    print("✓ Test predictions and probabilities saved.")

    return model


if __name__ == "__main__":
    # 모델 학습 및 테스트 수행
    model = train_and_test(
        model,
        train_loader,
        test_loader,
        criterion,
        num_epochs=30,
        log_file="efficientnet_training_log.txt",
    )
