import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import ResNet50_Weights, resnet50
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json

# ==========================================
# 1. 설정 및 하이퍼파라미터
# ==========================================
train_dir = "./data/train"  # 학습 데이터 경로
test_dir = "./data/test"    # 테스트 데이터 경로

# [변경] GPU 4개를 효율적으로 쓰기 위해 배치 사이즈를 늘림 (32 -> 128)
# 만약 "CUDA out of memory" 에러가 발생하면 64 또는 32로 줄이세요.
BATCH_SIZE = 128 
NUM_WORKERS = 16  # 데이터 로딩 속도 향상

# ==========================================
# 2. 데이터 전처리
# ==========================================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ==========================================
# 3. 데이터셋 및 데이터로더
# ==========================================
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    print("데이터 경로를 확인해주세요.")
    # exit()

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

# [변경] num_workers와 batch_size 적용
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"훈련셋 크기: {len(train_dataset)}")
print(f"테스트셋 크기: {len(test_dataset)}")

num_classes = len(train_dataset.classes)

# ==========================================
# 4. 모델 설정 및 GPU 병렬화
# ==========================================
# 사전 학습된 ResNet50 불러오기
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# 모든 파라미터 freeze (마지막 fc 제외)
for param in model.parameters():
    param.requires_grad = False

# 최상위 레이어 변경 (랜덤 초기화)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [핵심 변경] Multi-GPU 설정 (DataParallel)
if torch.cuda.device_count() > 1:
    print(f"🔥 [System] 감지된 GPU 개수: {torch.cuda.device_count()}개")
    print("🔥 DataParallel을 사용하여 병렬 학습을 시작합니다!")
    model = nn.DataParallel(model)
else:
    print("CUDA is not available or only 1 GPU found. Using single device.")

model = model.to(device)

# 손실 함수
criterion = nn.CrossEntropyLoss()

# ==========================================
# 5. 학습 및 테스트 함수
# ==========================================
def train_and_test(
    model,
    train_loader,
    test_loader,
    criterion,
    num_epochs=30,
    log_file="resnet_training_log.txt",
):
    best_test_acc = 0.0  # 최고 테스트 정확도 추적

    # 로그 파일 초기화 및 헤더 작성
    with open(log_file, mode="w") as file:
        file.write("Epoch\tTrain Loss\tTrain Acc\tTest Loss\tTest Acc\tTest F1\n")

    # [중요] DataParallel 사용 시, 실제 모델 파라미터에 접근하려면 .module을 사용해야 함
    # 모델이 DataParallel로 감싸져 있는지 확인하고 실제 모델 객체(real_model)를 참조
    real_model = model.module if hasattr(model, 'module') else model
    
    # 초기 Optimizer 설정 (임시)
    optimizer = optim.Adam(real_model.fc.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # ---------------------------------------
        # 동적 Learning Rate 및 Unfreezing 전략
        # ---------------------------------------
        if epoch == 0:
            print("첫 번째 epoch: 마지막 출력층만 학습")
            # 전체 동결
            for param in real_model.parameters(): 
                param.requires_grad = False
            # FC 레이어만 해제
            for param in real_model.fc.parameters(): 
                param.requires_grad = True
            
            optimizer = optim.Adam(real_model.fc.parameters(), lr=1e-4)
            
        elif epoch == 1:
            print("두 번째 epoch부터 전체 모델을 미세 조정 (Fine-tuning)")
            # 전체 해제
            for param in real_model.parameters():
                param.requires_grad = True
            
            optimizer = optim.Adam(real_model.parameters(), lr=1e-5)

        # ---------------------------------------
        # 학습 단계 (Training)
        # ---------------------------------------
        model.train()  # 학습 모드
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            # 데이터를 CUDA 장치로 이동
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

        # ---------------------------------------
        # 평가 단계 (Testing)
        # ---------------------------------------
        model.eval()  # 평가 모드
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
                
                # CPU로 이동하여 결과 모으기
                all_test_preds.extend(preds.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())

        epoch_test_loss = test_loss / len(test_loader.dataset)
        epoch_test_acc = test_corrects.double() / len(test_loader.dataset)
        epoch_test_f1 = f1_score(all_test_labels, all_test_preds, average="macro")
        
        print(
            f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.4f} | Test F1: {epoch_test_f1:.4f}"
        )

        # ---------------------------------------
        # 로그 및 모델 저장
        # ---------------------------------------
        with open(log_file, mode="a") as file:
            file.write(
                f"{epoch+1}\t{epoch_train_loss:.4f}\t{epoch_train_acc:.4f}\t{epoch_test_loss:.4f}\t{epoch_test_acc:.4f}\t{epoch_test_f1:.4f}\n"
            )

        if epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc
            # [중요] 저장할 때는 DataParallel 껍데기를 벗기고(real_model) 저장해야 나중에 불러오기 편함
            torch.save(real_model.state_dict(), "best_finetuned_resnet.pth")
            print("Best model saved.")

    print(f"\nTraining complete. Best Test Acc: {best_test_acc:.4f}")

    # ==========================================
    # 6. 최종 분석 결과 저장
    # ==========================================
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Final Evaluation"):
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

    with open("resnet_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\nClassification Report saved to resnet_classification_report.json")

    # Confusion Matrix 저장
    cm = confusion_matrix(all_labels, all_preds)
    np.save("resnet_confusion_matrix.npy", cm)
    print("Confusion Matrix saved to resnet_confusion_matrix.npy")

    # 예측 확률 저장 (ROC curve 분석용)
    np.save("resnet_test_predictions.npy", all_preds)
    np.save("resnet_test_labels.npy", all_labels)
    np.save("resnet_test_probs.npy", all_probs)
    print("Test predictions and probabilities saved.")

    return model


if __name__ == "__main__":
    # 모델 학습 및 테스트 수행
    model = train_and_test(
        model,
        train_loader,
        test_loader,
        criterion,
        num_epochs=30,
        log_file="resnet_training_log.txt",
    )