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

# ==========================================
# 1. 설정 및 하이퍼파라미터
# ==========================================
train_dir = "./data/train"  # 학습 데이터 경로
test_dir = "./data/test"    # 테스트 데이터 경로

# [변경] GPU 4개를 효율적으로 쓰기 위해 배치 사이즈 증가 (32 -> 128)
# 메모리 에러 발생 시 64 또는 32로 조정하세요.
BATCH_SIZE = 128
NUM_WORKERS = 16  # 데이터 로딩 속도 향상

# 로그 및 결과 파일 경로 설정 (ResNet과 겹치지 않게 설정)
LOG_FILE = "efficientnet_training_log.txt"
MODEL_SAVE_NAME = "best_finetuned_efficientnet.pth"
REPORT_SAVE_NAME = "efficientnet_classification_report.json"
CM_SAVE_NAME = "efficientnet_confusion_matrix.npy"

# ==========================================
# 2. 데이터 전처리
# ==========================================
# EfficientNet은 224x224 (B0 기준) 입력 사용
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

# [변경] 상향된 Batch Size와 Num Workers 적용
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"훈련셋 크기: {len(train_dataset)}")
print(f"테스트셋 크기: {len(test_dataset)}")
print(f"클래스 수: {len(train_dataset.classes)}")

# ==========================================
# 4. 모델 설정 및 GPU 병렬화
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EfficientNet-B0 모델 생성 (pretrained=True)
model = timm.create_model(
    "efficientnet_b0", pretrained=True, num_classes=len(train_dataset.classes)
)

print(f"\nEfficientNet-B0 모델 로드 완료.")

# [핵심 변경] Multi-GPU 설정 (DataParallel)
if torch.cuda.device_count() > 1:
    print(f"🔥 [System] 감지된 GPU 개수: {torch.cuda.device_count()}개")
    print("🔥 DataParallel을 사용하여 병렬 학습을 시작합니다!")
    model = nn.DataParallel(model)
else:
    print(f"Using single device: {device}")

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
    log_file=LOG_FILE,
):
    best_test_acc = 0.0

    # 로그 폴더 생성 및 헤더 작성
    os.makedirs("./log", exist_ok=True)
    log_path = f"./log/{log_file}"

    with open(log_path, mode="w") as file:
        file.write("Epoch\tTrain Loss\tTrain Acc\tTest Loss\tTest Acc\tTest F1\n")

    print("\n" + "=" * 70)
    print("EfficientNet-B0 학습 시작")
    print("=" * 70)

    # [중요] DataParallel 사용 시, 내부 파라미터 제어를 위해 .module로 접근
    real_model = model.module if hasattr(model, 'module') else model

    # 초기 옵티마이저 (임시)
    optimizer = optim.Adam(real_model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        
        # ---------------------------------------
        # 동적 Learning Rate 및 Layer Freezing
        # ---------------------------------------
        if epoch == 0:
            # 첫 번째 epoch: classifier만 학습
            print("첫 번째 epoch: Classifier만 학습 (Feature Extraction)")
            for name, param in real_model.named_parameters():
                if "classifier" in name: # timm EfficientNet의 마지막 층 이름은 'classifier'
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, real_model.parameters()), lr=1e-3
            )
            
        elif epoch == 1:
            # 두 번째 epoch부터 전체 모델 미세 조정
            print("두 번째 epoch부터 전체 모델을 미세 조정 (Fine-tuning)")
            for param in real_model.parameters():
                param.requires_grad = True
            
            optimizer = optim.Adam(real_model.parameters(), lr=1e-4)

        # ---------------------------------------
        # 학습 단계 (Training)
        # ---------------------------------------
        model.train()
        running_loss = 0.0
        running_corrects = 0

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

        # ---------------------------------------
        # 평가 단계 (Testing)
        # ---------------------------------------
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
                
                # CPU로 이동하여 결과 수집
                all_test_preds.extend(preds.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())

        epoch_test_loss = test_loss / len(test_loader.dataset)
        epoch_test_acc = test_corrects.double() / len(test_loader.dataset)
        epoch_test_f1 = f1_score(all_test_labels, all_test_preds, average="macro")
        
        print(
            f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.4f} | Test F1: {epoch_test_f1:.4f}"
        )

        # 로그 저장
        with open(log_path, mode="a") as file:
            file.write(
                f"{epoch+1}\t{epoch_train_loss:.4f}\t{epoch_train_acc:.4f}\t{epoch_test_loss:.4f}\t{epoch_test_acc:.4f}\t{epoch_test_f1:.4f}\n"
            )

        # 최고 성능 모델 저장 (real_model 저장)
        if epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc
            torch.save(real_model.state_dict(), MODEL_SAVE_NAME)
            print(f"✓ Best model saved to {MODEL_SAVE_NAME}")

    print("\n" + "=" * 70)
    print(f"학습 완료! Best Test Acc: {best_test_acc:.4f}")
    print("=" * 70)

    # ==========================================
    # 6. 최종 분석 결과 저장
    # ==========================================
    print("\n최종 평가 및 메트릭 저장 중...")
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

    # 1. Classification Report (JSON)
    class_names = train_loader.dataset.classes
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    with open(REPORT_SAVE_NAME, "w") as f:
        json.dump(report, f, indent=4)
    print(f"✓ Classification Report saved to {REPORT_SAVE_NAME}")

    # 2. Confusion Matrix (NPY)
    cm = confusion_matrix(all_labels, all_preds)
    np.save(CM_SAVE_NAME, cm)
    print(f"✓ Confusion Matrix saved to {CM_SAVE_NAME}")

    # 3. Probabilities (NPY) for ROC/Ensemble
    np.save("efficientnet_test_predictions.npy", all_preds)
    np.save("efficientnet_test_labels.npy", all_labels)
    np.save("efficientnet_test_probs.npy", all_probs)
    print("✓ Test predictions and probabilities saved (efficientnet_ prefix).")

    return model


if __name__ == "__main__":
    model = train_and_test(
        model,
        train_loader,
        test_loader,
        criterion,
        num_epochs=30,
        log_file=LOG_FILE,
    )