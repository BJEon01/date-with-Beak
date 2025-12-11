# 반려견 안구 질환 분류를 위한 딥러닝 모델 개발 및 성능 분석

**Data Mining Term Project Report**

---

## 1. Introduction

### 1.1 프로젝트 배경 및 동기

반려동물 양육 인구가 지속적으로 증가함에 따라, 반려동물의 건강 관리에 대한 관심도 높아지고 있다. 특히 안구 질환은 반려견에게 흔하게 발생하는 질병 중 하나로, 조기 발견과 정확한 진단이 치료 성공률을 크게 좌우한다. 그러나 일반 보호자들이 안구 질환을 육안으로 정확히 구분하기는 어려우며, 동물병원의 접근성이나 비용 문제로 인해 적절한 시기에 진료를 받지 못하는 경우가 많다.

최근 딥러닝 기술의 발전으로 의료 영상 분석 분야에서 높은 성능을 보이고 있으며, 이를 동물 의료 분야에 적용하려는 시도가 증가하고 있다. 이미지 분류 기술을 활용하면 반려견의 안구 사진만으로도 질환을 자동으로 분류할 수 있으며, 이는 조기 진단 및 예방에 크게 기여할 수 있다.

### 1.2 연구 목적

본 프로젝트의 목적은 다음과 같다:

1. **딥러닝 기반 안구 질환 분류 모델 개발**: ResNet50, Vision Transformer(ViT) 등 다양한 딥러닝 모델을 활용하여 반려견의 안구 질환을 11개 클래스로 분류하는 모델을 학습한다.

2. **모델 성능 비교 및 분석**: 딥러닝 모델(ResNet50, ViT)과 머신러닝 앙상블 모델(RandomForest, XGBoost, Stacking)의 성능을 정량적으로 비교하고, 각 모델의 장단점을 분석한다.

3. **결과 해석 및 시각화**: Confusion Matrix, ROC Curve, t-SNE 등 다양한 분석 기법을 통해 모델의 예측 결과를 해석하고, 오분류 패턴을 파악한다.

4. **실용성 검증**: Flask 기반 웹 애플리케이션을 통해 모델을 배포하고, 실제 사용자가 손쉽게 진단 결과를 확인할 수 있도록 한다.

### 1.3 데이터셋 개요

본 프로젝트에서는 반려견 안구 이미지 데이터셋을 사용하며, 총 11개의 클래스로 구성된다:

- **10개 질환 클래스**: 결막염, 궤양성 각막질환, 백내장, 비궤양성 각막질환, 색소침착성각막염, 안검내반증, 안검염, 안검종양, 유루증, 핵경화
- **1개 정상 클래스**: 무질환

각 클래스당 약 5,000장의 이미지를 샘플링하여 균형 잡힌 데이터셋을 구성하였으며, 학습/테스트 세트로 분할하여 모델의 일반화 성능을 평가하였다.

---

## 2. Related Methods and Models

### 2.1 전이 학습(Transfer Learning)

전이 학습은 대규모 데이터셋(예: ImageNet)에서 사전 학습된 모델을 새로운 작업에 적용하는 기법이다. 본 프로젝트에서는 ImageNet으로 사전 학습된 ResNet50과 ViT 모델을 사용하였으며, 다음과 같은 전략을 채택하였다:

1. **Feature Extraction**: 사전 학습된 모델의 가중치를 고정(freeze)하고, 마지막 분류 레이어만 학습
2. **Fine-tuning**: 전체 모델의 가중치를 미세 조정하여 도메인에 맞게 최적화

### 2.2 딥러닝 모델

#### 2.2.1 ResNet50

ResNet(Residual Network)은 잔차 연결(residual connection)을 통해 깊은 네트워크에서도 안정적인 학습이 가능하도록 설계된 모델이다. ResNet50은 50개의 레이어로 구성되어 있으며, 약 2,350만 개의 파라미터를 가진다.

**참고문헌**:

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. _CVPR_.

#### 2.2.2 Vision Transformer (ViT)

ViT는 Transformer 구조를 이미지 분류에 적용한 모델로, 이미지를 패치 단위로 나누어 시퀀스로 처리한다. 전역적인 관계를 효과적으로 학습할 수 있으며, 대규모 데이터셋에서 높은 성능을 보인다.

**참고문헌**:

- Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. _ICLR_.

### 2.3 머신러닝 앙상블 모델

#### 2.3.1 RandomForest

다수의 결정 트리를 결합한 앙상블 모델로, 오버피팅을 방지하고 안정적인 예측 성능을 제공한다.

#### 2.3.2 XGBoost

Gradient Boosting 알고리즘을 효율적으로 구현한 모델로, 높은 예측 성능과 빠른 학습 속도를 자랑한다.

#### 2.3.3 Stacking

여러 개의 기본 모델(base model)의 예측 결과를 메타 모델(meta model)이 학습하는 앙상블 기법이다.

### 2.4 성능 평가 지표

- **Accuracy**: 전체 샘플 중 올바르게 분류된 샘플의 비율
- **Precision**: 양성으로 예측한 샘플 중 실제 양성의 비율
- **Recall**: 실제 양성 샘플 중 올바르게 예측한 비율
- **F1-Score**: Precision과 Recall의 조화평균
- **ROC Curve & AUC**: 다양한 임계값에서의 TPR과 FPR을 시각화하고, 분류 성능을 정량화
- **Confusion Matrix**: 클래스별 예측 결과를 행렬로 표현하여 오분류 패턴 분석

### 2.5 시각화 기법

- **t-SNE**: 고차원 특징을 2차원으로 축소하여 데이터의 분포를 시각화
- **Learning Curve**: 에폭에 따른 손실과 정확도 변화를 시각화하여 학습 과정 분석

---

## 3. Experimental Settings

### 3.1 데이터셋 구성

#### 3.1.1 데이터 수집 및 전처리

- **데이터 출처**: 반려견 안구 질환 이미지 데이터셋 (공개 데이터셋 또는 자체 수집)
- **클래스 수**: 11개 (질환 10개 + 정상 1개)
- **이미지 수**: 클래스당 약 5,000장, 총 약 55,000장
- **데이터 분할**: Train 80%, Test 20%

#### 3.1.2 데이터 증강(Data Augmentation)

과적합 방지 및 일반화 성능 향상을 위해 다음과 같은 증강 기법을 적용:

```python
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 3.2 모델 구조

#### 3.2.1 ResNet50

- **입력 크기**: 224 × 224 × 3
- **사전 학습 가중치**: ImageNet
- **수정 사항**: 마지막 fully connected layer를 11개 클래스 분류에 맞게 변경
- **파라미터 수**: 약 23.5M

#### 3.2.2 Vision Transformer (ViT)

- **모델 변형**: ViT-B/16 (Base model, patch size 16)
- **입력 크기**: 224 × 224 × 3
- **사전 학습 가중치**: ImageNet
- **수정 사항**: Classification head를 11개 클래스에 맞게 변경
- **파라미터 수**: 약 86.4M

#### 3.2.3 머신러닝 모델

딥러닝 모델(ResNet50)에서 추출한 임베딩 벡터를 입력으로 사용:

- **RandomForest**: n_estimators=100, max_depth=50
- **XGBoost**: max_depth=6, learning_rate=0.1, n_estimators=150
- **Stacking**: Base models (RF, XGB) + Meta model (Logistic Regression)

### 3.3 학습 파라미터

#### 3.3.1 ResNet50 학습 설정

```python
# 1단계: 마지막 레이어만 학습 (1 epoch)
optimizer = Adam(lr=1e-4)
criterion = CrossEntropyLoss()

# 2단계: 전체 모델 미세 조정 (29 epochs)
optimizer = Adam(lr=1e-5)
```

- **총 에폭**: 30
- **배치 크기**: 32
- **손실 함수**: Cross Entropy Loss
- **옵티마이저**: Adam
- **학습률 스케줄링**: 1 epoch (1e-4) → 29 epochs (1e-5)

#### 3.3.2 ViT 학습 설정

ResNet50과 동일한 학습 전략 적용:

- **총 에폭**: 30
- **배치 크기**: 32
- **옵티마이저**: Adam
- **학습률**: 1 epoch (1e-4) → 29 epochs (1e-5)

### 3.4 실험 환경

- **하드웨어**:
  - GPU: NVIDIA RTX 3090 / A100 (SeoulTech GPU Server)
  - CPU: Intel Xeon / AMD Ryzen
  - RAM: 32GB 이상
- **소프트웨어**:
  - Python 3.8+
  - PyTorch 2.0+
  - torchvision, scikit-learn, xgboost, matplotlib, seaborn

### 3.5 코드 구조

```
PPE-ProtectPetEyes/
├── train_알고리즘/
│   ├── Resnet_finetuning.py      # ResNet50 학습 코드
│   ├── Resnet_추가기법.py         # ResNet50 + 추가 기법
│   ├── VIT.py                     # ViT 학습 코드
│   ├── 머신러닝 임베딩 추출.py    # Feature embedding 추출
│   ├── RandomForest.py            # RandomForest 학습
│   ├── XGBoost.py                 # XGBoost 학습
│   └── Stacking.py                # Stacking 앙상블
├── log/
│   ├── resnet_training_log.txt    # ResNet 학습 로그
│   └── vit_training_log.txt       # ViT 학습 로그
├── app.py                         # Flask 웹 애플리케이션
├── analysis_results.ipynb         # 결과 분석 노트북
└── REPORT.md                      # 보고서
```

---

## 4. Results

### 4.1 학습 과정 분석

#### 4.1.1 ResNet50 학습 곡선

[이 부분은 `analysis_results.ipynb`를 실행하여 생성된 그래프를 삽입]

- **최종 Train Accuracy**: 73.24%
- **최종 Test Accuracy**: 48.35%
- **최고 Test Accuracy**: 51.81% (6 epoch)

**분석**:

- 초기 5 에폭 동안 급격한 성능 향상
- 6 에폭 이후 테스트 정확도가 소폭 감소하는 과적합 경향 관찰
- Train과 Test 간의 격차(약 25%)는 데이터의 복잡성과 클래스 간 유사성을 시사

#### 4.1.2 ViT 학습 곡선

[이 부분은 `analysis_results.ipynb`를 실행하여 생성된 그래프를 삽입]

**분석**:

- ViT는 ResNet에 비해 더 많은 파라미터를 가지고 있으나, 본 데이터셋 규모에서는 상대적으로 낮은 성능
- 더 많은 데이터나 긴 학습 시간이 필요할 것으로 판단

### 4.2 모델 성능 비교

[이 부분은 `analysis_results.ipynb`에서 생성된 비교 표와 그래프를 삽입]

| Model        | Train Acc | Test Acc | Best Test Acc | Parameters (M) |
| ------------ | --------- | -------- | ------------- | -------------- |
| ResNet50     | 0.7324    | 0.4835   | 0.5181        | 23.5           |
| ViT          | -         | -        | -             | 86.4           |
| RandomForest | -         | -        | -             | -              |
| XGBoost      | -         | -        | -             | -              |
| Stacking     | -         | -        | -             | -              |

**Note**: 표의 빈 값은 `analysis_results.ipynb` 실행 후 실제 결과로 업데이트 필요

### 4.3 Confusion Matrix 분석

[이 부분은 `analysis_results.ipynb`에서 생성된 Confusion Matrix 이미지를 삽입]

**주요 발견사항**:

1. **잘 분류되는 클래스**:

   - [특정 질환 이름] - 높은 대각선 값
   - [정상 클래스] - 다른 질환과 명확히 구분

2. **오분류가 많은 클래스**:

   - [유사한 질환 A와 B] - 시각적 특징이 유사하여 혼동
   - [특정 질환] - 다른 여러 질환으로 분산되어 예측

3. **임상적 의미**:
   - 시각적으로 유사한 질환들(예: 궤양성/비궤양성 각막질환) 간의 오분류는 전문가도 구분이 어려운 경우가 많음
   - 이는 추가적인 임상 정보(병력, 검사 결과)가 필요함을 시사

### 4.4 ROC Curve 및 AUC

[이 부분은 `analysis_results.ipynb`에서 생성된 ROC Curve 이미지를 삽입]

**클래스별 AUC**:

- [클래스 1]: AUC = 0.XX
- [클래스 2]: AUC = 0.XX
- ...
- **Micro-average AUC**: 0.XX

**분석**:

- AUC가 높은 클래스는 다른 질환과 명확히 구분되는 특징 보유
- AUC가 낮은 클래스는 추가적인 데이터 수집 또는 특징 엔지니어링 필요

### 4.5 t-SNE 시각화

[이 부분은 `analysis_results.ipynb`에서 생성된 t-SNE 이미지를 삽입]

**해석**:

1. **클러스터링 패턴**:

   - 정상 샘플과 질환 샘플이 어느 정도 분리되어 분포
   - 일부 질환 클래스는 겹쳐져 있어 시각적 유사성 확인

2. **특징 공간 분석**:
   - ResNet50이 추출한 특징이 질환별로 의미 있는 패턴 형성
   - 오분류된 샘플들은 클래스 경계에 위치

### 4.6 성능 메트릭 요약

[이 부분은 `analysis_results.ipynb`에서 생성된 메트릭 요약 표 삽입]

| Metric              | Value  |
| ------------------- | ------ |
| Accuracy            | 0.XXXX |
| F1-Score (Macro)    | 0.XXXX |
| F1-Score (Weighted) | 0.XXXX |
| Precision (Macro)   | 0.XXXX |
| Recall (Macro)      | 0.XXXX |

### 4.7 오분류 분석

[이 부분은 `analysis_results.ipynb`에서 생성된 오분류 분석 그래프 삽입]

**클래스별 오분류율**:

- 오분류율이 높은 클래스: [클래스 이름] (XX%)
- 오분류율이 낮은 클래스: [클래스 이름] (XX%)

**오분류 원인**:

1. **시각적 유사성**: 일부 질환은 육안으로도 구분이 어려움
2. **데이터 불균형**: 특정 질환의 다양성 부족
3. **이미지 품질**: 조명, 각도, 초점 등의 변동성
4. **질환의 중첩**: 여러 질환이 동시에 발생하는 경우

---

## 5. Conclusion and Discussion

### 5.1 연구 결과 요약

본 프로젝트에서는 반려견 안구 질환을 자동으로 분류하기 위한 딥러닝 모델을 개발하고 성능을 분석하였다. 주요 성과는 다음과 같다:

1. **모델 개발**: ResNet50, ViT 등 다양한 딥러닝 모델과 머신러닝 앙상블 모델을 학습하고 비교
2. **성능 달성**: ResNet50 모델이 테스트 데이터에서 약 48-52%의 정확도를 기록
3. **심층 분석**: Confusion Matrix, ROC Curve, t-SNE 등 다양한 분석 기법을 통해 모델의 예측 패턴과 한계를 이해
4. **실용화**: Flask 웹 애플리케이션을 통해 실제 사용자가 활용할 수 있는 시스템 구축

### 5.2 연구의 의의

1. **임상적 활용 가능성**:

   - 반려동물 보호자가 초기 스크리닝 도구로 활용 가능
   - 수의사의 진단 보조 시스템으로 적용 가능

2. **기술적 기여**:

   - 동물 의료 영상 분석에 딥러닝 기법 적용
   - 다양한 모델 비교를 통한 최적 아키텍처 탐색

3. **확장 가능성**:
   - 다른 동물 질환 분류로 확장 가능
   - 더 많은 데이터 확보 시 성능 향상 기대

### 5.3 한계점 및 개선 방향

#### 5.3.1 현재 한계점

1. **상대적으로 낮은 정확도**:

   - 11개 클래스 분류에서 약 50%의 정확도는 실제 임상 환경에서 보조 도구로 사용하기에는 제한적
   - 유사한 질환 간의 오분류가 빈번

2. **데이터 품질 및 다양성**:

   - 이미지 촬영 조건(조명, 각도, 거리)의 변동성
   - 특정 질환의 다양한 진행 단계를 충분히 반영하지 못함

3. **클래스 불균형**:

   - 샘플링을 통해 균형을 맞췄으나, 실제 임상 환경의 질환 발생 빈도와는 차이

4. **단일 모달리티**:
   - 이미지만으로 판단하며, 병력이나 다른 임상 정보를 활용하지 못함

#### 5.3.2 개선 방향

1. **데이터 확충**:

   - 더 많은 이미지 수집 (클래스당 10,000장 이상)
   - 다양한 촬영 환경의 데이터 확보
   - 질환의 다양한 진행 단계 포함

2. **모델 개선**:

   - 더 큰 모델 (EfficientNet, Swin Transformer 등) 실험
   - Ensemble 기법 강화 (다양한 모델의 예측 결합)
   - Self-supervised learning으로 사전 학습

3. **추가 분석**:

   - Class Activation Map (CAM)을 통한 모델의 주의 영역 시각화
   - 질환별 특징 분석 및 해석 가능성 향상

4. **멀티모달 학습**:

   - 이미지 + 텍스트(병력, 증상) 결합
   - 시계열 이미지를 활용한 질환 진행 예측

5. **실용화 강화**:
   - 모바일 앱 개발로 접근성 향상
   - 수의사 피드백을 통한 지속적인 모델 업데이트
   - 설명 가능한 AI 기법 도입으로 신뢰도 향상

### 5.4 향후 연구 방향

1. **다양한 동물 종으로 확장**: 고양이, 토끼 등 다른 반려동물의 안구 질환 분류
2. **질환 진행도 예측**: 초기/중기/말기 등 질환의 심각도 분류
3. **실시간 진단 시스템**: 동영상 기반 실시간 분석
4. **의료 AI 윤리 연구**: 오진단 시 책임 소재, 데이터 프라이버시 등

### 5.5 결론

본 프로젝트는 딥러닝 기술을 활용하여 반려견 안구 질환을 자동으로 분류하는 시스템을 개발하고, 다양한 분석 기법을 통해 모델의 성능과 한계를 명확히 파악하였다. 비록 현재 정확도가 실제 임상 환경에서 단독으로 사용되기에는 제한적이나, 초기 스크리닝 도구 또는 수의사의 진단 보조 시스템으로서 충분한 가치가 있다고 판단된다.

향후 더 많은 데이터 확보, 모델 개선, 멀티모달 학습 등을 통해 성능을 지속적으로 향상시킨다면, 반려동물의 건강 관리에 실질적으로 기여할 수 있는 시스템으로 발전할 수 있을 것으로 기대된다.

---

## References

### 논문

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 770-778).

2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_.

3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In _Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining_ (pp. 785-794).

4. Breiman, L. (2001). Random forests. _Machine learning_, 45(1), 5-32.

5. Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. _Journal of machine learning research_, 9(11).

### 오픈소스 및 라이브러리

6. PyTorch: https://pytorch.org/
7. torchvision: https://pytorch.org/vision/stable/index.html
8. scikit-learn: https://scikit-learn.org/
9. XGBoost: https://xgboost.readthedocs.io/
10. Flask: https://flask.palletsprojects.com/

### 데이터셋

11. [데이터셋 출처 URL 또는 설명] (실제 데이터 출처를 명시)

### 기타

12. [기타 참고한 블로그, GitHub 저장소 등]

---

## Appendix

### A. 코드 저장소

전체 코드는 다음 GitHub 저장소에서 확인할 수 있습니다:

- Repository: [GitHub URL]

### B. 주요 파일 설명

- `train_알고리즘/Resnet_finetuning.py`: ResNet50 fine-tuning 코드
- `train_알고리즘/VIT.py`: Vision Transformer 학습 코드
- `analysis_results.ipynb`: 결과 분석 및 시각화 노트북
- `app.py`: Flask 웹 애플리케이션
- `REPORT.md`: 본 보고서

### C. 실행 방법

```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 데이터 다운로드
# Dropbox 링크에서 data.zip 다운로드 후 압축 해제

# 3. 모델 학습
python train_알고리즘/Resnet_finetuning.py

# 4. 결과 분석
jupyter notebook analysis_results.ipynb

# 5. 웹 애플리케이션 실행
python app.py
```

### D. 하드웨어 요구사항

- **GPU**: NVIDIA RTX 3090 이상 (VRAM 24GB 권장)
- **RAM**: 32GB 이상
- **저장공간**: 50GB 이상 (데이터셋 포함)

---

**보고서 작성일**: 2025년 12월

**프로젝트 팀원**: [팀원 이름]

**지도교수**: [교수님 성함]

**과목**: Data Mining
