# 2023 Samsung AI Challenge : Camera-Invariant Domain Adaptation
[2023 Samsung AI Challenge : Camera-Invariant Domain Adaptation](https://dacon.io/competitions/official/236132/overview/description)
대회의 PSSC팀의 코드입니다.

## PSSC팀
포항공과대학교 응용수학 학생 학술단체인 POSTECH SIAM Student Chapter (PSSC)에서 2023년 2학기 활동의 일환으로 대회에 참여하였습니다.
- 허은우 (POSTECH 수학과, 팀장)
- 이성헌 (POSTECH 인공지능대학원)
- 이동진 (POSTECH 인공지능대학원)

코드 검증에 어려움이 있을 시 `dongjinlee@postech.ac.kr`로 이메일 주시면 성실히 답변드리겠습니다.

## Our solution
아래 1번부터 4번 모델의 픽셀별 클래스 확률 분포로부터 평균을 내려 마스크를 만들고 이를 사후처리하여 최종 마스크를 생성

- **1번 모델:** 사전훈련된 SegFormer를 `train_source` 및 `val_source` 데이터를 사용하여 미세 조정 후, `train_target` 데이터 수도 레이블링 생성 및 추가 학습

- **2번 모델:** 사전훈련된 SegFormer를 DANN 알고리즘을 사용하여 미세 조정

- **3번 모델:** 사전훈련된 SegFormer를 dice loss와 함께 미세 조정

- **4번 모델:** 사전훈련된 Mask2Former를 미세 조정

- **배경 및 사후처리 모델:** 사전훈련된 OneFormer로 inference에 사용하여 사후처리

### Main Model
- SegFormer-b5 pretrained on CityScape dataset, which is implemented by [NVLabs](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024)
- Fine-tuning the network using both `train_source` and `val_source`
- AdamW optimizer with one cycle LR scheduler

### Domain Adaptation 방법
1. 데이터 레벨 DA
- Fisheye transformation 적용
- OneFormer를 이용한 Background 모델링

2. 모델 레벨 DA
- Pseudo labeling
- Domain adaptive neural network

### Inference
**Sliding window inference**

이미지를 왼쪽 상단부터 (512, 512) 크기로 잘라 (50, 50)씩 이동하며 모델에 입력하여 예측. 겹치는 영역은 클래스 확률 분포를 평균내어 사용.

**Multi-scale inference**
이미지의 크기를 (960, 540), (1200, 675), (1440, 810)으로 resize하여 각각에 대해 sliding window inference를 수행하고 (960, 540) 크기로 축소 후 각 픽셀별 예측 확률 분포를 평균하여 사용.

## 코드 재현 방법

**배경 모델링 코드 재현**
- `notebooks/(0)_Backgroud_Modeling.ipynb` 실행

<br>

**1번 모델 재현**
- `python train.py` 실행
- `notebooks/(1-1)_Construct_Pseudo_Labeling.ipynb` 실행
- `notebooks/(1-2)_Pseudo_Labeling_Training.ipynb` 실행
- `notebooks/(1-3)_MultiScale_Inference.ipynb` 실행

<br>

**2번 모델 재현**
- `python train_dann.py` 실행

<br>

**3번 모델 재현**
- `segformers/config.py`에서 `config['n_epochs']`을 41로, `config['creterion']['dice']`를 `True`로 변경
- `python train.py` 실행
- `notebooks/(1-3)_MultiScale_Inference.ipynb` (`dir_save` 및 `path_ckpt` 인자 변경 필요)

<br>

**4번 모델 훈련 및 예측**
- `notebooks/(4-1)_Mask2Former.ipynb` 실행
- `notebooks/(4-2)_Inference_Mask2Former.ipynb` 실행

<br>

**앙상블 및 제출**
- `notebooks/(5-1)_AverageEnsemble.ipynb` 실행
- `notebooks/(5-2)_Enhancement_1.ipynb` 실행
- `notebooks/(5-2)_Enhancement_2.ipynb` 실행

<br>
