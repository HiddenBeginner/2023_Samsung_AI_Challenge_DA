# 2023 Samsung AI Challenge : Camera-Invariant Domain Adaptation
[2023 Samsung AI Challenge : Camera-Invariant Domain Adaptation](https://dacon.io/competitions/official/236132/overview/description)
대회에서 (1등 / 64팀)의 코드입니다.

## PSSC팀
포항공과대학교 응용수학 학생 학술단체인 POSTECH SIAM Student Chapter (PSSC)에서 2023년 2학기 활동의 일환으로 대회에 참여하였습니다.
- 허은우 (POSTECH 수학과, 팀장)
- 이성헌 (POSTECH 인공지능대학원)
- 이동진 (POSTECH 인공지능대학원)

## Our solution

### Main Model 
- SegFormer-b5 pretrained on CityScape dataset, which is implemented by [NVLabs](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024)
- Fine-tuning the network using both `train_source` and `val_source`
- AdamW optimizer
- One cycle LR scheduler
- Focal loss

### Data Augmentation
Please, see `segformers/transforms.py`. 

1. Fisheye transformation (with a probability of 0.75), which is implemented by 이성헌
2. Randomly scaling the image with ratio 0.5 ~ 1.0 (with a probability of 1.0)
3. Randomly cropping the image into (512, 512) size of image (with a probability of 1.0)
4. Horizontally flipping the image (with a probability of 0.5)
5. One of the following transformation
    - RGBShift
    - RandomBrightnessContrast
    - HueSaturationValue

### Inference
- Sliding window inference scheme with a window size of (512, 512)

### Background 탐지 모델
- OneFormer