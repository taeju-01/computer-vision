# YOLOv8 Hard Hat Detection (Computer Vision)

본 프로젝트는 작업 현장에서 작업자의 **안전모 착용 여부(helmet)** 를 탐지하기 위한 Object Detection 모델을 YOLOv8으로 학습하고 평가한 프로젝트입니다.  
Roboflow의 **Hard Hat Workers dataset**을 기반으로 학습을 수행하였으며, 학습 epoch 및 입력 해상도(imgsz)에 따른 성능 변화를 분석했습니다.

---

## 1. Dataset
- Dataset: **Hard Hat Workers Dataset (Roboflow)**
- Task: Object Detection
- Classes:
  - `head`
  - `helmet`
  - `person`

학습 과정에서 train 데이터를 일부 valid로 이동시켜 validation set을 별도로 구성했습니다.

---

## 2. Environment
- Python 3.x
- ultralytics == 8.0.196
- roboflow
- opencv-python
- pyyaml

설치:
```bash
pip install -r requirements.txt
```


---


## 3. Project Structure
```text
project3/
├─ src/
│  ├─ download_data.py
│  ├─ prepare_data.py
│  ├─ train.py
│  └─ predict.py
│
├─ weight/
│  ├─ best.pt
│  ├─ best_2.pt
│  ├─ best_3.pt
│  └─ best_4.pt
│
├─ result/
│  ├─ results.png
│  ├─ BoxF1_curve.png
│  ├─ BoxPR_curve.png
│  ├─ BoxP_curve.png
│  └─ BoxR_curve.png
│
├─ test_image/
│  ├─ train1.jpeg
│  ├─ train2.jpeg
│  └─ train3.jpeg
│
├─ data.yaml
├─ requirements.txt
└─ .gitignore
```

---
## 4. Training

YOLOv8 학습 명령:

yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640

---

## 5. Inference (Prediction)
yolo task=detect mode=predict model=weight/best.pt source=test_image conf=0.25 save=True

---

## 6. Experiments

아래 조건으로 학습을 진행하고 성능을 비교했습니다.

| Experiment | epochs | imgsz | 목적 |
|----------:|------:|------:|------|
| train 1 | 100 | 640 | 기본 성능 최대화 |
| train 2 | 10  | 320 | 빠른 학습 + 저해상도 성능 |
| train 3 | 10  | 640 | 빠른 학습 + 고해상도 성능 |
| train 4 | 20  | 640 | 중간 학습 epoch 비교 |


---

## 7. Limitations & Future Work

person 클래스의 성능이 매우 낮음
→ 데이터 편향 및 라벨 부족 영향 가능성이 큼

향후 개선 방향:

person 데이터 추가 확보

oversampling / class weight 적용

YOLOv8s 이상 모델로 성능 비교

augmentation 전략 강화

---

## 8. References

Ultralytics YOLOv8

Roboflow Hard Hat Workers Dataset
