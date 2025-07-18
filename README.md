# 고양이 개체 추적 및 재식별 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 영상에서 다수의 고양이 객체를 탐지하고, 각 개체를 지속적으로 추적하며, 개체별로 고유 ID를 부여하여 재식별하는 시스템입니다. YOLOv11 모델을 사용하여 객체를 탐지하고, SORT 알고리즘을 dIoU(Distance-IoU) 기반으로 개선하여 추적 성능을 향상시켰습니다. 또한, Contrastive Learning 기반의 Re-ID(재식별) 모델을 통해 각 고양이에게 고유한 클래스 ID를 부여하고 관리합니다.

## 2. 주요 기능

- **객체 탐지**: `YOLOv11` 모델을 사용하여 영상 내 고양이와 사람을 탐지합니다.
- **객체 추적**: `SORT` 알고리즘을 기반으로 하되, dIoU와 유클리드 거리를 함께 사용하여 추적의 정확성과 안정성을 높였습니다.
- **객체 재식별 (Re-ID)**:
    - `DINOv2`를 백본으로 하는 `CatDiscriminationModel`을 통해 각 고양이의 특징(feature)을 추출합니다.
    - 사전에 계산된 클래스별 평균 특징점과의 거리를 비교하여 각 고양이에게 가장 가능성 높은 클래스 ID를 부여합니다.
    - Contrastive Loss와 Cross-Entropy Loss를 함께 사용하여 모델을 학습시켜, 유사한 개체는 가깝게, 다른 개체는 멀게 특징 공간에 임베딩합니다.
- **결과 시각화**: 추적된 객체의 경로, 바운딩 박스, 할당된 클래스 ID 및 신뢰도를 프레임별로 시각화하여 보여줍니다.
- **상세 분석 리포트**: 처리된 영상에 대한 상세 분석 결과를 JSON 파일과 시각화된 이미지(궤적, 통계)로 저장합니다.

## 3. 프로젝트 구조

```
.
├── config.py                       # 모델 및 학습 관련 주요 설정 관리
├── model.py                        # Re-ID 모델(CatDiscriminationModel) 정의
├── dataset.py                      # 데이터셋 및 데이터 로더 생성
├── train.py                        # Re-ID 모델 학습 및 검증 로직
├── run_training.py                 # Re-ID 모델 학습 실행 스크립트
├── cat_tracking_euclidean_modified.py # 고양이 추적 및 재식별 메인 실행 스크립트
├── requirements.txt                # 프로젝트 의존성 패키지 목록
├── origin_datasets/                # 원본 영상 데이터셋 디렉토리
└── sort_tracking_output/           # 추적 결과물이 저장되는 디렉토리
```

### 주요 파일 설명

- `config.py`: 학습률, 배치 사이즈, 모델 이름, 입출력 경로 등 프로젝트의 주요 하이퍼파라미터와 설정을 관리합니다.
- `model.py`: `DINOv2`를 백본으로 사용하여 고양이 특징을 추출하는 `CatDiscriminationModel`을 정의합니다. Contrastive Loss 계산 로직도 포함되어 있습니다.
- `train.py`: `model.py`와 `dataset.py`를 사용하여 Re-ID 모델의 학습 및 검증을 수행하는 메인 로직이 담겨있습니다.
- `run_training.py`: `train.py`를 실행하여 모델 학습을 시작하는 스크립트입니다.
- `cat_tracking_euclidean_modified.py`: YOLO 탐지, SORT 추적, Re-ID 분류 기능을 통합하여 동영상에서 고양이를 추적하고 재식별하는 핵심 스크립트입니다.

## 4. 설치 및 환경 설정

### 1. 저장소 복제
```bash
git clone https://github.com/[your-repository-url].git
cd cat_discrimination
```

### 2. 가상 환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows
```

### 3. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 사전 학습된 모델 준비
- **YOLO 모델**: `yolo11x.pt` 또는 다른 YOLO 모델을 프로젝트 루트에 다운로드합니다.
- **Re-ID 모델**: 학습된 `best_model.pth` 모델 가중치와 특징점 통계 파일(`detailed_results.json`)을 지정된 경로에 위치시킵니다. (필요시 `run_training.py`를 실행하여 직접 학습)

## 5. 사용 방법

### 1. Re-ID 모델 학습하기
`run_training.py`를 실행하여 `config.py`에 설정된 내용에 따라 Re-ID 모델 학습을 시작합니다.
```bash
python run_training.py
```
- 학습 과정은 TensorBoard를 통해 모니터링할 수 있습니다.
- 학습이 완료되면 `output` 디렉토리에 모델 가중치(`best_model.pth`)가 저장됩니다.

### 2. 동영상에서 고양이 추적 및 재식별 실행하기
`cat_tracking_euclidean_modified.py` 스크립트를 사용하여 동영상 처리를 실행합니다.

**기본 실행 (단일 영상, 실시간 표시)**
```bash
python cat_tracking_euclidean_modified.py --single_video "path/to/your/video.mp4"
```

**주요 실행 옵션:**
- `--yolo_model`: 사용할 YOLO 모델 경로 (기본값: `yolo11x.pt`)
- `--reid_model`: 학습된 Re-ID 모델 가중치 경로
- `--feature_stats`: 특징점 통계 파일 경로
- `--input_dir`: 처리할 영상들이 있는 디렉토리 (전체 영상 처리 시)
- `--output_dir`: 결과물을 저장할 디렉토리 (기본값: `sort_tracking_output`)
- `--save_video`: 처리된 영상을 저장할지 여부
- `--no_display`: 실시간 화면 표시 비활성화 (배치 처리 시 유용)
- `--confidence_threshold`: YOLO 객체 탐지 신뢰도 임계값 (기본값: 0.1)
- `--diou_threshold`: dIoU 매칭 임계값 (기본값: 0.3)
- `--max_age`: 객체 추적을 유지할 최대 프레임 수 (기본값: 30)
- `--reid_interval`: Re-ID를 재적용할 프레임 간격 (기본값: 10)
- `--save_detailed_visualization`: 상세 분석 시각화 자료 저장 여부

**결과물 저장 및 동시 처리 예시:**
```bash
python cat_tracking_euclidean_modified.py ^
    --input_dir "origin_datasets" ^
    --output_dir "sort_tracking_output" ^
    --save_video ^
    --save_detailed_visualization ^
    --no_display
```

## 6. 주요 의존성

- `torch`, `torchvision`
- `ultralytics` (YOLOv8, v11)
- `opencv-python`
- `numpy`
- `scipy`
- `tqdm`
- `tensorboard`
- `matplotlib`

자세한 내용은 `requirements.txt` 파일을 참고하세요.
