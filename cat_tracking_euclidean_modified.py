import torch
import cv2
import numpy as np
import os
import glob
from collections import defaultdict, deque
import json
from datetime import datetime
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torchvision import transforms

from ultralytics import YOLO

# ReID 모델 관련 import
try:
    from model import CatDiscriminationModel
    from config import Config
    REID_AVAILABLE = True
except ImportError:
    print("ReID 모델 관련 파일을 찾을 수 없습니다. ReID 기능이 비활성화됩니다.")
    REID_AVAILABLE = False

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression 적용
    Args:
        boxes: 바운딩 박스 리스트 [[x1, y1, x2, y2], ...]
        scores: 신뢰도 점수 리스트
        iou_threshold: IoU 임계값
    Returns:
        nms_indices: NMS 후 남은 박스들의 인덱스
    """
    if len(boxes) == 0:
        return []
    
    # numpy 배열로 변환
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 좌표 추출
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # 면적 계산
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # 신뢰도 순으로 정렬 (내림차순)
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        # 가장 높은 신뢰도를 가진 박스 선택
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # 나머지 박스들과의 IoU 계산
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # IoU 계산
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # IoU 임계값보다 작은 박스들만 유지
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

class ReIDClassifier:
    """ReID 분류기 - 특징점 거리 기반 분류"""
    
    def __init__(self, config, model_path, feature_stats_path):
        if not REID_AVAILABLE:
            raise ImportError("ReID 모델이 사용할 수 없습니다.")
        
        self.config = config
        self.device = config.device
        
        # ReID 모델 로드
        self.reid_model = CatDiscriminationModel(config).to(self.device)
        self.reid_model.eval()
        
        # 학습된 가중치 로드
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.reid_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"ReID 모델 로드 완료: {model_path}")
        else:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        # 특징점 통계 로드
        self.load_feature_statistics(feature_stats_path)
        
        # 이미지 전처리 변환
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_feature_statistics(self, stats_path):
        """저장된 특징점 통계 로드"""
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"특징점 통계 파일을 찾을 수 없습니다: {stats_path}")
        
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
        
        # 클래스별 평균 특징점 로드
        self.class_means = {}
        for class_id_str, class_stats in stats_data['statistics'].items():
            class_id = int(class_id_str)
            self.class_means[class_id] = np.array(class_stats['mean'])
        
        print(f"클래스별 평균 특징점 로드 완료: {len(self.class_means)}개 클래스")
    
    def extract_features_from_bbox(self, frame, bbox):
        """바운딩 박스 영역에서 특징점 추출"""
        try:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 바운딩 박스 영역 추출
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            
            # BGR to RGB 변환
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # 이미지 전처리
            image_tensor = self.transform(crop)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # 특징점 추출
            with torch.no_grad():
                features = self.reid_model(image_tensor, return_features=True)
                features = features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            print(f"특징점 추출 중 오류 발생: {e}")
            return None
    
    def classify_bbox(self, frame, bbox, threshold=1.0):
        """바운딩 박스 분류"""
        # 특징점 추출
        features = self.extract_features_from_bbox(frame, bbox)
        if features is None:
            return None
        
        # 각 클래스와의 거리 계산
        distances = {}
        for class_id, mean_features in self.class_means.items():
            # 유클리드 거리 계산
            distance = np.linalg.norm(features - mean_features)
            distances[class_id] = distance
        
        # 가장 가까운 클래스 찾기
        min_distance = min(distances.values())
        predicted_class = min(distances.keys(), key=lambda x: distances[x])
        
        # 임계값 적용
        if min_distance > threshold:
            predicted_class = 0  # 알 수 없는 클래스
            confidence = 0.0
        else:
            # 거리를 신뢰도로 변환 (거리가 가까울수록 높은 신뢰도)
            confidence = max(0.0, 1.0 - min_distance / threshold)
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'distances': distances,
            'min_distance': min_distance,
            'threshold': threshold
        }
        
        return result

class Track:
    """개별 추적 객체 (칼만 필터 없이)"""
    
    def __init__(self, detection, track_id):
        self.track_id = track_id
        
        # 초기 위치 설정
        center = self.calculate_center(detection)
        
        # 추적 정보
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        
        # 바운딩 박스 히스토리
        self.bbox_history = [detection]
        self.center_history = [center]
        
        # 현재 위치 (측정값 기반)
        self.current_center = center
        self.current_bbox = detection
        
        # 상태
        self.is_confirmed = False
        self.is_deleted = False
        
        # ReID 정보
        self.reid_class = None
        self.reid_confidence = 0.0
        self.last_reid_frame = 0
        
        # 고양이 클래스별 count 추가
        self.cat_class_counts = {}  # {class_id: count}
        self.dominant_cat_class = None  # 가장 많이 감지된 고양이 클래스
    
    def calculate_center(self, bbox):
        """바운딩 박스의 중심점 계산"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return np.array([center_x, center_y])
    
    def predict(self):
        """다음 위치 예측 (측정값 기반)"""
        # 칼만 필터 대신 최근 위치들의 평균 사용
        if len(self.center_history) >= 3:
            # 최근 3개 위치의 평균으로 예측
            recent_centers = self.center_history[-3:]
            predicted_center = np.mean(recent_centers, axis=0)
        else:
            # 히스토리가 부족하면 현재 위치 사용
            predicted_center = self.current_center
        
        self.age += 1
        self.time_since_update += 1
        return predicted_center
    
    def update(self, detection):
        """새로운 감지로 업데이트"""
        center = self.calculate_center(detection)
        
        # 현재 위치 업데이트
        self.current_center = center
        self.current_bbox = detection
        
        self.hits += 1
        self.time_since_update = 0
        self.bbox_history.append(detection)
        self.center_history.append(center)
        
        # 히스토리 길이 제한
        if len(self.bbox_history) > 30:
            self.bbox_history = self.bbox_history[-20:]
            self.center_history = self.center_history[-20:]
        
        # 확인된 track으로 설정
        if self.hits >= 3:
            self.is_confirmed = True
    
    def update_reid(self, reid_class, reid_confidence, frame_id):
        """ReID 정보 업데이트"""
        self.reid_class = reid_class
        self.reid_confidence = reid_confidence
        self.last_reid_frame = frame_id
        
        # 고양이 클래스 count 업데이트
        if reid_class is not None and reid_class > 0:
            if reid_class not in self.cat_class_counts:
                self.cat_class_counts[reid_class] = 0
            self.cat_class_counts[reid_class] += 1
            
            # 가장 많이 감지된 고양이 클래스 업데이트
            self.update_dominant_cat_class()
    
    def update_dominant_cat_class(self):
        """가장 많이 감지된 고양이 클래스 업데이트"""
        if self.cat_class_counts:
            self.dominant_cat_class = max(self.cat_class_counts.keys(), 
                                        key=lambda x: self.cat_class_counts[x])
        else:
            self.dominant_cat_class = None
    
    def get_dominant_cat_info(self):
        """가장 많이 감지된 고양이 클래스 정보 반환"""
        if self.dominant_cat_class is not None:
            total_detections = sum(self.cat_class_counts.values())
            dominant_count = self.cat_class_counts[self.dominant_cat_class]
            probability = (dominant_count / total_detections) * 100  # 백분율로 변환
            
            return {
                'class_id': self.dominant_cat_class,
                'count': dominant_count,
                'total_detections': total_detections,
                'probability': probability  # 확률 추가
            }
        return None
    
    def get_state(self):
        """현재 상태 반환 (측정값 기반)"""
        return self.current_center
    
    def get_latest_bbox(self):
        """최신 바운딩 박스 반환"""
        return self.current_bbox

def convert_numpy_types(obj):
    """numpy 타입을 Python 기본 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class SORTCatTracker:
    """SORT 알고리즘 기반 Euclidean Distance 고양이 추적 시스템 (측정값 기반)"""
    
    def __init__(self, yolo_model_path="yolo11s.pt", reid_model_path=None, feature_stats_path=None):
        # YOLO 모델 로드
        self.yolo_model = YOLO(yolo_model_path)
        print(f"YOLO 모델 로드 완료: {yolo_model_path}")
        print(f"사용 가능한 클래스: {self.yolo_model.names}")
        
        # 고양이 클래스 찾기
        self.cat_class_id = None
        for class_id, class_name in self.yolo_model.names.items():
            if 'cat' in class_name.lower():
                self.cat_class_id = class_id
                print(f"고양이 클래스 발견: ID={class_id}, 이름='{class_name}'")
                break
        
        if self.cat_class_id is None:
            print("고양이 클래스를 찾을 수 없습니다. 기본값으로 클래스 15를 사용합니다.")
            self.cat_class_id = 15
        
        # 사람 클래스 찾기
        self.person_class_id = None
        for class_id, class_name in self.yolo_model.names.items():
            if 'person' in class_name.lower():
                self.person_class_id = class_id
                print(f"사람 클래스 발견: ID={class_id}, 이름='{class_name}'")
                break
        
        if self.person_class_id is None:
            print("사람 클래스를 찾을 수 없습니다. 기본값으로 클래스 0을 사용합니다.")
            self.person_class_id = 0
        
        # ReID 분류기 초기화
        self.reid_classifier = None
        if reid_model_path and feature_stats_path and REID_AVAILABLE:
            try:
                config = Config()
                self.reid_classifier = ReIDClassifier(config, reid_model_path, feature_stats_path)
                print("ReID 분류기 초기화 완료")
            except Exception as e:
                print(f"ReID 분류기 초기화 실패: {e}")
                self.reid_classifier = None
        
        # SORT 설정
        self.tracks = []  # 활성 track 목록
        self.next_track_id = 0
        self.frame_count = 0
        
        # Track 관리 파라미터
        self.max_age = 30  # 최대 추적 유지 프레임 수
        self.min_hits = 3  # track 확인을 위한 최소 히트 수
        self.distance_threshold = 100  # Euclidean distance 임계값 (픽셀 단위) - 기존 호환성용
        self.confidence_threshold = 0.3  # YOLO 감지 신뢰도 임계값
        
        # dIoU 설정
        self.diou_threshold = 0.3  # dIoU 임계값 (0.3 이상이면 좋은 매칭)
        self.use_diou = True  # dIoU 사용 여부
        
        # NMS 설정
        self.nms_iou_threshold = 0.5  # NMS IoU 임계값
        
        # ReID 설정
        self.reid_interval = 10  # ReID 적용 간격 (프레임)
        self.reid_threshold = 1.0  # ReID 임계값
        
        # 고양이 클래스 할당 추적
        self.cat_class_assignments = {}  # {cat_class_id: track_id}
        self.track_cat_assignments = {}  # {track_id: cat_class_id}
        
        # 검출 결과 저장용
        self.cat_detections = []  # 고양이 검출 결과
        self.person_detections = []  # 사람 검출 결과
        self.all_detections = []  # 모든 검출 결과 (고양이 + 사람)
        
        # 통일된 고양이 클래스 색상 매핑 (BGR 형식 - OpenCV용)
        self.cat_colors_bgr = {
            1: (0, 255, 255),      # Cyan (밝은 청록) - Cat 1
            2: (255, 0, 255),      # Magenta (밝은 자홍) - Cat 2
            3: (0, 255, 0),        # Lime (밝은 초록) - Cat 3
            4: (255, 128, 0),      # Orange (밝은 주황) - Cat 4
            5: (128, 0, 255),      # Purple (밝은 보라) - Cat 5
            6: (0, 128, 255),      # Sky Blue (밝은 하늘) - Cat 6
            7: (255, 0, 128),      # Pink (밝은 분홍) - Cat 7
            8: (128, 255, 0),      # Light Green (밝은 연두) - Cat 8
            9: (255, 128, 128),    # Light Red - Cat 9
            10: (128, 255, 128),   # Light Green - Cat 10
            11: (128, 128, 255),   # Light Blue - Cat 11
            12: (255, 255, 0),     # Yellow (밝은 노랑) - Cat 12
        }
        
        # 통일된 고양이 클래스 색상 매핑 (HEX 형식 - matplotlib용)
        self.cat_colors_hex = {
            1: '#00FFFF',  # Cyan
            2: '#FF00FF',  # Magenta
            3: '#00FF00',  # Lime
            4: '#FF8000',  # Orange
            5: '#8000FF',  # Purple
            6: '#0080FF',  # Sky Blue
            7: '#FF0080',  # Pink
            8: '#80FF00',  # Light Green
            9: '#FF8080',  # Light Red
            10: '#80FF80', # Light Green
            11: '#8080FF', # Light Blue
            12: '#FFFF00', # Yellow
        }

    def get_cat_color_bgr(self, cat_class_id):
        """고양이 클래스 ID에 따른 BGR 색상 반환 (OpenCV용)"""
        if cat_class_id is None or cat_class_id <= 0:
            return (192, 192, 192)  # 할당되지 않은 경우 회색
        return self.cat_colors_bgr.get(cat_class_id, (192, 192, 192))
    
    def get_cat_color_hex(self, cat_class_id):
        """고양이 클래스 ID에 따른 HEX 색상 반환 (matplotlib용)"""
        if cat_class_id is None or cat_class_id <= 0:
            return '#C0C0C0'  # 할당되지 않은 경우 회색
        return self.cat_colors_hex.get(cat_class_id, '#C0C0C0')

    def get_cat_class_assignment_status(self, cat_class_id):
        """고양이 클래스 할당 상태 확인"""
        return cat_class_id in self.cat_class_assignments
    
    def assign_cat_class_to_track(self, track_id, cat_class_id):
        """고양이 클래스를 track에 할당"""
        # 기존 할당 해제
        if track_id in self.track_cat_assignments:
            old_cat_class = self.track_cat_assignments[track_id]
            if old_cat_class in self.cat_class_assignments:
                del self.cat_class_assignments[old_cat_class]
        
        # 새로운 할당
        if cat_class_id is not None and cat_class_id > 0:
            self.cat_class_assignments[cat_class_id] = track_id
            self.track_cat_assignments[track_id] = cat_class_id
        else:
            # Unknown으로 설정
            if track_id in self.track_cat_assignments:
                del self.track_cat_assignments[track_id]
    
    def resolve_cat_class_conflicts(self):
        """고양이 클래스 충돌 해결 (확률이 높은 track 우선, 나머지는 count reset)"""
        # 각 고양이 클래스별로 가장 높은 확률을 가진 track 찾기
        cat_class_candidates = {}  # {cat_class_id: [(track_id, probability), ...]}
        
        for track in self.tracks:
            if not track.is_deleted and track.is_confirmed:
                dominant_info = track.get_dominant_cat_info()
                if dominant_info and dominant_info['class_id'] > 0:
                    cat_class = dominant_info['class_id']
                    probability = dominant_info['probability']
                    
                    if cat_class not in cat_class_candidates:
                        cat_class_candidates[cat_class] = []
                    cat_class_candidates[cat_class].append((track.track_id, probability))
        
        # 각 고양이 클래스에 대해 가장 높은 확률의 track만 할당
        new_assignments = {}
        for cat_class, candidates in cat_class_candidates.items():
            # 확률 순으로 정렬하여 가장 높은 확률의 track 선택
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_track_id, best_probability = candidates[0]
            new_assignments[cat_class] = best_track_id
        
        # 기존 할당 초기화
        self.cat_class_assignments.clear()
        self.track_cat_assignments.clear()
        
        # 새로운 할당 적용
        for cat_class, track_id in new_assignments.items():
            self.cat_class_assignments[cat_class] = track_id
            self.track_cat_assignments[track_id] = cat_class
        
        # 할당되지 않은 track들의 충돌하는 고양이 클래스 count reset
        for track in self.tracks:
            if not track.is_deleted and track.is_confirmed:
                if track.track_id not in self.track_cat_assignments:
                    # 해당 track의 dominant cat class가 이미 다른 track에 할당되어 있으면 count reset
                    dominant_info = track.get_dominant_cat_info()
                    if dominant_info and dominant_info['class_id'] > 0:
                        cat_class = dominant_info['class_id']
                        if cat_class in self.cat_class_assignments and self.cat_class_assignments[cat_class] != track.track_id:
                            # 이미 다른 track에 할당되어 있으면 해당 고양이 클래스 count reset
                            if cat_class in track.cat_class_counts:
                                del track.cat_class_counts[cat_class]
                            
                            # dominant cat class 재계산
                            track.update_dominant_cat_class()
                            
                            # ReID 정보도 초기화
                            track.reid_class = None
                            track.reid_confidence = 0.0
    
    def calculate_center(self, bbox):
        """바운딩 박스의 중심점 계산"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return np.array([center_x, center_y])
    
    def calculate_iou(self, bbox1, bbox2):
        """두 바운딩 박스 간의 IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합 영역 계산
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 합집합 영역 계산
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_diou(self, bbox1, bbox2):
        """두 바운딩 박스 간의 dIoU(Distance-IoU) 계산"""
        # IoU 계산
        iou = self.calculate_iou(bbox1, bbox2)
        
        # 중심점 계산
        center1 = self.calculate_center(bbox1)
        center2 = self.calculate_center(bbox2)
        
        # 중심점 거리 계산
        center_distance = np.linalg.norm(center1 - center2)
        
        # 외접 직사각형 계산 (enclosing box)
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_enclosing = min(x1_1, x1_2)
        y1_enclosing = min(y1_1, y1_2)
        x2_enclosing = max(x2_1, x2_2)
        y2_enclosing = max(y2_1, y2_2)
        
        # 외접 직사각형의 대각선 길이
        c = np.sqrt((x2_enclosing - x1_enclosing) ** 2 + (y2_enclosing - y1_enclosing) ** 2)
        
        # dIoU 계산: IoU - (중심점 거리^2 / 외접 직사각형 대각선 길이^2)
        if c > 0:
            diou = iou - (center_distance ** 2) / (c ** 2)
        else:
            diou = iou
        
        return diou
    
    def calculate_diou_matrix(self, detections, tracks):
        """감지와 track 간의 dIoU 행렬 계산"""
        if len(detections) == 0 or len(tracks) == 0:
            return np.array([])
        
        diou_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, detection in enumerate(detections):
            for j, track in enumerate(tracks):
                if track.is_deleted:
                    diou_matrix[i, j] = -1.0  # dIoU는 -1 ~ 1 범위이므로 -1로 설정
                    continue
                
                # track의 예측 바운딩 박스 (최근 바운딩 박스 사용)
                track_bbox = track.get_latest_bbox()
                
                # dIoU 계산 (높을수록 좋은 매칭이므로 음수로 변환)
                diou = self.calculate_diou(detection, track_bbox)
                diou_matrix[i, j] = -diou  # 헝가리안 알고리즘은 최소화하므로 음수로 변환
        
        return diou_matrix
    
    def calculate_distance_matrix(self, detections, tracks):
        """감지와 track 간의 Euclidean distance 행렬 계산 (측정값 기반) - 기존 메서드 유지"""
        if len(detections) == 0 or len(tracks) == 0:
            return np.array([])
        
        distance_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, detection in enumerate(detections):
            detection_center = self.calculate_center(detection)
            
            for j, track in enumerate(tracks):
                if track.is_deleted:
                    distance_matrix[i, j] = float('inf')
                    continue
                
                # 측정값 기반 예측 위치 (최근 위치들의 평균)
                predicted_center = track.predict()
                
                # Euclidean distance 계산
                distance = np.linalg.norm(detection_center - predicted_center)
                distance_matrix[i, j] = distance
        
        return distance_matrix
    
    def associate_detections_to_tracks(self, detections, tracks):
        """헝가리안 알고리즘으로 감지와 track 매칭 (dIoU 사용)"""
        if len(detections) == 0:
            return [], [], [], list(range(len(tracks)))  # 4개 값 반환
        
        if len(tracks) == 0:
            return [], [], list(range(len(detections))), []  # 4개 값 반환
        
        # dIoU 행렬 계산
        diou_matrix = self.calculate_diou_matrix(detections, tracks)
        
        # 헝가리안 알고리즘으로 최적 할당 (dIoU 최대화)
        detection_indices, track_indices = linear_sum_assignment(diou_matrix)
        
        # 임계값을 초과하는 할당 제거
        matched_detections = []
        matched_tracks = []
        unmatched_detections = []
        unmatched_tracks = []
        
        # 매칭된 쌍들 확인
        for det_idx, track_idx in zip(detection_indices, track_indices):
            diou_score = -diou_matrix[det_idx, track_idx]  # 음수로 변환했으므로 다시 양수로
            
            # dIoU 임계값 확인 (0.3 이상이면 좋은 매칭)
            if diou_score >= 0.3:
                matched_detections.append(det_idx)
                matched_tracks.append(track_idx)
            else:
                unmatched_detections.append(det_idx)
                unmatched_tracks.append(track_idx)
        
        # 매칭되지 않은 감지들
        for i in range(len(detections)):
            if i not in matched_detections:
                unmatched_detections.append(i)
        
        # 매칭되지 않은 track들
        for i in range(len(tracks)):
            if i not in matched_tracks:
                unmatched_tracks.append(i)
        
        return matched_detections, matched_tracks, unmatched_detections, unmatched_tracks
    
    def apply_reid_to_tracks(self, frame, active_tracks):
        """활성 track들에 ReID 적용"""
        if self.reid_classifier is None:
            return
        
        for track in active_tracks:
            track_obj = None
            # track 객체 찾기
            for t in self.tracks:
                if t.track_id == track['track_id']:
                    track_obj = t
                    break
            
            if track_obj is None:
                continue
            
            # ReID 간격 확인
            if self.frame_count - track_obj.last_reid_frame >= self.reid_interval:
                # ReID 적용
                reid_result = self.reid_classifier.classify_bbox(
                    frame, track['bbox'], self.reid_threshold
                )
                
                if reid_result:
                    track_obj.update_reid(
                        reid_result['predicted_class'],
                        reid_result['confidence'],
                        self.frame_count
                    )
                    
                    # track 정보 업데이트
                    track['reid_class'] = reid_result['predicted_class']
                    track['reid_confidence'] = reid_result['confidence']
                    track['reid_distance'] = reid_result['min_distance']
    
    def update_tracks(self, detections, frame_width, frame_height):
        """track 업데이트"""
        self.frame_count += 1
        
        # 1. 모든 track에 대해 예측 (측정값 기반)
        for track in self.tracks:
            if not track.is_deleted:
                track.predict()
        
        # 2. 감지와 track 매칭
        matched_detections, matched_tracks, unmatched_detections, unmatched_tracks = \
            self.associate_detections_to_tracks(detections, self.tracks)
        
        # 3. 매칭된 track 업데이트
        for det_idx, track_idx in zip(matched_detections, matched_tracks):
            self.tracks[track_idx].update(detections[det_idx])
        
        # 4. 매칭되지 않은 감지로 새로운 track 생성
        for det_idx in unmatched_detections:
            new_track = Track(detections[det_idx], self.next_track_id)
            self.tracks.append(new_track)
            self.next_track_id += 1
        
        # 5. 오래된 track 정리
        tracks_to_remove = []
        for i, track in enumerate(self.tracks):
            if track.is_deleted:
                continue
            
            # 오래된 track 삭제
            if track.time_since_update > self.max_age:
                track.is_deleted = True
                tracks_to_remove.append(i)
        
        # 삭제된 track 제거
        self.tracks = [track for i, track in enumerate(self.tracks) if i not in tracks_to_remove]
        
        # 6. 고양이 클래스 충돌 해결
        # self.resolve_cat_class_conflicts()
        
        # 결과 반환 (매칭 정보 포함)
        active_tracks = []
        for track in self.tracks:
            if not track.is_deleted and track.is_confirmed:
                # 가장 많이 감지된 고양이 클래스 정보 가져오기
                dominant_info = track.get_dominant_cat_info()
                
                # 할당된 고양이 클래스 확인
                assigned_cat_class = self.track_cat_assignments.get(track.track_id)
                
                track_info = {
                    'track_id': track.track_id,
                    'bbox': track.get_latest_bbox(),
                    'center': track.get_state().tolist(),
                    'hits': track.hits,
                    'age': track.age,
                    'time_since_update': track.time_since_update,
                    'reid_class': track.reid_class,
                    'reid_confidence': track.reid_confidence,
                    'cat_class_counts': track.cat_class_counts.copy(),
                    'dominant_cat_class': track.dominant_cat_class,
                    'assigned_cat_class': assigned_cat_class
                }
                
                # 가장 많이 감지된 고양이 클래스 정보 추가
                if dominant_info:
                    track_info['dominant_cat_info'] = dominant_info
                
                active_tracks.append(track_info)
        
        return active_tracks, matched_detections, matched_tracks, unmatched_detections
    
    def reset_tracker(self):
        """tracker 상태 초기화"""
        self.tracks = []
        self.next_track_id = 0
        self.frame_count = 0
        self.cat_class_assignments.clear()
        self.track_cat_assignments.clear()
        print("Tracker가 초기화되었습니다.")
    
    def save_results(self, results, output_file):
        """결과를 JSON 파일로 저장"""
        # numpy 타입을 Python 기본 타입으로 변환
        serializable_results = convert_numpy_types(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"결과 저장 완료: {output_file}")
    
    def save_track_trajectories(self, results, output_dir):
        """Track 궤적 이미지 저장 (2D + 3D 시간 순서 표시 포함)"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.patches import FancyArrowPatch
            import matplotlib.colors as mcolors
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("matplotlib이 설치되지 않아 궤적 이미지를 저장할 수 없습니다.")
            return
        
        # 비디오 정보 가져오기
        video_path = results['video_path']
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 비디오 해상도 추정 (첫 번째 track의 바운딩 박스로부터)
        max_width, max_height = 1920, 1080  # 기본값
        if results['tracks']:
            for track_id, track_info in results['tracks'].items():
                for detection in track_info['detections']:
                    bbox = detection['bbox']
                    max_width = max(max_width, bbox[2])
                    max_height = max(max_height, bbox[3])
        
        # 1. 2D 궤적 시각화
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, max_width)
        ax.set_ylim(max_height, 0)  # OpenCV 좌표계 (y축 반전)
        
        # 각 track의 궤적 그리기
        for i, (track_id, track_info) in enumerate(results['tracks'].items()):
            # ReID 정보에서 dominant class 찾기
            reid_history = track_info.get('reid_history', [])
            class_assignments = {}
            
            for det in track_info['detections']:
                frame = det['frame']
                for reid_entry in reid_history:
                    if reid_entry['frame'] == frame:
                        class_assignments[frame] = reid_entry['reid_class']
                        break
            
            # 가장 많이 할당된 클래스 찾기
            if class_assignments:
                class_counts = {}
                for class_id in class_assignments.values():
                    if class_id > 0:
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                if class_counts:
                    dominant_class = max(class_counts.keys(), key=lambda x: class_counts[x])
                    base_color = self.get_cat_color_hex(dominant_class)
                else:
                    base_color = '#C0C0C0'  # 회색
            else:
                base_color = '#C0C0C0'  # 회색
            
            # 궤적 점들 (시간 순서대로)
            x_coords = [det['center'][0] for det in track_info['detections']]
            y_coords = [det['center'][1] for det in track_info['detections']]
            frames = [det['frame'] for det in track_info['detections']]
            
            if len(x_coords) < 2:
                continue
            
            # 1. 시간에 따른 색상 그라데이션으로 궤적 그리기
            colors = plt.cm.viridis(np.linspace(0, 1, len(x_coords)))
            for j in range(len(x_coords) - 1):
                ax.plot(x_coords[j:j+2], y_coords[j:j+2], 
                       color=base_color, linewidth=3, alpha=0.8)
            
            # 2. 화살표로 진행 방향 표시 (일정 간격으로)
            arrow_interval = max(1, len(x_coords) // 5)  # 5개 화살표 정도
            for j in range(0, len(x_coords) - 1, arrow_interval):
                if j + 1 < len(x_coords):
                    # 화살표 시작점과 끝점
                    start_x, start_y = x_coords[j], y_coords[j]
                    end_x, end_y = x_coords[j + 1], y_coords[j + 1]
                    
                    # 화살표 그리기
                    arrow = FancyArrowPatch(
                        (start_x, start_y), (end_x, end_y),
                        arrowstyle='->', mutation_scale=20,
                        color=base_color, linewidth=2, alpha=0.9
                    )
                    ax.add_patch(arrow)
            
            # 3. 시작점과 끝점 명확히 표시
            if x_coords:
                # 시작점 (녹색 원)
                ax.scatter(x_coords[0], y_coords[0], 
                          color='green', s=200, marker='o', 
                          edgecolors='black', linewidth=3, zorder=5,
                          label=f'Start {track_id}' if i == 0 else "")
                
                # 끝점 (빨간 사각형)
                ax.scatter(x_coords[-1], y_coords[-1], 
                          color='red', s=200, marker='s', 
                          edgecolors='black', linewidth=3, zorder=5,
                          label=f'End {track_id}' if i == 0 else "")
                
                # 시작점과 끝점에 프레임 번호 표시
                ax.annotate(f'F{frames[0]}', (x_coords[0], y_coords[0]), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, fontweight='bold', color='green',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                ax.annotate(f'F{frames[-1]}', (x_coords[-1], y_coords[-1]), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, fontweight='bold', color='red',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # 4. 중간 지점들에 시간 정보 표시 (일정 간격으로)
            time_interval = max(1, len(x_coords) // 3)  # 3개 지점 정도
            for j in range(time_interval, len(x_coords) - time_interval, time_interval):
                if j < len(x_coords):
                    ax.annotate(f'F{frames[j]}', (x_coords[j], y_coords[j]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=6, color='black',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
        
        # 범례 추가
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Start Point'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=10, label='End Point'),
            plt.Line2D([0], [0], color='black', linewidth=2, label='Trajectory Path'),
            plt.Line2D([0], [0], marker='>', color='black', linewidth=2, 
                      label='Movement Direction')
        ]
        
        ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
        ax.set_title(f'{video_name} - 2D Track Trajectories with Time Order', fontsize=14, fontweight='bold')
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 2D 이미지 저장
        trajectory_2d_file = os.path.join(output_dir, f'{video_name}_trajectories_2d.png')
        plt.tight_layout()
        plt.savefig(trajectory_2d_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 3D 궤적 시각화 (z축 = 프레임)
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 각 track의 3D 궤적 그리기
        for i, (track_id, track_info) in enumerate(results['tracks'].items()):
            # ReID 정보에서 dominant class 찾기
            reid_history = track_info.get('reid_history', [])
            class_assignments = {}
            
            for det in track_info['detections']:
                frame = det['frame']
                for reid_entry in reid_history:
                    if reid_entry['frame'] == frame:
                        class_assignments[frame] = reid_entry['reid_class']
                        break
            
            # 가장 많이 할당된 클래스 찾기
            if class_assignments:
                class_counts = {}
                for class_id in class_assignments.values():
                    if class_id > 0:
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                if class_counts:
                    dominant_class = max(class_counts.keys(), key=lambda x: class_counts[x])
                    base_color = self.get_cat_color_hex(dominant_class)
                else:
                    base_color = '#C0C0C0'  # 회색
            else:
                base_color = '#C0C0C0'  # 회색
            
            # 3D 궤적 점들
            x_coords = [det['center'][0] for det in track_info['detections']]
            y_coords = [det['center'][1] for det in track_info['detections']]
            z_coords = [det['frame'] for det in track_info['detections']]  # z축 = 프레임
            
            if len(x_coords) < 2:
                continue
            
            # 3D 궤적 선 그리기
            ax.plot(x_coords, y_coords, z_coords, 
                   color=base_color, linewidth=3, alpha=0.8,
                   label=f'Track {track_id} (Cat {dominant_class})' if dominant_class else f'Track {track_id} (Unknown)')
            
            # 3D 시작점과 끝점 표시
            if x_coords:
                # 시작점 (녹색 원)
                ax.scatter(x_coords[0], y_coords[0], z_coords[0], 
                          color='green', s=200, marker='o', 
                          edgecolors='black', linewidth=2, zorder=5)
                
                # 끝점 (빨간 사각형)
                ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                          color='red', s=200, marker='s', 
                          edgecolors='black', linewidth=2, zorder=5)
                
                # 시작점과 끝점에 프레임 번호 표시
                ax.text(x_coords[0], y_coords[0], z_coords[0], f'F{z_coords[0]}', 
                       fontsize=8, fontweight='bold', color='green',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                ax.text(x_coords[-1], y_coords[-1], z_coords[-1], f'F{z_coords[-1]}', 
                       fontsize=8, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # 3D 중간 지점들에 시간 정보 표시 (일정 간격으로)
            time_interval = max(1, len(x_coords) // 4)  # 4개 지점 정도
            for j in range(time_interval, len(x_coords) - time_interval, time_interval):
                if j < len(x_coords):
                    ax.text(x_coords[j], y_coords[j], z_coords[j], f'F{z_coords[j]}', 
                           fontsize=6, color='black',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
        
        # 3D 그래프 설정
        ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
        ax.set_zlabel('Frame Number', fontsize=12)
        ax.set_title(f'{video_name} - 3D Track Trajectories (Z-axis = Time)', fontsize=14, fontweight='bold')
        
        # 3D 범례 추가
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Start Point'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=10, label='End Point'),
            plt.Line2D([0], [0], color='black', linewidth=2, label='3D Trajectory Path')
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # 3D 뷰 설정
        ax.view_init(elev=20, azim=45)  # 시점 조정
        ax.grid(True, alpha=0.3)
        
        # 3D 이미지 저장
        trajectory_3d_file = os.path.join(output_dir, f'{video_name}_trajectories_3d.png')
        plt.tight_layout()
        plt.savefig(trajectory_3d_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 3D 궤적의 여러 각도에서 보기
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        
        for angle_idx, angle in enumerate(angles):
            ax = axes[angle_idx]
            
            # 각 track의 3D 궤적 그리기
            for track_id, track_info in results['tracks'].items():
                # ReID 정보에서 dominant class 찾기
                reid_history = track_info.get('reid_history', [])
                class_assignments = {}
                
                for det in track_info['detections']:
                    frame = det['frame']
                    for reid_entry in reid_history:
                        if reid_entry['frame'] == frame:
                            class_assignments[frame] = reid_entry['reid_class']
                            break
                
                # 가장 많이 할당된 클래스 찾기
                if class_assignments:
                    class_counts = {}
                    for class_id in class_assignments.values():
                        if class_id > 0:
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    if class_counts:
                        dominant_class = max(class_counts.keys(), key=lambda x: class_counts[x])
                        base_color = self.get_cat_color_hex(dominant_class)
                    else:
                        base_color = '#C0C0C0'  # 회색
                else:
                    base_color = '#C0C0C0'  # 회색
                
                # 3D 궤적 점들
                x_coords = [det['center'][0] for det in track_info['detections']]
                y_coords = [det['center'][1] for det in track_info['detections']]
                z_coords = [det['frame'] for det in track_info['detections']]  # z축 = 프레임
                
                if len(x_coords) < 2:
                    continue
                
                # 3D 궤적 선 그리기
                ax.plot(x_coords, y_coords, z_coords, 
                       color=base_color, linewidth=2, alpha=0.8)
                
                # 시작점과 끝점 표시
                if x_coords:
                    ax.scatter(x_coords[0], y_coords[0], z_coords[0], 
                              color='green', s=100, marker='o', alpha=0.8)
                    ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                              color='red', s=100, marker='s', alpha=0.8)
            
            # 각도별 뷰 설정
            ax.view_init(elev=20, azim=angle)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Frame')
            ax.set_title(f'Angle: {angle}°')
            ax.grid(True, alpha=0.3)
        
        # 다중 각도 3D 이미지 저장
        trajectory_3d_multi_file = os.path.join(output_dir, f'{video_name}_trajectories_3d_multi_angle.png')
        plt.tight_layout()
        plt.savefig(trajectory_3d_multi_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Trajectory images saved:")
        print(f"  - 2D trajectory: {trajectory_2d_file}")
        print(f"  - 3D trajectory: {trajectory_3d_file}")
        print(f"  - 3D multi-angle: {trajectory_3d_multi_file}")

    def save_detailed_track_visualization(self, results, output_dir):
        """Detailed visualization of cat trajectories and classes over time on empty background"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.colors import ListedColormap
            import matplotlib.dates as mdates
            from datetime import datetime, timedelta
            from matplotlib.patches import FancyArrowPatch
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("matplotlib is not installed. Cannot save detailed visualization.")
            return
        
        # Get video information
        video_path = results['video_path']
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        fps = results.get('fps', 30)
        
        # Estimate video resolution
        max_width, max_height = 1920, 1080  # Default values
        if results['tracks']:
            for track_id, track_info in results['tracks'].items():
                for detection in track_info['detections']:
                    bbox = detection['bbox']
                    max_width = max(max_width, bbox[2])
                    max_height = max(max_height, bbox[3])
        
        # 1. 2D + 3D Time-trajectory visualization
        fig = plt.figure(figsize=(20, 12))
        
        # 2D subplot (left)
        ax1 = fig.add_subplot(121)
        ax1.set_xlim(0, max_width)
        ax1.set_ylim(max_height, 0)  # OpenCV coordinate system (y-axis inverted)
        
        # 3D subplot (right)
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Draw trajectory for each track
        track_class_data = {}  # Data for JSON storage
        
        for track_id, track_info in results['tracks'].items():
            # Trajectory points
            x_coords = [det['center'][0] for det in track_info['detections']]
            y_coords = [det['center'][1] for det in track_info['detections']]
            frames = [det['frame'] for det in track_info['detections']]
            
            # Collect ReID information
            reid_history = track_info.get('reid_history', [])
            class_assignments = {}
            
            # Collect class assignment information for each frame
            for det in track_info['detections']:
                frame = det['frame']
                # Find class for this frame from ReID history
                class_at_frame = None
                for reid_entry in reid_history:
                    if reid_entry['frame'] == frame:
                        class_at_frame = reid_entry['reid_class']
                        break
                
                if class_at_frame and class_at_frame > 0:
                    class_assignments[frame] = class_at_frame
            
            # Find most frequently assigned class
            if class_assignments:
                class_counts = {}
                for class_id in class_assignments.values():
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                dominant_class = max(class_counts.keys(), key=lambda x: class_counts[x])
                color = self.get_cat_color_hex(dominant_class)
            else:
                dominant_class = None
                color = '#C0C0C0'
            
            # Draw 2D spatial trajectory (left)
            if len(x_coords) >= 2:
                ax1.plot(x_coords, y_coords, color=color, linewidth=3, alpha=0.8, 
                        label=f'Track {track_id} (Cat {dominant_class})' if dominant_class else f'Track {track_id} (Unknown)')
                
                # 화살표로 진행 방향 표시
                arrow_interval = max(1, len(x_coords) // 4)
                for j in range(0, len(x_coords) - 1, arrow_interval):
                    if j + 1 < len(x_coords):
                        start_x, start_y = x_coords[j], y_coords[j]
                        end_x, end_y = x_coords[j + 1], y_coords[j + 1]
                        
                        arrow = FancyArrowPatch(
                            (start_x, start_y), (end_x, end_y),
                            arrowstyle='->', mutation_scale=15,
                            color=color, linewidth=2, alpha=0.9
                        )
                        ax1.add_patch(arrow)
            
            # Draw 3D spatial trajectory (right)
            if len(x_coords) >= 2:
                ax2.plot(x_coords, y_coords, frames, color=color, linewidth=3, alpha=0.8)
            
            # Mark start and end points with time information
            if x_coords:
                # 2D 시작점과 끝점
                ax1.scatter(x_coords[0], y_coords[0], color='green', s=150, marker='o', 
                           edgecolors='black', linewidth=2, zorder=5)
                ax1.scatter(x_coords[-1], y_coords[-1], color='red', s=150, marker='s', 
                           edgecolors='black', linewidth=2, zorder=5)
                
                # 3D 시작점과 끝점
                ax2.scatter(x_coords[0], y_coords[0], frames[0], color='green', s=150, marker='o', 
                           edgecolors='black', linewidth=2, zorder=5)
                ax2.scatter(x_coords[-1], y_coords[-1], frames[-1], color='red', s=150, marker='s', 
                           edgecolors='black', linewidth=2, zorder=5)
                
                # 프레임 번호 표시
                ax1.annotate(f'F{frames[0]}', (x_coords[0], y_coords[0]), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, fontweight='bold', color='green',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                ax1.annotate(f'F{frames[-1]}', (x_coords[-1], y_coords[-1]), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, fontweight='bold', color='red',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                ax2.text(x_coords[0], y_coords[0], frames[0], f'F{frames[0]}', 
                        fontsize=8, fontweight='bold', color='green',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                ax2.text(x_coords[-1], y_coords[-1], frames[-1], f'F{frames[-1]}', 
                        fontsize=8, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Compose data for JSON storage
            track_class_data[track_id] = {
                'track_id': track_id,
                'dominant_class': dominant_class,
                'total_frames': len(track_info['detections']),
                'first_frame': track_info['first_frame'],
                'last_frame': track_info['last_frame'],
                'trajectory': {
                    'x_coords': x_coords,
                    'y_coords': y_coords,
                    'frames': frames
                },
                'class_assignments': class_assignments,
                'reid_history': reid_history
            }
        
        # 2D graph settings
        ax1.set_xlabel('X Coordinate (pixels)', fontsize=12)
        ax1.set_ylabel('Y Coordinate (pixels)', fontsize=12)
        ax1.set_title(f'{video_name} - 2D Cat Spatial Trajectories', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 3D graph settings
        ax2.set_xlabel('X Coordinate (pixels)', fontsize=12)
        ax2.set_ylabel('Y Coordinate (pixels)', fontsize=12)
        ax2.set_zlabel('Frame Number', fontsize=12)
        ax2.set_title(f'{video_name} - 3D Cat Spatial Trajectories (Z-axis = Time)', fontsize=14, fontweight='bold')
        ax2.view_init(elev=20, azim=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save combined 2D+3D image
        detailed_visualization_file = os.path.join(output_dir, f'{video_name}_detailed_track_visualization_2d3d.png')
        plt.savefig(detailed_visualization_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Class-based statistics visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Number of tracks per class
        class_track_counts = {}
        for track_data in track_class_data.values():
            class_id = track_data['dominant_class']
            if class_id:
                class_track_counts[class_id] = class_track_counts.get(class_id, 0) + 1
        
        if class_track_counts:
            classes = list(class_track_counts.keys())
            counts = list(class_track_counts.values())
            colors = [self.get_cat_color_hex(c) for c in classes]
            
            ax1.bar(classes, counts, color=colors, alpha=0.8)
            ax1.set_xlabel('Cat Class')
            ax1.set_ylabel('Number of Tracks')
            ax1.set_title('Number of Tracks per Class')
            ax1.grid(True, alpha=0.3)
        
        # Track duration distribution
        durations = [track_data['total_frames'] for track_data in track_class_data.values()]
        ax2.hist(durations, bins=min(10, len(durations)), alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Track Duration (frames)')
        ax2.set_ylabel('Number of Tracks')
        ax2.set_title('Track Duration Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Average duration per class
        class_durations = {}
        for track_data in track_class_data.values():
            class_id = track_data['dominant_class']
            if class_id:
                if class_id not in class_durations:
                    class_durations[class_id] = []
                class_durations[class_id].append(track_data['total_frames'])
        
        if class_durations:
            avg_durations = {c: np.mean(durs) for c, durs in class_durations.items()}
            classes = list(avg_durations.keys())
            avg_durs = list(avg_durations.values())
            colors = [self.get_cat_color_hex(c) for c in classes]
            
            ax3.bar(classes, avg_durs, color=colors, alpha=0.8)
            ax3.set_xlabel('Cat Class')
            ax3.set_ylabel('Average Duration (frames)')
            ax3.set_title('Average Track Duration per Class')
            ax3.grid(True, alpha=0.3)
        
        # Class activity over time
        all_frames = []
        all_classes = []
        for track_data in track_class_data.values():
            for frame, class_id in track_data['class_assignments'].items():
                if class_id > 0:
                    all_frames.append(frame)
                    all_classes.append(class_id)
        
        if all_frames:
            # Prepare heatmap data for class activity per frame
            frame_range = range(min(all_frames), max(all_frames) + 1)
            class_range = range(1, max(all_classes) + 1)
            
            activity_matrix = np.zeros((len(class_range), len(frame_range)))
            for frame, class_id in zip(all_frames, all_classes):
                if class_id in class_range and frame in frame_range:
                    class_idx = class_id - 1
                    frame_idx = frame - min(all_frames)
                    activity_matrix[class_idx, frame_idx] += 1
            
            im = ax4.imshow(activity_matrix, cmap='YlOrRd', aspect='auto')
            ax4.set_xlabel('Frame Number')
            ax4.set_ylabel('Cat Class')
            ax4.set_title('Class Activity Heatmap Over Time')
            ax4.set_yticks(range(len(class_range)))
            ax4.set_yticklabels([f'Cat {c}' for c in class_range])
            
            # Add colorbar
            plt.colorbar(im, ax=ax4, label='Activity Frequency')
        
        plt.tight_layout()
        
        # Save statistics visualization
        stats_visualization_file = os.path.join(output_dir, f'{video_name}_track_statistics.png')
        plt.savefig(stats_visualization_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Save detailed data as JSON file
        detailed_data = {
            'video_info': {
                'video_path': video_path,
                'video_name': video_name,
                'total_frames': results.get('total_frames', 0),
                'fps': fps,
                'resolution': f"{max_width}x{max_height}",
                'processing_time': results.get('processing_time', ''),
                'total_tracks': len(track_class_data)
            },
            'track_data': track_class_data,
            'class_statistics': {
                'class_track_counts': class_track_counts,
                'class_durations': class_durations,
                'avg_durations': {c: float(np.mean(durs)) for c, durs in class_durations.items()} if class_durations else {},
                'total_detections_by_class': {}
            },
            'visualization_files': {
                'detailed_track_visualization': detailed_visualization_file,
                'track_statistics': stats_visualization_file
            }
        }
        
        # Calculate total detections per class
        for track_data in track_class_data.values():
            for class_id in track_data['class_assignments'].values():
                if class_id > 0:
                    detailed_data['class_statistics']['total_detections_by_class'][class_id] = \
                        detailed_data['class_statistics']['total_detections_by_class'].get(class_id, 0) + 1
        
        # Save JSON file
        json_file = os.path.join(output_dir, f'{video_name}_detailed_track_analysis.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed visualization saved:")
        print(f"  - Spatial trajectory + Time-class visualization: {detailed_visualization_file}")
        print(f"  - Statistics visualization: {stats_visualization_file}")
        print(f"  - Detailed data JSON: {json_file}")
        
        return detailed_data

    def process_video(self, video_path, output_path=None, save_video=False, show_display=True, save_trajectories=True):
        """Process video"""
        # Reset tracker for new video
        self.reset_tracker()
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None
        
        # Video information
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output video settings
        out = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Results storage
        results = {
            'video_path': video_path,
            'total_frames': total_frames,
            'fps': fps,
            'resolution': f"{width}x{height}",
            'distance_threshold': self.distance_threshold,
            'confidence_threshold': self.confidence_threshold,
            'max_age': self.max_age,
            'min_hits': self.min_hits,
            'reid_interval': self.reid_interval,
            'reid_threshold': self.reid_threshold,
            'tracks': {},
            'frame_detections': []
        }
        
        print(f"Starting video processing: {os.path.basename(video_path)}")
        print(f"Total frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
        print(f"Distance threshold: {self.distance_threshold}px, Max tracking age: {self.max_age} frames")
        print(f"Cat class ID: {self.cat_class_id}, Person class ID: {self.person_class_id}")
        if self.reid_classifier:
            print(f"ReID interval: {self.reid_interval} frames, ReID threshold: {self.reid_threshold}")
        if show_display:
            print("Real-time display mode: Press 'q' to quit.")
        
        frame_id = 0
        pbar = tqdm(total=total_frames, desc="Processing frames", disable=show_display)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            if not show_display:
                pbar.update(1)
            
            # YOLO로 고양이와 사람 감지
            results_yolo = self.yolo_model(frame, verbose=False, iou=0.3)
            
            # 고양이 검출 결과
            cat_detections = []
            cat_confidences = []
            cat_frame_detections = []
            
            # 사람 검출 결과
            person_detections = []
            person_confidences = []
            person_frame_detections = []
            
            for result in results_yolo:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        detection_info = {
                            'frame': frame_id,
                            'bbox': [x1, y1, x2, y2],
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                            'yolo_confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.yolo_model.names[class_id]
                        }
                        
                        # 고양이 클래스 필터링
                        if class_id == self.cat_class_id and confidence > self.confidence_threshold:
                            cat_detections.append([x1, y1, x2, y2])
                            cat_confidences.append(confidence)
                            cat_frame_detections.append(detection_info)
                        
                        # 사람 클래스 필터링
                        elif class_id == self.person_class_id and confidence > self.confidence_threshold:
                            person_detections.append([x1, y1, x2, y2])
                            person_confidences.append(confidence)
                            person_frame_detections.append(detection_info)
            
            # 고양이에 대해 NMS 적용
            if len(cat_detections) > 0:
                cat_nms_indices = apply_nms(cat_detections, cat_confidences, self.nms_iou_threshold)
                cat_detections = [cat_detections[i] for i in cat_nms_indices]
                cat_confidences = [cat_confidences[i] for i in cat_nms_indices]
                cat_frame_detections = [cat_frame_detections[i] for i in cat_nms_indices]
            
            # 사람에 대해 NMS 적용
            if len(person_detections) > 0:
                person_nms_indices = apply_nms(person_detections, person_confidences, self.nms_iou_threshold)
                person_detections = [person_detections[i] for i in person_nms_indices]
                person_confidences = [person_confidences[i] for i in person_nms_indices]
                person_frame_detections = [person_frame_detections[i] for i in person_nms_indices]
            
            # 모든 검출 결과 합치기 (고양이만 추적)
            all_detections = cat_detections.copy()
            all_frame_detections = cat_frame_detections.copy()
            
            # 현재 프레임의 검출 결과 저장
            self.cat_detections = cat_detections
            self.person_detections = person_detections
            self.all_detections = all_detections
            
            # SORT 트래커로 track 업데이트 (고양이만 추적)
            active_tracks, matched_detections, matched_tracks, unmatched_detections = \
                self.update_tracks(all_detections, width, height)
            
            # ReID 적용 (10프레임 간격)
            if self.reid_classifier:
                self.apply_reid_to_tracks(frame, active_tracks)
            
            # 결과 저장
            for track in active_tracks:
                track_id = track['track_id']
                
                if track_id not in results['tracks']:
                    results['tracks'][track_id] = {
                        'first_frame': frame_id,
                        'last_frame': frame_id,
                        'detections': [],
                        'reid_history': []
                    }
                
                results['tracks'][track_id]['last_frame'] = frame_id
                results['tracks'][track_id]['detections'].append({
                    'frame': frame_id,
                    'bbox': track['bbox'],
                    'center': track['center'],
                    'hits': track['hits'],
                    'age': track['age']
                })
                
                # ReID 정보 저장
                if 'reid_class' in track and track['reid_class'] is not None:
                    results['tracks'][track_id]['reid_history'].append({
                        'frame': frame_id,
                        'reid_class': track['reid_class'],
                        'reid_confidence': track['reid_confidence'],
                        'reid_distance': track.get('reid_distance', 0.0)
                    })
            
            # 프레임별 감지 결과 저장 (고양이 + 사람)
            frame_detections = cat_frame_detections + person_frame_detections
            results['frame_detections'].append({
                'frame': frame_id,
                'cat_detections': cat_frame_detections,
                'person_detections': person_frame_detections,
                'all_detections': frame_detections,
                'active_tracks': active_tracks,
                'matched_detections': matched_detections,
                'unmatched_detections': unmatched_detections
            })
            
            # 결과 시각화 (고양이와 사람 모두 표시)
            annotated_frame = self.visualize_frame(
                frame, cat_detections, person_detections, matched_detections, frame_id, width, height
            )
            
            # 실시간 표시
            if show_display:
                cv2.imshow('SORT Euclidean Distance Cat Tracking with ReID', annotated_frame)
                
                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("사용자가 종료를 요청했습니다.")
                    break
            
            # 비디오 저장
            if save_video and out:
                out.write(annotated_frame)
        
        if not show_display:
            pbar.close()
        cap.release()
        if out:
            out.release()
        if show_display:
            cv2.destroyAllWindows()
        
        # 최종 통계
        results['total_tracks'] = len(results['tracks'])
        results['processing_time'] = datetime.now().isoformat()
        
        print(f"처리 완료: {len(results['tracks'])}개 track 발견")
        
        # Track 궤적 이미지 저장
        if save_trajectories and output_path:
            output_dir = os.path.dirname(output_path)
            self.save_track_trajectories(results, output_dir)
        
        return results

    def visualize_frame(self, frame, cat_detections, person_detections, matched_detections, frame_id, frame_width, frame_height):
        """Frame visualization (cats only)"""
        annotated_frame = frame.copy()
        
        # Map matched detections to corresponding track information
        matched_track_info = {}
        for i, track in enumerate(self.tracks):
            if not track.is_deleted and track.is_confirmed:
                track_bbox = track.get_latest_bbox()
                matched_track_info[tuple(track_bbox)] = {
                    'track_id': track.track_id,
                    'track_obj': track
                }
        
        # Draw tracker path points first (in background) - cats only
        for track in self.tracks:
            if not track.is_deleted and track.is_confirmed:
                # Check assigned cat class (priority: assigned class > dominant class)
                assigned_cat_class = self.track_cat_assignments.get(track.track_id)
                if assigned_cat_class is None or assigned_cat_class <= 0:
                    # Use dominant class if not assigned
                    dominant_info = track.get_dominant_cat_info()
                    assigned_cat_class = dominant_info['class_id'] if dominant_info else None
                
                # Select color using unified color mapping
                color = self.get_cat_color_bgr(assigned_cat_class)
                
                # Draw path points (show all history)
                path_points = track.center_history  # Show all positions
                for i, center in enumerate(path_points):
                    x, y = int(center[0]), int(center[1])
                    
                    # Adjust point size based on time (larger for recent)
                    point_size = max(2, min(6, 6 - (len(path_points) - 1 - i) // 8))
                    
                    # Adjust point transparency based on time (more transparent for older)
                    if len(path_points) > 1:
                        alpha = 0.3 + (i / len(path_points)) * 0.7
                    else:
                        alpha = 1.0
                    point_color = tuple(int(c * alpha) for c in color)
                    
                    # Draw point (smoother circle)
                    cv2.circle(annotated_frame, (x, y), point_size, point_color, -1)
                    
                    # Point border (for clarity)
                    if alpha > 0.8:
                        cv2.circle(annotated_frame, (x, y), point_size, (255, 255, 255), 1)
                
                # Draw path lines (connection lines)
                if len(path_points) > 1:
                    for i in range(len(path_points) - 1):
                        x1, y1 = int(path_points[i][0]), int(path_points[i][1])
                        x2, y2 = int(path_points[i+1][0]), int(path_points[i+1][1])
                        
                        # Adjust line transparency (more transparent for older lines)
                        if len(path_points) > 2:
                            alpha = 0.2 + (i / (len(path_points) - 1)) * 0.6
                        else:
                            alpha = 0.8
                        line_color = tuple(int(c * alpha) for c in color)
                        
                        # Draw line (smoother line)
                        cv2.line(annotated_frame, (x1, y1), (x2, y2), line_color, 2)
        
        # Display only matched cat detections (actual measured bounding boxes)
        for det_idx in matched_detections:
            if det_idx < len(cat_detections):
                x1, y1, x2, y2 = cat_detections[det_idx]
                bbox_tuple = (x1, y1, x2, y2)
                
                # Find track information for this detection
                if bbox_tuple in matched_track_info:
                    track_info = matched_track_info[bbox_tuple]
                    track_id = track_info['track_id']
                    track_obj = track_info['track_obj']
                    
                    # Check assigned cat class (priority: assigned class > dominant class)
                    assigned_cat_class = self.track_cat_assignments.get(track_id)
                    if assigned_cat_class is None or assigned_cat_class <= 0:
                        # Use dominant class if not assigned
                        dominant_info = track_obj.get_dominant_cat_info()
                        assigned_cat_class = dominant_info['class_id'] if dominant_info else None
                    
                    # Select color using unified color mapping
                    color = self.get_cat_color_bgr(assigned_cat_class)
                    
                    # Draw bounding box (actual measured position) - thicker and clearer
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Bounding box border (white border for clarity)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    
                    # Draw center point (current position larger, use same color)
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cv2.circle(annotated_frame, (center_x, center_y), 8, color, -1)
                    cv2.circle(annotated_frame, (center_x, center_y), 8, (255, 255, 255), 2)
                    
                    # Draw label (assigned cat class and probability)
                    if assigned_cat_class and assigned_cat_class > 0:
                        dominant_info = track_obj.get_dominant_cat_info()
                        if dominant_info and dominant_info['class_id'] == assigned_cat_class:
                            label = f"Cat {assigned_cat_class} ({dominant_info['probability']:.1f}%)"
                        else:
                            label = f"Cat {assigned_cat_class} (Assigned)"
                    else:
                        # Show current dominant info if not assigned
                        dominant_info = track_obj.get_dominant_cat_info()
                        if dominant_info and dominant_info['class_id'] > 0:
                            label = f"Track {track_id} (Cat {dominant_info['class_id']} {dominant_info['probability']:.1f}%)"
                        else:
                            label = f"Track {track_id} (No Cat)"
                    
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    
                    # Label background (semi-transparent effect, use same color)
                    label_bg_color = tuple(int(c * 0.8) for c in color)  # Slightly darker color
                    cv2.rectangle(annotated_frame, (x1, y1-label_height-15), (x1+label_width+10, y1), label_bg_color, -1)
                    
                    # Label border
                    cv2.rectangle(annotated_frame, (x1, y1-label_height-15), (x1+label_width+10, y1), (255, 255, 255), 1)
                    
                    # Label text (thicker and clearer)
                    cv2.putText(annotated_frame, label, (x1+5, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Information panel background (semi-transparent black)
        info_panel = np.zeros((220, frame_width, 3), dtype=np.uint8)
        info_panel = cv2.addWeighted(info_panel, 0.7, np.zeros_like(info_panel), 0.3, 0)
        
        # Overlay information panel on top of frame
        annotated_frame[0:220, 0:frame_width] = cv2.addWeighted(
            annotated_frame[0:220, 0:frame_width], 0.3, info_panel, 0.7, 0
        )
        
        # Frame information (larger font)
        info_text = f"Frame: {frame_id} | Matched Cats: {len(matched_detections)} | Total Tracks: {len(self.tracks)}"
        cv2.putText(annotated_frame, info_text, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Detection information display
        detection_info = f"Cats: {len(cat_detections)} | Total Detections: {len(cat_detections)}"
        cv2.putText(annotated_frame, detection_info, (15, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # SORT settings information
        sort_info = f"Distance Threshold: {self.distance_threshold}px | Max Age: {self.max_age} | Min Hits: {self.min_hits}"
        cv2.putText(annotated_frame, sort_info, (15, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ReID information display
        if self.reid_classifier:
            reid_info = f"ReID Interval: {self.reid_interval} | ReID Threshold: {self.reid_threshold}"
            cv2.putText(annotated_frame, reid_info, (15, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Path information display
        active_tracks = [t for t in self.tracks if not t.is_deleted and t.is_confirmed]
        total_path_points = sum(len(t.center_history) for t in active_tracks)
        path_info = f"Track Paths: {len(active_tracks)} active tracks | {total_path_points} total path points"
        cv2.putText(annotated_frame, path_info, (15, 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Class-based reconnection information display
        if hasattr(self, 'cat_class_track_history'):
            total_reconnections = sum(len(history) for history in self.cat_class_track_history.values())
            reconnection_info = f"Class Reconnections: {total_reconnections} total | {len(self.cat_class_track_history)} classes"
            cv2.putText(annotated_frame, reconnection_info, (15, 185), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # NMS information display
        nms_info = f"NMS IoU Threshold: {self.nms_iou_threshold} | Applied to Cats"
        cv2.putText(annotated_frame, nms_info, (15, 215), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame

def find_video_files(root_dir):
    """동영상 파일 찾기 (중복 제거)"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = set()
    
    for ext in video_extensions:
        # 대소문자 구분 없이 검색
        pattern = os.path.join(root_dir, '**', f'*{ext[1:]}')  # *.mp4 -> *mp4
        found_files = glob.glob(pattern, recursive=True)
        found_files.extend(glob.glob(pattern.upper(), recursive=True))
        found_files.extend(glob.glob(pattern.lower(), recursive=True))
        
        for file_path in found_files:
            video_files.add(os.path.abspath(file_path))
    
    return sorted(list(video_files))

def main():
    parser = argparse.ArgumentParser(description='SORT 알고리즘 기반 dIoU 고양이 추적 (ReID 포함)')
    parser.add_argument('--yolo_model', type=str, default='yolo11x.pt',
                       help='YOLO 모델 경로')
    parser.add_argument('--reid_model', type=str, default="C:/Users/w4d3r/git_archive/cat_discrimination/output/best_model.pth",
                       help='ReID 모델 경로')
    parser.add_argument('--feature_stats', type=str, default="C:/Users/w4d3r/git_archive/cat_discrimination/feature_analysis_output/detailed_results.json",
                       help='특징점 통계 파일 경로')
    parser.add_argument('--input_dir', type=str, default='origin_datasets',
                       help='입력 동영상 디렉토리')
    parser.add_argument('--output_dir', type=str, default='sort_tracking_output',
                       help='출력 디렉토리')
    parser.add_argument('--save_video', action='store_true',
                       help='결과 동영상 저장 여부')
    parser.add_argument('--no_display', action='store_true',
                       help='실시간 표시 비활성화')
    parser.add_argument('--single_video', type=str, default=None,#"./origin_datasets/2월28일/20250228_124520000_iOS.MP4",
                       help='단일 동영상 파일 처리')
    parser.add_argument('--confidence_threshold', type=float, default=0.1,
                       help='YOLO 감지 신뢰도 임계값')
    parser.add_argument('--distance_threshold', type=float, default=150,
                       help='Euclidean distance 임계값 (픽셀 단위) - 기존 호환성용')
    parser.add_argument('--diou_threshold', type=float, default=0.3,
                       help='dIoU 임계값 (0.3 이상이면 좋은 매칭)')
    parser.add_argument('--use_diou', action='store_true', default=True,
                       help='dIoU 사용 여부')
    parser.add_argument('--max_age', type=int, default=30,
                       help='최대 추적 유지 프레임 수')
    parser.add_argument('--min_hits', type=int, default=3,
                       help='track 확인을 위한 최소 히트 수')
    parser.add_argument('--reid_interval', type=int, default=10,
                       help='ReID 적용 간격 (프레임)')
    parser.add_argument('--reid_threshold', type=float, default=1.0,
                       help='ReID 임계값')
    parser.add_argument('--nms_iou_threshold', type=float, default=0.5,
                       help='NMS IoU 임계값')
    parser.add_argument('--save_trajectories', action='store_true',
                       help='Track 궤적 이미지 저장 여부')
    parser.add_argument('--save_detailed_visualization', action='store_true',
                       help='상세 시각화 (시간-경로, 클래스 통계) 저장 여부')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_video:
        os.makedirs(os.path.join(args.output_dir, 'videos'), exist_ok=True)
    
    # SORT 트래커 초기화 (ReID 포함)
    tracker = SORTCatTracker(
        args.yolo_model, 
        args.reid_model, 
        args.feature_stats
    )
    
    # 설정 적용
    tracker.confidence_threshold = args.confidence_threshold
    tracker.distance_threshold = args.distance_threshold
    tracker.diou_threshold = args.diou_threshold
    tracker.use_diou = args.use_diou
    tracker.max_age = args.max_age
    tracker.min_hits = args.min_hits
    tracker.reid_interval = args.reid_interval
    tracker.reid_threshold = args.reid_threshold
    tracker.nms_iou_threshold = args.nms_iou_threshold
    
    print(f"=== SORT 알고리즘 기반 dIoU 고양이 추적 시스템 (ReID 포함) ===")
    print(f"dIoU 임계값: {tracker.diou_threshold}")
    print(f"dIoU 사용: {tracker.use_diou}")
    print(f"신뢰도 임계값: {tracker.confidence_threshold}")
    print(f"최대 추적 유지 프레임: {tracker.max_age}")
    print(f"최소 히트 수: {tracker.min_hits}")
    print(f"NMS IoU 임계값: {tracker.nms_iou_threshold}")
    if tracker.reid_classifier:
        print(f"ReID 간격: {tracker.reid_interval}프레임")
        print(f"ReID 임계값: {tracker.reid_threshold}")
    else:
        print("ReID 기능이 비활성화되었습니다.")
    
    if args.single_video:
        # 단일 동영상 처리
        if not os.path.exists(args.single_video):
            print(f"동영상 파일을 찾을 수 없습니다: {args.single_video}")
            return
        
        print(f"\n단일 동영상 처리: {args.single_video}")
        
        # 출력 파일 경로 설정
        video_name = os.path.splitext(os.path.basename(args.single_video))[0]
        output_video_path = None
        if args.save_video:
            output_video_path = os.path.join(args.output_dir, 'videos', f'{video_name}_sort_reid_tracking.mp4')
        
        # 처리
        results = tracker.process_video(
            args.single_video, 
            output_video_path, 
            args.save_video, 
            show_display=not args.no_display,
            save_trajectories=args.save_trajectories or args.save_detailed_visualization
        )
        
        # 결과 저장
        if results:
            result_file = os.path.join(args.output_dir, f'{video_name}_sort_reid_results.json')
            tracker.save_results(results, result_file)
            
            # 상세 시각화 추가 저장
            if args.save_detailed_visualization:
                output_dir = os.path.dirname(output_video_path) if output_video_path else args.output_dir
                tracker.save_detailed_track_visualization(results, output_dir)
        
    else:
        # 전체 디렉토리 처리
        video_files = find_video_files(args.input_dir)
        print(f"\n발견된 동영상 파일 수: {len(video_files)}")
        
        if len(video_files) == 0:
            print("처리할 동영상 파일을 찾을 수 없습니다.")
            return
        
        # 각 동영상 처리
        all_results = []
        
        for i, video_path in enumerate(video_files):
            print(f"\n처리 중 ({i+1}/{len(video_files)}): {os.path.basename(video_path)}")
            
            # 출력 파일 경로 설정
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = None
            if args.save_video:
                output_video_path = os.path.join(args.output_dir, 'videos', f'{video_name}_sort_reid_tracking.mp4')
            
            # 처리
            results = tracker.process_video(
                video_path, 
                output_video_path, 
                args.save_video, 
                show_display=not args.no_display,
                save_trajectories=args.save_trajectories or args.save_detailed_visualization
            )
            
            if results:
                all_results.append(results)
                
                # 개별 결과 저장
                result_file = os.path.join(args.output_dir, f'{video_name}_sort_reid_results.json')
                tracker.save_results(results, result_file)
                
                # 상세 시각화 추가 저장
                if args.save_detailed_visualization:
                    output_dir = os.path.dirname(output_video_path) if output_video_path else args.output_dir
                    tracker.save_detailed_track_visualization(results, output_dir)
        
        # 전체 결과 요약
        if all_results:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_videos': len(video_files),
                'total_tracks': sum(len(result['tracks']) for result in all_results),
                'average_tracks_per_video': np.mean([len(result['tracks']) for result in all_results]),
                'settings': {
                    'distance_threshold': tracker.distance_threshold,
                    'confidence_threshold': tracker.confidence_threshold,
                    'max_age': tracker.max_age,
                    'min_hits': tracker.min_hits,
                    'reid_interval': tracker.reid_interval,
                    'reid_threshold': tracker.reid_threshold,
                    'nms_iou_threshold': tracker.nms_iou_threshold
                },
                'videos': [{
                    'path': result['video_path'],
                    'total_frames': result['total_frames'],
                    'total_tracks': len(result['tracks'])
                } for result in all_results]
            }
            
            summary_file = os.path.join(args.output_dir, 'sort_reid_tracking_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n=== SORT Euclidean Distance 추적 + ReID 완료 ===")
            print(f"처리된 동영상: {len(video_files)}개")
            print(f"총 고양이 track: {summary['total_tracks']}개")
            print(f"동영상당 평균 track: {summary['average_tracks_per_video']:.2f}개")
            print(f"결과 저장 위치: {args.output_dir}")

if __name__ == "__main__":
    main() 