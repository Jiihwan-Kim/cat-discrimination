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

from ultralytics import YOLO
from model import CatDiscriminationModel
from config import Config
from utils import load_checkpoint

class CatReIDSystem:
    """고양이 ReID 시스템 (Feature 기반 거리 비교)"""
    
    def __init__(self, config, model_path, yolo_model_path="yolo11s.pt", feature_stats_path=None):
        self.config = config
        self.device = config.device
        
        # YOLO 모델 로드
        self.yolo_model = YOLO(yolo_model_path)
        
        # YOLO 모델 정보 출력
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
            print("고양이 클래스를 찾을 수 없습니다. 모든 클래스를 출력합니다:")
            for class_id, class_name in self.yolo_model.names.items():
                print(f"  {class_id}: {class_name}")
            # 기본값으로 15번 클래스 사용 (COCO 데이터셋의 고양이 클래스)
            self.cat_class_id = 15
            print(f"기본값으로 클래스 {self.cat_class_id}를 사용합니다.")
        
        # ReID 모델 로드
        self.reid_model = CatDiscriminationModel(config).to(self.device)
        self.reid_model.eval()
        
        # 학습된 가중치 로드
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.reid_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"ReID 모델 로드 완료: {model_path}")
        else:
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return
        
        # 특징점 통계 로드 (선택적)
        self.class_means = {}
        if feature_stats_path and os.path.exists(feature_stats_path):
            self.load_feature_statistics(feature_stats_path)
            print(f"특징점 통계 로드 완료: {len(self.class_means)}개 클래스")
        else:
            print("특징점 통계 파일이 없습니다. 실시간 특징점 비교를 사용합니다.")
        
        # 고양이 ID 추적을 위한 변수들
        self.cat_tracks = {}  # {track_id: {'features': [], 'frames': [], 'last_seen': frame_id, 'predicted_id': id}}
        self.next_track_id = 0
        self.max_track_age = 30  # 최대 추적 유지 프레임 수
        self.distance_threshold = 0.8  # 거리 임계값 (L2 정규화된 특징점 기준)
        
        # 실시간 표시를 위한 변수들
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # 디버깅을 위한 변수들
        self.debug_mode = True
        self.total_detections = 0
        self.cat_detections = 0
        
        # 클래스 이름 매핑
        self.class_names = ['Cat 1', 'Cat 2', 'Cat 3']
        
    def load_feature_statistics(self, stats_path):
        """저장된 특징점 통계 로드"""
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
        
        # 클래스별 평균 특징점 로드
        for class_id_str, class_stats in stats_data['statistics'].items():
            class_id = int(class_id_str)
            self.class_means[class_id] = np.array(class_stats['mean'])
    
    def extract_features(self, image):
        """이미지에서 특징점 추출"""
        with torch.no_grad():
            # 이미지 전처리
            image_tensor = self.preprocess_image(image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # 특징점 추출
            features = self.reid_model(image_tensor, return_features=True)
            features = features.cpu().numpy().flatten()
            
            return features
    
    def classify_cat(self, image):
        """이미지에서 고양이 분류 (기존 호환성 유지)"""
        features = self.extract_features(image)
        
        if len(self.class_means) > 0:
            # 저장된 특징점 통계와 비교
            distances = {}
            for class_id, mean_features in self.class_means.items():
                distance = np.linalg.norm(features - mean_features)
                distances[class_id] = distance
            
            # 가장 가까운 클래스 찾기
            min_distance = min(distances.values())
            predicted_class = min(distances.keys(), key=lambda x: distances[x])
            
            # 임계값 적용
            if min_distance > self.distance_threshold:
                predicted_class = 0  # 알 수 없는 클래스
                confidence = 0.0
            else:
                confidence = max(0.0, 1.0 - min_distance / self.distance_threshold)
            
            # 모든 클래스의 확률 계산 (거리 기반)
            all_probabilities = []
            for class_id in range(len(self.class_names)):
                if class_id in distances:
                    prob = max(0.0, 1.0 - distances[class_id] / self.distance_threshold)
                else:
                    prob = 0.0
                all_probabilities.append(prob)
            
            return int(predicted_class), float(confidence), all_probabilities
        else:
            # 기존 분류 방식 (호환성)
            with torch.no_grad():
                image_tensor = self.preprocess_image(image)
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                logits, _ = self.reid_model(image_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).cpu().numpy()[0]
                confidence = probabilities[0][predicted_class].cpu().numpy()
                all_probabilities = probabilities[0].cpu().numpy()
                
                return int(predicted_class), float(confidence), all_probabilities.tolist()
    
    def preprocess_image(self, image):
        """이미지 전처리"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image)
    
    def assign_track_id(self, features, frame_id):
        """특징점 기반으로 track ID 할당"""
        if len(self.cat_tracks) == 0:
            # 첫 번째 고양이
            track_id = self.next_track_id
            self.cat_tracks[track_id] = {
                'features': [features],
                'frames': [frame_id],
                'last_seen': frame_id,
                'predicted_id': 0  # 기본 ID
            }
            self.next_track_id += 1
            return track_id
        
        # 기존 track들과 특징점 거리 비교
        best_track_id = None
        min_distance = float('inf')
        
        for track_id, track_info in self.cat_tracks.items():
            # 최근 특징점들과 비교
            recent_features = track_info['features'][-5:]  # 최근 5개 특징점
            
            # 평균 거리 계산
            distances = []
            for track_feature in recent_features:
                distance = np.linalg.norm(features - track_feature)
                distances.append(distance)
            
            avg_distance = np.mean(distances)
            
            if avg_distance < min_distance and avg_distance < self.distance_threshold:
                min_distance = avg_distance
                best_track_id = track_id
        
        if best_track_id is not None:
            # 기존 track 업데이트
            self.cat_tracks[best_track_id]['features'].append(features)
            self.cat_tracks[best_track_id]['frames'].append(frame_id)
            self.cat_tracks[best_track_id]['last_seen'] = frame_id
            
            # 특징점이 너무 많아지면 오래된 것 제거
            if len(self.cat_tracks[best_track_id]['features']) > 20:
                self.cat_tracks[best_track_id]['features'] = self.cat_tracks[best_track_id]['features'][-10:]
                self.cat_tracks[best_track_id]['frames'] = self.cat_tracks[best_track_id]['frames'][-10:]
            
            return best_track_id
        else:
            # 새로운 track 생성
            track_id = self.next_track_id
            self.cat_tracks[track_id] = {
                'features': [features],
                'frames': [frame_id],
                'last_seen': frame_id,
                'predicted_id': len(self.cat_tracks)  # 새로운 ID
            }
            self.next_track_id += 1
            return track_id
    
    def cleanup_old_tracks(self, current_frame_id):
        """오래된 track 정리"""
        tracks_to_remove = []
        for track_id, track_info in self.cat_tracks.items():
            if current_frame_id - track_info['last_seen'] > self.max_track_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.cat_tracks[track_id]
    
    def update_fps(self):
        """FPS 계산"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # 30프레임마다 FPS 업데이트
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
    
    def convert_to_serializable(self, obj):
        """JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_to_serializable(item) for item in obj)
        else:
            return obj
    
    def process_video_realtime(self, video_path, output_path=None, show_display=True):
        """실시간 동영상 처리"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"동영상을 열 수 없습니다: {video_path}")
            return
        
        # 출력 동영상 설정
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_id = 0
        results_summary = {
            'video_path': video_path,
            'total_frames': 0,
            'detected_cats': [],
            'tracks': {}
        }
        
        print(f"실시간 동영상 처리 시작: {video_path}")
        print("'q' 키를 누르면 종료됩니다.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            results_summary['total_frames'] = frame_id
            
            # FPS 업데이트
            self.update_fps()
            
            # YOLO로 고양이 감지
            results = self.yolo_model(frame, verbose=False)
            
            detected_cats = []
            all_detections = []  # 모든 감지 결과 저장
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0])
                        
                        # 모든 감지 결과 저장
                        all_detections.append({
                            'class_id': class_id,
                            'class_name': self.yolo_model.names[class_id],
                            'confidence': confidence
                        })
                        
                        # 고양이 클래스 필터링
                        if class_id == self.cat_class_id and confidence > 0.3:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # 고양이 영역 추출
                            cat_roi = frame[y1:y2, x1:x2]
                            if cat_roi.size == 0:
                                continue
                            
                            # 특징점 추출
                            features = self.extract_features(cat_roi)
                            
                            # Track ID 할당 (특징점 기반)
                            track_id = self.assign_track_id(features, frame_id)
                            
                            # 분류 결과 (호환성)
                            predicted_class, class_confidence, all_probabilities = self.classify_cat(cat_roi)
                            
                            detected_cats.append({
                                'bbox': (x1, y1, x2, y2),
                                'track_id': track_id,
                                'yolo_confidence': confidence,
                                'class_confidence': class_confidence,
                                'predicted_class': predicted_class,
                                'all_probabilities': all_probabilities,
                                'features': features.tolist()  # 특징점 저장
                            })
                            
                            # 결과 저장
                            if track_id not in results_summary['tracks']:
                                results_summary['tracks'][track_id] = {
                                    'first_frame': frame_id,
                                    'last_frame': frame_id,
                                    'detections': []
                                }
                            
                            results_summary['tracks'][track_id]['last_frame'] = frame_id
                            results_summary['tracks'][track_id]['detections'].append({
                                'frame': frame_id,
                                'bbox': (x1, y1, x2, y2),
                                'yolo_confidence': confidence,
                                'class_confidence': class_confidence,
                                'predicted_class': predicted_class,
                                'all_probabilities': all_probabilities,
                                'features': features.tolist()
                            })
            
            # 통계 업데이트
            self.total_detections += len(all_detections)
            self.cat_detections += len(detected_cats)
            
            # 디버깅 정보 출력 (10프레임마다)
            if self.debug_mode and frame_id % 10 == 0:
                print(f"Frame {frame_id}: 총 감지={len(all_detections)}, 고양이={len(detected_cats)}")
                if len(detected_cats) > 0:
                    cat_info = []
                    for cat in detected_cats:
                        cat_info.append(f"Track {cat['track_id']}({cat['class_confidence']:.2f})")
                    print(f"  추적된 고양이들: {cat_info}")
            
            # 오래된 track 정리
            self.cleanup_old_tracks(frame_id)
            
            # 결과 시각화
            annotated_frame = self.visualize_detections_realtime(frame, detected_cats, frame_id, all_detections)
            
            # 출력 동영상에 저장
            if output_path:
                out.write(annotated_frame)
            
            # 실시간 표시
            if show_display:
                cv2.imshow('Cat ReID - Real-time', annotated_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # q 키로 종료
                    print("사용자가 종료를 요청했습니다.")
                    break
                elif key == ord('p'):  # p 키로 일시정지
                    print("일시정지. 아무 키나 누르면 계속...")
                    cv2.waitKey(0)
                elif key == ord('s'):  # s 키로 스크린샷 저장
                    screenshot_path = f"screenshot_frame_{frame_id}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"스크린샷 저장: {screenshot_path}")
                elif key == ord('d'):  # d 키로 디버그 모드 토글
                    self.debug_mode = not self.debug_mode
                    print(f"디버그 모드: {'ON' if self.debug_mode else 'OFF'}")
        
        cap.release()
        if output_path:
            out.release()
        if show_display:
            cv2.destroyAllWindows()
        
        print(f"동영상 처리 완료: {video_path}")
        print(f"총 {len(self.cat_tracks)}개의 고양이 track 발견")
        print(f"전체 감지: {self.total_detections}, 고양이 감지: {self.cat_detections}")
        
        return results_summary
    
    def visualize_detections_realtime(self, frame, detections, frame_id, all_detections=None):
        """실시간 감지 결과 시각화"""
        annotated_frame = frame.copy()
        
        # 색상 팔레트 (Track ID별 색상)
        colors = [
            (255, 0, 0),    # Track 0 - 빨강
            (0, 255, 0),    # Track 1 - 초록
            (0, 0, 255),    # Track 2 - 파랑
            (255, 255, 0),  # 추가 색상들
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128)
        ]
        
        # 고양이 감지 결과 그리기
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection['track_id']
            predicted_class = detection['predicted_class']
            class_confidence = detection['class_confidence']
            yolo_confidence = detection['yolo_confidence']
            all_probabilities = detection['all_probabilities']
            
            # 색상 선택 (Track ID별)
            color = colors[track_id % len(colors)]
            
            # 바운딩 박스 그리기
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 배경 (더 큰 배경으로 확장)
            label = f"Track {track_id} (Cat {predicted_class})"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1-label_height-40), (x1+label_width, y1), color, -1)
            
            # 라벨 텍스트
            cv2.putText(annotated_frame, label, (x1, y1-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 신뢰도 정보
            conf_label = f"Class: {class_confidence:.2f}, YOLO: {yolo_confidence:.2f}"
            cv2.putText(annotated_frame, conf_label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 모든 클래스의 확률 표시
            prob_text = f"Cat1:{all_probabilities[0]:.2f} Cat2:{all_probabilities[1]:.2f} Cat3:{all_probabilities[2]:.2f}"
            cv2.putText(annotated_frame, prob_text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # 정보 패널 추가
        info_panel = np.zeros((200, annotated_frame.shape[1], 3), dtype=np.uint8)
        
        # 프레임 정보
        cv2.putText(info_panel, f"Frame: {frame_id}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS 정보
        cv2.putText(info_panel, f"FPS: {self.current_fps:.1f}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 감지된 고양이 수
        cv2.putText(info_panel, f"Detected Cats: {len(detections)}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 총 track 수
        cv2.putText(info_panel, f"Total Tracks: {len(self.cat_tracks)}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 거리 임계값 정보
        cv2.putText(info_panel, f"Distance Threshold: {self.distance_threshold:.2f}", (10, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Track별 통계
        track_stats = f"Active Tracks: {len(self.cat_tracks)}"
        cv2.putText(info_panel, track_stats, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 평균 신뢰도 정보
        if len(detections) > 0:
            avg_class_conf = np.mean([d['class_confidence'] for d in detections])
            avg_yolo_conf = np.mean([d['yolo_confidence'] for d in detections])
            avg_conf_text = f"Avg Class: {avg_class_conf:.2f}, Avg YOLO: {avg_yolo_conf:.2f}"
            cv2.putText(info_panel, avg_conf_text, (10, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 컨트롤 안내
        cv2.putText(info_panel, "Controls: q=quit, p=pause, s=screenshot, d=debug", 
                   (annotated_frame.shape[1]-450, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # 디버그 정보 (모든 감지 결과 표시)
        if all_detections and self.debug_mode:
            debug_text = f"All detections: {len(all_detections)}"
            cv2.putText(info_panel, debug_text, (annotated_frame.shape[1]-450, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # 상위 3개 감지 결과 표시
            for i, det in enumerate(all_detections[:3]):
                det_text = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(info_panel, det_text, (annotated_frame.shape[1]-450, 75+i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        # 정보 패널을 프레임에 추가
        annotated_frame = np.vstack([info_panel, annotated_frame])
        
        return annotated_frame
    
    def process_video(self, video_path, output_path=None):
        """기존 동영상 처리 (호환성 유지)"""
        return self.process_video_realtime(video_path, output_path, show_display=False)
    
    def visualize_detections(self, frame, detections):
        """기존 시각화 (호환성 유지)"""
        return self.visualize_detections_realtime(frame, detections, 0)
    
    def save_results(self, results, output_file):
        """결과를 JSON 파일로 저장"""
        # JSON 직렬화 가능한 형태로 변환
        serializable_results = self.convert_to_serializable(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"결과 저장 완료: {output_file}")

def find_video_files(root_dir):
    """동영상 파일 찾기"""
    video_extensions = ['*.mp4', '*.MP4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
    
    return video_files

def main():
    parser = argparse.ArgumentParser(description='고양이 ReID 실시간 추론 (Feature 기반)')
    parser.add_argument('--model_path', type=str, default='output/best_model.pth',
                       help='학습된 ReID 모델 경로')
    parser.add_argument('--yolo_model', type=str, default='yolo11s.pt',
                       help='YOLO 모델 경로')
    parser.add_argument('--feature_stats', type=str, default="C:/Users/w4d3r/git_archive/cat_discrimination/feature_analysis_output/detailed_results.json",
                       help='특징점 통계 파일 경로 (선택적)')
    parser.add_argument('--input_dir', type=str, default='origin_datasets',
                       help='입력 동영상 디렉토리')
    parser.add_argument('--output_dir', type=str, default='reid_output',
                       help='출력 디렉토리')
    parser.add_argument('--save_video', action='store_true',
                       help='결과 동영상 저장 여부')
    parser.add_argument('--no_display', action='store_true',
                       help='실시간 표시 비활성화')
    parser.add_argument('--single_video', type=str, default=None,
                       help='단일 동영상 파일 처리')
    parser.add_argument('--confidence_threshold', type=float, default=0.3,
                       help='YOLO 감지 신뢰도 임계값')
    parser.add_argument('--distance_threshold', type=float, default=1.0,
                       help='특징점 거리 임계값')
    
    args = parser.parse_args()
    
    # 설정
    config = Config()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_video:
        os.makedirs(os.path.join(args.output_dir, 'videos'), exist_ok=True)
    
    # ReID 시스템 초기화
    reid_system = CatReIDSystem(config, args.model_path, args.yolo_model, args.feature_stats)
    
    # 거리 임계값 설정
    if args.distance_threshold:
        reid_system.distance_threshold = args.distance_threshold
    
    if args.single_video:
        # 단일 동영상 처리
        if not os.path.exists(args.single_video):
            print(f"동영상 파일을 찾을 수 없습니다: {args.single_video}")
            return
        
        print(f"단일 동영상 처리: {args.single_video}")
        
        # 출력 파일 경로 설정
        video_name = os.path.splitext(os.path.basename(args.single_video))[0]
        output_video_path = None
        if args.save_video:
            output_video_path = os.path.join(args.output_dir, 'videos', f'{video_name}_reid.mp4')
        
        # 실시간 처리
        results = reid_system.process_video_realtime(
            args.single_video, 
            output_video_path, 
            show_display=not args.no_display
        )
        
        # 결과 저장
        result_file = os.path.join(args.output_dir, f'{video_name}_results.json')
        reid_system.save_results(results, result_file)
        
    else:
        # 전체 디렉토리 처리
        video_files = find_video_files(args.input_dir)
        print(f"발견된 동영상 파일 수: {len(video_files)}")
        
        if len(video_files) == 0:
            print("처리할 동영상 파일을 찾을 수 없습니다.")
            return
        
        # 각 동영상 처리
        all_results = []
        
        for i, video_path in enumerate(video_files):
            print(f"\n처리 중 ({i+1}/{len(video_files)}): {video_path}")
            
            # 출력 파일 경로 설정
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = None
            if args.save_video:
                output_video_path = os.path.join(args.output_dir, 'videos', f'{video_name}_reid.mp4')
            
            # 실시간 처리
            results = reid_system.process_video_realtime(
                video_path, 
                output_video_path, 
                show_display=not args.no_display
            )
            all_results.append(results)
            
            # 개별 결과 저장
            result_file = os.path.join(args.output_dir, f'{video_name}_results.json')
            reid_system.save_results(results, result_file)
        
        # 전체 결과 요약
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_videos': len(video_files),
            'total_tracks': sum(len(result['tracks']) for result in all_results),
            'videos': all_results
        }
        
        summary_file = os.path.join(args.output_dir, 'reid_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== ReID 처리 완료 ===")
        print(f"처리된 동영상: {len(video_files)}개")
        print(f"총 고양이 track: {summary['total_tracks']}개")
        print(f"결과 저장 위치: {args.output_dir}")

if __name__ == "__main__":
    main() 