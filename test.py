import torch
import cv2
import numpy as np
import json
import os
from pathlib import Path
from model import CatDiscriminationModel
from config import Config

class CatClassifier:
    """고양이 분류기 - 특징점 거리 기반 분류"""
    
    def __init__(self, config, model_path, feature_stats_path):
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
        for class_id, mean_features in self.class_means.items():
            print(f"  Cat {class_id}: {mean_features.shape} 차원")
    
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
    
    def extract_features(self, image_path):
        """이미지에서 특징점 추출"""
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                print(f"이미지를 로드할 수 없습니다: {image_path}")
                return None
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 이미지 전처리
            image_tensor = self.preprocess_image(image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # 특징점 추출
            with torch.no_grad():
                features = self.reid_model(image_tensor, return_features=True)
                features = features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            print(f"특징점 추출 중 오류 발생: {image_path}, 오류: {e}")
            return None
    
    def classify_image(self, image_path, threshold=1.0):
        """이미지 분류"""
        print(f"이미지 분류 시작: {image_path}")
        
        # 특징점 추출
        features = self.extract_features(image_path)
        if features is None:
            return None
        
        print(f"추출된 특징점 차원: {features.shape}")
        
        # 각 클래스와의 거리 계산
        distances = {}
        for class_id, mean_features in self.class_means.items():
            # 유클리드 거리 계산
            distance = np.linalg.norm(features - mean_features)
            distances[class_id] = distance
            print(f"  Cat {class_id}와의 거리: {distance:.4f}")
        
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
    
    def print_classification_result(self, result):
        """분류 결과 출력"""
        if result is None:
            print("분류 실패")
            return
        
        print("\n=== 분류 결과 ===")
        predicted_class = result['predicted_class']
        class_name = '알 수 없음' if predicted_class == 0 else f'Cat {predicted_class}'
        
        print(f"예측 클래스: {class_name}")
        print(f"신뢰도: {result['confidence']:.4f}")
        print(f"최소 거리: {result['min_distance']:.4f}")
        print(f"임계값: {result['threshold']:.4f}")
        
        print("\n클래스별 거리:")
        for class_id, distance in result['distances'].items():
            status = "✓" if class_id == result['predicted_class'] and result['predicted_class'] != 0 else ""
            print(f"  Cat {class_id}: {distance:.4f} {status}")

def main():
    # 설정 로드
    config = Config()
    
    # 파일 경로 설정
    model_path = "output/best_model.pth"
    feature_stats_path = "feature_analysis_output/detailed_results.json"
    
    # 테스트할 이미지 경로 (사용자가 수정 필요)
    test_image_path = "./datasets/2/2025-02-28.20-06-33.ecbb4323-21e4-420f-ac03-aa59045f0ab1_frame0_cat_0.jpg"  # 실제 테스트 이미지 경로로 수정
    
    # 임계값 설정
    threshold = 1.0  # 필요에 따라 조정
    
    try:
        # 분류기 생성
        classifier = CatClassifier(config, model_path, feature_stats_path)
        
        # 이미지 분류
        result = classifier.classify_image(test_image_path, threshold)
        
        # 결과 출력
        classifier.print_classification_result(result)
        
    except FileNotFoundError as e:
        print(f"파일 오류: {e}")
        print("다음 파일들이 존재하는지 확인하세요:")
        print(f"  - 모델 파일: {model_path}")
        print(f"  - 특징점 통계 파일: {feature_stats_path}")
        print(f"  - 테스트 이미지: {test_image_path}")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main() 