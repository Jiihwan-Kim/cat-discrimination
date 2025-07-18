import torch
import cv2
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from datetime import datetime

from model import CatDiscriminationModel
from config import Config
from utils import load_checkpoint

class FeatureAnalyzer:
    """ReID 모델을 사용한 특징점 분석 및 분류 성능 평가"""
    
    def __init__(self, config, model_path):
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
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return
        
        # 특징점 저장용 딕셔너리
        self.features_by_class = defaultdict(list)
        self.class_names = ['Cat 1', 'Cat 2', 'Cat 3']
        
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
    
    def load_dataset_features(self, datasets_dir):
        """datasets 폴더에서 모든 이미지의 특징점 추출"""
        print("데이터셋 특징점 추출 시작...")
        
        for class_id in [1, 2, 3]:
            class_dir = os.path.join(datasets_dir, str(class_id))
            if not os.path.exists(class_dir):
                print(f"클래스 {class_id} 디렉토리를 찾을 수 없습니다: {class_dir}")
                continue
            
            # 이미지 파일들 찾기
            image_files = glob.glob(os.path.join(class_dir, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(class_dir, "*.jpeg")))
            image_files.extend(glob.glob(os.path.join(class_dir, "*.png")))
            
            print(f"클래스 {class_id}: {len(image_files)}개 이미지 발견")
            
            # 각 이미지에서 특징점 추출
            for image_path in image_files:
                features = self.extract_features(image_path)
                if features is not None:
                    self.features_by_class[class_id].append({
                        'path': image_path,
                        'features': features
                    })
            
            print(f"클래스 {class_id}: {len(self.features_by_class[class_id])}개 특징점 추출 완료")
        
        print("모든 특징점 추출 완료!")
    
    def calculate_statistics(self):
        """각 클래스별 특징점 통계 계산"""
        print("통계 계산 시작...")
        
        stats = {}
        
        for class_id in [1, 2, 3]:
            if class_id not in self.features_by_class or len(self.features_by_class[class_id]) == 0:
                print(f"클래스 {class_id}의 특징점이 없습니다.")
                continue
            
            # 특징점들을 numpy 배열로 변환
            features_array = np.array([item['features'] for item in self.features_by_class[class_id]])
            
            # 평균과 표준편차 계산
            mean_features = np.mean(features_array, axis=0)
            std_features = np.std(features_array, axis=0)
            
            stats[class_id] = {
                'count': len(features_array),
                'mean': mean_features,
                'std': std_features,
                'features_array': features_array
            }
            
            print(f"클래스 {class_id}: {len(features_array)}개 샘플")
            print(f"  평균 특징점 차원: {mean_features.shape}")
            print(f"  평균 특징점 범위: [{mean_features.min():.4f}, {mean_features.max():.4f}]")
            print(f"  표준편차 범위: [{std_features.min():.4f}, {std_features.max():.4f}]")
        
        return stats
    
    def evaluate_classification_performance(self, stats, thresholds=None):
        """다양한 임계값에서 분류 성능 평가"""
        print("분류 성능 평가 시작...")
        
        if thresholds is None:
            # 다양한 임계값 설정
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
        
        results = []
        
        # 각 클래스의 평균 특징점
        class_means = {class_id: stats[class_id]['mean'] for class_id in stats.keys()}
        
        # 모든 특징점과 실제 라벨 준비
        all_features = []
        all_labels = []
        
        for class_id in stats.keys():
            features_array = stats[class_id]['features_array']
            all_features.extend(features_array)
            all_labels.extend([class_id] * len(features_array))
        
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        
        # 각 임계값에 대해 성능 평가
        for threshold in thresholds:
            print(f"임계값 {threshold}에서 평가 중...")
            
            # 거리 기반 분류
            predicted_labels = []
            
            for features in all_features:
                min_distance = float('inf')
                predicted_class = None
                
                for class_id, mean_features in class_means.items():
                    # 유클리드 거리 계산
                    distance = np.linalg.norm(features - mean_features)
                    
                    if distance < min_distance:
                        min_distance = distance
                        predicted_class = class_id
                
                # 임계값 적용
                if min_distance > threshold:
                    predicted_class = 0  # 알 수 없는 클래스
                
                predicted_labels.append(predicted_class)
            
            predicted_labels = np.array(predicted_labels)
            
            # 성능 메트릭 계산
            accuracy = accuracy_score(all_labels, predicted_labels)
            
            # 알 수 없는 클래스 제외하고 계산
            valid_mask = predicted_labels != 0
            if np.sum(valid_mask) > 0:
                valid_accuracy = accuracy_score(all_labels[valid_mask], predicted_labels[valid_mask])
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels[valid_mask], predicted_labels[valid_mask], average='weighted'
                )
            else:
                valid_accuracy = 0
                precision = recall = f1 = 0
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'valid_accuracy': valid_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'unknown_rate': np.mean(predicted_labels == 0)
            })
            
            print(f"  정확도: {accuracy:.4f}, 유효 정확도: {valid_accuracy:.4f}, 알 수 없음 비율: {np.mean(predicted_labels == 0):.4f}")
        
        return results
    
    def evaluate_ml_classifiers(self, stats):
        """머신러닝 분류기 성능 평가"""
        print("머신러닝 분류기 성능 평가 시작...")
        
        # 데이터 준비
        all_features = []
        all_labels = []
        
        for class_id in stats.keys():
            features_array = stats[class_id]['features_array']
            all_features.extend(features_array)
            all_labels.extend([class_id] * len(features_array))
        
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            all_features, all_labels, test_size=0.3, random_state=42, stratify=all_labels
        )
        
        # 분류기들
        classifiers = {
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'SVM': SVC(kernel='rbf', probability=True),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, clf in classifiers.items():
            print(f"{name} 분류기 훈련 중...")
            
            # 훈련
            clf.fit(X_train, y_train)
            
            # 예측
            y_pred = clf.predict(X_test)
            
            # 성능 평가
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'true_labels': y_test
            }
            
            print(f"  {name} - 정확도: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def visualize_results(self, stats, threshold_results, ml_results, output_dir):
        """결과 시각화"""
        print("결과 시각화 시작...")
        
        # 출력 디렉토리 생성
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. 특징점 분포 시각화 (PCA 사용)
        from sklearn.decomposition import PCA
        
        plt.figure(figsize=(15, 10))
        
        # 모든 특징점을 하나의 배열로
        all_features = []
        all_labels = []
        
        for class_id in stats.keys():
            features_array = stats[class_id]['features_array']
            all_features.extend(features_array)
            all_labels.extend([class_id] * len(features_array))
        
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        
        # PCA로 2차원으로 축소
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(all_features)
        
        # 산점도
        plt.subplot(2, 3, 1)
        for class_id in [1, 2, 3]:
            mask = all_labels == class_id
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       label=f'Cat {class_id}', alpha=0.7)
        
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Feature Distribution (PCA)')
        plt.legend()
        
        # 2. 임계값별 성능 그래프
        plt.subplot(2, 3, 2)
        thresholds = [r['threshold'] for r in threshold_results]
        accuracies = [r['accuracy'] for r in threshold_results]
        valid_accuracies = [r['valid_accuracy'] for r in threshold_results]
        
        plt.plot(thresholds, accuracies, 'o-', label='Overall Accuracy')
        plt.plot(thresholds, valid_accuracies, 's-', label='Valid Accuracy')
        plt.xlabel('Distance Threshold')
        plt.ylabel('Accuracy')
        plt.title('Classification Performance vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 클래스별 특징점 평균 비교
        plt.subplot(2, 3, 3)
        feature_dims = list(range(len(stats[1]['mean'])))
        
        for class_id in [1, 2, 3]:
            if class_id in stats:
                plt.plot(feature_dims, stats[class_id]['mean'], 
                        label=f'Cat {class_id} Mean', alpha=0.8)
        
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Value')
        plt.title('Mean Features by Class')
        plt.legend()
        
        # 4. 클래스별 표준편차 비교
        plt.subplot(2, 3, 4)
        for class_id in [1, 2, 3]:
            if class_id in stats:
                plt.plot(feature_dims, stats[class_id]['std'], 
                        label=f'Cat {class_id} Std', alpha=0.8)
        
        plt.xlabel('Feature Dimension')
        plt.ylabel('Standard Deviation')
        plt.title('Feature Standard Deviation by Class')
        plt.legend()
        
        # 5. 머신러닝 분류기 성능 비교
        plt.subplot(2, 3, 5)
        ml_names = list(ml_results.keys())
        ml_accuracies = [ml_results[name]['accuracy'] for name in ml_names]
        
        bars = plt.bar(ml_names, ml_accuracies, alpha=0.7)
        plt.ylabel('Accuracy')
        plt.title('ML Classifier Performance')
        plt.ylim(0, 1)
        
        # 값 표시
        for bar, acc in zip(bars, ml_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 6. 혼동 행렬 (최고 성능 분류기)
        plt.subplot(2, 3, 6)
        best_classifier = max(ml_results.keys(), key=lambda x: ml_results[x]['accuracy'])
        best_result = ml_results[best_classifier]
        
        cm = confusion_matrix(best_result['true_labels'], best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Cat 1', 'Cat 2', 'Cat 3'],
                   yticklabels=['Cat 1', 'Cat 2', 'Cat 3'])
        plt.title(f'Confusion Matrix ({best_classifier})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'feature_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 상세 결과 저장
        detailed_results = {
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                str(class_id): {
                    'count': stats[class_id]['count'],
                    'mean': stats[class_id]['mean'].tolist(),  # mean range 대신 mean 저장
                    'std': stats[class_id]['std'].tolist(),    # std range 대신 std 저장
                    'mean_range': [float(stats[class_id]['mean'].min()), float(stats[class_id]['mean'].max())],
                    'std_range': [float(stats[class_id]['std'].min()), float(stats[class_id]['std'].max())]
                } for class_id in stats.keys()
            },
            'threshold_analysis': threshold_results,
            'ml_classifier_results': {
                name: {
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1_score': result['f1_score']
                } for name, result in ml_results.items()
            }
        }
        
        with open(Path(output_dir) / 'detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"결과 저장 완료: {output_dir}")
    
    def run_analysis(self, datasets_dir, output_dir):
        """전체 분석 실행"""
        print("=== ReID 특징점 분석 및 분류 성능 평가 시작 ===")
        
        # 1. 데이터셋 특징점 추출
        self.load_dataset_features(datasets_dir)
        
        # 2. 통계 계산
        stats = self.calculate_statistics()
        
        if not stats:
            print("분석할 데이터가 없습니다.")
            return
        
        # 3. 임계값별 분류 성능 평가
        threshold_results = self.evaluate_classification_performance(stats)
        
        # 4. 머신러닝 분류기 성능 평가
        ml_results = self.evaluate_ml_classifiers(stats)
        
        # 5. 결과 시각화 및 저장
        self.visualize_results(stats, threshold_results, ml_results, output_dir)
        
        # 6. 최적 임계값 찾기
        best_threshold_result = max(threshold_results, key=lambda x: x['accuracy'])
        print(f"\n=== 최적 임계값 분석 ===")
        print(f"최적 임계값: {best_threshold_result['threshold']}")
        print(f"최고 정확도: {best_threshold_result['accuracy']:.4f}")
        print(f"최고 유효 정확도: {best_threshold_result['valid_accuracy']:.4f}")
        print(f"알 수 없음 비율: {best_threshold_result['unknown_rate']:.4f}")
        
        # 7. 최고 성능 머신러닝 분류기
        best_ml_classifier = max(ml_results.keys(), key=lambda x: ml_results[x]['accuracy'])
        best_ml_result = ml_results[best_ml_classifier]
        print(f"\n=== 최고 성능 머신러닝 분류기 ===")
        print(f"분류기: {best_ml_classifier}")
        print(f"정확도: {best_ml_result['accuracy']:.4f}")
        print(f"F1 점수: {best_ml_result['f1_score']:.4f}")
        
        print(f"\n분석 완료! 결과 저장 위치: {output_dir}")

def main():
    # 설정 로드
    config = Config()
    
    # 모델 경로 설정
    model_path = "output/best_model.pth"  # 실제 모델 경로로 수정 필요
    
    # 분석기 생성
    analyzer = FeatureAnalyzer(config, model_path)
    
    # 분석 실행
    datasets_dir = "datasets"
    output_dir = "feature_analysis_output"
    
    analyzer.run_analysis(datasets_dir, output_dir)

if __name__ == "__main__":
    main()
