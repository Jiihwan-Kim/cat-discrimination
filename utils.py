import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class AverageMeter:
    """평균값을 계산하는 유틸리티 클래스"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """정확도 계산"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(model, optimizer, epoch, accuracy, filename):
    """체크포인트 저장"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename):
    """체크포인트 로드"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    return epoch, accuracy

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """혼동 행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_features(features, labels, save_path=None):
    """특징 벡터 시각화 (t-SNE)"""
    from sklearn.manifold import TSNE
    
    # t-SNE로 2D로 차원 축소
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('Feature Visualization (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_loader, config):
    """모델 평가"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)
            
            logits, features = model(images)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features.extend(features.cpu().numpy())
    
    # 분류 리포트 출력
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['Cat 1', 'Cat 2', 'Cat 3']))
    
    # 혼동 행렬 시각화
    plot_confusion_matrix(all_labels, all_predictions, 
                         ['Cat 1', 'Cat 2', 'Cat 3'],
                         os.path.join(config.save_dir, 'confusion_matrix.png'))
    
    # 특징 시각화
    visualize_features(np.array(all_features), np.array(all_labels),
                      os.path.join(config.save_dir, 'feature_visualization.png'))
    
    return all_predictions, all_labels, all_features 