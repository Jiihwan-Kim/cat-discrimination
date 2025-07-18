import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class CatDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            if self.is_training:
                # 대조 학습을 위해 두 개의 다른 증강을 적용
                image1 = self.transform(image)
                image2 = self.transform(image)
                return (image1, image2), label
            else:
                return self.transform(image), label
            
        return image, label

def get_transforms(config, is_training=True):
    """데이터 증강 및 전처리 변환"""
    if is_training:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def create_data_loaders(config):
    """데이터셋을 train/validation으로 분할하고 DataLoader 생성"""
    image_paths = []
    labels = []
    
    # 각 클래스별로 이미지 경로와 라벨 수집
    for class_id in range(1, config.num_classes + 1):
        class_dir = os.path.join(config.data_root, str(class_id))
        if os.path.exists(class_dir):
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(class_id - 1)  # 0-based indexing
    
    # train/validation 분할
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, 
        test_size=1-config.train_ratio, 
        stratify=labels, 
        random_state=42
    )
    
    # 변환 생성
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    # 데이터셋 생성
    train_dataset = CatDataset(train_paths, train_labels, train_transform, is_training=True)
    val_dataset = CatDataset(val_paths, val_labels, val_transform, is_training=False)
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset) 