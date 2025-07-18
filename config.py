import os
from dataclasses import dataclass
import torch
@dataclass
class Config:
    # 데이터셋 설정
    data_root = "datasets"
    train_ratio = 0.8
    num_classes = 3
    
    # 모델 설정
    dino_model_name = "dinov2_vitb14"  # dinov2_vitb14, dinov2_vitl14, dinov2_vits14
    projection_dim = 128
    temperature = 0.07
    
    # 학습 설정
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-4
    
    # 데이터 증강 설정
    image_size = 224
    crop_size = 224
    
    # 기타 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    save_dir = "output"
    log_dir = "logs"
    
    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True) 