print("Starting training script...")
#!/usr/bin/env python3
"""
고양이 분류 모델 학습 실행 스크립트
"""

import torch
import os
from train import main
from config import Config

if __name__ == "__main__":
    # 설정 확인
    config = Config()
    
    print("=== 고양이 분류 모델 학습 시작 ===")
    print(f"Device: {config.device}")
    print(f"DINO Model: {config.dino_model_name}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Projection Dim: {config.projection_dim}")
    print(f"Temperature: {config.temperature}")
    
    # GPU 메모리 확인
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 학습 시작
    try:
        main()
        print("=== 학습 완료 ===")
    except Exception as e:
        print(f"학습 중 오류 발생: {e}")
        raise 