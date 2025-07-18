import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm
import numpy as np

from config import Config
from dataset import create_data_loaders
from model import CatDiscriminationModel, contrastive_loss
from utils import AverageMeter, accuracy, save_checkpoint

def train_epoch(model, train_loader, criterion, optimizer, config, epoch):
    """한 에포크 학습"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, ((images1, images2), labels) in enumerate(pbar):
        images = torch.cat([images1, images2], dim=0).to(config.device)
        labels = labels.to(config.device)
        
        # Forward pass
        logits, features = model(images)
        
        # 특징 벡터를 원래 페어로 분리
        f1, f2 = torch.chunk(features, 2, dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # Loss 계산
        ce_loss = criterion(logits, torch.cat([labels, labels], dim=0))
        contrastive_loss_val = contrastive_loss(features, labels, config.temperature)
        
        # 전체 loss (contrastive loss와 classification loss 결합)
        total_loss = ce_loss + 0.1 * contrastive_loss_val
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 메트릭 업데이트
        prec1 = accuracy(logits, torch.cat([labels, labels], dim=0), topk=(1,))[0]
        losses.update(total_loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        
        # Progress bar 업데이트
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{top1.avg:.2f}%',
            'CE': f'{ce_loss.item():.4f}',
            'CL': f'{contrastive_loss_val.item():.4f}'
        })
    
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, config):
    """검증"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(config.device)
            labels = labels.to(config.device)
            
            # Forward pass
            logits, _ = model(images)
            
            # Loss 계산 (Cross-Entropy만)
            loss = criterion(logits, labels)
            
            # 메트릭 업데이트
            prec1 = accuracy(logits, labels, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
    
    return losses.avg, top1.avg

def main():
    config = Config()
    
    # 데이터 로더 생성
    train_loader, val_loader, train_size, val_size = create_data_loaders(config)
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    # 모델 생성
    model = CatDiscriminationModel(config).to(config.device)
    print(f"Model loaded: {config.dino_model_name}")
    
    # Loss function과 optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    # TensorBoard
    writer = SummaryWriter(config.log_dir)
    
    # 학습 루프
    best_acc = 0.0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # 학습
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config, epoch
        )
        
        # 검증
        val_loss, val_acc = validate(model, val_loader, criterion, config)
        
        # Learning rate 업데이트
        scheduler.step()
        
        # 로깅
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                os.path.join(config.save_dir, 'best_model.pth')
            )
            print(f"New best model saved! Accuracy: {best_acc:.2f}%")
        
        # 주기적 저장
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                os.path.join(config.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    writer.close()
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main() 