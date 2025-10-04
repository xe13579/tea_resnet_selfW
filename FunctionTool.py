"""
工具函数模块
作用：提供常用的辅助函数
"""
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def set_seed(seed=42):
    """设置随机种子，确保实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 随机种子设置为: {seed}")

def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 精度曲线
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 训练曲线保存至: {save_path}")
    
    plt.show()

def print_model_summary(model, input_size):
    """打印模型摘要"""
    total_params, trainable_params = count_parameters(model)
    
    print("=" * 60)
    print("🤖 TEA-ResNet 模型摘要")
    print("=" * 60)
    print(f"📊 总参数量: {total_params:,}")
    print(f"🎯 可训练参数: {trainable_params:,}")
    print(f"💾 模型大小: {total_params * 4 / (1024**2):.2f} MB")
    print(f"📐 输入尺寸: {input_size}")
    print("=" * 60)