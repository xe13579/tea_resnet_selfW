"""
UCF-101数据集加载器
作用：加载视频数据，进行预处理，支持训练和验证
"""
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from pathlib import Path

class UCF101Dataset(Dataset):
    def __init__(self, data_root, split_file, num_segments=8, transform=None, is_train=True, augment_factor=1):
        """
        Args:
            data_root: UCF-101数据集根目录
            split_file: 分割文件路径 (trainlist01.txt 或 testlist01.txt)
            num_segments: 每个视频采样的帧数
            transform: 数据增强变换
            is_train: 是否为训练模式
            augment_factor: 训练时每个视频的增强倍数（此处忽略，固定为1）
        """
        self.data_root = Path(data_root)
        self.num_segments = num_segments
        self.transform = transform
        self.is_train = is_train
        # 去掉增强：训练与验证均不做样本倍增
        self.augment_factor = 1

        # 加载类别映射
        self.class_to_idx = self._load_class_mapping()

        # 加载视频列表
        self.video_list = self._load_video_list(split_file)

        # 不使用累加策略：样本总数等于视频数
        self.total_samples = len(self.video_list)

        print(f"📊 {'训练' if is_train else '验证'}集: {len(self.video_list)} 个视频（无数据增强）")
    
    def _load_class_mapping(self):
        """加载类别映射"""
        class_file = Path("ucfTrainTestlist/classInd.txt")
        class_to_idx = {}
        
        with open(class_file, 'r') as f:
            for line in f:
                idx, class_name = line.strip().split()
                class_to_idx[class_name] = int(idx) - 1  # 转换为0-based索引
        
        return class_to_idx
    
    def _load_video_list(self, split_file):
        """加载视频列表"""
        video_list = []
        
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if self.is_train:
                    # 训练集格式: video_path class_index
                    parts = line.split()
                    video_path = parts[0]
                    label = int(parts[1]) - 1  # 转换为0-based
                else:
                    # 测试集格式: video_path (需要从路径推断标签)
                    video_path = line
                    class_name = video_path.split('/')[0]
                    label = self.class_to_idx[class_name]
                
                # 检查视频文件是否存在
                full_path = self.data_root / video_path
                if full_path.exists():
                    video_list.append((video_path, label))
                else:
                    print(f"⚠️  视频文件不存在: {full_path}")
        
        return video_list
    
    def _load_video(self, video_path):
        """加载视频并采样帧"""
        full_path = self.data_root / video_path
        
        # 使用OpenCV读取视频
        cap = cv2.VideoCapture(str(full_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 转换BGR到RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"无法读取视频: {video_path}")
        
        # TSN风格随机/中心采样指定数量的帧
        return self._sample_frames(frames)
    
    def _sample_frames(self, frames):
        """
        TSN风格的随机采样：将视频分为num_segments个片段，每个片段内随机选择一帧
        """
        total_frames = len(frames)
        
        if total_frames <= self.num_segments:
            # 🔥 修复短视频问题：使用更智能的重复策略
            indices = []
            # 先均匀分布现有帧
            step = total_frames / self.num_segments
            for i in range(self.num_segments):
                idx = min(int(i * step), total_frames - 1)
                indices.append(idx)
        else:
            # 🎲 TSN风格随机采样
            # 将视频分为num_segments个片段，每个片段内随机选择一帧
            indices = []
            segment_length = total_frames / self.num_segments
            
            for i in range(self.num_segments):
                # 计算当前片段的起始和结束位置
                start_idx = int(i * segment_length)
                end_idx = int((i + 1) * segment_length)
                
                # 确保不越界
                end_idx = min(end_idx, total_frames)
                
                if start_idx >= end_idx:
                    # 边界情况，使用start_idx
                    indices.append(start_idx - 1 if start_idx > 0 else 0)
                else:
                    # 在当前片段内随机选择一帧
                    if self.is_train:
                        # 训练时随机
                        random_idx = random.randint(start_idx, end_idx - 1)
                    else:
                        # 验证时固定选择中间帧，增加一致性
                        random_idx = start_idx + (end_idx - start_idx) // 2
                    indices.append(random_idx)
        
        sampled_frames = [frames[i] for i in indices]
        return sampled_frames
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # 简化：无增强，直接按索引取视频
        video_idx = idx
        
        video_path, label = self.video_list[video_idx]
        
        try:
            # 加载视频帧
            frames = self._load_video(video_path)
            
            # 单一变换（训练/验证一致）
            frames = [self.transform(frame) for frame in frames]
            
            # 将帧列表转换为张量 (T, C, H, W)
            video_tensor = torch.stack(frames)
            
            return video_tensor, label
            
        except Exception as e:
            print(f"❌ 加载视频失败: {video_path} (样本{idx}), 错误: {e}")
            # 返回第一个视频作为默认值
            return self.__getitem__(0)

def get_transforms(is_train=True):
    """
    简化版数据变换（无数据增强，训练/验证一致）
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
    # 保持纵横比：将短边缩放到256，再中心裁剪到224，避免拉伸变形
    transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform

def custom_collate_fn(batch):
    """自定义批次处理函数"""
    videos, labels = zip(*batch)
    
    # 将视频数据重塑为TEA模型需要的格式
    batch_size = len(videos)
    num_segments = videos[0].size(0)
    
    # (B, T, C, H, W) -> (B*T, C, H, W)
    videos = torch.stack(videos)  # (B, T, C, H, W)
    videos = videos.view(-1, videos.size(2), videos.size(3), videos.size(4))  # (B*T, C, H, W)
    
    labels = torch.LongTensor(labels)
    
    return videos, labels

def get_ucf101_loaders(config):
    """创建UCF-101数据加载器"""
    
    # 数据变换
    # 训练/验证使用同一套无增强变换
    train_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)
    
    # 数据集
    train_dataset = UCF101Dataset(
        data_root=config.data_root,
        split_file=f"ucfTrainTestlist/trainlist0{config.split}.txt",
        num_segments=config.num_segments,
    transform=train_transform,
    is_train=True,
    augment_factor=1
    )
    
    val_dataset = UCF101Dataset(
        data_root=config.data_root,
        split_file=f"ucfTrainTestlist/testlist0{config.split}.txt",
        num_segments=config.num_segments,
    transform=val_transform,
    is_train=False,
    augment_factor=1
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=custom_collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=custom_collate_fn,
        drop_last=False
    )
    
    print(f"✅ 数据加载器创建完成!")
    print(f"   训练批次数: {len(train_loader)}")
    print(f"   验证批次数: {len(val_loader)}")
    
    return train_loader, val_loader

# 测试数据加载器
if __name__ == "__main__":
    from ConfigBlock import get_config
    
    config = get_config()
    config.data_root = "UCF-101"  # 根据你的路径调整
    config.batch_size = 2  # 测试用小batch
    
    try:
        train_loader, val_loader = get_ucf101_loaders(config)
        
        # 测试加载一个批次
        for videos, labels in train_loader:
            print(f"视频数据形状: {videos.shape}")
            print(f"标签形状: {labels.shape}")
            print(f"标签值: {labels}")
            break
            
        print("🎉 数据加载器测试成功!")
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")