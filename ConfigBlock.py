"""
配置管理模块
作用：管理所有训练参数和模型配置
"""
import argparse
from pathlib import Path


class Config:
    def __init__(self):
        # 模型参数
        self.num_classes = 101
        self.num_segments = 8
        self.reset_optimizer = False
        self.eval_only = False

        # 训练参数
        self.epochs = 100
        self.batch_size = 8
        self.lr = 0.002
        self.momentum = 0.9
        self.weight_decay = 0.0006

        # 优化器和调度器
        self.optimizer = 'sgd'  # sgd, adam
        self.scheduler = 'cosine'  # step, cosine
        self.step_size = 30
        self.gamma = 0.1

        # 数据参数
        self.data_root = 'UCF-101'
        self.split = 1  # 1, 2, 3

        # 训练控制
        self.print_freq = 20
        self.save_freq = 2
        self.save_dir = 'checkpoints'
        self.resume = None

        # 硬件
        self.num_workers = 4
        self.pin_memory = True


def get_config():
    """获取配置，支持命令行参数覆盖"""
    config = Config()

    parser = argparse.ArgumentParser(description='TEA-ResNet训练配置')

    # 模型参数
    parser.add_argument('--num_classes', type=int, default=config.num_classes,
                       help='分类类别数 (default: 101)')
    parser.add_argument('--num_segments', type=int, default=config.num_segments,
                       help='视频段数 (default: 8)')
    parser.add_argument('--reset_optimizer', action='store_true', 
                       help='重置优化器状态，从新学习率开始')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=config.epochs,
                       help='训练轮数 (default: 100)')
    parser.add_argument('--batch_size', type=int, default=config.batch_size,
                       help='批次大小 (default: 8)')
    parser.add_argument('--lr', type=float, default=config.lr,
                       help='学习率 (default: 0.002)')
    parser.add_argument('--momentum', type=float, default=config.momentum,
                       help='SGD动量 (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay,
                       help='权重衰减 (default: 6e-4)')

    # 优化器选择
    parser.add_argument('--optimizer', type=str, default=config.optimizer,
                       choices=['sgd', 'adam'], help='优化器选择')
    parser.add_argument('--scheduler', type=str, default=config.scheduler,
                       choices=['step', 'cosine'], help='学习率调度器')
    parser.add_argument('--step_size', type=int, default=config.step_size,
                       help='StepLR步长 (default: 30)')
    parser.add_argument('--gamma', type=float, default=config.gamma,
                       help='学习率衰减系数 (default: 0.1)')

    # 数据路径
    parser.add_argument('--data_root', type=str, default=config.data_root,
                       help='数据集根目录')
    parser.add_argument('--split', type=int, default=config.split,
                       choices=[1, 2, 3], help='UCF-101数据分割')

    # 训练控制
    parser.add_argument('--print_freq', type=int, default=config.print_freq,
                       help='打印频率 (default: 20)')
    parser.add_argument('--save_freq', type=int, default=config.save_freq,
                       help='保存频率 (default: 2)')
    parser.add_argument('--save_dir', type=str, default=config.save_dir,
                       help='模型保存目录')
    parser.add_argument('--resume', type=str, default=config.resume,
                       help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true', default=config.eval_only,
                       help='仅评估（不会训练），需要配合 --resume 指定检查点')

    # 硬件配置
    parser.add_argument('--num_workers', type=int, default=config.num_workers,
                       help='数据加载进程数 (default: 4)')
    parser.add_argument('--pin_memory', action='store_true', default=config.pin_memory,
                       help='是否使用pin_memory')

    args = parser.parse_args()

    # 更新配置
    for key, value in vars(args).items():
        setattr(config, key, value)

    return config