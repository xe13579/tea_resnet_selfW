"""
TEA训练器模块
作用：封装训练逻辑，支持检查点保存/加载，日志记录等
"""
import os
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


class TEATrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"🚀 使用设备: {self.device}")

        # 损失函数（关闭 label smoothing）
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

        # 优化器/调度器
        self._setup_optimizer()

        # 不使用梯度裁剪
        self.grad_clip = None

        # 训练状态
        self.start_epoch = 0
        self.best_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        # 创建保存目录
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def _setup_optimizer(self):
        """设置优化器和学习率调度器"""
        self.scheduler = None

        if getattr(self.config, 'optimizer', 'adam') == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=getattr(self.config, 'momentum', 0.9),
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )

        sched = getattr(self.config, 'scheduler', None)
        if sched == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma,
            )
        elif sched == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )

    def save_checkpoint(self, epoch, is_best=False):
        """保存训练检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_acc': self.best_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'config': self.config,
        }

        latest_path = self.save_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.save_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 保存最佳模型! 验证精度: {self.best_acc:.2f}%")

    def load_checkpoint(self, checkpoint_path, reset_optimizer=False):
        """加载检查点，支持选择是否重置优化器/调度器状态"""
        print(f"📥 加载检查点: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"❌ 检查点文件不存在: {checkpoint_path}")
            return 0

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # 模型
            state = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state)
            print("✅ 模型权重加载成功")

            if not reset_optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ 优化器状态加载成功")

                sched_state = checkpoint.get('scheduler_state_dict', None)
                if self.scheduler is not None and sched_state is not None:
                    self.scheduler.load_state_dict(sched_state)
                    print("✅ 调度器状态加载成功")

                start_epoch = int(checkpoint.get('epoch', 0)) + 1
                print(f"🔄 继续训练，从第{start_epoch}轮开始")
            else:
                print("🔄 重置优化器和调度器状态")
                print(f"📊 新的初始学习率: {self.optimizer.param_groups[0]['lr']}")
                start_epoch = 0

            # 历史信息
            prev_best = float(checkpoint.get('best_acc', 0.0))
            self.best_acc = prev_best
            self.train_losses = checkpoint.get('train_losses', []) or []
            self.val_losses = checkpoint.get('val_losses', []) or []
            self.train_accs = checkpoint.get('train_accs', []) or []
            self.val_accs = checkpoint.get('val_accs', []) or []

            self.start_epoch = start_epoch
            print(f"📊 之前的最佳精度: {prev_best:.2f}% (作为起点)")

            return start_epoch
        except Exception as e:
            print(f"❌ 检查点加载失败: {e}")
            return 0

    def train_epoch(self, epoch):
        """单个epoch训练"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 前向/反向
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # 统计梯度范数（无裁剪，仅用于观察）
            total_sq = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_sq += p.grad.detach().pow(2).sum().item()
            grad_norm = math.sqrt(total_sq)

            # 无裁剪
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % self.config.print_freq == 0:
                print(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Acc: {100.*correct/total:.2f}% | "
                    f"Grad: {grad_norm:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

        avg_loss = total_loss / max(1, len(self.train_loader))
        accuracy = 100.0 * correct / max(1, total)
        return avg_loss, accuracy

    def validate(self):
        """验证函数"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / max(1, len(self.val_loader))
        accuracy = 100.0 * correct / max(1, total)
        return avg_loss, accuracy

    def evaluate_loader(self, loader):
        """在 eval 模式下评估任意 DataLoader（如训练集）"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / max(1, len(loader))
        accuracy = 100.0 * correct / max(1, total)
        return avg_loss, accuracy

    def train(self):
        """主训练循环"""
        print("🎯 开始训练...")
        print(f"训练参数: {vars(self.config)}")
        print("-" * 80)

        for epoch in range(self.start_epoch, self.config.epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            epoch_time = time.time() - start_time

            print(f"\n📊 Epoch {epoch+1}/{self.config.epochs} 结果:")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Time: {epoch_time:.2f}s | LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
            self.save_checkpoint(epoch, is_best)

            print("-" * 80)

        print(f"🎉 训练完成! 最佳验证精度: {self.best_acc:.2f}%")
        self.save_checkpoint(self.config.epochs - 1, False)
        return self.best_acc