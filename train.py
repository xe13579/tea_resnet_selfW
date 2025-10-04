"""
TEA-ResNet主训练脚本
作用：模块化训练入口，支持命令行参数调整
使用方法：
    python train.py --epochs 50 --lr 0.001 --batch_size 8
    python train.py --resume ./checkpoints/checkpoint_best.pth
"""
import sys
import os
sys.path.append('.')

# 导入模块
from ConfigBlock import get_config
from TEAtrainer import TEATrainer
from FunctionTool import set_seed, print_model_summary, plot_training_curves
from TEA_ResNet_Pre import tea_resnet50_pretrained
from dataset import get_ucf101_loaders
import torch

def main():
    # 获取配置
    config = get_config()
    
    # 检查数据集路径
    if not os.path.exists(config.data_root):
        print(f"❌ 数据集路径不存在: {config.data_root}")
        print("请确保UCF-101文件夹在当前目录下")
        return
    
    if not os.path.exists("ucfTrainTestlist"):
        print(f"❌ 标注文件路径不存在: ucfTrainTestlist")
        print("请确保ucfTrainTestlist文件夹在当前目录下")
        return
    
    # 设置随机种子
    set_seed(42)
    
    # 创建模型
    print("🔧 创建TEA-ResNet50模型...")
    used_pretrained = True
    try:
        model = tea_resnet50_pretrained(
            num_classes=config.num_classes,
            num_segments=config.num_segments,
            pretrained=True,        # 优先尝试预训练
            freeze_backbone=False   # 全模型微调模式
        )
    except Exception as e:
        print(f"⚠️  预训练权重加载失败，回退到随机初始化: {e}")
        used_pretrained = False
        model = tea_resnet50_pretrained(
            num_classes=config.num_classes,
            num_segments=config.num_segments,
            pretrained=False,
            freeze_backbone=False
        )
    print(f"🧠 使用预训练权重: {'是' if used_pretrained else '否'}")
    
    # 打印模型信息
    input_size = (config.batch_size * config.num_segments, 3, 224, 224)
    print_model_summary(model, input_size)
    
    # 创建数据加载器
    print("📁 创建数据加载器...")
    try:
        train_loader, val_loader = get_ucf101_loaders(config)
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return
    
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 创建训练器
    print("🎯 初始化训练器...")
    trainer = TEATrainer(model, train_loader, val_loader, config)
    
    # 如果有检查点，加载它
    if config.resume:
        if config.reset_optimizer:
            print("🔄 重置优化器模式：只加载模型权重，重新开始学习率调度")
            trainer.load_checkpoint(config.resume, reset_optimizer=True)
        else:
            print("🔄 继续训练模式：加载完整状态")
            trainer.load_checkpoint(config.resume, reset_optimizer=False)

    # 仅评估模式：直接在验证/测试集上跑一次，不进行训练
    if getattr(config, 'eval_only', False):
        ckpt_path = config.resume or os.path.join(config.save_dir, 'checkpoint_best.pth')
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"🧪 仅评估模式：加载检查点 {ckpt_path}")
            trainer.load_checkpoint(ckpt_path, reset_optimizer=True)
        else:
            print(f"❌ 未找到检查点：{ckpt_path}。请使用 --resume 指定已有模型。")
            return

        val_loss, val_acc = trainer.validate()
        print(f"[Eval/Test] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        try:
            train_eval_loss, train_eval_acc = trainer.evaluate_loader(trainer.train_loader)
            print(f"[Train Eval] Loss: {train_eval_loss:.4f} | Acc: {train_eval_acc:.2f}%")
        except Exception as e:
            print(f"⚠️  训练集Eval评估失败: {e}")
        return
    
    # 开始训练
    print("🚀 开始训练...")
    best_acc = trainer.train()
    
    # 绘制训练曲线
    plot_training_curves(
        trainer.train_losses, trainer.val_losses,
        trainer.train_accs, trainer.val_accs,
        save_path=f"{config.save_dir}/training_curves.png"
    )
    
    # 追加：在训练集上以 eval 模式再评估一次，检查BN/Dropout差异
    try:
        print("\n🧪 训练集Eval评估(关闭Dropout/使用BN running stats)...")
        train_eval_loss, train_eval_acc = trainer.evaluate_loader(trainer.train_loader)
        print(f"[Train Eval] Loss: {train_eval_loss:.4f} | Acc: {train_eval_acc:.2f}%")
    except Exception as e:
        print(f"⚠️  训练集Eval评估失败: {e}")

    print(f"🏆 训练完成! 最佳精度: {best_acc:.2f}%")

if __name__ == "__main__":
    main()