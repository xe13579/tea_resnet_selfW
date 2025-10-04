"""
TEA-ResNet50预训练模型 - 基于ImageNet预训练权重
作用：在TEA-ResNet50基础上加载ResNet50预训练权重（仅映射兼容层），TEA模块随机初始化
统一 Dropout=0.1
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from MeModel import MotionExcitation
from MtaModel import MultipleTemporalAggregation


class TEABottleneckBlockPre(nn.Module):
    """TEA增强的Bottleneck块（预训练友好）"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, num_segments=8, me_use_gn: bool = True):
        super().__init__()
        self.num_segments = num_segments
        self.stride = stride

        # 1x1 降维（可加载预训练）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # TEA 模块（统一 Dropout=0.1）
        self.me = MotionExcitation(out_channels, dropout_rate=0.1, use_gn=me_use_gn)
        self.mta = MultipleTemporalAggregation(out_channels, num_segments, dropout_rate=0.1)

        # TEA 输出 BN
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 升维（可加载预训练）
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # 残差下采样（可加载预训练）
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B*T, C, H, W)
        assert x.dim() == 4, f"期望输入(B*T,C,H,W)，得到{x.shape}"
        assert x.size(0) % self.num_segments == 0, f"batch维{ x.size(0) }不能被num_segments={self.num_segments}整除"

        identity = x

        # 1x1 降维
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # ME + MTA 在时间维上运行
        BT, C, H, W = out.size()
        B = BT // self.num_segments
        T = self.num_segments
        out_5d = out.view(B, T, C, H, W)
        out_5d = self.me(out_5d)
        out_5d = self.mta(out_5d)
        # 可能非连续，使用 reshape 还原
        out = out_5d.reshape(BT, C, H, W)

        # 等效下采样（保持与原ResNet步幅一致）
        if self.stride != 1:
            out = F.avg_pool2d(out, kernel_size=1, stride=self.stride)

        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 升维
        out = self.conv3(out)
        out = self.bn3(out)

        # 残差分支
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class TEAResNet50Pre(nn.Module):
    """TEA-ResNet50 预训练模型（仅映射兼容层）"""

    def __init__(self, num_classes=101, num_segments=8, dropout_rate=0.1,
                 pretrained=True, freeze_backbone=False, me_use_gn: bool = True):
        super().__init__()
        self.num_segments = num_segments
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        self.me_use_gn = me_use_gn

        # stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet50 布局 [3,4,6,3]
        self.layer1 = self._make_tea_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_tea_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_tea_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_tea_layer(1024, 512, 3, stride=2)

        # 分类头（Dropout=0.1）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(2048, num_classes)

        # 预训练 or 随机初始化
        if pretrained:
            self._load_pretrained_weights()
        else:
            self._initialize_weights()

        if freeze_backbone:
            self._freeze_backbone()

    def _make_tea_layer(self, in_channels, base_channels, num_blocks, stride):
        layers = []
        layers.append(TEABottleneckBlockPre(in_channels, base_channels, stride, self.num_segments, me_use_gn=self.me_use_gn))
        in_channels = base_channels * TEABottleneckBlockPre.expansion
        for _ in range(1, num_blocks):
            layers.append(TEABottleneckBlockPre(in_channels, base_channels, 1, self.num_segments, me_use_gn=self.me_use_gn))
        return nn.Sequential(*layers)

    def _load_pretrained_weights(self):
        print("📦 正在加载ImageNet ResNet50预训练权重(仅映射兼容层)...")
        try:
            from torchvision.models import ResNet50_Weights
            pretrained_resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            pretrained_resnet = models.resnet50(pretrained=True)

        pretrained_dict: Dict[str, torch.Tensor] = pretrained_resnet.state_dict()
        model_dict = self.state_dict()

        loaded_count = 0
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v
                loaded_count += 1
        self.load_state_dict(model_dict, strict=False)
        print(f"✅ 预训练权重加载完成: {loaded_count}/{len(model_dict)} 个参数匹配 (TEA相关层已跳过)")

    def _initialize_weights(self):
        print("🎲 使用Kaiming/默认策略随机初始化权重...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _freeze_backbone(self):
        print("🔒 冻结backbone参数(仅训练fc)...")
        freeze_count = 0
        for name, param in self.named_parameters():
            if not name.startswith('fc.'):
                param.requires_grad = False
                freeze_count += 1
        print(f"   🧊 已冻结 {freeze_count} 个参数")

    def unfreeze_backbone(self):
        print("🔓 解冻backbone参数...")
        unfreeze_count = 0
        for param in self.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfreeze_count += 1
        print(f"   🔥 已解冻 {unfreeze_count} 个参数")

    def forward(self, x):
        # x: (B*T, C, H, W)
        assert x.dim() == 4, f"期望输入为(B*T,C,H,W)，得到{x.shape}"
        assert x.size(0) % self.num_segments == 0, f"batch维{ x.size(0) }不能被num_segments={self.num_segments}整除"

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 时间维聚合
        BT, feat = x.size()
        B = BT // self.num_segments
        x = x.view(B, self.num_segments, feat).mean(dim=1)

        x = self.dropout(x)
        x = self.fc(x)
        return x


def tea_resnet50_pretrained(num_classes=101, num_segments=8, dropout_rate=0.1,
                            pretrained=True, freeze_backbone=False, me_use_gn: bool = True):
    """创建TEA-ResNet50预训练模型（统一Dropout=0.1）"""
    return TEAResNet50Pre(num_classes, num_segments, dropout_rate, pretrained, freeze_backbone, me_use_gn)

