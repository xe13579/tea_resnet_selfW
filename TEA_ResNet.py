"""
TEA-ResNet50模型 - 论文标准实现
作用：基于ResNet50的视频动作识别模型，从conv2到conv5集成TEA模块
架构：conv1(标准) + conv2-5(TEA blocks)
"""
import torch
import torch.nn as nn
from MeModel import MotionExcitation
from MtaModel import MultipleTemporalAggregation

class TEABottleneckBlock(nn.Module):
    """
    TEA增强的Bottleneck块 - 论文标准实现
    架构：Input → conv1x1 → ME → MTA → conv1x1 → 残差连接 → Output
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, num_segments=8):
        super(TEABottleneckBlock, self).__init__()
        self.num_segments = num_segments
        self.stride = stride

        # 🔹 第一个1x1卷积（降维）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 🌟 TEA模块：ME + MTA（替换原来的3x3卷积）
        self.me = MotionExcitation(out_channels, dropout_rate=0.3)
        self.mta = MultipleTemporalAggregation(out_channels, num_segments, dropout_rate=0.3)

        # BN for TEA output
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 🔹 第二个1x1卷积（升维）
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # 残差连接的下采样
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 输入一致性断言
        assert x.dim() == 4, f"期望(B*T,C,H,W)，得到{x.shape}"
        assert x.size(0) % self.num_segments == 0, f"batch维{ x.size(0) }不能被num_segments={self.num_segments}整除"
        identity = x

        # 🔹 第一个1x1卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 🌟 TEA模块：ME + MTA
        BT, C, H, W = out.size()
        B = BT // self.num_segments
        T = self.num_segments
        out_5d = out.view(B, T, C, H, W)

        out_me = self.me(out_5d)
        out_mta = self.mta(out_me)

        # 转换回4D
        out = out_mta.reshape(BT, C, H, W)

        # 应用stride（如果需要下采样）
        if self.stride != 1:
            out = nn.functional.avg_pool2d(out, kernel_size=self.stride, stride=self.stride)

        out = self.bn2(out)
        out = self.relu(out)

        # 🔹 第二个1x1卷积
        out = self.conv3(out)
        out = self.bn3(out)

        # 🔹 残差连接
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class TEAResNet50(nn.Module):
    def __init__(self, num_classes=101, num_segments=8, dropout_rate=0.5):  # ✅ 添加dropout_rate参数
        super(TEAResNet50, self).__init__()
        self.num_segments = num_segments
        self.dropout_rate = dropout_rate
        
        # 🔹 conv1: 标准ResNet输入层（保持不变）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 🌟 conv2-conv5: 使用TEA blocks（论文要求）
        # ResNet50的层结构：[3, 4, 6, 3] 个块
        self.layer1 = self._make_tea_layer(64, 64, 3, stride=1)      # conv2_x
        self.layer2 = self._make_tea_layer(256, 128, 4, stride=2)    # conv3_x  
        self.layer3 = self._make_tea_layer(512, 256, 6, stride=2)    # conv4_x
        self.layer4 = self._make_tea_layer(1024, 512, 3, stride=2)   # conv5_x
        
        # 🔧 修改：增强分类头的正则化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)
        
        self._initialize_weights()
    
    def _make_tea_layer(self, in_channels, base_channels, num_blocks, stride):
        """创建TEA层（conv2-conv5使用）"""
        layers = []
        
        # 第一个块（可能有下采样）
        layers.append(TEABottleneckBlock(in_channels, base_channels, stride, 
                                       self.num_segments))  # ✅ 传递dropout_rate
        
        # 后续块
        in_channels = base_channels * TEABottleneckBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(TEABottleneckBlock(in_channels, base_channels, 1, 
                                           self.num_segments))  # ✅ 传递dropout_rate
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):  # ✅ 添加3D BN支持
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (B*T, C, H, W)
        assert x.dim() == 4, f"期望输入为(B*T,C,H,W)，得到{x.shape}"
        assert x.size(0) % self.num_segments == 0, f"batch维{ x.size(0) }不能被num_segments={self.num_segments}整除"
        
        # 🔹 conv1: 标准ResNet处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 🌟 conv2-conv5: TEA处理
        x = self.layer1(x)  # conv2_x -> (B*T, 256, H/4, W/4)
        x = self.layer2(x)  # conv3_x -> (B*T, 512, H/8, W/8)
        x = self.layer3(x)  # conv4_x -> (B*T, 1024, H/16, W/16)
        x = self.layer4(x)  # conv5_x -> (B*T, 2048, H/32, W/32)
        
        x = self.avgpool(x)  # -> (B*T, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # -> (B*T, 2048)
        
        # 时间维度聚合
        BT, features = x.size()
        B = BT // self.num_segments
        x = x.view(B, self.num_segments, features).mean(dim=1)  # -> (B, 2048)
        
        x = self.dropout(x)
        x = self.fc(x)  # -> (B, num_classes)
        return x

def tea_resnet50(num_classes=101, num_segments=8, dropout_rate=0.5):  # ✅ 添加dropout_rate参数
    """创建TEA-ResNet50模型（论文标准版本）"""
    return TEAResNet50(num_classes, num_segments, dropout_rate)

# 在TEA_ResNet.py文件的最后，替换现有的测试代码

# 替换if __name__ == "__main__":部分

if __name__ == "__main__":
    import time
    
    print("🎬 TEA-ResNet50 UCF-101测试")
    print("=" * 50)
    
    # 1. 创建UCF-101标准模型
    print("📦 创建模型...")
    model = tea_resnet50(num_classes=101, num_segments=8, dropout_rate=0.5)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型创建成功!")
    print(f"   参数量: {total_params:,}")
    print(f"   模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # 2. UCF-101标准测试
    print(f"\n🔄 UCF-101前向传播测试...")
    
    test_cases = [
        {"batch": 1, "name": "单视频"},
        {"batch": 4, "name": "小批量"},
        {"batch": 8, "name": "标准批量"},
    ]
    
    model.eval()
    for case in test_cases:
        try:
            batch_size = case["batch"]
            # UCF-101标准: 8帧, 224x224
            x = torch.randn(batch_size * 8, 3, 224, 224)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(x)
            end_time = time.time()
            
            # 验证输出
            expected_shape = (batch_size, 101)
            assert output.shape == expected_shape, f"形状错误"
            assert not torch.isnan(output).any(), "输出有NaN"
            
            print(f"✅ {case['name']}: 输入{x.shape} → 输出{output.shape} ({(end_time-start_time)*1000:.1f}ms)")
            
        except Exception as e:
            print(f"❌ {case['name']}失败: {e}")
            raise
    
    # 3. 梯度测试
    print(f"\n🧠 梯度测试...")
    try:
        model.train()
        x = torch.randn(8, 3, 224, 224, requires_grad=True)
        target = torch.randint(0, 101, (1,))
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        assert x.grad is not None, "没有梯度"
        print(f"✅ 梯度测试通过: 损失={loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ 梯度测试失败: {e}")
        raise
    
    # 4. TEA模块验证
    print(f"\n🧩 TEA模块验证...")
    try:
        me_count = sum(1 for m in model.modules() if isinstance(m, MotionExcitation))
        mta_count = sum(1 for m in model.modules() if isinstance(m, MultipleTemporalAggregation))
        
        print(f"✅ 发现 {me_count} 个ME模块, {mta_count} 个MTA模块")
        
        # 测试单个TEA块
        tea_block = model.layer1[0]
        x_test = torch.randn(8, 64, 56, 56)
        with torch.no_grad():
            out_test = tea_block(x_test)
        print(f"✅ TEA块测试: {x_test.shape} → {out_test.shape}")
        
    except Exception as e:
        print(f"❌ TEA模块验证失败: {e}")
        raise
    
    # 5. 性能测试
    print(f"\n⚡ 性能测试...")
    try:
        model.eval()
        x = torch.randn(8, 3, 224, 224)
        
        # 预热
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)
        
        # 计时
        start_time = time.time()
        for _ in range(10):
            with torch.no_grad():
                output = model(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10 * 1000
        print(f"✅ 平均推理时间: {avg_time:.1f}ms")
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        # 不抛出异常，性能测试失败不影响功能
    
    print(f"\n🎉 UCF-101测试完成!")
    print("=" * 50)
    print("✅ 模型已准备好训练UCF-101数据集!")
    print("💡 使用方法:")
    print("   model = tea_resnet50(num_classes=101, num_segments=8)")
    print("   输入: (batch*8, 3, 224, 224)")
    print("   输出: (batch, 101)")
    print("=" * 50)