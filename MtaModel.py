"""
多重时间聚合(Multiple Temporal Aggregation, MTA)模块 - 论文标准实现
作用：将输入特征分成四组，每组采用不同的时序聚合策略
"""
import torch
import torch.nn as nn

class MultipleTemporalAggregation(nn.Module):
    def __init__(self, channels, num_segments=8, dropout_rate=0.1):
        super(MultipleTemporalAggregation, self).__init__()
        self.channels = channels
        self.num_segments = num_segments
        assert channels % 4 == 0, f"通道数 {channels} 必须能被4整除"
        self.group_channels = channels // 4

        # 第二片段：ConvT -> ConvS -> GN -> ReLU -> Dropout
        self.conv_temp_2 = nn.Conv3d(
            self.group_channels, self.group_channels,
            kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False
        )
        self.conv_spa_2 = nn.Conv3d(
            self.group_channels, self.group_channels,
            kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
        )
        self.gn_spa_2 = nn.GroupNorm(8, self.group_channels)
        self.dropout_2 = nn.Dropout3d(dropout_rate)

        # 第三片段：ConvT -> ConvS -> GN -> ReLU -> Dropout（带残差）
        self.conv_temp_3 = nn.Conv3d(
            self.group_channels, self.group_channels,
            kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False
        )
        self.conv_spa_3 = nn.Conv3d(
            self.group_channels, self.group_channels,
            kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
        )
        self.gn_spa_3 = nn.GroupNorm(8, self.group_channels)
        self.dropout_3 = nn.Dropout3d(dropout_rate)

        # 第四片段：ConvT -> ConvS -> GN -> ReLU -> Dropout（带残差）
        self.conv_temp_4 = nn.Conv3d(
            self.group_channels, self.group_channels,
            kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False
        )
        self.conv_spa_4 = nn.Conv3d(
            self.group_channels, self.group_channels,
            kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
        )
        self.gn_spa_4 = nn.GroupNorm(8, self.group_channels)
        self.dropout_4 = nn.Dropout3d(dropout_rate)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # (N, C, T, H, W)

        group_size = C // 4
        x1 = x[:, 0*group_size:1*group_size, :, :, :]
        x2 = x[:, 1*group_size:2*group_size, :, :, :]
        x3 = x[:, 2*group_size:3*group_size, :, :, :]
        x4 = x[:, 3*group_size:4*group_size, :, :, :]

        # 1) 第一片段：恒等映射
        out1 = x1

        # 2) 第二片段
        temp_out2 = self.conv_temp_2(x2)
        out2 = self.conv_spa_2(temp_out2)
        out2 = self.gn_spa_2(out2)
        out2 = self.relu(out2)
        out2 = self.dropout_2(out2)

        # 3) 第三片段：残差 + 时序->空间
        residual_input3 = x3 + out2
        temp_out3 = self.conv_temp_3(residual_input3)
        out3 = self.conv_spa_3(temp_out3)
        out3 = self.gn_spa_3(out3)
        out3 = self.relu(out3)
        out3 = self.dropout_3(out3)

        # 4) 第四片段：残差 + 时序->空间
        residual_input4 = x4 + out3
        temp_out4 = self.conv_temp_4(residual_input4)
        out4 = self.conv_spa_4(temp_out4)
        out4 = self.gn_spa_4(out4)
        out4 = self.relu(out4)
        out4 = self.dropout_4(out4)

        output = torch.cat([out1, out2, out3, out4], dim=1)
        output = output.permute(0, 2, 1, 3, 4)
        return output