"""
运动激励(Motion Excitation, ME)模块（无BN/无Dropout；可选GN）
改动：
- 去除 BN（小 batch 更稳），但在 1x1 降维与 1x1 升维后各加一个 GroupNorm（可选，默认开启）
- 差分前不做 ReLU（保留运动方向/符号）
- 注意力门控使用 y = sigmoid(att) - 0.5，范围 [-0.5, 0.5]
- 不在注意力权重上做 Dropout
"""
import torch
import torch.nn as nn

def _make_gn(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """选择不超过 max_groups 的分组数，且能整除通道数；若不整除则退化为 GN(1,C)。"""
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return nn.GroupNorm(g, num_channels)


class MotionExcitation(nn.Module):
    def __init__(self, channels, reduction=16, dropout_rate: float = 0.0, use_gn: bool = True, gn_max_groups: int = 32):
        super(MotionExcitation, self).__init__()
        self.channels = channels
        self.reduction = reduction

        mid = channels // reduction
        assert mid > 0 and channels % reduction == 0, "channels 应能被 reduction 整除"

        # 1x1 降维
        self.conv_red = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.gn_red = _make_gn(mid, gn_max_groups) if use_gn else nn.Identity()

        # 对 t+1 卷积
        self.conv_trans = nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False)

        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1x1 升维
        self.conv_exp = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.gn_exp = _make_gn(channels, gn_max_groups) if use_gn else nn.Identity()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (N, T, C, H, W)
        N, T, C, H, W = x.size()

        # 降维
        x_flat = x.reshape(N * T, C, H, W)
        x_r_flat = self.conv_red(x_flat)  # (N*T, C//r, H, W)
        x_r_flat = self.gn_red(x_r_flat)
        x_r = x_r_flat.view(N, T, -1, H, W)

        # 运动差分：仅对 t+1 做 depthwise，再与 t 做差
        if T > 1:
            x_r_prev = x_r[:, :-1]  # (N, T-1, C//r, H, W)
            x_r_next = x_r[:, 1:]   # (N, T-1, C//r, H, W)

            x_r_next_flat = x_r_next.reshape(-1, x_r_next.size(2), H, W)
            x_r_next_trans = self.conv_trans(x_r_next_flat)  # (N*(T-1), C//r, H, W)
            x_r_next_trans = x_r_next_trans.view(N, T-1, -1, H, W)

            motion = x_r_next_trans - x_r_prev  # (N, T-1, C//r, H, W)
            zero_motion = torch.zeros(N, 1, motion.size(2), H, W, device=x.device, dtype=x.dtype)
            motion = torch.cat([motion, zero_motion], dim=1)  # (N, T, C//r, H, W)
        else:
            motion = torch.zeros_like(x_r)

        # 池化 + 升维
        motion_flat = motion.reshape(N * T, -1, H, W)
        motion_pooled = self.global_avg_pool(motion_flat)  # (N*T, C//r, 1, 1)
        att_logits = self.conv_exp(motion_pooled)          # (N*T, C, 1, 1)
        att_logits = self.gn_exp(att_logits)

        # [-0.5, 0.5]
        gate = self.sigmoid(att_logits) - 0.5              # (N*T, C, 1, 1)
        gate = gate.view(N, T, C, 1, 1)

        out = x + x * gate
        return out
