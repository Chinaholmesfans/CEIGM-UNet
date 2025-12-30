from __future__ import annotations
from collections import OrderedDict
import torch
import torch.nn as nn
from einops import rearrange
from model.vmamba.vmamba import VSSBlock,VSSBlock2, LayerNorm2d, Linear2d
from typing import Sequence, Type, Optional
import torch.nn.functional as F

class MSConv(nn.Module):
    def __init__(self, dim: int, kernel_sizes: Sequence[int] = (1, 3, 5)) -> None:
        super(MSConv, self).__init__()
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
            for kernel_size in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + sum([conv(x) for conv in self.dw_convs])

class InceptionDWConv2d(nn.Module): # 92.77

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        gc = int(in_channels * branch_ratio)  # branch的通道数
        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        # 计算每个分支的输入输出通道数：idc (原始通道数 - 3个分支的通道数)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

        self.dw_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels, bias=False)
            for kernel_size in kernel_sizes
        ])

    def forward(self, x):

        self.dx = x

        # 根据计算的split_indexes拆分输入x
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)

        # 将原始通道x_id与经过3个卷积分支处理后的x_hw、x_w、x_h合并
        x = torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,  # 在通道维度上拼接
        )

        return self.dx + x

        # return self.dx + sum([conv(x) for conv in self.dw_convs])

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels = None, kernel_sizes = [1,3,5], rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)


    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x + x1
        

class InceptionDWConv2d2(nn.Module):
    def __init__(self, in_channels, kernel_sizes = [1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Sequential(
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
            nn.Conv2d(gc, gc, 5, padding=5 // 2, groups=gc),  # 新增5x5卷积核
            nn.Conv2d(gc, gc, 7, padding=7 // 2, groups=gc),  # 新增7x7卷积核
        )
        self.dwconv_w = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.Conv2d(gc, gc, kernel_size=(1, 5), padding=(0, 5 // 2), groups=gc),  # 新增5x5卷积核
        )
        self.dwconv_h = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Conv2d(gc, gc, kernel_size=(5, 1), padding=(5 // 2, 0), groups=gc),  # 新增5x1卷积核
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        self.dx = x
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x = torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )
        return self.dx + x


class InceptionDWConv2d3(nn.Module):
    def __init__(self, in_channels, kernel_sizes = [1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Sequential(
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
        )
        self.dwconv_w = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
        )
        self.dwconv_h = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        self.dx = x
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        residual_hw = self.dwconv_hw(x_hw)
        residual_w = self.dwconv_w(x_w)
        residual_h = self.dwconv_h(x_h)
        x = torch.cat(
            (x_id, residual_hw + x_hw, residual_w + x_w, residual_h + x_h),  # 加入残差连接
            dim=1,
        )
        return self.dx + x


class InceptionDWConv2d4(nn.Module):
    def __init__(self, in_channels, kernel_sizes = [1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Sequential(
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
            nn.SiLU(inplace=True),  # 使用Swish激活函数
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
        )
        self.dwconv_w = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.SiLU(inplace=True),
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
        )
        self.dwconv_h = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.SiLU(inplace=True),
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        self.dx = x
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x = torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )
        return self.dx + x

class InceptionDWConv2d_MultiScale(nn.Module): # 92.74
    def __init__(self, in_channels,kernel_sizes = [1,3,5], branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)

        # 多尺度卷积核
        self.dwconv_3x3 = nn.Conv2d(gc, gc, kernel_size=3, padding=1, groups=gc)
        self.dwconv_5x5 = nn.Conv2d(gc, gc, kernel_size=5, padding=2, groups=gc)
        self.dwconv_7x7 = nn.Conv2d(gc, gc, kernel_size=7, padding=3, groups=gc)

        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):

        self.dx = x
        
        x_id, x_3x3, x_5x5, x_7x7 = torch.split(x, self.split_indexes, dim=1)

        # 多尺度特征融合
        out_3x3 = self.dwconv_3x3(x_3x3)
        out_5x5 = self.dwconv_5x5(x_5x5)
        out_7x7 = self.dwconv_7x7(x_7x7)

        return self.dx + torch.cat((x_id, out_3x3, out_5x5, out_7x7), dim=1)


class InceptionDWConv2d5(nn.Module): # 92.71
    def __init__(self, in_channels, kernel_sizes=[1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Sequential(
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
        )
        self.dwconv_w = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
        )
        self.dwconv_h = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return x + torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class InceptionDWConv2d6(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Sequential(
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
        )
        self.dwconv_w = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
        )
        self.dwconv_h = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

        # Feature Pyramid Fusion Layer
        self.fpn = nn.Conv2d(3 * gc, gc, 1, padding=0)
        self.fpn2 = nn.Conv2d(gc, gc * 3, 1, padding=0)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        f_hw = self.dwconv_hw(x_hw)
        f_w = self.dwconv_w(x_w)
        f_h = self.dwconv_h(x_h)
        fused_features = torch.cat((f_hw, f_w, f_h), dim=1)
        fused_features = self.fpn2(self.fpn(fused_features)) + fused_features
        return x + torch.cat((x_id, fused_features), dim=1)


class InceptionDWConv2d7(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        
        # 卷积分支
        self.dwconv_hw = nn.Sequential(
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
        )
        self.dwconv_w = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
        )
        self.dwconv_h = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
        )

        # 计算输入输出通道数的索引
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

        # Feature Pyramid Fusion Layer: 输出通道数仍然为 gc
        self.fpn = nn.Conv2d(3 * gc, gc, 1, padding=0)

    def forward(self, x):
        # 拆分输入特征图
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        
        # 每个分支的卷积操作
        f_hw = self.dwconv_hw(x_hw)
        f_w = self.dwconv_w(x_w)
        f_h = self.dwconv_h(x_h)
        
        # 融合各分支的特征
        fused_features = torch.cat((f_hw, f_w, f_h), dim=1)
        
        # Feature Pyramid Fusion
        fused_features = self.fpn(fused_features)
        
        # 合并调整后的特征与x_id
        return x + torch.cat((x_id, f_w, fused_features, f_h), dim=1)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class InceptionDWConv2d8(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Sequential(
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
            Swish(),  # 使用Swish激活
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
        )
        self.dwconv_w = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            Swish(),  # 使用Swish激活
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
        )
        self.dwconv_h = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            Swish(),  # 使用Swish激活
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return x + torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class InceptionDWConv2d9(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)

        self.dwconv_hw_3x3 = nn.Conv2d(gc, gc, 3, padding=1, groups=gc)
        self.dwconv_hw_5x5 = nn.Conv2d(gc, gc, 5, padding=2, groups=gc)
        self.dwconv_hw_7x7 = nn.Conv2d(gc, gc, 7, padding=3, groups=gc)

        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)

        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)

        f_hw = self.dwconv_hw_3x3(x_hw) + self.dwconv_hw_5x5(x_hw) + self.dwconv_hw_7x7(x_hw)
        return x + torch.cat((x_id, f_hw, self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)



class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class InceptionDWConv2d10(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)

        self.dwconv_hw = nn.Sequential(
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
            Mish()
        )
        self.dwconv_w = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            Mish()
        )
        self.dwconv_h = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            Mish()
        )

        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return x + torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)


class InceptionDWConv2d11(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)

        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)

        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

        # 添加跳跃连接
        self.skip_connection_hw = nn.Conv2d(gc, gc, 1)
        self.skip_connection_w = nn.Conv2d(gc, gc, 1)
        self.skip_connection_h = nn.Conv2d(gc, gc, 1)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        f_hw = self.dwconv_hw(x_hw) + self.skip_connection_hw(x_hw)
        f_w = self.dwconv_w(x_w) + self.skip_connection_w(x_w)
        f_h = self.dwconv_h(x_h) + self.skip_connection_h(x_h)
        return x + torch.cat((x_id, f_hw, f_w, f_h), dim=1)



class DepthwiseSeparableDilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):  # 修改dilation为1，避免过大的卷积核
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=dilation, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))  # 深度可分离卷积 + 空洞卷积

class InceptionDWConv2d12(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)

        # 修改卷积核尺寸和空洞大小
        self.dwconv_hw = DepthwiseSeparableDilatedConv2d(gc, gc, square_kernel_size, dilation=1)  # 使用 dilation=1
        self.dwconv_w = DepthwiseSeparableDilatedConv2d(gc, gc, kernel_size=(1, band_kernel_size), dilation=1)  # 使用 dilation=1
        self.dwconv_h = DepthwiseSeparableDilatedConv2d(gc, gc, kernel_size=(band_kernel_size, 1), dilation=1)  # 使用 dilation=1

        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        f_hw = self.dwconv_hw(x_hw)
        f_w = self.dwconv_w(x_w)
        f_h = self.dwconv_h(x_h)

        size = x_id.shape[2:]
        
        f_hw = F.interpolate(f_hw, size=size, mode='bilinear', align_corners=False)
        f_w = F.interpolate(f_w, size=size, mode='bilinear', align_corners=False)
        f_h = F.interpolate(f_h, size=size, mode='bilinear', align_corners=False)

        return x + torch.cat((x_id, f_hw, f_w, f_h), dim=1)


class InceptionDWConv2d13(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1,3,5], square_kernel_size=3, band_kernel_size=(3,7,11), branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)

        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)

        
        self.dwconv_w_3 = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size[0]), padding=(0,band_kernel_size[0]//2), groups = gc)
        self.dwconv_w_7 = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size[1]), padding=(0,band_kernel_size[1]//2), groups = gc)
        self.dwconv_w_11 = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size[2]), padding=(0,band_kernel_size[2]//2), groups = gc)
        
        self.dwconv_h_3 = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size[0], 1), padding=(band_kernel_size[0]//2,0), groups = gc)
        self.dwconv_h_7 = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size[1], 1), padding=(band_kernel_size[1]//2,0), groups = gc)
        self.dwconv_h_11 = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size[2], 1), padding=(band_kernel_size[2]//2,0), groups = gc)

        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        f_hw = self.dwconv_hw(x_hw)
        
        f_w = self.dwconv_w_3(x_w) + self.dwconv_w_7(x_w) + self.dwconv_w_11(x_w) 
        f_h = self.dwconv_h_3(x_h) + self.dwconv_h_7(x_h) + self.dwconv_h_11(x_h)

        return x + torch.cat((x_id, f_hw, f_w, f_h), dim=1)

"""

# 深度可分离卷积 + 扩张卷积组合
class DilatedSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, groups=None):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=dilation, dilation=dilation, groups=groups)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class InceptionDWConv2d14(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        # 计算每个卷积分支的输出通道数
        gc = int(in_channels * branch_ratio)

        # 定义四个卷积分支，包含深度可分卷积和扩张卷积组合
        self.dwconv_hw = DilatedSeparableConv(gc, gc, square_kernel_size, dilation=1, groups=gc)  # 3x3深度可分卷积
        self.dwconv_w = DilatedSeparableConv(gc, gc, kernel_size=(1, band_kernel_size), dilation=1, groups=gc)  # 横向扩张卷积
        self.dwconv_h = DilatedSeparableConv(gc, gc, kernel_size=(band_kernel_size, 1), dilation=1, groups=gc)  # 纵向扩张卷积
        self.dwconv_large = DilatedSeparableConv(gc, gc, kernel_size=7, dilation=1, groups=gc)  # 大尺度的扩张卷积

        # 计算每个分支的输入输出通道数
        self.split_indexes = (in_channels - 4 * gc, gc, gc, gc, gc)

        # 使用残差连接
        # self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 根据计算的split_indexes拆分输入
        x_id, x_hw, x_w, x_h, x_large = torch.split(x, self.split_indexes, dim=1)

        # 通过卷积分支进行特征提取
        f_hw = self.dwconv_hw(x_hw)
        f_w = self.dwconv_w(x_w)
        f_h = self.dwconv_h(x_h)
        f_large = self.dwconv_large(x_large)

        size = x_id.shape[2:]

        f_hw = F.interpolate(f_hw, size=size, mode='bilinear', align_corners=False)
        f_w = F.interpolate(f_w, size=size, mode='bilinear', align_corners=False)
        f_h = F.interpolate(f_h, size=size, mode='bilinear', align_corners=False)
        f_large = F.interpolate(f_large, size=size, mode='bilinear', align_corners=False)

        # 合并卷积后的特征
        f = torch.cat((x_id,f_hw, f_w, f_h, f_large), dim=1)

        # 使用残差连接将原始输入与提取的特征相加
        # x_res = self.residual_conv(x_id)  # 保持输入维度不变
        return x + f
"""

class HierarchicalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups = out_channels)
        self.layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups = out_channels)  # Downsample
        self.layer3 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, stride=2, padding=1, groups = out_channels)

    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = F.relu(self.layer3(x2))
        return x1, x2, x3  # 返回不同层次的特征

class DilatedSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, groups=None):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=dilation, dilation=dilation, groups=groups)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class InceptionDWConv2d15(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        # 计算每个卷积分支的输出通道数
        gc = int(in_channels * branch_ratio)

        # 定义卷积分支
        self.dwconv_hw = DilatedSeparableConv(gc, gc, square_kernel_size, dilation=1, groups=gc)
        self.dwconv_w = DilatedSeparableConv(gc, gc, kernel_size=(1, band_kernel_size), dilation=1, groups=gc)
        self.dwconv_h = DilatedSeparableConv(gc, gc, kernel_size=(band_kernel_size, 1), dilation=1, groups=gc)
        self.dwconv_large = DilatedSeparableConv(gc, gc, kernel_size=7, dilation=1, groups=gc)

        # 引入层次化特征提取模块
        self.hierarchical_feature_extractor = HierarchicalFeatureExtractor(in_channels - 4 * gc, gc)

        # 计算每个分支的输入输出通道数
        self.split_indexes = (in_channels - 4 * gc, gc, gc, gc, gc)

        # 引入残差连接
        # self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 根据计算的split_indexes拆分输入
        x_id, x_hw, x_w, x_h, x_large = torch.split(x, self.split_indexes, dim=1)

        # 通过卷积分支进行特征提取
        f_hw = self.dwconv_hw(x_hw)
        f_w = self.dwconv_w(x_w)
        f_h = self.dwconv_h(x_h)
        f_large = self.dwconv_large(x_large)

        # 层次化特征提取
        h1, h2, h3 = self.hierarchical_feature_extractor(x_id)

        size = x_id.shape[2:]

        f_hw = F.interpolate(f_hw, size=size, mode='bilinear', align_corners=False)
        f_w = F.interpolate(f_w, size=size, mode='bilinear', align_corners=False)
        f_h = F.interpolate(f_h, size=size, mode='bilinear', align_corners=False)
        f_large = F.interpolate(f_large, size=size, mode='bilinear', align_corners=False)
        h2 = F.interpolate(h2, size=size, mode='bilinear', align_corners=False)
        h3 = F.interpolate(h3, size=size, mode='bilinear', align_corners=False)

        h31 = h3[:,:h3.shape[1]//2]
        h32 = h3[:,h3.shape[1]//2:]

        # 合并卷积后的特征
        f = torch.cat((f_hw, f_w, f_h, f_large, h1, h2,h31,h32), dim=1)

        # 使用残差连接
        # x_res = self.residual_conv(x_id)
        return x + f


class InceptionDWConv2d16(nn.Module):

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.dwconv_large = nn.Conv2d(gc,gc,kernel_size=(band_kernel_size, band_kernel_size), padding=(band_kernel_size // 2, band_kernel_size // 2),groups=gc)

    def forward(self, x):
        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_large, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_large = self.dwconv_large(x_large)
        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)

        x_cat = self.dx + torch.cat((elx,x_large,x_hw,x_w,x_h), dim=1)

        return x_cat


class InceptionDWConv2d17(nn.Module):

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.ap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        size = self.dx.shape[-1]

        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_ap, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_ap_add = self.ap(x_ap)
        x_ap_add = x_ap_add.repeat(1,1,size,size)
        x_ap_add = x_ap_add.repeat(1,4,1,1)
        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)

        x_cat = self.dx + torch.cat((elx + x_ap_add,x_ap,x_hw,x_w,x_h), dim=1)

        return x_cat


class InceptionDWConv2d18(nn.Module): # 92.70

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.ap = nn.AdaptiveAvgPool2d(1)

        self.conv_ap = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 8, kernel_size=1)

    def forward(self, x):
        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        size = self.dx.shape[-1]

        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_ap, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_ap_add = self.ap(elx)
        x_ap_add = self.conv_ap(x_ap_add)
        x_ap_add.repeat(1, 1, size, size)
        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)

        x_cat = self.dx + torch.cat((elx ,x_ap + x_ap_add,x_hw,x_w,x_h), dim=1)

        return x_cat


class InceptionDWConv2d19(nn.Module):

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.ap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        b,c,w,h = self.dx.shape
        c = c // 2

        # print(self.dx.shape)

        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_ap, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_ap_add = self.ap(elx)
        x_ap_add = x_ap_add.view(b,c//4,4,1)
        x_ap_add = x_ap_add.mean(dim=2,keepdim=True)
        # print(x_ap_add.shape)
        x_ap_add = x_ap_add.repeat(1, 1, w, h)
        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)

        x_cat = self.dx + torch.cat((elx ,x_ap + x_ap_add,x_hw,x_w,x_h), dim=1)

        return x_cat


class InceptionDWConv2d20(nn.Module):

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

    def forward(self, x):
        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        elx,erx = self.dx[:,1::2,:,:],self.dx[:,::2,:,:]

        x_lt, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)

        x_cat = self.dx + torch.cat((elx ,x_lt,x_hw,x_w,x_h), dim=1)

        return x_cat

class InceptionDWConv2d21(nn.Module):

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.ap = nn.AdaptiveMinPool2d()

    def forward(self, x):
        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        b,c,w,h = self.dx.shape
        c = c // 2

        # print(self.dx.shape)

        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_ap, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_ap_add = self.ap(elx)
        x_ap_add = x_ap_add.view(b,c//4,4,1)
        x_ap_add = x_ap_add.mean(dim=2,keepdim=True)
        # print(x_ap_add.shape)
        x_ap_add = x_ap_add.repeat(1, 1, w, h)
        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)

        x_cat = self.dx + torch.cat((elx ,x_ap + x_ap_add,x_hw,x_w,x_h), dim=1)

        return x_cat


class InceptionDWConv2d22(nn.Module):

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

    def forward(self, x):
        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        elx,erx = self.dx[:,1::2,:,:],self.dx[:,::2,:,:]

        x_lt, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)

        x_s1 = torch.cat((elx,x_lt),dim=1)

        shuffle_indices = torch.randperm(x_s1.size(1))

        x_s2 = x_s1[:, shuffle_indices, :, :]

        x_cat = self.dx + torch.cat((x_s2,x_hw,x_w,x_h), dim=1)

        return x_cat

class AdaptiveMinPool2d(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveMinPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        # 使用 F.unfold 获取每个局部区域
        unfolded = F.unfold(x, kernel_size=x.size(2), stride=1)  # 展开整个图像
        unfolded = unfolded.view(x.size(0), x.size(1), -1)

        # 对每个局部区域取最小值
        min_out = unfolded.min(dim=2)[0].view(x.size(0), x.size(1), 1, 1)

        return min_out


class InceptionDWConv2d_MultiScale2(nn.Module):
    def __init__(self, in_channels,kernel_sizes = [1,3,5], branch_ratio=0.125):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8
        
        # 多尺度卷积核
        self.dwconv_3x3 = nn.Conv2d(gc, gc, kernel_size=3, padding=1, groups=gc)
        self.dwconv_5x5 = nn.Conv2d(gc, gc, kernel_size=5, padding=2, groups=gc)
        self.dwconv_7x7 = nn.Conv2d(gc, gc, kernel_size=7, padding=3, groups=gc)

        self.ap = nn.AdaptiveAvgPool2d(1)
        self.conv_ap = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 8, kernel_size=1)

    def forward(self, x):

        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        size = self.dx.shape[-1]
        
        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_ap,x_3x3,x_5x5,x_7x7 = torch.chunk(erx, 4, dim=1)

        x_ap_add = self.ap(elx)
        x_ap_add = self.conv_ap(x_ap_add)
        x_ap_add.repeat(1, 1, size, size)
        
        # 多尺度特征融合
        out_3x3 = self.dwconv_3x3(x_3x3)
        out_5x5 = self.dwconv_5x5(x_5x5)
        out_7x7 = self.dwconv_7x7(x_7x7)

        return self.dx + torch.cat((elx, x_ap + x_ap_add, out_3x3, out_5x5, out_7x7), dim=1)



class InceptionDWConv2d24(nn.Module): 
    def __init__(self, in_channels, kernel_sizes=[1,3,5], square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8
        self.dwconv_hw = nn.Sequential(
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
        )
        self.dwconv_w = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
        )
        self.dwconv_h = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        
        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        size = self.dx.shape[-1]
        
        elx,erx = self.dx[:,::2,:,:],self.dx[:,1::2,:,:]

        x_id,x_hw,x_w,x_h = torch.chunk(erx, 4, dim=1)
        
        return x + torch.cat(
            (elx, x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class InceptionDWConv2d25(nn.Module):

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8
        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.bn = nn.BatchNorm2d(gc)
        self.am = nn.AdaptiveMaxPool2d(1)
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.LeakyReLU(inplace=False)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):

        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_lt, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        f_lt = self.bn(self.relu(x_lt))

        x_lt = x_lt * self.sig(self.am(f_lt) + self.ap(f_lt))


        # 将原始通道x_id与经过3个卷积分支处理后的x_hw、x_w、x_h合并
        x = torch.cat(
            (elx, x_lt, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,  # 在通道维度上拼接
        )

        return self.dx + x


class InceptionDWConv2d26(nn.Module):

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.ap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.LeakyReLU(inplace=False)

        self.conv_ap = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 8, kernel_size=1)

    def forward(self, x):
        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        size = self.dx.shape[-1]

        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_ap, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_ap_add = self.ap(elx)
        x_ap_add = self.relu(self.conv_ap(x_ap_add))
        x_ap_add.repeat(1, 1, size, size)
        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)

        x_cat = self.dx + torch.cat((elx ,x_ap + x_ap_add,x_hw,x_w,x_h), dim=1)

        return x_cat


class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class InceptionDWConv2d27(nn.Module):

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8
        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.fem = FEM(gc,gc)

    
        
    def forward(self, x):

        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_lt, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_lt = self.fem(x_lt)

        # 将原始通道x_id与经过3个卷积分支处理后的x_hw、x_w、x_h合并
        x = torch.cat(
            (elx, x_lt, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,  # 在通道维度上拼接
        )

        return self.dx + x


class InceptionDWConv2d28(nn.Module): #92.62

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8
        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        
        self.conv1 = nn.Conv2d(gc,gc*2,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(gc*2)
        self.bn2 = nn.BatchNorm2d(gc)
        self.relu = nn.LeakyReLU(inplace=False)
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(gc*2, gc*2, kernel_size, padding=kernel_size // 2, groups=gc, bias=False)
            for kernel_size in kernel_sizes
        ])
        self.conv2 = nn.Conv2d(gc*2,gc,kernel_size=1)
        
    
        
    def forward(self, x):

        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_lt, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_lt2 = self.relu(self.bn1(self.conv1(x_lt)))

        x_lt2 = sum([conv(x_lt2) for conv in self.dw_convs])

        x_lt = self.bn2(self.conv2(x_lt2)) + x_lt

        # 将原始通道x_id与经过3个卷积分支处理后的x_hw、x_w、x_h合并
        x = torch.cat(
            (elx, x_lt, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,  # 在通道维度上拼接
        )

        return self.dx + x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class Pinwheel_shapedConv(nn.Module):  
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()
        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)
    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class InceptionDWConv2d29(nn.Module):

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8
        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.fem = Pinwheel_shapedConv(c1=gc,c2=gc,k=3,s=1)

    
        
    def forward(self, x):

        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_lt, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_lt = self.fem(x_lt)

        # 将原始通道x_id与经过3个卷积分支处理后的x_hw、x_w、x_h合并
        x = torch.cat(
            (elx, x_lt, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,  # 在通道维度上拼接
        )

        return self.dx + x


from einops import rearrange
from torch import einsum

class GCSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(GCSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class InceptionDWConv2d30(nn.Module):

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8
        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.fem = GCSA(dim=gc, num_heads=4, bias=True)

    
        
    def forward(self, x):

        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_lt, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_lt = self.fem(x_lt)

        # 将原始通道x_id与经过3个卷积分支处理后的x_hw、x_w、x_h合并
        x = torch.cat(
            (elx, x_lt, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,  # 在通道维度上拼接
        )

        return self.dx + x

import math

class InceptionDWConv2d31(nn.Module): # 92.71
    def __init__(self, in_features, kernel_sizes, square_kernel_size = 3, band_kernel_size = 11):
        super(InceptionDWConv2d31, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        
        gc = math.ceil(self.out_features / 7.0)

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

    def forward(self, x):

        self.dx = x

        elx1, elx2, dw, dhw, dh, erx1,erx2  = torch.chunk(self.dx, 7, dim=1)

        dw = self.dwconv_w(dw)
        dhw = self.dwconv_hw(dhw)
        dh = self.dwconv_h(dh)

        x_cat = self.dx + torch.cat((elx1, elx2, dw, dhw, dh, erx1,erx2), dim=1)

        return x_cat


class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Sigmoid()
        )
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))

    def forward(self, x):
        b, _, h, w = x.shape
        attention = self.attention(x)  # 生成动态权重
        weight = self.weight.unsqueeze(0) * attention.view(b, -1, 1, 1, 1)  # 动态调整卷积核
        weight = weight.view(-1, self.weight.size(1), self.weight.size(2), self.weight.size(3))
        x = x.view(1, -1, h, w)
        out = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.groups * b)
        out = out.view(b, -1, out.size(2), out.size(3))
        return out

class InceptionDWConv2d32(nn.Module):
    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super(InceptionDWConv2d32, self).__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = DynamicConv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = DynamicConv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = DynamicConv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return x + torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)


class InceptionDWConv2d33(nn.Module):
    def __init__(self, in_features, kernel_sizes,square_kernel_size = 3, band_kernel_size = 11):
        super(InceptionDWConv2d33, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        gc = self.out_features // 8

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc*2, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc*2, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc*2, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.change_linear_hw = nn.Linear(gc, gc * 2, bias=True)
        self.change_linear_w = nn.Linear(gc, gc * 2, bias=True)
        self.change_linear_h = nn.Linear(gc, gc * 2, bias=True)

        self.change_linear_c = nn.Linear(gc, gc, bias=True)

        self.gc = gc

    def forward(self, x):

        self.dx = x

        b,c,w,h = self.dx.shape

        elx, erx  = torch.chunk(self.dx, 2, dim=1)

        dl, dw, dhw, dh = torch.chunk(erx, 4, dim=1)

        dl = dl.reshape(-1,self.gc)
        dw = dw.reshape(-1,self.gc)
        dhw = dhw.reshape(-1,self.gc)
        dh = dh.reshape(-1,self.gc)

        dw = self.change_linear_w(dw)
        dhw = self.change_linear_hw(dhw)
        dh = self.change_linear_h(dh)

        dl = self.change_linear_c(dl)

        dw = dw.reshape(b,self.gc*2,w,h)
        dhw = dhw.reshape(b,self.gc*2,w,h)
        dh = dw.reshape(b,self.gc*2,w,h)
        dl = dl.reshape(b,self.gc,w,h)

        dw = self.dwconv_w(dw)
        dhw = self.dwconv_hw(dhw)
        dh = self.dwconv_h(dh)

        x_cat = self.dx + torch.cat((elx,dl, dw, dhw, dh), dim=1)

        return x_cat


class InceptionDWConv2d34(nn.Module): # exit
    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super(InceptionDWConv2d34, self).__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.shape
        x = x.view(b, groups, c // groups, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        out = torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)
        return x + self.channel_shuffle(out, groups=4)


class InceptionDWConv2d35(nn.Module):
    def __init__(self, in_features, kernel_sizes,square_kernel_size = 3, band_kernel_size = 11):
        super(InceptionDWConv2d35, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        gc = self.out_features // 8

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.change_linear_hw = nn.Linear(gc, gc, bias=True)
        self.change_linear_w = nn.Linear(gc, gc, bias=True)
        self.change_linear_h = nn.Linear(gc, gc, bias=True)

        self.change_linear_c = nn.Linear(gc, gc, bias=True)

    def forward(self, x):

        self.dx = x

        elx, erx  = torch.chunk(self.dx, 2, dim=1)

        dl, dw, dhw, dh = torch.chunk(erx, 4, dim=1)

        dw = self.dwconv_w(self.change_linear_w(dw) + dw)
        dhw = self.dwconv_hw(self.change_linear_hw(dhw) + dhw)
        dh = self.dwconv_h(self.change_linear_h(dh) + dh)

        dl = self.change_linear_c(dl)

        x_cat = self.dx + torch.cat((elx,dl, dw, dhw, dh), dim=1)

        return x_cat

"""
class ContextAwareConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.context(x)
        return self.conv(x) * context
"""


class ContextAwareConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 深度可分离卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels)
        # self.pointwise = nn.Conv2d(in_channels, out_channels, 1)  # 1x1 卷积
        
        # 上下文卷积层（使用深度可分离卷积）
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels),  # 深度卷积
            # nn.Conv2d(in_channels, in_channels, 1),  # 逐点卷积
            nn.Sigmoid()
        )

    def forward(self, x):
        # 深度可分离卷积
        x_conv = self.depthwise(x)
        # x_conv = self.pointwise(x_conv)
        
        # 上下文权重
        context = self.context(x)
        
        # 应用上下文权重
        return x_conv * context

class InceptionDWConv2d36(nn.Module):
    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super(InceptionDWConv2d36, self).__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = ContextAwareConv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2)
        self.dwconv_w = ContextAwareConv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2))
        self.dwconv_h = ContextAwareConv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0))
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return x + torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)


class InceptionDWConv2d37(nn.Module):
    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super(InceptionDWConv2d37, self).__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        self.recalibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        out = torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)
        recalibrated = self.recalibration(out)
        return x + out * recalibrated

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        batch, c, h, w = x.size()
        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        return output
    
class Freq_Fusion(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=[1,3,5,7],
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4
    ):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim*scale_ratio//spilt_num
        self.conv_init_1 = nn.Sequential(  # PW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_init_2 = nn.Sequential(  # DW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)
        self.bn = torch.nn.BatchNorm2d(dim*2)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        x_1, x_2 = torch.split(x, self.dim, dim=1)
        x_1 = self.conv_init_1(x_1)
        x_2 = self.conv_init_2(x_2)
        x0 = torch.cat([x_1, x_2], dim=1)
        x = self.FFC(x0) + x0
        x = self.relu(self.bn(x))
        return x
    
class Fused_Fourier_Conv_Mixer(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_gloal=Freq_Fusion,
            mixer_kernel_size=[1,3,5,7],
            local_size=8
    ):
        super(Fused_Fourier_Conv_Mixer, self).__init__()
        self.dim = dim
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, kernel_size=mixer_kernel_size,
                                 se_ratio=8, local_size=local_size)
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=3 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=5, padding=5 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )
    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local_1 = self.dw_conv_1(x[0])
        x_local_2 = self.dw_conv_2(x[0])
        x_gloal = self.mixer_gloal(torch.cat([x_local_1, x_local_2], dim=1))
        x = self.ca_conv(x_gloal)
        x = self.ca(x) * x
        return x

class InceptionDWConv2d38(nn.Module):

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8
        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.fem = Fused_Fourier_Conv_Mixer(gc)

    
        
    def forward(self, x):

        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_lt, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_lt = self.fem(x_lt)

        # 将原始通道x_id与经过3个卷积分支处理后的x_hw、x_w、x_h合并
        x = torch.cat(
            (elx, x_lt, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,  # 在通道维度上拼接
        )

        return self.dx + x

import numbers
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
class FSAS(nn.Module):
    def __init__(self, dim, bias=False):
        super(FSAS, self).__init__()
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')
        self.patch_size = 8
    def forward(self, x):
        hidden = self.to_hidden(x)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)
        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)
        out = self.norm(out)
        output = v * out
        output = self.project_out(output)
        return output

class InceptionDWConv2d39(nn.Module):

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8
        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.fem = FSAS(gc)

    
        
    def forward(self, x):

        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_lt, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_lt = self.fem(x_lt)

        # 将原始通道x_id与经过3个卷积分支处理后的x_hw、x_w、x_h合并
        x = torch.cat(
            (elx, x_lt, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,  # 在通道维度上拼接
        )

        return self.dx + x


class AdaptiveNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.adaptive = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 3, padding = 1, groups = channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        norm_out = self.norm(x)
        adaptive_weight = self.adaptive(x)
        return norm_out * adaptive_weight

class InceptionDWConv2d40(nn.Module):
    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        self.adaptive_norm = AdaptiveNorm(in_channels)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        out = torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)
        out = self.adaptive_norm(out)
        return x + out

class InceptionDWConv2d41(nn.Module): 

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8
        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.dwconv_3x3 = nn.Conv2d(gc, gc, kernel_size=3, padding=1, groups=gc)
        self.dwconv_5x5 = nn.Conv2d(gc, gc, kernel_size=5, padding=2, groups=gc)
        self.dwconv_7x7 = nn.Conv2d(gc, gc, kernel_size=7, padding=3, groups=gc)

    def forward(self, x):

        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        # 根据计算的split_indexes拆分输入x
        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_3x3, x_5x5, x_7x7, x_rt = torch.chunk(elx, 4, dim=1)
        x_lt, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_3x3 = self.dwconv_3x3(x_3x3)
        x_5x5 = self.dwconv_5x5(x_5x5)
        x_7x7 = self.dwconv_7x7(x_7x7)

        x1 = self.dx + torch.cat(
            (x_3x3, x_5x5, x_7x7, x_rt, x_lt, x_hw, x_w, x_h), dim=1,  # 在通道维度上拼接
        )

        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)

        # 将原始通道x_id与经过3个卷积分支处理后的x_hw、x_w、x_h合并
        x2 = x1 + torch.cat(
            (x_3x3, x_5x5, x_7x7, x_rt, x_lt, x_hw, x_w, x_h), dim=1,  # 在通道维度上拼接
        )

        return x2

        # return self.dx + sum([conv(x) for conv in self.dw_convs])


class InceptionDWConv2d42(nn.Module): 

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        self.in_channels = in_channels
        self.change = in_channels % 8 != 0
        if self.change:
            while in_channels % 8 != 0:
                in_channels += 1
            self.change_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels, kernel_size=1)

        gc = in_channels // 8
        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.dwconv_3x3 = nn.Conv2d(gc, gc, kernel_size=3, padding=1, groups=gc)
        self.dwconv_5x5 = nn.Conv2d(gc, gc, kernel_size=5, padding=2, groups=gc)
        self.dwconv_7x7 = nn.Conv2d(gc, gc, kernel_size=7, padding=3, groups=gc)

    def forward(self, x):

        if self.change:
            self.dx = self.change_conv(x)
        else:
            self.dx = x

        # 根据计算的split_indexes拆分输入x
        elx,erx = torch.chunk(self.dx, 2, dim=1)

        x_3x3, x_5x5, x_7x7, x_rt = torch.chunk(elx, 4, dim=1)
        x_lt, x_hw, x_w, x_h = torch.chunk(erx, 4, dim=1)

        x_3x3 = self.dwconv_3x3(x_3x3)
        x_5x5 = self.dwconv_5x5(x_5x5)
        x_7x7 = self.dwconv_7x7(x_7x7)

        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)
        

        # 将原始通道x_id与经过3个卷积分支处理后的x_hw、x_w、x_h合并
        x2 = self.dx + torch.cat(
            (x_3x3, x_5x5, x_7x7, x_rt, x_lt, x_hw, x_w, x_h), dim=1,  # 在通道维度上拼接
        )

        return x2

class MS_MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
        channels_first: bool = False,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ) -> None:
        super(MS_MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.multiscale_conv = InceptionDWConv2d2(hidden_features, kernel_sizes=kernel_sizes)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout( )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.multiscale_conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MS_MLP2(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
        channels_first: bool = False,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ) -> None:
        super(MS_MLP2, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.multiscale_conv = InceptionDWConv2d(hidden_features, kernel_sizes=kernel_sizes)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.multiscale_conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MS_MLP3(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
        channels_first: bool = False,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ) -> None:
        super(MS_MLP3, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.multiscale_conv = InceptionDWConv2d31(hidden_features, kernel_sizes=kernel_sizes)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.multiscale_conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MS_MLP4(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
        channels_first: bool = False,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ) -> None:
        super(MS_MLP4, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear

        self.in_features = in_features
        self.wh = (768 // in_features) * 7

        self.gc = InceptionDWConv2d31(in_features, kernel_sizes=kernel_sizes)
        # self.ln = nn.LayerNorm(in_features)
        self.ln = nn.LayerNorm([self.in_features,self.wh,self.wh])
        
        
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.multiscale_conv = InceptionDWConv2d31(hidden_features, kernel_sizes=kernel_sizes)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b,c,w,h = x.shape
        
        x = self.gc(x)
        # x = x.permute(0, 2, 3, 1).reshape(b, -1, c)
        x = self.ln(x)
        # x = x.reshape(b, w, h, c).permute(0, 3, 1, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.multiscale_conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MSVSS(nn.Sequential):
    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path: Sequence[float] | float = 0.0,
        use_checkpoint: bool = False,
        norm_layer: Type[nn.Module] = LayerNorm2d,
        channel_first: bool = True,
        ssm_d_state: int = 1,
        ssm_ratio: float = 1.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer: Type[nn.Module] = nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "v05_noz",
        mlp_ratio: float = 4.0,
        mlp_act_layer: Type[nn.Module] = nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
    ) -> None:
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d] if isinstance(drop_path, Sequence) else drop_path,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                customized_mlp=MS_MLP
            ))
        super(MSVSS, self).__init__(OrderedDict(
            blocks=nn.Sequential(*blocks),
        ))

class MSVSS2(nn.Sequential):
    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path: Sequence[float] | float = 0.0,
        use_checkpoint: bool = False,
        norm_layer: Type[nn.Module] = LayerNorm2d,
        channel_first: bool = True,
        ssm_d_state: int = 1,
        ssm_ratio: float = 1.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer: Type[nn.Module] = nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "v05_noz",
        mlp_ratio: float = 4.0,
        mlp_act_layer: Type[nn.Module] = nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
    ) -> None:
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d] if isinstance(drop_path, Sequence) else drop_path,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                customized_mlp=MS_MLP2
            ))
        super(MSVSS2, self).__init__(OrderedDict(
            blocks=nn.Sequential(*blocks),
        ))

class MSVSS3(nn.Sequential):
    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path: Sequence[float] | float = 0.0,
        use_checkpoint: bool = False,
        norm_layer: Type[nn.Module] = LayerNorm2d,
        channel_first: bool = True,
        ssm_d_state: int = 1,
        ssm_ratio: float = 1.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer: Type[nn.Module] = nn.SiLU,
        ssm_conv: int = 300,
        ssm_conv_bias: bool = False,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "v05_noz",
        mlp_ratio: float = 4.0,
        mlp_act_layer: Type[nn.Module] = nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
    ) -> None:
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock2(
                hidden_dim=dim,
                drop_path=drop_path[d] if isinstance(drop_path, Sequence) else drop_path,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                customized_mlp=MS_MLP3
            ))
        super(MSVSS3, self).__init__(OrderedDict(
            blocks=nn.Sequential(*blocks),
        ))


class MSVSS4(nn.Sequential):
    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path: Sequence[float] | float = 0.0,
        use_checkpoint: bool = False,
        norm_layer: Type[nn.Module] = LayerNorm2d,
        channel_first: bool = True,
        ssm_d_state: int = 1,
        ssm_ratio: float = 1.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer: Type[nn.Module] = nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "v05_noz",
        mlp_ratio: float = 4.0,
        mlp_act_layer: Type[nn.Module] = nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
    ) -> None:
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d] if isinstance(drop_path, Sequence) else drop_path,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                customized_mlp=MS_MLP4
            ))
        super(MSVSS4, self).__init__(OrderedDict(
            blocks=nn.Sequential(*blocks),
        ))

class LKPE(nn.Module):
    def __init__(self, dim: int, dim_scale: int = 2, norm_layer: Type[nn.Module] = nn.LayerNorm):
        super(LKPE, self).__init__()
        self.dim = dim
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=True)
        )
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)

        x = rearrange(x, pattern="b h w c -> b c h w")
        return x

class FLKPE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dim_scale: int = 4,
        norm_layer: Type[nn.Module] = nn.LayerNorm
    ):
        super(FLKPE, self).__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 16, dim * 16, kernel_size=3, padding=1, groups=dim * 16, bias=True)
        )

        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
        self.out = nn.Conv2d(self.output_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        x = rearrange(x, pattern="b h w c -> b c h w")
        return self.out(x)

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        drop_path: Sequence[float] | float,
    ) -> None:
        super(UpBlock, self).__init__()
        self.up = LKPE(in_channels)
        self.concat_layer = Linear2d(2 * out_channels, out_channels)
        self.vss_layer = MSVSS(dim=out_channels, depth=depth, drop_path=drop_path)

    def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.up(input)
        out = torch.cat(tensors=(out, skip), dim=1)
        out = self.concat_layer(out)
        out = self.vss_layer(out)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        num_classes: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        drop_path_rate: float = 0.2,
    ) -> None:
        super(Decoder, self).__init__()
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (len(dims) - 1) * 2)]

        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(
                UpBlock(
                    in_channels=dims[i - 1],
                    out_channels=dims[i],
                    depth=depths[i],
                    drop_path=dpr[sum(depths[: i - 1]): sum(depths[: i])],
                ))

        self.out_layers = nn.Sequential(FLKPE(dims[-1], num_classes))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        out = features[0]
        features = features[1:]
        for i, layer in enumerate(self.layers):
            out = layer(out, features[i])
        return self.out_layers[0](out)
