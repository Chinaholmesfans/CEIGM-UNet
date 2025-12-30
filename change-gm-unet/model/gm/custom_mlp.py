import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

import math

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class InceptionDWConv2d18(nn.Module): # 92.70

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()

        self.gc = int(in_channels // 8)
        self.half = int(in_channels // 2)
        self.ap_gc = self.half - self.gc * 3

        gc = self.gc

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.ap = nn.AdaptiveAvgPool2d(1)

        self.conv_ap = nn.Conv2d(in_channels=self.half, out_channels=self.ap_gc, kernel_size=1)

    def forward(self, x, H, W):

        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        self.dx = x

        elx, x_ap, x_hw, x_w, x_h = torch.split(x, (self.half, self.ap_gc, self.gc, self.gc, self.gc), dim=1)

        x_ap_add = self.ap(elx)
        x_ap_add = self.conv_ap(x_ap_add)
        x_ap_add.repeat(1, 1, H, W)
        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)

        x_cat = self.dx + torch.cat((elx ,x_ap + x_ap_add,x_hw,x_w,x_h), dim=1)

        return x_cat.flatten(2).transpose(1, 2)


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

class InceptionDWConv2d18m(nn.Module): # 92.70

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()

        self.gc = int(in_channels // 8)
        self.half = int(in_channels // 2)
        self.ap_gc = self.half - self.gc * 3

        gc = self.gc

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.ap = nn.AdaptiveMinPool2d(1)

        self.conv_ap = nn.Conv2d(in_channels=self.half, out_channels=self.ap_gc, kernel_size=1)

    def forward(self, x, H, W):

        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        self.dx = x

        elx, x_ap, x_hw, x_w, x_h = torch.split(x, (self.half, self.ap_gc, self.gc, self.gc, self.gc), dim=1)

        x_ap_add = self.ap(elx)
        x_ap_add = self.conv_ap(x_ap_add)
        x_ap_add.repeat(1, 1, H, W)
        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)

        x_cat = self.dx + torch.cat((elx ,x_ap + x_ap_add,x_hw,x_w,x_h), dim=1)

        return x_cat.flatten(2).transpose(1, 2)


class InceptionDWConv2d18m2(nn.Module): # 92.70

    def __init__(self, in_channels, kernel_sizes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()

        self.gc = int(in_channels // 8)
        self.half = int(in_channels // 2)
        self.ap_gc = self.half - self.gc * 3

        gc = self.gc

        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.ap = nn.AdaptiveMaxPool2d(1)

        self.conv_ap = nn.Conv2d(in_channels=self.half, out_channels=self.ap_gc, kernel_size=1)

    def forward(self, x, H, W):

        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        self.dx = x

        elx, x_ap, x_hw, x_w, x_h = torch.split(x, (self.half, self.ap_gc, self.gc, self.gc, self.gc), dim=1)

        x_ap_add = self.ap(elx)
        x_ap_add = self.conv_ap(x_ap_add)
        x_ap_add.repeat(1, 1, H, W)
        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)

        x_cat = self.dx + torch.cat((elx ,x_ap + x_ap_add,x_hw,x_w,x_h), dim=1)

        return x_cat.flatten(2).transpose(1, 2)


class InceptionDWConv2d31(nn.Module): # 92.71
    def __init__(self, in_features, kernel_sizes, square_kernel_size = 3, band_kernel_size = 11):
        super(InceptionDWConv2d31, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc = in_features // 7
        self.add = in_features - self.gc * 7

        gc = self.gc
        
        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

    def forward(self, x, H, W):

        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        self.dx = x

        elx1, elx2, dw, dhw, dh, erx1,erx2  = torch.split(self.dx, (self.gc,self.gc,self.gc,self.gc,self.gc,self.gc,self.gc+self.add), dim=1)

        dw = self.dwconv_w(dw)
        dhw = self.dwconv_hw(dhw)
        dh = self.dwconv_h(dh)

        x_cat = self.dx + torch.cat((elx1, elx2, dw, dhw, dh, erx1,erx2), dim=1)

        return x_cat.flatten(2).transpose(1, 2)

class InceptionDWConv2d_MultiScale(nn.Module): # 92.74
    def __init__(self, in_channels,kernel_sizes = [1,3,5], branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)

        # 多尺度卷积核
        self.dwconv_3x3 = nn.Conv2d(gc, gc, kernel_size=3, padding=1, groups=gc)
        self.dwconv_5x5 = nn.Conv2d(gc, gc, kernel_size=5, padding=2, groups=gc)
        self.dwconv_7x7 = nn.Conv2d(gc, gc, kernel_size=7, padding=3, groups=gc)

        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x, H, W):
        
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        
        self.dx = x
        
        x_id, x_3x3, x_5x5, x_7x7 = torch.split(x, self.split_indexes, dim=1)

        # 多尺度特征融合
        out_3x3 = self.dwconv_3x3(x_3x3)
        out_5x5 = self.dwconv_5x5(x_5x5)
        out_7x7 = self.dwconv_7x7(x_7x7)

        return (self.dx + torch.cat((x_id, out_3x3, out_5x5, out_7x7), dim=1)).flatten(2).transpose(1, 2)


class InceptionDWConv2d_MultiScale2(nn.Module): # 92.74
    def __init__(self, in_channels,kernel_sizes = [1,3,5], branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)

        # 多尺度卷积核
        self.dwconv_3x3 = nn.Conv2d(gc, gc, kernel_size=3, padding=1, groups=gc)
        self.dwconv_5x5 = nn.Conv2d(gc, gc, kernel_size=(1,11), padding=(0,11//2), groups=gc)
        self.dwconv_7x7 = nn.Conv2d(gc, gc, kernel_size=(11,1), padding=(11//2,0), groups=gc)

        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x, H, W):
        
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        
        self.dx = x
        
        x_id, x_3x3, x_5x5, x_7x7 = torch.split(x, self.split_indexes, dim=1)

        # 多尺度特征融合
        out_3x3 = self.dwconv_3x3(x_3x3)
        out_5x5 = self.dwconv_5x5(x_5x5)
        out_7x7 = self.dwconv_7x7(x_7x7)

        return (self.dx + torch.cat((x_id, out_3x3, out_5x5, out_7x7), dim=1)).flatten(2).transpose(1, 2)


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

    def forward(self, x, H, W):

        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return (x + torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )).flatten(2).transpose(1, 2)

class InceptionDWConv2d_MultiScale(nn.Module): # 92.74
    def __init__(self, in_channels,kernel_sizes = [1,3,5], branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)

        # 多尺度卷积核
        self.dwconv_3x3 = nn.Conv2d(gc, gc, kernel_size=3, padding=1, groups=gc)
        self.dwconv_5x5 = nn.Conv2d(gc, gc, kernel_size=5, padding=2, groups=gc)
        self.dwconv_7x7 = nn.Conv2d(gc, gc, kernel_size=7, padding=3, groups=gc)

        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x, H, W):
        
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        
        self.dx = x
        
        x_id, x_3x3, x_5x5, x_7x7 = torch.split(x, self.split_indexes, dim=1)

        # 多尺度特征融合
        out_3x3 = self.dwconv_3x3(x_3x3)
        out_5x5 = self.dwconv_5x5(x_5x5)
        out_7x7 = self.dwconv_7x7(x_7x7)

        return (self.dx + torch.cat((x_id, out_3x3, out_5x5, out_7x7), dim=1)).flatten(2).transpose(1, 2)

class custom_ffn(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        
        self.custom = InceptionDWConv2d_MultiScale(hidden_features,[])
        
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.custom(x, H, W)
        x = self.fc2(x)
        return x