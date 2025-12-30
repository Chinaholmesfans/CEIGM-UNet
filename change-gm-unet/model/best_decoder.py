import torch
import torch.nn as nn
from torch.functional import F
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

from model.gm.custom_module import cm


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


#   Large-kernel grouped attention gate (LGAG)
class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1

        self.W_g_1 = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, groups=groups,
                               bias=True)
        self.W_g_3 = nn.Conv2d(F_g, F_int, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=True)
        self.W_g_5 = nn.Conv2d(F_g, F_int, kernel_size=5, stride=1, padding=2, groups=groups,
                               bias=True)

        self.W_x_1 = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, groups=groups,
                               bias=True)
        self.W_x_3 = nn.Conv2d(F_g, F_int, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=True)
        self.W_x_5 = nn.Conv2d(F_g, F_int, kernel_size=5, stride=1, padding=2, groups=groups,
                               bias=True)

        self.bn = nn.BatchNorm2d(F_int)

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g_1(g)
        g2 = self.W_g_3(g)
        g3 = self.W_g_5(g)

        x1 = self.W_x_1(g)
        x2 = self.W_x_3(g)
        x3 = self.W_x_5(g)

        gs = self.bn(g1 + g2 + g3)
        xs = self.bn(x1 + x2 + x3)

        psi = self.activation(gs + xs)
        psi = self.psi(psi)

        return x * psi


class OptimizedMultiScaleCAB(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(OptimizedMultiScaleCAB, self).__init__()
        # 通过调整reduced_channels的值来减小参数量
        self.reduced_channels = max(1, in_channels // ratio)  # 减小通道数

        # 分支1：平均池化 + 卷积
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv1 = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1, bias=False)

        # 分支2：最大池化 + 多尺度卷积
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv2_1 = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1, padding=0, bias=False)
        self.conv2_2 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=3, padding=1, bias=False)

        # 融合与重映射
        self.fc = nn.Sequential(
            nn.Conv2d(self.reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 分支1：平均池化分支
        avg_out = self.avg_pool(x)
        # avg_out = self.conv1(avg_out)
        avg_out = self.conv2_1(avg_out)
        avg_out = self.conv2_2(avg_out)

        # 分支2：最大池化分支 + 多尺度卷积
        max_out = self.max_pool(x)
        max_out = self.conv2_1(max_out)
        max_out = self.conv2_2(max_out)

        combined = avg_out + max_out

        # 融合
        # combined = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.fc(combined)
        return attention_map

class AdaptiveMinPool2d(nn.Module):
    def __init__(self):
        super(AdaptiveMinPool2d, self).__init__()

    def forward(self, x):
        # 使用 F.unfold 获取每个局部区域
        unfolded = F.unfold(x, kernel_size=x.size(2), stride=1)  # 展开整个图像
        unfolded = unfolded.view(x.size(0), x.size(1), -1)

        # 对每个局部区域取最小值
        min_out = unfolded.min(dim=2)[0].view(x.size(0), x.size(1), 1, 1)

        return min_out

class OptimizedMultiScaleCAB2(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(OptimizedMultiScaleCAB2, self).__init__()

        factor = in_channels  // ratio // 3

        while in_channels % factor != 0:
            factor += 1

        # Further reduce parameters by optimizing channel dimensions and operations
        self.reduced_channels = max(1, factor)  # Further reduce channels
        # Branch 1: Average Pooling + Convolution
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1, bias=False)

        # Branch 2: Max Pooling + Multi-scale Convolutions with Efficiency Enhancements
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv2_1 = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1, padding=0, groups=self.reduced_channels, bias=False)  # Depthwise convolution
        self.conv2_2 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=1, padding=0, groups=self.reduced_channels, bias=False)  # Depthwise convolution

        # Additional Branch 3: Dilated Convolution for enhanced receptive field
        self.min_pool = AdaptiveMinPool2d()
        # self.min_pool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1, padding=0, bias=False)

        # Fusion and Remapping
        self.fc = nn.Sequential(
            nn.Conv2d(self.reduced_channels * 3, in_channels, kernel_size=1, bias=False),
        )

        self.init_weights('normal')

        self.sigmoid = nn.Sigmoid()

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Branch 1: Average Pooling Branch
        avg_out = self.avg_pool(x)
        avg_out = self.conv1(avg_out)

        # Branch 2: Max Pooling Branch + Multi-scale Convolutions
        max_out = self.max_pool(x)
        max_out = self.conv2_1(max_out)
        max_out = self.conv2_2(max_out)

        # Branch 3: Dilated Convolution Branch
        dilated_out = self.conv3(self.min_pool(x))


        # Fusion
        combined = torch.cat([avg_out, max_out, dilated_out], dim=1)
        attention_map = self.sigmoid(self.fc(combined) + x)
        return attention_map


class ImprovedSAB(nn.Module):
    def __init__(self, kernel_sizes=(3, 7, 11)):
        super(ImprovedSAB, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=3 // 2, bias=False)
        self.conv7 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=7 // 2, bias=False)
        self.conv11 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=11, stride=1, padding=11 // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)

        x3 = self.conv3(x_cat)
        x7 = self.conv7(x_cat)
        x11 = self.conv11(x_cat)

        spatial_features = sum([x3, x7, x11])


        return self.sigmoid(spatial_features)


class LightweightParallelAttentionFusion(nn.Module):

    def __init__(self, in_channels, kernel_sizes=(3, 7, 11), ratio=16, activation='relu', reduced_factor=2):
        super(LightweightParallelAttentionFusion, self).__init__()
        # 通道注意力模块
        self.channel_attention = OptimizedMultiScaleCAB2(in_channels, ratio=ratio)
        # 空间注意力模块
        self.spatial_attention = ImprovedSAB()

        # 动态权重生成器
        self.x = nn.Parameter(torch.tensor(0.0))

        # 通道与空间交互模块（减少输出通道）
        self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x1, x2):
        # 并行执行 CAB 和 SAB
        ca_out = self.channel_attention(x1)  # 通道注意力特征
        sa_out = self.spatial_attention(x2)  # 空间注意力特征

        # 动态权重生成

        ca_weight = .5 + (torch.atan(torch.pi * self.x) / torch.pi)
        sa_weight = 1 - ca_weight

        # 加权特征
        ca_out = x1 * ca_out * ca_weight
        sa_out = x2 * sa_out * sa_weight

        # 通道与空间特征交互（减少通道数）
        fusion = torch.cat([ca_out, sa_out], dim=1)  # 在通道维度拼接
        fusion = self.final_conv(fusion)  # 恢复到输入通道维度
        fusion = self.sigmoid(fusion)

        # 输出融合特征
        out = (x1 + x2) * fusion  # 将融合特征作用于原始输入
        return out




class SplitChannelsOddEven(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(SplitChannelsOddEven, self).__init__()
        out_channels = out_channels if out_channels is not None else in_channels

        self.cw = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # 奇数通道索引
        out1 = x[:, 0::2, :, :]  # 从第 0 个通道开始，每隔 2 个选一个
        out1 = self.cw(out1)
        # 偶数通道索引
        out2 = x[:, 1::2, :, :]  # 从第 1 个通道开始，每隔 2 个选一个
        out2 = self.cw(out2)
        return out1, out2


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, other_out_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups

        assert in_channels >= groups and in_channels % groups == 0

        out_channels = 2 * groups * scale ** 2

        self.offset = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        )

        normal_init(self.offset, std=0.001)

        self.register_buffer('init_pos', self._init_pos())

        self.eu = EUCB2(in_channels, other_out_channels)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H) + torch.sin(torch.pi * torch.arange(1, H + 1, 1) / H)
        coords_w = torch.arange(W) + torch.sin(torch.pi * torch.arange(1, W + 1, 1) / W)

        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.contiguous().view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward_lp(self, x):
        offset = self.offset(x) * 1.0/self.groups + self.init_pos
        return self.sample(x, offset)


    def forward(self, x):
        # print(x.shape)

        out = self.forward_lp(x)
        out = self.eu(out)
        return out


class EUCB2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

class Front(nn.Module):
    def __init__(self, in_channels, out_channels=None, depths = (3, 2, 2, 2), drop_path_rate = 0.2, ilayer = 1, channels = [512, 320, 128, 64]):
        super(Front, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, sum(depths) - depths[-1])] # 9
        self.cm_layer = cm(dim=self.out_channels, depth=depths[ilayer-1], drop_path=dpr[sum(depths[: ilayer - 1]): sum(depths[: ilayer])])
        self.m = self.cm_layer

        self.init_weights('normal')
        
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
    
    def forward(self, x):
        if self.m is not None:
            return self.m(x)
        else:
            return x


#   Efficient multi-scale convolutional attention decoding (EMCAD)
class EMCAD(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], num_classes=4, kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,
                 add=True, lgag_ks=3, activation='relu6'):
        super(EMCAD, self).__init__()
        eucb_ks = 3  # kernel size for eucb

        self.lgag3 = LGAG(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2, kernel_size=lgag_ks,
                          groups=channels[1] // 2)

        self.lgag2 = LGAG(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2, kernel_size=lgag_ks,
                          groups=channels[2] // 2)

        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2), kernel_size=lgag_ks,
                          groups=int(channels[3] / 2))

        self.para4 = LightweightParallelAttentionFusion(in_channels=channels[0])
        self.para3 = LightweightParallelAttentionFusion(in_channels=channels[1])
        self.para2 = LightweightParallelAttentionFusion(in_channels=channels[2])
        self.para1 = LightweightParallelAttentionFusion(in_channels=channels[3])

        self.cc4 = SplitChannelsOddEven(channels[0])
        self.cc3 = SplitChannelsOddEven(channels[1])
        self.cc2 = SplitChannelsOddEven(channels[2])
        self.cc1 = SplitChannelsOddEven(channels[3])


        self.eucb3 = DySample(channels[0], channels[1])
        self.eucb2 = DySample(channels[1], channels[2])
        self.eucb1 = DySample(channels[2], channels[3])

        self.f1 = Front(channels[1],ilayer = 1, channels = channels)
        self.f2 = Front(channels[2],ilayer = 2, channels = channels)
        self.f3 = Front(channels[3],ilayer = 3, channels = channels)

        # self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        # self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        # self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)

        # self.gconv = nn.Conv2d(num_classes * 4, num_classes * 4, kernel_size = 3, padding = 1, groups = num_classes * 4)
        # self.bn = nn.BatchNorm2d(num_classes * 4)
        # self.relu = nn.ReLU()
        # self.conv3 = nn.Conv2d(num_classes * 4, num_classes, kernel_size = 3, padding = 1)
    

    def forward(self, x, skips=[]):
        
        skips = [x[1],x[2],x[3]]
        x = x[0]
        
        # MSCAM4
        d4 = x
        c4, s4 = self.cc4(x)
        d4 = self.para4(c4, s4)
        # d4 = self.m4(d4)
        # d4 = self.mscb4(d4)

        # EUCB3
        d3 = self.eucb3(d4)

        # LGAG3
        x3 = self.lgag3(g=d3, x=skips[0])

        # Additive aggregation 3
        d3 = d3 + x3
        d3 = self.f1(d3)

        # MSCAM3
        c3, s3 = self.cc3(d3)
        d3 = self.para3(c3, s3)
        # d3 = self.m3(d3)
        # d3 = self.mscb3(d3)

        # EUCB2
        d2 = self.eucb2(d3)

        # LGAG2
        x2 = self.lgag2(g=d2, x=skips[1])

        # Additive aggregation 2
        d2 = d2 + x2
        d2 = self.f2(d2)

        # MSCAM2
        c2, s2 = self.cc2(d2)
        d2 = self.para2(c2, s2)
        # d2 = self.m2(d2)
        # d2 = self.mscb2(d2)

        # EUCB1
        d1 = self.eucb1(d2)

        # LGAG1
        x1 = self.lgag1(g=d1, x=skips[2])

        # Additive aggregation 1
        d1 = d1 + x1
        d1 = self.f3(d1)

        # MSCAM1
        c1, s1 = self.cc1(d1)
        d1 = self.para1(c1, s1)
        # d1 = self.m1(d1)
        # d1 = self.mscb1(d1)

        dec_outs = [d4, d3, d2, d1]

        # p4 = self.out_head4(dec_outs[0])
        # p3 = self.out_head3(dec_outs[1])
        # p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])

        # p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
        # p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
        # p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear')

        # ps = self.conv3(self.relu(self.bn(self.gconv(torch.cat((p1,p2,p3,p4),dim = 1)))))

        return p1
        
        return [p4,p3,p2,p1]


if __name__ == '__main__':
    decoder = EMCAD(expansion_factor=2)

    print('Model %s created, param count: %d' %
          ('EMCAD decoder: ', sum([m.numel() for m in decoder.parameters()])))

