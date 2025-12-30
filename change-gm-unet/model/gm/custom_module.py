import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Sequence, Type

import math

try:
    from .groupmamba import Block_mamba
    from .custom_mlp import custom_ffn
except:
    from groupmamba import Block_mamba
    from custom_mlp import custom_ffn

"""
        dim, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm,
        custom_mlp=None
"""
"""
        in_chans=3, 
        num_classes=1000, 
        stem_hidden_dim = 32,
        embed_dims=[64, 128, 348, 448],
        mlp_ratios=[8, 8, 4, 4], 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        num_stages=4,
        distillation=True,
        **kwargs
"""

class cm(nn.Module):
    def __init__(self,
                 dim:int = 0,
                 mlp_ratio = 4.0,
                 depth = 2,
                 drop_path = 0.0,
                 norm_layer = nn.LayerNorm,
                ):
        super(cm, self).__init__()

        self.blocks = nn.ModuleList([Block_mamba(
                    dim = dim,
                    mlp_ratio = mlp_ratio,
                    drop_path = drop_path[each] if isinstance(drop_path, Sequence) else drop_path,
                    norm_layer = norm_layer,
                    custom_mlp = custom_ffn
                ) for each in range(depth)])
        
    def forward(self,x):
        B,C,H,W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        for each in self.blocks:
            x = each(x,H,W)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x
        
    