from __future__ import annotations
import os
import re
import copy
import torch
from loguru import logger
from functools import partial
from typing import Optional, Any
from model.gm.groupmamba import GroupMamba
from torch import nn

__all__ = ["GroupMamba", "ENCODERS", "build_model"]

DEFAULT_CONFIG = {
    "stem_hidden_dim": 32,
    "embed_dims": [64,128,348,448],
    "mlp_ratios" : [8, 8, 4, 4],
    "norm_layer" : partial(nn.LayerNorm, eps=1e-6), 
    "depths" : [3, 4, 9, 3],
}

def get_config(config: dict[str, Any]) -> dict[str, Any]:
    target = copy.deepcopy(DEFAULT_CONFIG)
    target.update(config)
    return target

def load_pretrained_ckpt(model: GroupMamba, ckpt: str) -> GroupMamba:
    logger.info(f"Loading weights from: {ckpt}")
    skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias","dist_head.weight","dist_head.bias"]

    t_device = next(model.parameters()).device
    model = model.cpu()
    ckpt = torch.load(ckpt, map_location="cpu")
    # print(ckpt.keys())
    model_dict = model.state_dict()
    loaded_key_set = set()
    for kr, v in ckpt.items():
        if kr in skip_params:
            logger.info(f"Skipping weights: {kr}")
            continue
        if "downsample" in kr:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
            assert kr in model_dict.keys()
        if "ln_1" in kr:
            kr = kr.replace("ln_1", "norm")
        if "self_attention" in kr:
            kr = kr.replace("self_attention", "op")
        if kr in model_dict.keys():
            assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
            model_dict[kr] = v
            loaded_key_set.add(kr)
            # logger.info(f"Loaded weights: {kr}")
        else:
            logger.info(f"Passing weights: {kr}")

    model.load_state_dict(model_dict)
    return model.to(t_device)

def build_model(config: dict[str, Any], ckpt: Optional[str] = None, **kwargs: Any) -> GroupMamba:
    config = get_config(config)
    model = GroupMamba(
        stem_hidden_dim = config["stem_hidden_dim"],
        embed_dims = config["embed_dims"], 
        mlp_ratios = config["mlp_ratios"],
        norm_layer = config["norm_layer"], 
        depths = config["depths"],
        **kwargs
    )

    print(ckpt)
    if ckpt and os.path.exists(ckpt):
        model = load_pretrained_ckpt(model=model, ckpt=ckpt)
    return model

def build_gm_tiny(**kwargs: Any) -> GroupMamba:
    return build_model({
        "stem_hidden_dim": 32,
        "embed_dims": [64,128,348,448],
        "mlp_ratios" : [8, 8, 4, 4],
        "norm_layer" : partial(nn.LayerNorm, eps=1e-6), 
        "depths" : [3, 4, 9, 3],
    }, **kwargs)


root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENCODERS = {
    "gm_tiny": partial(
        build_gm_tiny,
        ckpt=os.path.join(root, "pretrain/groupmamba_tiny_ema.pth"),
    )
}
