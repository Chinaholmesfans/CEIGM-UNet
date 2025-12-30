from __future__ import annotations
import cv2
import numpy as np
import torch
from typing import Sequence

'''
# 改进后的 SYNAPSE_COLORMAP
SYNAPSE_COLORMAP = {
    1: [220, 130, 120][::-1],  # (202, 98, 81) -> 更柔和明亮
    2: [120, 130, 160][::-1],  # (97, 107, 142) -> 更柔和明亮
    3: [220, 200, 80][::-1],   # (205, 189, 0) -> 更柔和明亮
    4: [30, 150, 80][::-1],    # (0, 130, 40) -> 更柔和明亮
    5: [30, 160, 200][::-1],   # (0, 146, 178) -> 更柔和明亮
    6: [190, 200, 190][::-1],  # (176, 189, 181) -> 更柔和明亮
    7: [90, 90, 140][::-1],    # (72, 77, 126) -> 更柔和明亮
    8: [80, 180, 180][::-1]    # (58, 163, 167) -> 更柔和明亮
}
'''
'''
SYNAPSE_COLORMAP = {
    1: [255, 100, 100][::-1],  # 更鲜艳的红色
    2: [100, 255, 100][::-1],  # 更鲜艳的绿色
    3: [255, 255, 100][::-1],  # 更鲜艳的黄色
    4: [100, 100, 255][::-1],  # 更鲜艳的蓝色
    5: [255, 100, 255][::-1],  # 更鲜艳的紫色
    6: [100, 255, 255][::-1],  # 更鲜艳的青色
    7: [255, 150, 50][::-1],   # 更鲜艳的橙色
    8: [150, 50, 255][::-1]    # 更鲜艳的紫红色
}
'''

SYNAPSE_COLORMAP = {
    1:(254,112,106)[::-1],
    2:(120,124,186)[::-1],
    3:(255,218,0)[::-1],
    4:(0,150,54)[::-1],
    5:(0,168,233)[::-1],
    6:(219,218,237)[::-1],
    7:(115,43,245)[::-1],
    8:(73,188,219)[::-1]
}

ACDC_COLORMAP = {
    1:(254,112,106)[::-1],
    2:(120,124,186)[::-1],
    3:(255,218,0)[::-1],
}

class2colormap = {
    9: SYNAPSE_COLORMAP,
    4: ACDC_COLORMAP
}

def make_rgb_darker(color: Sequence[int, int, int], percentage: float = 0.5) -> tuple[int, int, int]:
    def _dark(c: int) -> int:
        return int(max(0., c - c * percentage))
    r, g, b = color
    return _dark(r), _dark(g), _dark(b)

def is_grayscale(image: np.ndarray | torch.Tensor) -> bool:
    return not (len(image.shape) > 2 and image.shape[2] > 1)


def save_x_y(x: np.ndarray, y: np.ndarray, colormap: dict, out: str) -> None:
    """
    改进后的渲染函数，轮廓线颜色与填充颜色区分度更大。
    """
    assert all([x.dtype == np.uint8, y.dtype == np.uint8])
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if is_grayscale(x) else cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    
    # 创建一个空白图像用于叠加颜色
    overlay = np.zeros_like(x, dtype=np.uint8)
    
    for i, color in colormap.items():
        mask = (y == i).all(axis=2)  # 获取当前类别的掩码
        overlay[mask] = color  # 在 overlay 上填充颜色
        
        # 计算填充颜色的亮度
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        
        # 根据亮度选择轮廓线颜色
        if brightness > 128:  # 填充颜色较亮，使用深色轮廓线
            contour_color = (0, 0, 0)  # 黑色
        else:  # 填充颜色较暗，使用浅色轮廓线
            contour_color = (255, 255, 255)  # 白色
        
        # 绘制轮廓（细线条）
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(x, contours, -1, contour_color, thickness=1)  # 细线条
    
    # 叠加颜色并保存
    alpha = 0.6  # 透明度
    x = cv2.addWeighted(overlay, alpha, x, 1 - alpha, 0)
    cv2.imwrite(out, x)


def save_x_y_hat(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, colormap: dict, out: str) -> None:
    """
    改进后的渲染函数，轮廓线颜色与填充颜色区分度更大。
    """
    assert all([x.dtype == np.uint8, y.dtype == np.uint8, y_hat.dtype == np.uint8])
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if is_grayscale(x) else cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    y_hat = cv2.cvtColor(y_hat, cv2.COLOR_GRAY2BGR)
    
    # 创建一个空白图像用于叠加颜色
    overlay = np.zeros_like(x, dtype=np.uint8)
    
    for i, color in colormap.items():
        mask = (y_hat == i).all(axis=2)  # 获取当前类别的掩码
        overlay[mask] = color  # 在 overlay 上填充颜色
        
        # 计算填充颜色的亮度
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        
        # 根据亮度选择轮廓线颜色
        if brightness > 128:  # 填充颜色较亮，使用深色轮廓线
            contour_color = (0, 0, 0)  # 黑色
        else:  # 填充颜色较暗，使用浅色轮廓线
            contour_color = (255, 255, 255)  # 白色
        
        # 绘制轮廓（细线条）
        contours, _ = cv2.findContours((y == i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(x, contours, -1, contour_color, thickness=1)  # 细线条
    
    # 叠加颜色并保存
    alpha = 0.6  # 透明度
    x = cv2.addWeighted(overlay, alpha, x, 1 - alpha, 0)
    cv2.imwrite(out, x)

'''
def save_x_y(x: np.ndarray, y: np.ndarray, colormap: dict, out: str) -> None:
    """
    改进后的渲染函数，增加颜色平滑度和轮廓清晰度。
    """
    assert all([x.dtype == np.uint8, y.dtype == np.uint8])
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if is_grayscale(x) else cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    
    # 创建一个空白图像用于叠加颜色
    overlay = np.zeros_like(x, dtype=np.uint8)
    
    for i, color in colormap.items():
        mask = (y == i).all(axis=2)  # 获取当前类别的掩码
        overlay[mask] = color  # 在 overlay 上填充颜色
        
        # 绘制轮廓
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(x, contours, -1, make_rgb_darker(color, percentage=0.3), thickness=2)
    
    # 叠加颜色并保存
    alpha = 0.6  # 透明度
    x = cv2.addWeighted(overlay, alpha, x, 1 - alpha, 0)
    cv2.imwrite(out, x)


def save_x_y_hat(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, colormap: dict, out: str) -> None:
    """
    改进后的渲染函数，增加颜色平滑度和轮廓清晰度。
    """
    assert all([x.dtype == np.uint8, y.dtype == np.uint8, y_hat.dtype == np.uint8])
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if is_grayscale(x) else cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    y_hat = cv2.cvtColor(y_hat, cv2.COLOR_GRAY2BGR)
    
    # 创建一个空白图像用于叠加颜色
    overlay = np.zeros_like(x, dtype=np.uint8)
    
    for i, color in colormap.items():
        mask = (y_hat == i).all(axis=2)  # 获取当前类别的掩码
        overlay[mask] = color  # 在 overlay 上填充颜色
        
        # 绘制轮廓
        contours, _ = cv2.findContours((y == i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(x, contours, -1, make_rgb_darker(color, percentage=0.3), thickness=2)
    
    # 叠加颜色并保存
    alpha = 0.6  # 透明度
    x = cv2.addWeighted(overlay, alpha, x, 1 - alpha, 0)
    cv2.imwrite(out, x)
'''

def save_x_y_tensor(x: torch.Tensor, y: torch.Tensor, colormap: dict, out: str) -> None:
    """
    输入张量并保存渲染结果。
    """
    x = x if is_grayscale(x) else x.permute(1, 2, 0)
    x = x.detach().cpu().numpy().astype(np.uint8)
    y = y.detach().cpu().numpy().astype(np.uint8)
    save_x_y(x, y, colormap, out)