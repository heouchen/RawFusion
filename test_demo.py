import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
import os, sys, time, shutil
from PIL import Image
from torchvision.transforms import transforms
to_pil_image = transforms.ToPILImage()
import torchvision.transforms.functional as F
# from DataLoader.custom_data_class import CustomDataset
# from utils.utils import *
# from utils.checkpoint import *



def analyze_train_data(train_root="./trainset/"):
    """
    加载训练数据，打印整体最小/最大值，以及高中低曝光的均值。
    假设：
      - train_root 下都是 tif/png 等图片
      - 文件名中包含 'low' / 'mid' / 'high' 用来区分曝光档
    """
    files = [f for f in os.listdir(train_root) if not ('gt.tif' in f)]
    print(f"训练集文件总数: {len(files)}")

    all_pixels = []
    exposure_pixels = {i: [] for i in range(9)}

    for name in files:
        path = os.path.join(train_root, name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 保留原始bit深度
        if img is None:
            print(f"警告: 无法读取 {path}")
            continue

        # 转为 float32 做统计，避免溢出
        arr = img.astype(np.float32)

        all_pixels.append(arr)

        lname = name.lower()
        for k in range(9):
            if f"{k}.tif" in lname:
                exposure_pixels[k].append(arr)
                break

    if not all_pixels:
        print("未成功读取到训练图像，检查 train_root 路径和文件格式。")
        return

    all_stack = np.concatenate([a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(-1, 1)
                                for a in all_pixels], axis=0)
    print(f"训练集像素整体最小值: {all_stack.min()}")
    print(f"训练集像素整体最大值: {all_stack.max()}")

    def print_stats(name, arr_list):
        if not arr_list:
            print(f"{name} 曝光: 未找到样本")
            return
        stack = np.concatenate([a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(-1, 1)
                                for a in arr_list], axis=0)
        mean_val = stack.mean(axis=0)
        min_val = stack.min(axis=0)
        max_val = stack.max(axis=0)
        print(f"{name} 曝光均值: {mean_val}, 曝光最小值: {min_val}, 曝光最大值: {max_val}")

    for k in range(9):
        print_stats(str(k), exposure_pixels[k])

if __name__ == '__main__':
    analyze_train_data('/home/chen/data/ntire2026/hdr/validation/')