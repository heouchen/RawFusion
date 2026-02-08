import torch
import torchvision.transforms as transforms
import os
import cv2
from tqdm import tqdm
from typing import Optional


class HDRBurstAugment:
    """
    Burst HDR/ISP 任务的数据增广（按样本随机、可配置开关）。

    重要说明（RAW Bayer 安全性）：
    - 输入是单通道 Bayer mosaic（通过 squeeze(2) 变成 9 通道喂给网络）。
    - 对 Bayer mosaic 做翻转/旋转/奇数像素平移会改变 CFA 相位（GRBG->RGGB/BGGR/...），与真实测试分布不一致。
    - 因此默认仅提供“Bayer-safe”的增广：偶数偏移裁剪 + （可选）加噪。
    - 如果你明确希望做翻转/旋转，请在模型侧做 pack/unpack 或显式处理 CFA 相位后再打开。
    """

    def __init__(
        self,
        enable: bool = False,
        crop_enable: bool = True,
        crop_size: int = 512,
        crop_even_offset: bool = True,
        noise_enable: bool = False,
        noise_std: float = 0.0,
        clamp: bool = True,
    ):
        self.enable = bool(enable)
        self.crop_enable = bool(crop_enable)
        self.crop_size = int(crop_size)
        self.crop_even_offset = bool(crop_even_offset)
        self.noise_enable = bool(noise_enable)
        self.noise_std = float(noise_std)
        self.clamp = bool(clamp)

    def _random_crop(self, inputs: torch.Tensor, target: torch.Tensor):
        # inputs: (9, C, H, W)  target: (C, H, W)
        _, _, h, w = inputs.shape
        ps = self.crop_size
        if ps <= 0 or ps > h or ps > w:
            return inputs, target

        max_y = h - ps
        max_x = w - ps
        if max_y == 0 and max_x == 0:
            return inputs, target

        if self.crop_even_offset:
            # Bayer-safe：保持 (x,y) parity，不改变 CFA 相位
            y = int(torch.randint(0, max_y + 1, (1,)).item())
            x = int(torch.randint(0, max_x + 1, (1,)).item())
            y = (y // 2) * 2
            x = (x // 2) * 2
            y = min(y, max_y)
            x = min(x, max_x)
        else:
            y = int(torch.randint(0, max_y + 1, (1,)).item())
            x = int(torch.randint(0, max_x + 1, (1,)).item())

        inputs = inputs[:, :, y : y + ps, x : x + ps]
        target = target[:, y : y + ps, x : x + ps]
        return inputs, target

    def _add_noise(self, inputs: torch.Tensor):
        if self.noise_std <= 0:
            return inputs
        # 简单高斯读噪（对 RAW 更贴近真实；GT 不加噪）
        noise = torch.randn_like(inputs) * self.noise_std
        return inputs + noise

    def __call__(self, inputs: torch.Tensor, target: torch.Tensor):
        if not self.enable:
            return inputs, target

        # 注意：self.cache 里的 Tensor 会在多进程间共享；这里绝不能原地修改缓存
        if self.crop_enable:
            inputs, target = self._random_crop(inputs, target)

        if self.noise_enable:
            inputs = self._add_noise(inputs)

        if self.clamp:
            inputs = inputs.clamp(0.0, 1.0)
            target = target.clamp(0.0, 1.0)
        return inputs.contiguous(), target.contiguous()


class CustomDataset(torch.utils.data.Dataset):
    """
    预加载全部图像到内存，供 DataLoader 多 worker 共享访问。
    - Linux: 主进程 __init__ 中加载后，fork 出的 worker 通过 copy-on-write 共享同一份内存，只读访问不复制。
    - 所有图像保存在 self.cache 中，__getitem__ 仅做索引返回，无磁盘 IO。
    """

    def __init__(
        self,
        root_dir,
        transform=transforms.ToTensor(),
        train=True,
        augment: Optional[HDRBurstAugment] = None,
    ):
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir.rstrip("/") + "/"
        self.transform = transform
        self.train = train
        self.augment = augment

        # 扫描实际存在的场景编号
        all_files = os.listdir(self.root_dir)
        self.scene_ids = sorted(
            int(f.split("-")[1]) for f in all_files if f.endswith("-gt.tif")
        )
        n_scenes = len(self.scene_ids)
        print(f"Found {n_scenes} scenes in {self.root_dir}")

        # 预加载所有图像到内存（主进程执行，worker fork 后共享此内存）
        self.cache = []
        for idx in tqdm(range(n_scenes), desc="Loading dataset into memory", ncols=80):
            scene_id = self.scene_ids[idx]
            input_tensors = []
            for i in range(9):
                img_path = f"{self.root_dir}Scene-{scene_id:03d}-in-{i}.tif"
                arr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if arr is None:
                    raise FileNotFoundError(f"Cannot read: {img_path}")
                input_tensors.append(
                    self.transform(arr) if self.transform else torch.from_numpy(arr)
                )
            gt_path = f"{self.root_dir}Scene-{scene_id:03d}-gt.tif"
            gt_arr = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt_arr is None:
                raise FileNotFoundError(f"Cannot read: {gt_path}")
            gt_tensor = (
                self.transform(gt_arr) if self.transform else torch.from_numpy(gt_arr)
            )
            inputs = torch.stack(input_tensors)  # (9, C, H, W)
            self.cache.append((inputs, gt_tensor))

        print(f"Cached {len(self.cache)} scenes in memory.")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        """
        Returns:
            inputs (Tensor): (9, C, H, W)
            target (Tensor): (C, H, W)
        """
        inputs, target = self.cache[idx]
        if self.train and (self.augment is not None):
            inputs, target = self.augment(inputs, target)
        return inputs, target
