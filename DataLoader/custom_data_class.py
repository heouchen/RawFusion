import torch
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Tuple


class HDRBurstAugment:
    """
    Burst HDR/ISP 任务的数据增广。
    现在数据已 Packing 为 (H/2, W/2, 4)，不再受 Bayer 相位限制。
    """

    def __init__(
        self,
        enable: bool = False,
        crop_enable: bool = True,
        crop_size: int = 256,  # 减半 (512 -> 256)
        crop_sizes: Optional[List[Tuple[int, int]]] = None,
        crop_even_offset: bool = False, # Packing 后不再强制偶数
        geo_enable: bool = False,
        geo_flip_enable: bool = True,
        geo_rot90_enable: bool = True,
        exp_enable: bool = False,
        exp_range_low: Tuple[float, float] = (0.9, 1.1),
        exp_range_mid: Tuple[float, float] = (0.9, 1.1),
        exp_range_high: Tuple[float, float] = (0.9, 1.1),
        exp_global: bool = False,
        wb_enable: bool = False,
        wb_gain_delta: float = 0.05,
        noise_enable: bool = False,
        noise_std: float = 0.0,
        clamp: bool = True,
    ):
        self.enable = bool(enable)
        self.crop_enable = bool(crop_enable)
        self.crop_size = int(crop_size)
        self.crop_sizes = crop_sizes
        self.crop_even_offset = bool(crop_even_offset)
        self.geo_enable = bool(geo_enable)
        self.geo_flip_enable = bool(geo_flip_enable)
        self.geo_rot90_enable = bool(geo_rot90_enable)
        self.exp_enable = bool(exp_enable)
        self.exp_range_low = exp_range_low
        self.exp_range_mid = exp_range_mid
        self.exp_range_high = exp_range_high
        self.exp_global = bool(exp_global)
        self.wb_enable = bool(wb_enable)
        self.wb_gain_delta = float(wb_gain_delta)
        self.noise_enable = bool(noise_enable)
        self.noise_std = float(noise_std)
        self.clamp = bool(clamp)

    def _pick_crop_size(self):
        if not self.crop_sizes:
            return self.crop_size, self.crop_size
        idx = int(torch.randint(0, len(self.crop_sizes), (1,)).item())
        h, w = self.crop_sizes[idx]
        return int(h), int(w)

    def _random_crop_with_size(
        self, inputs: torch.Tensor, target: torch.Tensor, ch: int, cw: int
    ):
        # inputs: (9, 4, H, W)  target: (3, H*2, W*2)
        _, _, h, w = inputs.shape
        if ch <= 0 or cw <= 0 or ch > h or cw > w:
            return inputs, target

        max_y = h - ch
        max_x = w - cw
        if max_y == 0 and max_x == 0:
            return inputs, target

        y = int(torch.randint(0, max_y + 1, (1,)).item())
        x = int(torch.randint(0, max_x + 1, (1,)).item())

        inputs = inputs[:, :, y : y + ch, x : x + cw]
        # 注意：target 是全分辨率，需要对应裁剪
        target = target[:, y * 2 : (y + ch) * 2, x * 2 : (x + cw) * 2]
        return inputs, target

    def _random_crop(self, inputs: torch.Tensor, target: torch.Tensor):
        ch, cw = self._pick_crop_size()
        return self._random_crop_with_size(inputs, target, ch, cw)

    def _sample_geom(self):
        hflip = self.geo_flip_enable and bool(torch.randint(0, 2, (1,)).item())
        vflip = self.geo_flip_enable and bool(torch.randint(0, 2, (1,)).item())
        if self.geo_rot90_enable:
            rot_k = int(torch.randint(0, 4, (1,)).item())
        else:
            rot_k = 0
        return hflip, vflip, rot_k

    def _apply_geom(self, inputs: torch.Tensor, target: torch.Tensor, geom):
        hflip, vflip, rot_k = geom
        if hflip:
            inputs = torch.flip(inputs, dims=[-1])
            target = torch.flip(target, dims=[-1])
        if vflip:
            inputs = torch.flip(inputs, dims=[-2])
            target = torch.flip(target, dims=[-2])
        if rot_k:
            inputs = torch.rot90(inputs, k=rot_k, dims=[-2, -1])
            target = torch.rot90(target, k=rot_k, dims=[-2, -1])
        return inputs, target

    def _apply_exposure_jitter(self, inputs: torch.Tensor, target: torch.Tensor):
        # inputs: (9, 4, H, W), target: (3, H*2, W*2)

        if self.exp_global:
            # 全局增益：所有帧和 GT 乘同一个系数 (模拟场景亮度变化)
            # 使用 mid range 作为全局范围
            lo, hi = self.exp_range_mid
            g = (hi - lo) * torch.rand(1).item() + lo
            return inputs * g, target * g
        else:
            # 独立增益：每帧独立随机 (模拟帧间曝光抖动，GT 不变)
            ranges = [
                self.exp_range_low, self.exp_range_low, self.exp_range_low,
                self.exp_range_mid, self.exp_range_mid, self.exp_range_mid,
                self.exp_range_high, self.exp_range_high, self.exp_range_high,
            ]
            gains = []
            for lo, hi in ranges:
                g = (hi - lo) * torch.rand(1).item() + lo
                gains.append(g)
            gains = torch.tensor(gains, dtype=inputs.dtype, device=inputs.device).view(9, 1, 1, 1)
            return inputs * gains, target

    def _apply_wb_jitter(self, inputs: torch.Tensor):
        # inputs: (9, 4, H, W), packing order: [R, G1, G2, B] (index 0,1,2,3)
        delta = self.wb_gain_delta
        if delta <= 0:
            return inputs

        # Burst 内所有帧共享相同的 WB 增益（相机 WB 设置在 burst 内不变）
        r_gain = (1.0 - delta) + (2.0 * delta) * torch.rand(1).item()
        b_gain = (1.0 - delta) + (2.0 * delta) * torch.rand(1).item()

        gains = torch.ones((1, 4, 1, 1), dtype=inputs.dtype, device=inputs.device)
        gains[0, 0, 0, 0] = r_gain  # R channel (index 0)
        gains[0, 3, 0, 0] = b_gain  # B channel (index 3)

        return inputs * gains  # broadcast (1,4,1,1) -> (9,4,H,W)

    def __call__(self, inputs: torch.Tensor, target: torch.Tensor):
        if not self.enable:
            return inputs, target

        if self.crop_enable:
            inputs, target = self._random_crop(inputs, target)

        if self.geo_enable:
            geom = self._sample_geom()
            inputs, target = self._apply_geom(inputs, target, geom)

        if self.exp_enable:
            inputs, target = self._apply_exposure_jitter(inputs, target)

        if self.wb_enable:
            inputs = self._apply_wb_jitter(inputs)

        if self.noise_enable:
            noise = torch.randn_like(inputs) * self.noise_std
            inputs = inputs + noise

        if self.clamp:
            inputs = inputs.clamp(0.0, 1.0)
            target = target.clamp(0.0, 1.0)
        return inputs.contiguous(), target.contiguous()

    def augment_with_crop_size(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        crop_size: Tuple[int, int],
        geom=None,
    ):
        if not self.enable:
            return inputs, target

        ch, cw = crop_size
        if self.crop_enable:
            inputs, target = self._random_crop_with_size(inputs, target, ch, cw)

        if self.geo_enable:
            if geom is None:
                geom = self._sample_geom()
            inputs, target = self._apply_geom(inputs, target, geom)

        if self.exp_enable:
            inputs, target = self._apply_exposure_jitter(inputs, target)

        if self.wb_enable:
            inputs = self._apply_wb_jitter(inputs)

        if self.noise_enable:
            noise = torch.randn_like(inputs) * self.noise_std
            inputs = inputs + noise

        if self.clamp:
            inputs = inputs.clamp(0.0, 1.0)
            target = target.clamp(0.0, 1.0)
        return inputs.contiguous(), target.contiguous()


class CustomDataset(torch.utils.data.Dataset):
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

        all_files = os.listdir(self.root_dir)
        self.scene_ids = sorted(
            int(f.split("-")[1]) for f in all_files if f.endswith("-gt.tif")
        )
        n_scenes = len(self.scene_ids)
        print(f"Found {n_scenes} scenes in {self.root_dir}")

        self.cache = []
        for idx in tqdm(range(n_scenes), desc="Loading dataset and packing Bayer", ncols=80):
            scene_id = self.scene_ids[idx]
            input_tensors = []
            for i in range(9):
                img_path = f"{self.root_dir}Scene-{scene_id:03d}-in-{i}.tif"
                arr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if arr is None:
                    raise FileNotFoundError(f"Cannot read: {img_path}")
                
                # 处理伪三通道
                if len(arr.shape) == 3:
                    arr = arr[:, :, 0]
                
                # Bayer Packing (GRBG pattern)
                # G1 R
                # B G2
                g1 = arr[0::2, 0::2]
                r  = arr[0::2, 1::2]
                b  = arr[1::2, 0::2]
                g2 = arr[1::2, 1::2]
                
                packed = np.stack([r, g1, g2, b], axis=0) # (4, 384, 768)
                input_tensors.append((torch.from_numpy(packed).float() / 65535.0)**(1/2.2) if arr.dtype == np.uint16 else (torch.from_numpy(packed).float())**(1/2.2))

            gt_path = f"{self.root_dir}Scene-{scene_id:03d}-gt.tif"
            gt_arr = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt_arr is None:
                raise FileNotFoundError(f"Cannot read: {gt_path}")
            
            # GT (768, 1536, 3) -> (3, 768, 1536)
            gt_tensor = self.transform(gt_arr) if self.transform else torch.from_numpy(gt_arr).permute(2,0,1).float()

            inputs = torch.stack(input_tensors)  # (9, 4, 384, 768)
            self.cache.append((inputs, gt_tensor))

        print(f"Cached {len(self.cache)} scenes in memory.")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        inputs, target = self.cache[idx]
        if self.train and (self.augment is not None):
            inputs, target = self.augment(inputs, target)
        return inputs, target
