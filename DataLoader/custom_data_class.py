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

    def __call__(self, inputs: torch.Tensor, target: torch.Tensor):
        if not self.enable:
            return inputs, target

        if self.crop_enable:
            inputs, target = self._random_crop(inputs, target)

        if self.geo_enable:
            geom = self._sample_geom()
            inputs, target = self._apply_geom(inputs, target, geom)

        if self.clamp:
            if torch.is_floating_point(inputs):
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

        if self.clamp:
            if torch.is_floating_point(inputs):
                inputs = inputs.clamp(0.0, 1.0)
            target = target.clamp(0.0, 1.0)
        return inputs.contiguous(), target.contiguous()


def pack_raw_bayer(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    g1 = arr[0::2, 0::2]
    r = arr[0::2, 1::2]
    b = arr[1::2, 0::2]
    g2 = arr[1::2, 1::2]
    packed = np.stack([r, g1, g2, b], axis=0)
    return torch.from_numpy(np.ascontiguousarray(packed))


def to_float_tensor(arr: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(arr))
    if tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)
    tensor = tensor.float()
    if arr.dtype == np.uint16:
        tensor.div_(65535.0)
    elif arr.dtype == np.uint8:
        tensor.div_(255.0)
    return tensor


def finalize_input_tensor(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.dtype == torch.uint16:
        inputs = inputs.float().div_(65535.0)
    else:
        inputs = inputs.float()
    inputs.clamp_(min=0.0)
    return inputs.pow_(1 / 2.2)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        transform=transforms.ToTensor(),
        train=True,
        augment: Optional[HDRBurstAugment] = None,
        finalize_inputs: bool = True,
    ):
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir.rstrip("/") + "/"
        self.transform = transform
        self.train = train
        self.augment = augment
        self.finalize_inputs = bool(finalize_inputs)

        all_files = os.listdir(self.root_dir)

        # ----- 1) 先尝试标准「有 GT」模式 -----
        gt_scene_ids = sorted(
            int(f.split("-")[1]) for f in all_files if f.endswith("-gt.tif")
        )
        self.input_only_mode = len(gt_scene_ids) == 0
        self.samples = []
        self.cache = []
        if not self.input_only_mode:
            self.scene_ids = gt_scene_ids
            n_scenes = len(self.scene_ids)
            print(f"Found {n_scenes} scenes in {self.root_dir}")
            for scene_id in tqdm(
                self.scene_ids,
                desc="Loading dataset and packing Bayer",
                ncols=80,
            ):
                input_tensors = []
                for i in range(9):
                    img_path = f"{self.root_dir}Scene-{scene_id:03d}-in-{i}.tif"
                    arr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if arr is None:
                        raise FileNotFoundError(f"Cannot read: {img_path}")
                    input_tensors.append(pack_raw_bayer(arr))

                gt_path = f"{self.root_dir}Scene-{scene_id:03d}-gt.tif"
                gt_arr = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                if gt_arr is None:
                    raise FileNotFoundError(f"Cannot read: {gt_path}")

                inputs = torch.stack(input_tensors, dim=0)
                target = to_float_tensor(gt_arr)
                self.cache.append((inputs, target))

            print(f"Cached {len(self.cache)} scenes in memory.")
        else:
            # ----- 2) 兼容「无 GT、仅输入」的推理模式 -----
            # 约定：存在 Scene-XXX-in-0.tif 即认为是一个 scene
            input_scene_ids = sorted(
                int(f.split("-")[1])
                for f in all_files
                if f.endswith("-in-0.tif")
            )
            self.scene_ids = input_scene_ids
            n_scenes = len(self.scene_ids)
            print(
                f"Found {n_scenes} scenes in {self.root_dir} "
                f"(input-only, no GT files)"
            )
            for scene_id in self.scene_ids:
                input_paths = [
                    f"{self.root_dir}Scene-{scene_id:03d}-in-{i}.tif"
                    for i in range(9)
                ]
                self.samples.append((input_paths, None))

    def __len__(self):
        if self.input_only_mode:
            return len(self.samples)
        return len(self.cache)

    def __getitem__(self, idx):
        if self.input_only_mode:
            input_paths, gt_path = self.samples[idx]
            input_tensors = []
            for img_path in input_paths:
                arr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if arr is None:
                    raise FileNotFoundError(f"Cannot read: {img_path}")
                input_tensors.append(pack_raw_bayer(arr))

            inputs = torch.stack(input_tensors, dim=0)

            # 推理模式没有 GT，直接返回 burst 输入
            if gt_path is None:
                if self.finalize_inputs:
                    inputs = finalize_input_tensor(inputs)
                return inputs

            gt_arr = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt_arr is None:
                raise FileNotFoundError(f"Cannot read: {gt_path}")
            target = to_float_tensor(gt_arr)
        else:
            inputs, target = self.cache[idx]
            # clone cached tensors before any in-place ops (augment/finalize)
            inputs = inputs.clone()
            target = target.clone()
        if self.train and (self.augment is not None):
            inputs, target = self.augment(inputs, target)
            if self.augment.clamp:
                target = target.clamp(0.0, 1.0)
        if self.finalize_inputs:
            inputs = finalize_input_tensor(inputs)
        return inputs, target
