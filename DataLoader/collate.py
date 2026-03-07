import torch
from DataLoader.custom_data_class import HDRBurstAugment


def parse_crop_sizes(s: str):
    """
    Parse "HxW,HxW,..." to List[Tuple[int,int]].
    Example: "192x384,384x768,768x1536"
    """
    if not s:
        return None
    sizes = []
    for part in s.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Invalid crop size: {part}. Expect HxW.")
        h_str, w_str = part.split("x", 1)
        sizes.append((int(h_str), int(w_str)))
    return sizes if sizes else None


def parse_progressive_crop_schedule(s: str):
    """
    Parse "HxW@r,HxW@r,..." to [((H, W), cumulative_ratio)].
    Example: "96x192@0.3,192x384@0.7,384x768@1.0"
    """
    if not s:
        return None

    schedule = []
    prev_ratio = 0.0
    for part in s.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "@" not in part:
            raise ValueError(f"Invalid progressive crop entry: {part}. Expect HxW@ratio.")
        size_part, ratio_part = part.split("@", 1)
        sizes = parse_crop_sizes(size_part)
        if not sizes or len(sizes) != 1:
            raise ValueError(f"Invalid progressive crop size: {size_part}. Expect one HxW value.")
        ratio = float(ratio_part)
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"Invalid progressive crop ratio: {ratio}. Expect 0 < ratio <= 1.")
        if ratio <= prev_ratio:
            raise ValueError("Progressive crop ratios must be strictly increasing.")
        schedule.append((sizes[0], ratio))
        prev_ratio = ratio

    if not schedule:
        return None
    if abs(schedule[-1][1] - 1.0) > 1e-6:
        raise ValueError("Progressive crop schedule must end at ratio 1.0.")
    return schedule


def parse_progressive_batch_sizes(s: str):
    """
    Parse "HxW@batch,HxW@batch,..." to {(H, W): batch_size}.
    Example: "96x192@16,192x384@8,384x768@4"
    """
    if not s:
        return None

    batch_sizes = {}
    for part in s.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "@" not in part:
            raise ValueError(f"Invalid progressive batch entry: {part}. Expect HxW@batch.")
        size_part, batch_part = part.split("@", 1)
        sizes = parse_crop_sizes(size_part)
        if not sizes or len(sizes) != 1:
            raise ValueError(f"Invalid progressive batch crop size: {size_part}. Expect one HxW value.")
        batch_size = int(batch_part)
        if batch_size <= 0:
            raise ValueError(f"Invalid progressive batch size: {batch_size}. Expect > 0.")
        batch_sizes[sizes[0]] = batch_size

    return batch_sizes if batch_sizes else None


class CropScheduleController:
    def __init__(
        self,
        random_crop_sizes=None,
        progressive_enable=False,
        progressive_schedule=None,
        default_batch_size=1,
        progressive_batch_enable=False,
        progressive_batch_sizes=None,
    ):
        self.random_crop_sizes = random_crop_sizes
        self.progressive_enable = bool(progressive_enable)
        self.progressive_schedule = progressive_schedule or []
        self.default_batch_size = int(default_batch_size)
        self.progressive_batch_enable = bool(progressive_batch_enable)
        self.progressive_batch_sizes = progressive_batch_sizes or {}
        self.current_epoch = 0
        self.total_epochs = 1

    def set_epoch(self, epoch: int, total_epochs: int):
        self.current_epoch = int(epoch)
        self.total_epochs = max(int(total_epochs), 1)

    def current_crop_size(self):
        if not self.progressive_enable:
            return None
        progress = float(self.current_epoch + 1) / float(self.total_epochs)
        for crop_size, ratio in self.progressive_schedule:
            if progress <= ratio:
                return crop_size
        return self.progressive_schedule[-1][0] if self.progressive_schedule else None

    def current_batch_size(self):
        crop_size = self.current_crop_size()
        if (not self.progressive_batch_enable) or (crop_size is None):
            return self.default_batch_size
        return int(self.progressive_batch_sizes.get(crop_size, self.default_batch_size))

    def describe_mode(self):
        crop_size = self.current_crop_size()
        if crop_size is not None:
            return "progressive", f"{crop_size[0]}x{crop_size[1]}"
        if self.random_crop_sizes:
            return "random", ",".join(f"{h}x{w}" for h, w in self.random_crop_sizes)
        return "fixed", ""


def make_train_collate(augment: HDRBurstAugment, crop_sizes, batch_geom=False, crop_controller: CropScheduleController = None):
    """
    Batch-level multi-scale crop: ensure same HxW within a batch.
    """
    def _collate(batch):
        # batch: List[(inputs, target)]
        if (augment is None) or (not augment.enable):
            inputs = torch.stack([b[0] for b in batch], dim=0)
            targets = torch.stack([b[1] for b in batch], dim=0)
            return inputs, targets

        geom = None
        if batch_geom and augment.geo_enable:
            geom = augment._sample_geom()

        progressive_crop_size = crop_controller.current_crop_size() if crop_controller is not None else None
        if progressive_crop_size is not None:
            ch, cw = progressive_crop_size
            augmented = [
                augment.augment_with_crop_size(inp, tgt, (ch, cw), geom=geom)
                for (inp, tgt) in batch
            ]
        elif crop_sizes:
            idx = int(torch.randint(0, len(crop_sizes), (1,)).item())
            ch, cw = crop_sizes[idx]
            augmented = [
                augment.augment_with_crop_size(inp, tgt, (ch, cw), geom=geom)
                for (inp, tgt) in batch
            ]
        else:
            if augment.crop_enable:
                ch, cw = augment.crop_size, augment.crop_size
            else:
                ch, cw = batch[0][0].shape[-2], batch[0][0].shape[-1]
            augmented = [
                augment.augment_with_crop_size(inp, tgt, (ch, cw), geom=geom)
                for (inp, tgt) in batch
            ]

        inputs = torch.stack([a[0] for a in augmented], dim=0)
        targets = torch.stack([a[1] for a in augmented], dim=0)
        return inputs, targets

    return _collate
