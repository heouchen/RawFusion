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


def make_train_collate(augment: HDRBurstAugment, crop_sizes, batch_geom=False):
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

        if crop_sizes:
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
