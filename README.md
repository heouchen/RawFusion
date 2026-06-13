# RawFusion

RawFusion is a PyTorch workspace for the NTIRE Efficient Burst HDR and Restoration task. It trains and evaluates lightweight RAW burst fusion models that reconstruct one HDR RGB image from nine degraded RAW input frames.

## Repository Layout

```text
DataLoader/          dataset loading, augmentation, and progressive crop collate logic
models/              model registry and model implementations
utils/               checkpoint, metric, EMA, and loss helpers
scripts/             training and submission convenience scripts
scoring_program/     challenge-style PSNR/SSIM scorer
train.py             main training entry point
eval.py              validation/test inference and result.zip packaging
reparam_model.py     structural re-parameterization helper for supported models
test_demo.py         lightweight dataset inspection utility
```

Generated files such as checkpoints, logs, output previews, datasets, and submission images are intentionally ignored by Git.

## Environment

The code is developed with Python 3 and PyTorch. Install the runtime dependencies with:

```bash
pip install -r requirements.txt
```

CUDA is recommended for training and full-frame evaluation.

## Data Layout

Each scene is expected to be stored as flat TIFF files:

```text
Scene-000-in-0.tif
Scene-000-in-1.tif
...
Scene-000-in-8.tif
Scene-000-gt.tif
```

Input RAW files are single-channel Bayer images with GRBG layout. The loader packs each RAW frame to four channels and maps one scene to a tensor shaped `(9, 4, H/2, W/2)`. GT files are read as 3-channel float RGB/BGR tensors in `[0, 1]`.

Default local paths used by scripts:

```text
/home/chen/data/ntire2026/hdr/train/
/home/chen/data/ntire2026/hdr/validation/
/home/chen/data/ntire2026/hdr/test/
```

Override these paths with `--train_root`, `--val_root`, or the corresponding environment variables in `scripts/run.sh`.

## Training

Use `train.py` directly for controlled experiments:

```bash
python train.py \
  --model safnet_claude_33_v3 \
  --exp_name model_submit_claude33_v3 \
  --train_root /home/chen/data/ntire2026/hdr/train/ \
  --val_root /home/chen/data/ntire2026/hdr/validation/ \
  --epochs 1000 \
  --batch_size 4 \
  --lr 2e-4 \
  --loss mse \
  --aug_enable 1 \
  --aug_crop_enable 1 \
  --aug_crop_sizes 128x128 \
  --aug_geo_enable 1
```

The convenience wrapper `scripts/run.sh` contains the current experiment defaults and can be launched with:

```bash
bash scripts/run.sh --gpu 0 --num_workers 4 --val_every 1
```

Checkpoints are written under `checkpoint_dir/checkpoint_dir_<model>_<exp_name>/`. Training logs are written under `output_log/`.

## Evaluation and Submission

Run validation or test inference with `eval.py`:

```bash
python eval.py \
  --model safnet_claude_33_v3 \
  --exp_name model_submit_claude33_v3 \
  --val_root /home/chen/data/ntire2026/hdr/validation/
```

For final submission-style inference with re-parameterization, TTA, and TLC:

```bash
bash scripts/test.sh safnet_claude_33_v3 model_submit_claude33_v3 128 128
```

Outputs are saved to the experiment checkpoint directory:

```text
checkpoint_dir/checkpoint_dir_<model>_<exp_name>/img/
checkpoint_dir/checkpoint_dir_<model>_<exp_name>/result.zip
```

## Model Registry

Models are registered in `models/__init__.py` and can be selected with `--model`. Current notable entries include:

```text
safnet_claude_33
safnet_claude_33_v2
safnet_claude_33_v3
safnet_claude_33_v4
rawnet
unet
```

When adding a new model, implement it under `models/`, import it in `models/__init__.py`, and add it to `_MODEL_MAP`.

## Notes

- `output_log/`, `checkpoint_dir/`, `output/`, datasets, and generated TIFF/ZIP files are local artifacts and are ignored by Git.
- Do not commit private keys, pretrained checkpoints, datasets, or generated submissions.
- If a private key has ever been pushed to a remote repository, remove it from active use and rotate it even after deleting it from the current tree.
