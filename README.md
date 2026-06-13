# RawFusion

RawFusion 是面向 NTIRE 2026 Efficient Burst HDR and Restoration 任务的 PyTorch 工程。输入为 9 帧退化 RAW burst，模型输出一张 HDR RGB 图像。本仓库当前整理为主线复现实验代码，保留 SAFNet Claude-33 系列、RawNet 复现实验和 UNet baseline，历史试错快照已从代码树中移除。

## VIBE Coding 精度提升

这里的 VIBE Coding 指通过大模型辅助进行模型结构、训练脚本、增广策略和验证流程的快速迭代，而不是一个额外运行库。下表来自本机 `output_log/` 训练验证日志；该目录被 Git 忽略，表中 PSNR 是有 GT 验证集上的本地验证结果，部分结果属于 crop/full-crop 验证口径。

| 阶段 | 模型/实验 | 本地 Val PSNR | 相对 31.013 dB baseline | 说明 |
| --- | --- | ---: | ---: | --- |
| 起点 | challenge/starter UNet baseline | 31.013 dB | +0.00 dB | 参考基线 |
| 早期 VIBE 迭代 | archived SAFNet Claude branch | 35.204 dB | +4.19 dB | 结构快速搜索阶段 |
| 主线收敛 | `safnet_claude_33_v2` | 39.323 dB | +8.31 dB | Claude-33 主干成型 |
| 最终提交主模型 | `safnet_claude_33_v3` | 39.726 dB | +8.71 dB | 9-frame group-prepared variant |
| 后期微调最好日志 | `safnet_claude_33_v2_fullcrop448_ft_lr1e6` | 39.842 dB | +8.83 dB | full-crop fine-tune 日志最好值 |
| 二名结构复现对照 | `rawnet` | 37.725 dB | +6.71 dB | 直接训练 RawNet 低于本项目主线 |

最终隐藏 test 集没有提供 GT。本项目最终提交的隐藏 test PSNR 为 38.4 dB；它与上表本地 Val PSNR 不是同一评测口径，不能逐行等价比较。

从结果看，收益主要来自持续迭代后的任务适配，而不是简单替换为外部模型结构。直接使用第二名开源结构 RawNet 时，训练 recipe、数据处理、验证 crop/full-frame 口径、checkpoint 选择、TTA/TLC 推理细节和隐藏 test 分布都可能与原方案不一致，因此本地复现只有 37.725 dB，低于 `safnet_claude_33_v3` 的 39.726 dB。

## Repository Layout

```text
DataLoader/          RAW burst dataset, Bayer packing, augmentation, collate
models/              compact model registry and model implementations
utils/               checkpoint, metric, EMA, loss, and loader helpers
scripts/             training and submission helper scripts
scoring_program/     challenge-style PSNR/SSIM scorer
train.py             training entry point
eval.py              validation/test inference and result.zip packaging
reparam_model.py     structural re-parameterization helper
test_demo.py         lightweight dataset inspection utility
```

Generated artifacts such as checkpoints, logs, datasets, local outputs, TIFF submissions, and ZIP files are ignored by Git.

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

CUDA is recommended for training and full-frame evaluation.

## Data Layout

Each scene is stored as flat TIFF files:

```text
Scene-000-in-0.tif
Scene-000-in-1.tif
...
Scene-000-in-8.tif
Scene-000-gt.tif
```

Input RAW files are single-channel Bayer images with GRBG layout. The loader packs each RAW frame to 4 channels and maps one scene to a tensor shaped `(9, 4, H/2, W/2)`. GT files are read as 3-channel float tensors in `[0, 1]`.

Default local paths used by scripts:

```text
/home/chen/data/ntire2026/hdr/train/
/home/chen/data/ntire2026/hdr/validation/
/home/chen/data/ntire2026/hdr/test/
```

Override them with `TRAIN_ROOT`, `VAL_ROOT`, `TEST_ROOT`, or the corresponding CLI arguments.

## Training

Recommended mainline run:

```bash
bash scripts/run.sh --gpu 0 --num_workers 4 --val_every 1
```

Equivalent direct command:

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

Checkpoints are written under `checkpoint_dir/checkpoint_dir_<model>_<exp_name>/`. Training metrics are written under `output_log/`.

## Evaluation and Submission

Validation/test inference:

```bash
python eval.py \
  --model safnet_claude_33_v3 \
  --exp_name model_submit_claude33_v3 \
  --val_root /home/chen/data/ntire2026/hdr/validation/
```

Final submission-style inference with re-parameterization, TTA, and TLC:

```bash
bash scripts/test.sh safnet_claude_33_v3 model_submit_claude33_v3 128 128
```

Outputs are saved to:

```text
checkpoint_dir/checkpoint_dir_<model>_<exp_name>/img/
checkpoint_dir/checkpoint_dir_<model>_<exp_name>/result.zip
```

## Model Registry

Registered model names:

```text
unet
safnet_claude_33
safnet_claude_33_v2
safnet_claude_33_v3
safnet_claude_33_v4
rawnet
```

Add new models by implementing the module under `models/`, importing it in `models/__init__.py`, and adding it to `_MODEL_MAP`.

## Cleanup Notes

- Historical SAFNet Claude trial snapshots, challenge factsheet files, unused starting-kit figures, and the legacy `utils/custom_data_class.py` stub were removed.
- Local artifacts stay ignored: `checkpoint_dir/`, `output_log/`, `output/`, datasets, generated TIFF files, model weights, and submission ZIPs.
- Do not commit private keys, pretrained checkpoints, datasets, or generated submissions.
