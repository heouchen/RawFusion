import torch
import torch.nn as nn
import time

from tqdm import tqdm
from PIL import Image
from torchvision.transforms import transforms

from utils.utils import calculate_psnr, calculate_ssim
to_pil_image = transforms.ToPILImage()

def train_one_epoch(model, data_loader, optimizer, scaler, use_amp, cuda,
                    epoch, n_epoch, output_dir, loss_fn=None, ema=None):
    """单 epoch 训练，返回 (avg_loss, step_count)"""
    model.train()
    lr_current = optimizer.param_groups[0]['lr']
    base_model = model.module if hasattr(model, 'module') else model
    if hasattr(loss_fn, 'set_epoch'):
        loss_fn.set_epoch(epoch, n_epoch)
    if hasattr(base_model, 'set_spd_weight'):
        base_model.set_spd_weight(float(getattr(loss_fn, 'current_w_spd', 0.0)))

    # 训练循环（使用 tqdm 显示进度）
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}/{n_epoch} [LR={lr_current:.6f}]', ncols=100)
    epoch_loss = 0.0
    step_count = 0

    for step, (burst_noise, gt) in enumerate(pbar):
        if cuda:
            burst_noise = burst_noise.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)

        # burst_noise: (B, 9, 4, H, W) -> (B, 36, H, W)
        b, f, c, h, w = burst_noise.shape
        burst_noise = burst_noise.view(b, f * c, h, w)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            pred = model(burst_noise)
            loss = loss_fn(pred, gt)
            # Add auxiliary losses from model (e.g. flow regularization)
            if hasattr(base_model, '_aux_losses') and base_model._aux_losses:
                for v in base_model._aux_losses.values():
                    loss = loss + v
                base_model._aux_losses = {}
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        loss_val = loss.item()
        epoch_loss += loss_val

        # 更新 tqdm 显示的损失
        pbar.set_postfix({'loss': f'{loss_val:.4f}'})

        # 保存样例图片（仅在特定 epoch）
        if (epoch % 50 == 0) and (step < 5):
            with torch.no_grad():
                pil_image = to_pil_image(gt[0].cpu())
                pil_image.save(f'./{output_dir}/E{epoch}_Batch{step}_gt.png')
                pil_image = to_pil_image(pred[0].cpu().clamp(0, 1))
                pil_image.save(f'./{output_dir}/E{epoch}_Batch{step}_output.png')

        step_count += 1

    pbar.close()
    avg_loss = epoch_loss / len(data_loader)
    return avg_loss, step_count


def validate(model, val_loader, use_amp, cuda):
    """验证循环，返回 (psnr, ssim)。按样本数加权聚合，保证任意 batch_size 下与逐图平均同口径。"""
    model.eval()
    val_psnr_sum = 0.0
    val_ssim_sum = 0.0
    val_n_samples = 0

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc='Validation', ncols=100, leave=False)
        for burst_noise, gt in val_pbar:
            if cuda:
                burst_noise = burst_noise.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)

            # burst_noise: (B, 9, 4, H, W) -> (B, 36, H, W)
            b, f, c, h, w = burst_noise.shape
            burst_noise = burst_noise.view(b, f * c, h, w)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(burst_noise)
                pred = torch.clamp(pred, 0.0, 1.0)
            batch_psnr = calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
            batch_ssim = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
            val_psnr_sum += batch_psnr * b
            val_ssim_sum += batch_ssim * b
            val_n_samples += b
        val_pbar.close()

    val_psnr = val_psnr_sum / val_n_samples if val_n_samples > 0 else 0.0
    val_ssim = val_ssim_sum / val_n_samples if val_n_samples > 0 else 0.0
    return val_psnr, val_ssim
