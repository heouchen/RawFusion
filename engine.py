import torch
import torch.nn as nn
import time

from tqdm import tqdm
from PIL import Image
from torchvision.transforms import transforms

from utils.utils import calculate_psnr, calculate_ssim
to_pil_image = transforms.ToPILImage()


def train_one_epoch(model, data_loader, optimizer, scaler, use_amp, cuda,
                    epoch, n_epoch, output_dir, loss_fn=None, ema=None, consist_weight=0.1):
    """单 epoch 训练，返回 (avg_loss, step_count)"""

    model.train()
    lr_current = optimizer.param_groups[0]['lr']
    if hasattr(model, 'module'):
        base_model = model.module
    elif hasattr(model, '_orig_mod'):
        base_model = model._orig_mod
    else:
        base_model = model
    if hasattr(loss_fn, 'set_epoch'):
        loss_fn.set_epoch(epoch, n_epoch)
    if hasattr(base_model, 'set_spd_weight'):
        base_model.set_spd_weight(float(getattr(loss_fn, 'current_w_spd', 0.0)))

    # 训练循环（使用 tqdm 显示进度）
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}/{n_epoch} [LR={lr_current:.6f}]', ncols=100)
    epoch_loss = 0.0
    step_count = 0

    for step, batch_data in enumerate(pbar):
        if len(batch_data) == 4:
            burst_noise, gt, consist_inputs_list, consist_bboxes_list = batch_data
        else:
            burst_noise, gt = batch_data
            consist_inputs_list, consist_bboxes_list = None, None
            

        if cuda:
            burst_noise = burst_noise.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)

        # burst_noise: (B, 9, 4, H, W) -> (B, 36, H, W)
        b, f, c, h, w = burst_noise.shape
        burst_noise = burst_noise.view(b, f * c, h, w)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            pred = model(burst_noise)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = loss_fn(pred, gt)
            
            # Add multi-scale consistency loss if enabled
            if consist_inputs_list is not None and consist_bboxes_list is not None:
                # Randomly select one scale to compute consistency loss to save memory/compute
                num_scales = len(consist_inputs_list)
                if num_scales > 0:
                    scale_idx = torch.randint(0, num_scales, (1,)).item()
                    c_inputs = consist_inputs_list[scale_idx]
                    c_bboxes = consist_bboxes_list[scale_idx]
                    
                    if cuda:
                        c_inputs = c_inputs.cuda(non_blocking=True)
                    
                    # c_inputs: (B, 9, 4, ch, cw) -> (B, 36, ch, cw)
                    cb, cf, cc, ch, cw = c_inputs.shape
                    c_inputs = c_inputs.view(cb, cf * cc, ch, cw)
                    
                    pred_crop = model(c_inputs)
                    
                    # Calculate consistency loss per item in batch
                    consist_loss = 0.0
                    for b_idx in range(cb):
                        # Extract region from full prediction
                        # c_bbox is [y, x, ch, cw] relative to inputs (which are half res of target)
                        # pred_full is same res as target, so coordinates are * 2
                        by, bx, bch, bcw = c_bboxes[b_idx].tolist()
                        # Ignore 4 pixels boundary to alleviate padding effects
                        margin = 4
                        if bch * 2 > 2 * margin and bcw * 2 > 2 * margin:
                            roi_y_start = by * 2 + margin
                            roi_y_end = (by + bch) * 2 - margin
                            roi_x_start = bx * 2 + margin
                            roi_x_end = (bx + bcw) * 2 - margin
                            
                            pred_full_roi = pred[b_idx:b_idx+1, :, roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                            pred_crop_roi = pred_crop[b_idx:b_idx+1, :, margin:-margin, margin:-margin]
                        else:
                            pred_full_roi = pred[b_idx:b_idx+1, :, by*2:(by+bch)*2, bx*2:(bx+bcw)*2]
                            pred_crop_roi = pred_crop[b_idx:b_idx+1]
                        
                        consist_loss += torch.nn.functional.mse_loss(pred_crop_roi, pred_full_roi)
                    
                    consist_loss = consist_loss / cb
                    loss = loss + consist_weight * consist_loss

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

    # HARD CODE:
    # Training-time validation uses a fixed center crop so the validation input
    # matches the train crop size. Remove or parameterize this block if you want
    # full-image validation again.
    val_crop_h = 128
    val_crop_w = 128

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc='Validation', ncols=100, leave=False)
        for burst_noise, gt in val_pbar:
            if cuda:
                burst_noise = burst_noise.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)

            b, f, c, h, w = burst_noise.shape
            if h >= val_crop_h and w >= val_crop_w:
                crop_top = (h - val_crop_h) // 2
                crop_left = (w - val_crop_w) // 2
                burst_noise = burst_noise[
                    :,
                    :,
                    :,
                    crop_top:crop_top + val_crop_h,
                    crop_left:crop_left + val_crop_w,
                ]
                gt = gt[
                    :,
                    :,
                    crop_top * 2:(crop_top + val_crop_h) * 2,
                    crop_left * 2:(crop_left + val_crop_w) * 2,
                ]

            # burst_noise: (B, 9, 4, H, W) -> (B, 36, H, W)
            b, f, c, h, w = burst_noise.shape
            burst_noise = burst_noise.view(b, f * c, h, w)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(burst_noise)
                if isinstance(pred, tuple):
                    pred = pred[0]
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
