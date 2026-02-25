"""
We refer the code made from
https://github.com/z-bingo/kernel-prediction-networks-PyTorch/blob/master/train_eval_syn.py
"""

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

import numpy as np
import argparse

import os, time, shutil
from datetime import datetime

from torchvision.transforms import transforms

from DataLoader.custom_data_class import CustomDataset, HDRBurstAugment
from DataLoader.collate import make_train_collate, parse_crop_sizes
from models import build_model, MODEL_NAMES
from utils.checkpoint import save_checkpoint
from utils.utils import resume_or_load
from utils.loss import build_loss, LOSS_NAMES
from engine import train_one_epoch, validate


def train(
    num_threads,
    cuda,
    restart_train,
    mGPU,
    model_name='unet',
    exp_name='default',
    train_root="/home/chen/data/ntire2026/hdr/train/",
    val_root="/home/chen/data/ntire2026/hdr/validation/",
    n_epoch=300,
    batch_size=2,
    lr=2e-4,
    lr_decay=0.95,
    aug_enable=False,
    aug_crop_enable=True,
    aug_crop_size=256,
    aug_crop_sizes=None,
    aug_crop_even_offset=False,
    aug_geo_enable=False,
    aug_geo_flip_enable=True,
    aug_geo_rot90_enable=True,
    aug_exp_enable=False,
    aug_exp_low_min=0.9,
    aug_exp_low_max=1.1,
    aug_exp_mid_min=0.9,
    aug_exp_mid_max=1.1,
    aug_exp_high_min=0.9,
    aug_exp_high_max=1.1,
    aug_exp_global=False,
    aug_wb_enable=False,
    aug_wb_gain_delta=0.05,
    aug_noise_enable=False,
    aug_noise_std=0.0,
    pretrained_path=None,
    loss_name='mulaw_l1',
):
    torch.set_num_threads(num_threads)

    use_amp = cuda  # 混合精度训练，加速并减显存
    exp_name = (exp_name or "default").strip().replace(" ", "_")

    # checkpoint path（不同模型使用不同目录）
    checkpoint_dir = f'./checkpoint_dir/checkpoint_dir_{model_name}_{exp_name}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # output path
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 训练指标日志目录（每个 epoch 的 loss / PSNR / SSIM），文件名带训练开始时间戳
    output_log_dir = 'output_log'
    if not os.path.exists(output_log_dir):
        os.makedirs(output_log_dir)
    train_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_txt_path = os.path.join(output_log_dir, f'{train_timestamp}_train_log_{model_name}_{exp_name}.txt')
    log_header_written = False  # 续训时从 checkpoint 恢复
    print(f"=> Log file: {log_txt_path}")
    # logs path
    logs_dir = 'logs_dir'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    shutil.rmtree(logs_dir)

    # 训练集（可选增广）
    train_aug = HDRBurstAugment(
        enable=aug_enable,
        crop_enable=aug_crop_enable,
        crop_size=aug_crop_size,
        crop_sizes=None,  # 多尺度时由 collate_fn 统一选择
        crop_even_offset=aug_crop_even_offset,
        geo_enable=aug_geo_enable,
        geo_flip_enable=aug_geo_flip_enable,
        geo_rot90_enable=aug_geo_rot90_enable,
        exp_enable=aug_exp_enable,
        exp_range_low=(aug_exp_low_min, aug_exp_low_max),
        exp_range_mid=(aug_exp_mid_min, aug_exp_mid_max),
        exp_range_high=(aug_exp_high_min, aug_exp_high_max),
        exp_global=aug_exp_global,
        wb_enable=aug_wb_enable,
        wb_gain_delta=aug_wb_gain_delta,
        noise_enable=aug_noise_enable,
        noise_std=aug_noise_std,
        clamp=True,
    )
    use_batch_crop = bool(aug_enable and aug_crop_enable and aug_crop_sizes)
    use_batch_geom = bool(aug_enable and aug_geo_enable and aug_geo_rot90_enable)
    use_batch_collate = use_batch_crop or use_batch_geom
    train_set = CustomDataset(
        root_dir=train_root,
        transform=transforms.ToTensor(),
        train=True,
        augment=None if use_batch_collate else train_aug,
    )
    num_workers = 2
    data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cuda,
        persistent_workers=(num_workers > 0),
        collate_fn=make_train_collate(train_aug, aug_crop_sizes, batch_geom=use_batch_geom) if use_batch_collate else None,
    )
    print("Train loader length:", len(data_loader))

    # 验证集（每个 epoch 后计算 PSNR/SSIM）
    # 注意：验证集不做增广
    val_set = CustomDataset(
        root_dir=val_root,
        transform=transforms.ToTensor(),
        train=False,
        augment=None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=cuda,
        persistent_workers=True,
    )
    print("Val loader length:", len(val_loader))

    # 模型选择
    model = build_model(model_name)
    print(f'=> Using {model_name} model')

    # 损失函数
    loss_fn = build_loss(loss_name)
    print(f'=> Loss function: {loss_name}')

    print('\n-------Training started -------\n')

    if cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    # # torch.compile 加速训练（PyTorch 2.0+）
    # model = torch.compile(model)
    # print('=> torch.compile enabled')

    if mGPU:
        model = nn.DataParallel(model)
    model.train()


    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, global_step, best_psnr, log_txt_path, log_header_written = resume_or_load(
        model, optimizer, scheduler, scaler, use_amp,
        restart_train, checkpoint_dir, pretrained_path, log_txt_path
    )

    print(f"=> Experiment: {exp_name}")
    crop_desc = aug_crop_sizes if aug_crop_sizes else aug_crop_size
    print(
        f"=> Augment: enable={aug_enable}, crop={aug_crop_enable}({crop_desc}, even={aug_crop_even_offset}), "
        f"geo={aug_geo_enable}(flip={aug_geo_flip_enable}, rot90={aug_geo_rot90_enable}), "
        f"exp={aug_exp_enable}(low={aug_exp_low_min}-{aug_exp_low_max}, "
        f"mid={aug_exp_mid_min}-{aug_exp_mid_max}, high={aug_exp_high_min}-{aug_exp_high_max}, "
        f"global={aug_exp_global}), "
        f"wb={aug_wb_enable}(delta={aug_wb_gain_delta}), "
        f"noise={aug_noise_enable}(std={aug_noise_std})"
    )

    for epoch in range(start_epoch, n_epoch):
        epoch_start_time = time.time()
        lr_current = optimizer.param_groups[0]['lr']

        avg_train_loss, step_count = train_one_epoch(
            model, data_loader, optimizer, scaler, use_amp, cuda,
            epoch, n_epoch, output_dir, loss_fn=loss_fn
        )
        global_step += step_count
        epoch_time = time.time() - epoch_start_time

        # 验证集评估
        val_psnr, val_ssim = validate(model, val_loader, use_amp, cuda)

        # 打印 epoch 总结
        print(f'\n[Epoch {epoch:04d}] Time: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.5f} | '
              f'Val PSNR: {val_psnr:.3f} dB | Val SSIM: {val_ssim:.4f}\n')

        # 保存当前 epoch 的 loss / PSNR / SSIM 到 output_log（txt 格式，文件名带时间戳）
        with open(log_txt_path, 'a', encoding='utf-8') as f:
            if not log_header_written:
                f.write('epoch\tloss\tpsnr\tssim\ttime_s\tlr\n')
                log_header_written = True
            f.write(f'{epoch}\t{avg_train_loss:.6f}\t{val_psnr:.6f}\t{val_ssim:.6f}\t{epoch_time:.2f}\t{lr_current:.8f}\n')

        # 基于 val PSNR 判断 is_best（而非 train loss，避免增广实验偏置）
        if val_psnr > best_psnr:
            is_best = True
            best_psnr = val_psnr
        else:
            is_best = False

        if epoch % 5 == 0 or is_best:
            save_dict = {
                'epoch': epoch + 1,
                'global_iter': global_step,
                'state_dict': model.state_dict(),
                'best_psnr': best_psnr,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'log_path': log_txt_path,
            }
            if use_amp:
                save_dict['scaler'] = scaler.state_dict()
            save_checkpoint(
                save_dict, is_best, checkpoint_dir, global_step, max_keep=5
            )


        # decay the learning rate
        scheduler.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ulite', choices=MODEL_NAMES)
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--train_root', type=str, default="/home/chen/data/ntire2026/hdr/train/")
    parser.add_argument('--val_root', type=str, default="/home/chen/data/ntire2026/hdr/validation/")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--mgpu', type=int, default=1)
    parser.add_argument('--restart_train', type=int, default=1)
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint for transfer learning (partial weight loading)')
    parser.add_argument('--loss', type=str, default='mulaw_l1', choices=LOSS_NAMES,
                        help='Loss function name')

    # 数据增广（训练集用，验证集强制不用）
    parser.add_argument('--aug_enable', type=int, default=0)
    parser.add_argument('--aug_crop_enable', type=int, default=1)
    parser.add_argument('--aug_crop_size', type=int, default=256) # 减半
    parser.add_argument('--aug_crop_sizes', type=str, default="96x192,192x384,384x768") # 减半
    parser.add_argument('--aug_crop_even_offset', type=int, default=0) # Packing后无需强制偶数
    parser.add_argument('--aug_geo_enable', type=int, default=0)
    parser.add_argument('--aug_geo_flip_enable', type=int, default=1)
    parser.add_argument('--aug_geo_rot90_enable', type=int, default=1)
    parser.add_argument('--aug_exp_enable', type=int, default=0)
    parser.add_argument('--aug_exp_low_min', type=float, default=0.9)
    parser.add_argument('--aug_exp_low_max', type=float, default=1.1)
    parser.add_argument('--aug_exp_mid_min', type=float, default=0.9)
    parser.add_argument('--aug_exp_mid_max', type=float, default=1.1)
    parser.add_argument('--aug_exp_high_min', type=float, default=0.9)
    parser.add_argument('--aug_exp_high_max', type=float, default=1.1)
    parser.add_argument('--aug_exp_global', type=int, default=0)
    parser.add_argument('--aug_wb_enable', type=int, default=0)
    parser.add_argument('--aug_wb_gain_delta', type=float, default=0.05)
    parser.add_argument('--aug_noise_enable', type=int, default=0)
    parser.add_argument('--aug_noise_std', type=float, default=0.0)

    args = parser.parse_args()

    train(
        num_threads=args.num_threads,
        cuda=bool(args.cuda),
        restart_train=bool(args.restart_train),
        mGPU=bool(args.mgpu),
        model_name=args.model,
        exp_name=args.exp_name,
        train_root=args.train_root,
        val_root=args.val_root,
        n_epoch=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay=args.lr_decay,
        aug_enable=bool(args.aug_enable),
        aug_crop_enable=bool(args.aug_crop_enable),
        aug_crop_size=args.aug_crop_size,
        aug_crop_sizes=parse_crop_sizes(args.aug_crop_sizes),
        aug_crop_even_offset=bool(args.aug_crop_even_offset),
        aug_geo_enable=bool(args.aug_geo_enable),
        aug_geo_flip_enable=bool(args.aug_geo_flip_enable),
        aug_geo_rot90_enable=bool(args.aug_geo_rot90_enable),
        aug_exp_enable=bool(args.aug_exp_enable),
        aug_exp_low_min=args.aug_exp_low_min,
        aug_exp_low_max=args.aug_exp_low_max,
        aug_exp_mid_min=args.aug_exp_mid_min,
        aug_exp_mid_max=args.aug_exp_mid_max,
        aug_exp_high_min=args.aug_exp_high_min,
        aug_exp_high_max=args.aug_exp_high_max,
        aug_exp_global=bool(args.aug_exp_global),
        aug_wb_enable=bool(args.aug_wb_enable),
        aug_wb_gain_delta=args.aug_wb_gain_delta,
        aug_noise_enable=bool(args.aug_noise_enable),
        aug_noise_std=args.aug_noise_std,
        pretrained_path=args.pretrained,
        loss_name=args.loss,
    )
