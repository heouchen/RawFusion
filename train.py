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
from DataLoader.collate import (
    CropScheduleController,
    make_train_collate,
    parse_crop_sizes,
    parse_progressive_batch_sizes,
    parse_progressive_crop_schedule,
)
from models import build_model, MODEL_NAMES
from utils.checkpoint import save_checkpoint
from utils.utils import resume_or_load
from utils.loss import build_loss, LOSS_NAMES
from utils.ema import ModelEMA
from engine import train_one_epoch, validate


def train(
    num_threads,
    cuda,
    restart_train,
    mGPU,
    num_workers=4,
    val_every=1,
    cudnn_benchmark=False,
    compile_model=False,
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
    aug_progressive_crop_enable=False,
    aug_progressive_crop_schedule=None,
    aug_progressive_batch_enable=False,
    aug_progressive_batch_sizes=None,
    aug_crop_even_offset=False,
    aug_geo_enable=False,
    aug_geo_flip_enable=True,
    aug_geo_rot90_enable=True,
    pretrained_path=None,
    loss_name='mulaw_l1',
    ema_enable=False,
    ema_decay=0.999,
    consist_enable=False,
    consist_sizes=None,
    consist_weight=0.1,
    val_num_workers=0,
):
    torch.set_num_threads(num_threads)
    if val_every <= 0:
        raise ValueError(f"val_every must be >= 1, got {val_every}")

    use_amp = cuda  # 混合精度训练，加速并减显存
    exp_name = (exp_name or "default").strip().replace(" ", "_")
    if cuda:
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

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
        clamp=True,
    )
    use_progressive_crop = bool(aug_enable and aug_crop_enable and aug_progressive_crop_enable and aug_progressive_crop_schedule)
    use_batch_crop = bool(aug_enable and aug_crop_enable and (aug_crop_sizes or use_progressive_crop))
    use_batch_geom = bool(aug_enable and aug_geo_enable and aug_geo_rot90_enable)
    use_batch_collate = use_batch_crop or use_batch_geom
    use_progressive_batch = bool(use_progressive_crop and aug_progressive_batch_enable)
    crop_controller = CropScheduleController(
        random_crop_sizes=aug_crop_sizes,
        progressive_enable=use_progressive_crop,
        progressive_schedule=aug_progressive_crop_schedule,
        default_batch_size=batch_size,
        progressive_batch_enable=use_progressive_batch,
        progressive_batch_sizes=aug_progressive_batch_sizes,
    )
    train_set = CustomDataset(
        root_dir=train_root,
        transform=transforms.ToTensor(),
        train=True,
        augment=None if use_batch_collate else train_aug,
        finalize_inputs=not (use_batch_collate or consist_enable),
    )
    #num_workers = num_threads

    def build_train_loader(curr_batch_size):
        loader_kwargs = dict(
            dataset=train_set,
            batch_size=curr_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=cuda,
            collate_fn=make_train_collate(
                train_aug,
                aug_crop_sizes,
                batch_geom=use_batch_geom,
                crop_controller=crop_controller,
                consist_enable=consist_enable,
                consist_sizes=consist_sizes,
            ) if use_batch_collate or consist_enable else None,
        )
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = True
        return torch.utils.data.DataLoader(
            **loader_kwargs,
        )

    # 验证集（每个 epoch 后计算 PSNR/SSIM）
    # 注意：验证集不做增广
    val_set = CustomDataset(
        root_dir=val_root,
        transform=transforms.ToTensor(),
        train=False,
        augment=None,
        finalize_inputs=True,
    )
    val_loader_kwargs = dict(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=cuda,
    )
    if val_num_workers > 0:
        val_loader_kwargs['persistent_workers'] = True
    val_loader = torch.utils.data.DataLoader(**val_loader_kwargs)
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

    if compile_model:
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
            print('=> torch.compile enabled')
        else:
            print('=> torch.compile requested but unavailable in this PyTorch build')

    if mGPU:
        model = nn.DataParallel(model)
    model.train()

    # EMA 权重平均
    ema = None
    if ema_enable:
        ema = ModelEMA(model, decay=ema_decay)
        print(f'=> EMA enabled (decay={ema_decay})')


    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epoch - 1, 1), eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, global_step, best_psnr, log_txt_path, log_header_written = resume_or_load(
        model, optimizer, scheduler, scaler, use_amp,
        restart_train, checkpoint_dir, pretrained_path, log_txt_path, ema=ema
    )

    print(f"=> Experiment: {exp_name}")
    crop_desc = aug_crop_sizes if aug_crop_sizes else aug_crop_size
    if aug_progressive_crop_enable and aug_progressive_crop_schedule:
        crop_desc = " -> ".join(
            f"{h}x{w}@{ratio:.2f}" for (h, w), ratio in aug_progressive_crop_schedule
        )
    batch_desc = str(batch_size)
    if use_progressive_batch and aug_progressive_batch_sizes:
        batch_desc = " -> ".join(
            f"{h}x{w}@{aug_progressive_batch_sizes.get((h, w), batch_size)}"
            for (h, w), _ in aug_progressive_crop_schedule
        )
    print(
        f"=> Augment: enable={aug_enable}, crop={aug_crop_enable}({crop_desc}, even={aug_crop_even_offset}), "
        f"geo={aug_geo_enable}(flip={aug_geo_flip_enable}, rot90={aug_geo_rot90_enable})"
    )
    print(f"=> Train batch: {batch_desc}")
    print(
        f"=> Runtime: num_workers={num_workers}, val_every={val_every}, "
        f"val_num_workers={val_num_workers}, "
        f"cudnn_benchmark={bool(cuda and cudnn_benchmark)}, compile={bool(compile_model)}, "
        f"mgpu={mGPU}"
    )

    crop_controller.set_epoch(start_epoch, n_epoch)
    current_batch_size = crop_controller.current_batch_size()
    data_loader = build_train_loader(current_batch_size)
    print(f"Train loader length: {len(data_loader)} (batch_size={current_batch_size})")

    for epoch in range(start_epoch, n_epoch):
        epoch_start_time = time.time()
        lr_current = optimizer.param_groups[0]['lr']
        crop_controller.set_epoch(epoch, n_epoch)
        crop_mode, crop_value = crop_controller.describe_mode()
        target_batch_size = crop_controller.current_batch_size()
        if target_batch_size != current_batch_size:
            current_batch_size = target_batch_size
            data_loader = build_train_loader(current_batch_size)
            print(f"=> Rebuilt train loader for batch_size={current_batch_size} at epoch {epoch:04d}")
        if aug_enable and aug_crop_enable:
            print(
                f"=> Epoch {epoch:04d} crop mode: {crop_mode} | crop: {crop_value} | "
                f"batch_size: {current_batch_size}"
            )

        avg_train_loss, step_count = train_one_epoch(
            model, data_loader, optimizer, scaler, use_amp, cuda,
            epoch, n_epoch, output_dir, loss_fn=loss_fn, ema=ema, consist_weight=consist_weight
        )
        global_step += step_count
        epoch_time = time.time() - epoch_start_time

        should_validate = ((epoch + 1) % val_every == 0) or (epoch == n_epoch - 1)
        if should_validate:
            # 验证集评估（使用 EMA 权重）
            if ema is not None:
                ema.apply(model)
            val_psnr, val_ssim = validate(model, val_loader, use_amp, cuda)
            if ema is not None:
                ema.restore(model)
            print(f'\n[Epoch {epoch:04d}] Time: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.5f} | '
                  f'Val PSNR: {val_psnr:.3f} dB | Val SSIM: {val_ssim:.4f}\n')
        else:
            val_psnr = float('nan')
            val_ssim = float('nan')
            print(f'\n[Epoch {epoch:04d}] Time: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.5f} | '
                  f'Val: skipped (val_every={val_every})\n')

        # 保存当前 epoch 的 loss / PSNR / SSIM 到 output_log（txt 格式，文件名带时间戳）
        with open(log_txt_path, 'a', encoding='utf-8') as f:
            if not log_header_written:
                f.write('epoch\tloss\tpsnr\tssim\ttime_s\tlr\tcrop_mode\tcrop_size\tbatch_size\n')
                log_header_written = True
            f.write(
                f'{epoch}\t{avg_train_loss:.6f}\t{val_psnr:.6f}\t{val_ssim:.6f}\t'
                f'{epoch_time:.2f}\t{lr_current:.8f}\t{crop_mode}\t{crop_value}\t{current_batch_size}\n'
            )

        # 基于 val PSNR 判断 is_best（而非 train loss，避免增广实验偏置）
        if should_validate and val_psnr > best_psnr:
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
            if ema is not None:
                save_dict['ema'] = ema.state_dict()
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
    parser.add_argument('--num_threads', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--mgpu', type=int, default=0)
    parser.add_argument('--val_every', type=int, default=1,
                        help='Run validation every N epochs')
    parser.add_argument('--cudnn_benchmark', type=int, default=0,
                        help='Enable torch.backends.cudnn.benchmark (0/1)')
    parser.add_argument('--compile', dest='compile_model', type=int, default=0,
                        help='Enable torch.compile when available (0/1)')
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
    parser.add_argument('--aug_progressive_crop_enable', type=int, default=0,
                        help='Enable epoch-wise progressive crop schedule (0/1)')
    parser.add_argument('--aug_progressive_crop_schedule', type=str,
                        default="96x192@0.3,192x384@0.7,384x768@1.0",
                        help='Progressive crop schedule as HxW@ratio,HxW@ratio,...')
    parser.add_argument('--aug_progressive_batch_enable', type=int, default=0,
                        help='Enable progressive batch sizes tied to progressive crop stages (0/1)')
    parser.add_argument('--aug_progressive_batch_sizes', type=str,
                        default="96x192@16,192x384@8,384x768@4",
                        help='Progressive batch sizes as HxW@batch,HxW@batch,...')
    parser.add_argument('--aug_crop_even_offset', type=int, default=0) # Packing后无需强制偶数
    parser.add_argument('--aug_geo_enable', type=int, default=0)
    parser.add_argument('--aug_geo_flip_enable', type=int, default=1)
    parser.add_argument('--aug_geo_rot90_enable', type=int, default=1)

    # EMA
    parser.add_argument('--ema', type=int, default=0, help='Enable EMA weight averaging (0/1)')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')

    # Consistency Constraint
    parser.add_argument('--consist_enable', type=int, default=0, help='Enable multi-scale consistency constraint')
    parser.add_argument('--consist_sizes', type=str, default="96x192,192x384", help='Crop sizes for consistency constraint')
    parser.add_argument('--consist_weight', type=float, default=0.1, help='Weight for consistency loss')
    parser.add_argument('--val_num_workers', type=int, default=0,
                        help='Validation dataloader workers; default 0 to avoid extra worker residency')

    args = parser.parse_args()

    train(
        num_threads=args.num_threads,
        num_workers=args.num_workers,
        cuda=bool(args.cuda),
        restart_train=bool(args.restart_train),
        mGPU=bool(args.mgpu),
        val_every=args.val_every,
        cudnn_benchmark=bool(args.cudnn_benchmark),
        compile_model=bool(args.compile_model),
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
        aug_progressive_crop_enable=bool(args.aug_progressive_crop_enable),
        aug_progressive_crop_schedule=parse_progressive_crop_schedule(args.aug_progressive_crop_schedule),
        aug_progressive_batch_enable=bool(args.aug_progressive_batch_enable),
        aug_progressive_batch_sizes=parse_progressive_batch_sizes(args.aug_progressive_batch_sizes),
        aug_crop_even_offset=bool(args.aug_crop_even_offset),
        aug_geo_enable=bool(args.aug_geo_enable),
        aug_geo_flip_enable=bool(args.aug_geo_flip_enable),
        aug_geo_rot90_enable=bool(args.aug_geo_rot90_enable),
        pretrained_path=args.pretrained,
        loss_name=args.loss,
        ema_enable=bool(args.ema),
        ema_decay=args.ema_decay,
        consist_enable=bool(args.consist_enable),
        consist_sizes=parse_crop_sizes(args.consist_sizes),
        consist_weight=args.consist_weight,
        val_num_workers=args.val_num_workers,
    )
