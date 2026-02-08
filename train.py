"""
We refer the code made from  
https://github.com/z-bingo/kernel-prediction-networks-PyTorch/blob/master/train_eval_syn.py
"""



import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import argparse

import os, sys, time, shutil
from datetime import datetime
from tqdm import tqdm

from PIL import Image
from torchvision.transforms import transforms
to_pil_image = transforms.ToPILImage()

from DataLoader.custom_data_class import CustomDataset
from DataLoader.custom_data_class import HDRBurstAugment
from models.unet_model import UNet
from models.ULite import ULite
import pdb

from utils.utils import *
from utils.checkpoint import *

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
    aug_crop_size=512,
    aug_crop_even_offset=True,
    aug_noise_enable=False,
    aug_noise_std=0.0,
):
    torch.set_num_threads(num_threads)

    use_amp = cuda  # 混合精度训练，加速并减显存
    exp_name = (exp_name or "default").strip().replace(" ", "_")

    # checkpoint path（不同模型使用不同目录）
    checkpoint_dir = f'checkpoint_dir_{model_name}_{exp_name}'
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
        crop_even_offset=aug_crop_even_offset,
        noise_enable=aug_noise_enable,
        noise_std=aug_noise_std,
        clamp=True,
    )
    train_set = CustomDataset(
        root_dir=train_root,
        transform=transforms.ToTensor(),
        train=True,
        augment=train_aug,
    )
    num_workers = 2
    data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cuda,
        persistent_workers=(num_workers > 0),
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
    if model_name.lower() == 'ulite':
        model = ULite()
        print('=> Using ULite model')
    elif model_name.lower() == 'unet':
        model = UNet(
            in_channels=9,  # 9 frames considered as channel dimension
            n_classes=3,    # out channels (RGB)
            depth=4,
            wf=6,
            padding=True,
            batch_norm=False,
            up_mode='upconv'
        )
        print('=> Using UNet model')
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'unet' or 'ulite'")

    print('\n-------Training started -------\n')

    if cuda:
        model = model.cuda()

    if mGPU:
        model = nn.DataParallel(model)
    model.train()


    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    average_loss = MovingAverage(200)
    if not restart_train:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, 'best')
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if use_amp and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            if 'log_path' in checkpoint:
                log_txt_path = checkpoint['log_path']
                log_header_written = True  # 续训时已有表头，只追加
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            print('=> no checkpoint file to be loaded.')
    else:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        if os.path.exists(checkpoint_dir):
            pass
        else:
            os.mkdir(checkpoint_dir)
        print('=> training')

    MSE_loss = nn.MSELoss()

    print(f"=> Experiment: {exp_name}")
    print(f"=> Augment: enable={aug_enable}, crop={aug_crop_enable}({aug_crop_size}, even={aug_crop_even_offset}), "
          f"noise={aug_noise_enable}(std={aug_noise_std})")

    for epoch in range(start_epoch, n_epoch):
        model.train()
        epoch_start_time = time.time()
        lr_current = optimizer.param_groups[0]['lr']
        
        # 训练循环（使用 tqdm 显示进度）
        pbar = tqdm(data_loader, desc=f'Epoch {epoch}/{n_epoch} [LR={lr_current:.6f}]', ncols=100)
        epoch_loss = 0.0
        
        for step, (burst_noise, gt) in enumerate(pbar):
            if cuda:
                burst_noise = burst_noise.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
            burst_noise = burst_noise.squeeze(2)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(burst_noise)
                loss = MSE_loss(pred, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            average_loss.update(loss)
            epoch_loss += loss_val
            
            # 更新 tqdm 显示的损失
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})

            # 保存样例图片（仅在特定 epoch）
            if (epoch % 50 == 0) and (step < 5):
                with torch.no_grad():
                    for frame in range(9):
                        pil_image = to_pil_image(burst_noise[0][frame].cpu())
                        pil_image.save(f'./{output_dir}/Batch{step}_input{frame}.png')
                    pil_image = to_pil_image(gt[0].cpu())
                    pil_image.save(f'./{output_dir}/Batch{step}_gt.png')
                    pil_image = to_pil_image(pred[0].cpu())
                    pil_image.save(f'./{output_dir}/Batch{step}_output_E{epoch}.png')
            
            global_step += 1
        
        pbar.close()
        avg_train_loss = epoch_loss / len(data_loader)
        epoch_time = time.time() - epoch_start_time

        # 验证集评估
        model.eval()
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation', ncols=100, leave=False)
            for burst_noise, gt in val_pbar:
                if cuda:
                    burst_noise = burst_noise.cuda(non_blocking=True)
                    gt = gt.cuda(non_blocking=True)
                burst_noise = burst_noise.squeeze(2)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred = model(burst_noise)
                val_psnr_sum += calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
                val_ssim_sum += calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
                val_count += 1
            val_pbar.close()
        
        val_psnr = val_psnr_sum / val_count if val_count > 0 else 0.0
        val_ssim = val_ssim_sum / val_count if val_count > 0 else 0.0
        
        # 打印 epoch 总结
        print(f'\n[Epoch {epoch:04d}] Time: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.5f} | '
              f'Val PSNR: {val_psnr:.3f} dB | Val SSIM: {val_ssim:.4f}\n')

        # 保存当前 epoch 的 loss / PSNR / SSIM 到 output_log（txt 格式，文件名带时间戳）
        with open(log_txt_path, 'a', encoding='utf-8') as f:
            if not log_header_written:
                f.write('epoch\tloss\tpsnr\tssim\ttime_s\tlr\n')
                log_header_written = True
            f.write(f'{epoch}\t{avg_train_loss:.6f}\t{val_psnr:.6f}\t{val_ssim:.6f}\t{epoch_time:.2f}\t{lr_current:.8f}\n')

        if epoch % 5 == 0:
            if average_loss.get_value() < best_loss:
                is_best = True
                best_loss = average_loss.get_value()
            else:
                is_best = False

            save_dict = {
                'epoch': epoch,
                'global_iter': global_step,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
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
        lr_cur = [param['lr'] for param in optimizer.param_groups]
        if lr_cur[0] > 5e-6:
            scheduler.step()
        else:
            for param in optimizer.param_groups:
                param['lr'] = 5e-6



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ulite', choices=['unet', 'ulite'])
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

    # 数据增广（训练集用，验证集强制不用）
    parser.add_argument('--aug_enable', type=int, default=0)
    parser.add_argument('--aug_crop_enable', type=int, default=1)
    parser.add_argument('--aug_crop_size', type=int, default=512)
    parser.add_argument('--aug_crop_even_offset', type=int, default=1)
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
        aug_crop_even_offset=bool(args.aug_crop_even_offset),
        aug_noise_enable=bool(args.aug_noise_enable),
        aug_noise_std=args.aug_noise_std,
    )
