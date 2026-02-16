import torch
import numpy as np
from skimage import metrics

from skimage.color import rgb2gray

import os
import cv2

from utils.checkpoint import load_checkpoint, load_model_state_dict, load_pretrained_weights
class MovingAverage(object):
    def __init__(self, n):
        self.n = n
        self._cache = []
        self.mean = 0

    def update(self, val):
        self._cache.append(val)
        if len(self._cache) > self.n:
            del self._cache[0]
        self.mean = sum(self._cache) / len(self._cache)

    def get_value(self):
        return self.mean

def torch2numpy(tensor, gamma=None):
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # Convert to 0 - 255
    if gamma is not None:
        tensor = torch.pow(tensor, gamma)
    tensor *= 255.0
    tensor=torch.clamp(tensor,0,255).to(torch.uint8)
    tensor = tensor.squeeze(1)
    return tensor.permute(0, 2, 3, 1).cpu().data.numpy()

def calculate_psnr(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    psnr = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        psnr += metrics.peak_signal_noise_ratio(target_tf[im_idx, ...],
                                             output_tf[im_idx, ...])
   
        n += 1.0
    return psnr / n

def calculate_ssim(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    ssim = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        ssim += metrics.structural_similarity(
            rgb2gray(cv2.cvtColor(target_tf[im_idx, ...], cv2.COLOR_BGR2RGB)),
            rgb2gray(cv2.cvtColor(output_tf[im_idx, ...], cv2.COLOR_BGR2RGB)),
            data_range=1.0)
        n += 1.0
    return ssim / n


def resume_or_load(model, optimizer, scheduler, scaler, use_amp,
                   restart_train, checkpoint_dir, pretrained_path,
                   log_txt_path):
    """
    模型加载与重训练逻辑。
    根据 restart_train / pretrained_path 决定：
      - 续训：从 checkpoint 恢复 model/optimizer/scheduler/scaler 状态
      - 预训练迁移：部分加载权重
      - 从零开始训练

    Returns:
        (start_epoch, global_step, best_loss, log_txt_path, log_header_written)
    """
    if not restart_train:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, 'best')
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            load_model_state_dict(model, checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if use_amp and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            log_header_written = False
            if 'log_path' in checkpoint:
                log_txt_path = checkpoint['log_path']
                log_header_written = True  # 续训时已有表头，只追加
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
            return start_epoch, global_step, best_loss, log_txt_path, log_header_written
        except Exception as e:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            print(f'=> no checkpoint file to be loaded. Reason: {e}')
            return start_epoch, global_step, best_loss, log_txt_path, False
    elif pretrained_path:
        # 从预训练模型加载权重（支持跨模型迁移，只加载匹配的key）
        try:
            load_pretrained_weights(model, pretrained_path)
        except Exception as e:
            print(f'=> failed to load pretrained weights: {e}')
        return 0, 0, np.inf, log_txt_path, False
    else:
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        print('=> training from scratch')
        return 0, 0, np.inf, log_txt_path, False
