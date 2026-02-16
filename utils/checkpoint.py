import torch
import torch.nn as nn

import os, shutil
import glob
import numbers


def _represent_int(s):
    """判断字符串是否可转为整数"""
    try:
        int(s)
        return True
    except ValueError:
        return False


def save_checkpoint(state, is_best, checkpoint_dir, n_iter, max_keep=10):
    filename = os.path.join(checkpoint_dir, "{:06d}.pth.tar".format(n_iter))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        os.path.join(checkpoint_dir,
                                     'model_best.pth.tar'))
    files = sorted(os.listdir(checkpoint_dir))
    rm_files = files[0:max(0, len(files) - max_keep)]
    for f in rm_files:
        os.remove(os.path.join(checkpoint_dir, f))

def load_checkpoint(checkpoint_dir, best_or_latest='best'):
    if best_or_latest == 'best':
        checkpoint_file = os.path.join(checkpoint_dir, 'model_best.pth.tar')
    elif isinstance(best_or_latest, numbers.Number):
        checkpoint_file = os.path.join(checkpoint_dir,
                                       '{:06d}.pth.tar'.format(best_or_latest))
        if not os.path.exists(checkpoint_file):
            files = glob.glob(os.path.join(checkpoint_dir, '*.pth.tar'))
            basenames = [os.path.basename(f).split('.')[0] for f in files]
            iters = sorted([int(b) for b in basenames if _represent_int(b)])
            raise ValueError('Available iterations are ({} requested): {}'.format(best_or_latest, iters))
    else:
        files = glob.glob(os.path.join(checkpoint_dir, '*.pth.tar'))
        basenames = [os.path.basename(f).split('.')[0] for f in files]
        iters = sorted([int(b) for b in basenames if _represent_int(b)])
        checkpoint_file = os.path.join(checkpoint_dir,
                                       '{:06d}.pth.tar'.format(iters[-1]))
    return torch.load(checkpoint_file)


def load_model_state_dict(model, state_dict):
    """加载 state_dict，自动处理 DataParallel 的 module. 前缀不匹配问题"""
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # checkpoint 有 module. 前缀但当前模型没有，或反过来
        is_parallel = isinstance(model, nn.DataParallel)
        ckpt_has_module = any(k.startswith('module.') for k in state_dict.keys())
        if is_parallel and not ckpt_has_module:
            # 模型是 DataParallel 但 checkpoint 没有 module. 前缀
            new_sd = {'module.' + k: v for k, v in state_dict.items()}
            model.load_state_dict(new_sd)
        elif not is_parallel and ckpt_has_module:
            # 模型不是 DataParallel 但 checkpoint 有 module. 前缀
            new_sd = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            model.load_state_dict(new_sd)
        else:
            raise


def load_pretrained_weights(model, path):
    """从预训练模型加载权重（支持跨模型迁移，只加载匹配的key）"""
    ckpt = torch.load(path, map_location='cpu')
    pretrained_sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    # 去除 module. 前缀
    cleaned_sd = {}
    for k, v in pretrained_sd.items():
        new_k = k.replace('module.', '', 1) if k.startswith('module.') else k
        cleaned_sd[new_k] = v
    # 获取当前模型的 state_dict（去 module. 前缀）
    is_parallel = isinstance(model, nn.DataParallel)
    current_sd = model.module.state_dict() if is_parallel else model.state_dict()
    # 只加载 shape 匹配的 key
    loaded_keys = []
    skipped_keys = []
    for k, v in cleaned_sd.items():
        if k in current_sd and current_sd[k].shape == v.shape:
            current_sd[k] = v
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)
    if is_parallel:
        model.module.load_state_dict(current_sd)
    else:
        model.load_state_dict(current_sd)
    print(f'=> loaded pretrained weights from {path}')
    print(f'   loaded {len(loaded_keys)} keys, skipped {len(skipped_keys)} keys')
    if skipped_keys:
        print(f'   skipped keys (first 10): {skipped_keys[:10]}')
