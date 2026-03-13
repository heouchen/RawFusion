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
    return torch.load(checkpoint_file, map_location='cpu', weights_only=False)


def _strip_state_dict_prefixes(state_dict, prefixes):
    """Remove known prefixes (e.g., module./_orig_mod.) from every key in the state dict."""
    if not prefixes:
        return state_dict

    def _strip(key):
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        return key

    return {_strip(k): v for k, v in state_dict.items()}


def _unwrap_model(model):
    """Peel off wrappers like DataParallel or torch.compile OptimizedModule."""
    base = model
    while True:
        if isinstance(base, nn.DataParallel):
            base = base.module
            continue
        if hasattr(base, '_orig_mod'):
            base = base._orig_mod
            continue
        break
    return base


def load_model_state_dict(model, state_dict):
    """加载 state_dict，自动处理 DataParallel 的 module. 前缀不匹配问题"""
    base_model = _unwrap_model(model)
    cleaned_sd = _strip_state_dict_prefixes(state_dict, ('module.', '_orig_mod.'))
    base_model.load_state_dict(cleaned_sd)


def load_pretrained_weights(model, path):
    """从预训练模型加载权重（支持跨模型迁移，只加载匹配的key）"""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    pretrained_sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    # 去除常见包装产生的前缀
    cleaned_sd = _strip_state_dict_prefixes(pretrained_sd, ('module.', '_orig_mod.'))
    # 获取当前模型（去除 DataParallel/compile 包装）
    base_model = _unwrap_model(model)
    current_sd = base_model.state_dict()
    # 只加载 shape 匹配的 key
    loaded_keys = []
    skipped_keys = []
    for k, v in cleaned_sd.items():
        if k in current_sd and current_sd[k].shape == v.shape:
            current_sd[k] = v
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)
    base_model.load_state_dict(current_sd)
    print(f'=> loaded pretrained weights from {path}')
    print(f'   loaded {len(loaded_keys)} keys, skipped {len(skipped_keys)} keys')
    if skipped_keys:
        print(f'   skipped keys (first 10): {skipped_keys[:10]}')
