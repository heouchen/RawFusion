"""Exponential Moving Average (EMA) for model weights."""
import torch
import torch.nn as nn
from copy import deepcopy


class ModelEMA:
    """Maintains exponential moving average of model parameters.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        # after each optimizer step:
        ema.update(model)
        # before validation:
        ema.apply(model)   # swap EMA weights into model
        # after validation:
        ema.restore(model) # restore original training weights
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        # Store EMA weights as plain dict (no grad, no graph)
        self.shadow = {}
        self.backup = {}
        self._init_shadow(model)

    def _init_shadow(self, model):
        for name, param in self._get_params(model):
            self.shadow[name] = param.data.clone()

    @staticmethod
    def _get_params(model):
        """Yield (name, param) handling DataParallel transparently."""
        m = model.module if isinstance(model, nn.DataParallel) else model
        return m.named_parameters()

    @torch.no_grad()
    def update(self, model):
        """Update EMA weights after an optimizer step."""
        for name, param in self._get_params(model):
            if name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply(self, model):
        """Swap EMA weights into model (backup originals for restore)."""
        self.backup = {}
        for name, param in self._get_params(model):
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original training weights after validation."""
        for name, param in self._get_params(model):
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
