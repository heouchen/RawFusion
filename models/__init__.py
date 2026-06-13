from models.UNet import UNet
from models.SAFNet_Claude_33 import SAFNet_Claude_33
from models.SAFNet_Claude_33_v2 import SAFNet_Claude_33_v2
from models.SAFNet_Claude_33_v3 import SAFNet_Claude_33_v3
from models.SAFNet_Claude_33_v4 import SAFNet_Claude_33_v4
from models.rawnet import RawNet

_MODEL_MAP = {
    'unet': UNet,
    'safnet_claude_33': SAFNet_Claude_33,
    'safnet_claude_33_v2': SAFNet_Claude_33_v2,
    'safnet_claude_33_v3': SAFNet_Claude_33_v3,
    'safnet_claude_33_v4': SAFNet_Claude_33_v4,
    'rawnet': RawNet
}

MODEL_NAMES = list(_MODEL_MAP.keys())


def build_model(name):
    """Build model by name. Raises ValueError if unknown."""
    key = name.lower()
    if key not in _MODEL_MAP:
        raise ValueError(f"Unknown model: {name}. Choose from: {MODEL_NAMES}")
    return _MODEL_MAP[key]()


__all__ = ['MODEL_NAMES', 'build_model']
