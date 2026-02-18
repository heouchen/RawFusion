from models.SAFNet import SAFNet
from models.SAFNet_Opt_V2 import SAFNet_Opt_V2
from models.SAFNet_Opt_V4 import SAFNet_Opt_V4
from models.SAFNet_Claude_1 import SAFNet_Claude_1
from models.SAFNet_Claude_2 import SAFNet_Claude_2
from models.SAFNet_Claude_3 import SAFNet_Claude_3
from models.SAFNet_Claude_4 import SAFNet_Claude_4
from models.SAFNet_Claude_5 import SAFNet_Claude_5
from models.SAFNet_Claude_6 import SAFNet_Claude_6
from models.SAFNet_Claude_7 import SAFNet_Claude_7
from models.SAFNet_Claude_8 import SAFNet_Claude_8
from models.SAFNet_Claude_9 import SAFNet_Claude_9
from models.SAFNet_Claude_10 import SAFNet_Claude_10
from models.SAFNet_Claude_5_pconv import SAFNet_Claude_5_pconv

_MODEL_MAP = {
    'safnet': SAFNet,
    'safnet_opt_v2': SAFNet_Opt_V2,
    'safnet_opt_v4': SAFNet_Opt_V4,
    'safnet_claude_1': SAFNet_Claude_1,
    'safnet_claude_2': SAFNet_Claude_2,
    'safnet_claude_3': SAFNet_Claude_3,
    'safnet_claude_4': SAFNet_Claude_4,
    'safnet_claude_5': SAFNet_Claude_5,
    'safnet_claude_6': SAFNet_Claude_6,
    'safnet_claude_7': SAFNet_Claude_7,
    'safnet_claude_8': SAFNet_Claude_8,
    'safnet_claude_9': SAFNet_Claude_9,
    'safnet_claude_10': SAFNet_Claude_10,
    'safnet_claude_5_pconv': SAFNet_Claude_5_pconv,
}

MODEL_NAMES = list(_MODEL_MAP.keys())


def build_model(name):
    """Build model by name. Raises ValueError if unknown."""
    key = name.lower()
    if key not in _MODEL_MAP:
        raise ValueError(f"Unknown model: {name}. Choose from: {MODEL_NAMES}")
    return _MODEL_MAP[key]()
