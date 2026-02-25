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
from models.SAFNet_Claude_11 import SAFNet_Claude_11
from models.SAFNet_Claude_12 import SAFNet_Claude_12
from models.SAFNet_Claude_13 import SAFNet_Claude_13
from models.SAFNet_Claude_14 import SAFNet_Claude_14
from models.SAFNet_Claude_15 import SAFNet_Claude_15
from models.SAFNet_Claude_16 import SAFNet_Claude_16
from models.SAFNet_Claude_17 import SAFNet_Claude_17
from models.SAFNet_Claude_18 import SAFNet_Claude_18
from models.SAFNet_Claude_19 import SAFNet_Claude_19
from models.SAFNet_Claude_20 import SAFNet_Claude_20
from models.SAFNet_Claude_21 import SAFNet_Claude_21
from models.SAFNet_Claude_22 import SAFNet_Claude_22
from models.SAFNet_Claude_23 import SAFNet_Claude_23
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
    'safnet_claude_11': SAFNet_Claude_11,
    'safnet_claude_12': SAFNet_Claude_12,
    'safnet_claude_13': SAFNet_Claude_13,
    'safnet_claude_14': SAFNet_Claude_14,
    'safnet_claude_15': SAFNet_Claude_15,
    'safnet_claude_16': SAFNet_Claude_16,
    'safnet_claude_17': SAFNet_Claude_17,
    'safnet_claude_18': SAFNet_Claude_18,
    'safnet_claude_19': SAFNet_Claude_19,
    'safnet_claude_20': SAFNet_Claude_20,
    'safnet_claude_21': SAFNet_Claude_21,
    'safnet_claude_22': SAFNet_Claude_22,
    'safnet_claude_23': SAFNet_Claude_23,
    'safnet_claude_5_pconv': SAFNet_Claude_5_pconv,
}

MODEL_NAMES = list(_MODEL_MAP.keys())


def build_model(name):
    """Build model by name. Raises ValueError if unknown."""
    key = name.lower()
    if key not in _MODEL_MAP:
        raise ValueError(f"Unknown model: {name}. Choose from: {MODEL_NAMES}")
    return _MODEL_MAP[key]()
