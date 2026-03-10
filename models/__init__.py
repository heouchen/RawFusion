from models.UNet import UNet
from models.SAFNet_Claude_27 import SAFNet_Claude_27
from models.SAFNet_Claude_29 import SAFNet_Claude_29
from models.SAFNet_Claude_30 import SAFNet_Claude_30
from models.SAFNet_Claude_31 import SAFNet_Claude_31
from models.SAFNet_Claude_32 import SAFNet_Claude_32
from models.SAFNet_Claude_33 import SAFNet_Claude_33
from models.SAFNet_Claude_34 import SAFNet_Claude_34
from models.SAFNet_Claude_35 import SAFNet_Claude_35
from models.SAFNet_Claude_35_v2 import SAFNet_Claude_35_v2
from models.SAFNet_Claude_36 import SAFNet_Claude_36
from models.SAFNet_Claude_37 import SAFNet_Claude_37
from models.SAFNet_Claude_38 import SAFNet_Claude_38
from models.SAFNet_Claude_39 import SAFNet_Claude_39
from models.SAFNet_Claude_40 import SAFNet_Claude_40
from models.SAFNet_Claude_41 import SAFNet_Claude_41
from models.SAFNet_Claude_42 import SAFNet_Claude_42
from models.SAFNet_Claude_43 import SAFNet_Claude_43
from models.SAFNet_Claude_44 import SAFNet_Claude_44
from models.SAFNet_Claude_45 import SAFNet_Claude_45
from models.SAFNet_Claude_46 import SAFNet_Claude_46
from models.SAFNet_Claude_47 import SAFNet_Claude_47
from models.SAFNet_Claude_48 import SAFNet_Claude_48
from models.SAFNet_Claude_49 import SAFNet_Claude_49
from models.SAFNet_Claude_50 import SAFNet_Claude_50
from models.SAFNet_Claude_50_v2 import SAFNet_Claude_50_v2
from models.SAFNet_Claude_51 import SAFNet_Claude_51
from models.SAFNet_Claude_52 import SAFNet_Claude_52
from models.SAFNet_Claude_53 import SAFNet_Claude_53
from models.SAFNet_Claude_54 import SAFNet_Claude_54

_MODEL_MAP = {
    'unet': UNet,
    'safnet_claude_27': SAFNet_Claude_27,
    'safnet_claude_29': SAFNet_Claude_29,
    'safnet_claude_30': SAFNet_Claude_30,
    'safnet_claude_31': SAFNet_Claude_31,
    'safnet_claude_32': SAFNet_Claude_32,
    'safnet_claude_33': SAFNet_Claude_33,
    'safnet_claude_34': SAFNet_Claude_34,
    'safnet_claude_35': SAFNet_Claude_35,
    'safnet_claude_35_v2': SAFNet_Claude_35_v2,
    'safnet_claude_36': SAFNet_Claude_36,
    'safnet_claude_37': SAFNet_Claude_37,
    'safnet_claude_38': SAFNet_Claude_38,
    'safnet_claude_39': SAFNet_Claude_39,
    'safnet_claude_40': SAFNet_Claude_40,
    'safnet_claude_41': SAFNet_Claude_41,
    'safnet_claude_42': SAFNet_Claude_42,
    'safnet_claude_43': SAFNet_Claude_43,
    'safnet_claude_44': SAFNet_Claude_44,
    'safnet_claude_45': SAFNet_Claude_45,
    'safnet_claude_46': SAFNet_Claude_46,
    'safnet_claude_47': SAFNet_Claude_47,
    'safnet_claude_48': SAFNet_Claude_48,
    'safnet_claude_49': SAFNet_Claude_49,
    'safnet_claude_50': SAFNet_Claude_50,
    'safnet_claude_50_v2': SAFNet_Claude_50_v2,
    'safnet_claude_51': SAFNet_Claude_51,
    'safnet_claude_52': SAFNet_Claude_52,
    'safnet_claude_53': SAFNet_Claude_53,
    'safnet_claude_54': SAFNet_Claude_54,
}

MODEL_NAMES = list(_MODEL_MAP.keys())


def build_model(name):
    """Build model by name. Raises ValueError if unknown."""
    key = name.lower()
    if key not in _MODEL_MAP:
        raise ValueError(f"Unknown model: {name}. Choose from: {MODEL_NAMES}")
    return _MODEL_MAP[key]()
