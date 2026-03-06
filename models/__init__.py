from models.SAFNet_Claude_27 import SAFNet_Claude_27
from models.SAFNet_Claude_29 import SAFNet_Claude_29
from models.SAFNet_Claude_30 import SAFNet_Claude_30
from models.SAFNet_Claude_31 import SAFNet_Claude_31
from models.SAFNet_Claude_32 import SAFNet_Claude_32
from models.SAFNet_Claude_33 import SAFNet_Claude_33
from models.SAFNet_Claude_34 import SAFNet_Claude_34
from models.SAFNet_Claude_35 import SAFNet_Claude_35
from models.SAFNet_Claude_36 import SAFNet_Claude_36
from models.SAFNet_Claude_37 import SAFNet_Claude_37
from models.SAFNet_Claude_38 import SAFNet_Claude_38
from models.SAFNet_Claude_39 import SAFNet_Claude_39
from models.SAFNet_Claude_40 import SAFNet_Claude_40

_MODEL_MAP = {
    'safnet_claude_27': SAFNet_Claude_27,
    'safnet_claude_29': SAFNet_Claude_29,
    'safnet_claude_30': SAFNet_Claude_30,
    'safnet_claude_31': SAFNet_Claude_31,
    'safnet_claude_32': SAFNet_Claude_32,
    'safnet_claude_33': SAFNet_Claude_33,
    'safnet_claude_34': SAFNet_Claude_34,
    'safnet_claude_35': SAFNet_Claude_35,
    'safnet_claude_36': SAFNet_Claude_36,
    'safnet_claude_37': SAFNet_Claude_37,
    'safnet_claude_38': SAFNet_Claude_38,
    'safnet_claude_39': SAFNet_Claude_39,
    'safnet_claude_40': SAFNet_Claude_40,
}

MODEL_NAMES = list(_MODEL_MAP.keys())


def build_model(name):
    """Build model by name. Raises ValueError if unknown."""
    key = name.lower()
    if key not in _MODEL_MAP:
        raise ValueError(f"Unknown model: {name}. Choose from: {MODEL_NAMES}")
    return _MODEL_MAP[key]()
