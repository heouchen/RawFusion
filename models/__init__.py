from models.SAFNet_Claude_27_v2 import SAFNet_Claude_27_v2
from models.SAFNet_Claude_29 import SAFNet_Claude_29
from models.SAFNet_Claude_30 import SAFNet_Claude_30
from models.SAFNet_Claude_31 import SAFNet_Claude_31
from models.SAFNet_Claude_32 import SAFNet_Claude_32
from models.SAFNet_Claude_33 import SAFNet_Claude_33

_MODEL_MAP = {
    'safnet_claude_27_v2': SAFNet_Claude_27_v2,
    'safnet_claude_29': SAFNet_Claude_29,
    'safnet_claude_30': SAFNet_Claude_30,
    'safnet_claude_31': SAFNet_Claude_31,
    'safnet_claude_32': SAFNet_Claude_32,
    'safnet_claude_33': SAFNet_Claude_33,
}

MODEL_NAMES = list(_MODEL_MAP.keys())


def build_model(name):
    """Build model by name. Raises ValueError if unknown."""
    key = name.lower()
    if key not in _MODEL_MAP:
        raise ValueError(f"Unknown model: {name}. Choose from: {MODEL_NAMES}")
    return _MODEL_MAP[key]()
