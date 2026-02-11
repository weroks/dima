from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_ROOT / "configs"
DATA_PATH = PACKAGE_ROOT


def get_config_path() -> str:
    """Returns the absolute path to the config directory as a string."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config directory not found at {CONFIG_PATH}")
    return str(CONFIG_PATH)


def get_data_path() -> Path:
    """Returns the package root where checkpoints/statistics are resolved from."""
    return DATA_PATH


def get_stats_path(encoder_name: str = "CHEAP_shorten_1_dim_1024") -> Path:
    """Returns the absolute path to the normalization statistics for a given encoder."""
    from utils.pretrained_utils import PRETRAINED_MODELS_PATHS

    if encoder_name not in PRETRAINED_MODELS_PATHS:
        raise ValueError(f"Unknown encoder: {encoder_name}. Available: {list(PRETRAINED_MODELS_PATHS.keys())}")
    return DATA_PATH / PRETRAINED_MODELS_PATHS[encoder_name]["stats"]