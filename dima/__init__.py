from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

CONFIG_PATH = PACKAGE_ROOT / "configs"

def get_config_path():
    """Returns the absolute path to the config directory as a string."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config directory not found at {CONFIG_PATH}")
    return str(CONFIG_PATH)