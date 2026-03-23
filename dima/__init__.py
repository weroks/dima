import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_ROOT / "configs"
DATA_PATH = PACKAGE_ROOT

_S3_BUCKET = os.environ.get("DIMA_S3_BUCKET", "dima-protein-diffusion")
_S3_REGION = os.environ.get("DIMA_S3_REGION", "eu-north-1")


def _get_cache_dir() -> Path:
    """Returns the cache directory: DIMA_CACHE_DIR env var, or falls back to package root."""
    cache_dir = os.environ.get("DIMA_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return DATA_PATH


def get_config_path() -> str:
    """Returns the absolute path to the config directory as a string."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config directory not found at {CONFIG_PATH}")
    return str(CONFIG_PATH)


def get_data_path() -> Path:
    """Returns the package root where checkpoints/statistics are resolved from."""
    return _get_cache_dir()


def get_stats_path(encoder_name: str = "CHEAP_shorten_1_dim_1024") -> Path:
    """Returns the path to normalization statistics, downloading from S3 if needed."""
    from dima.utils.pretrained_utils import PRETRAINED_MODELS_PATHS

    if encoder_name not in PRETRAINED_MODELS_PATHS:
        raise ValueError(f"Unknown encoder: {encoder_name}. Available: {list(PRETRAINED_MODELS_PATHS.keys())}")

    relative_path = PRETRAINED_MODELS_PATHS[encoder_name]["stats"]
    local_path = _get_cache_dir() / relative_path

    if not local_path.exists():
        _ensure_downloaded(relative_path, local_path)

    return local_path


def _ensure_downloaded(relative_path: str, local_path: Path):
    """Downloads a file from S3."""
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if _S3_REGION and _S3_REGION != "us-east-1":
        url = f"https://{_S3_BUCKET}.s3.{_S3_REGION}.amazonaws.com/{relative_path}"
    else:
        url = f"https://{_S3_BUCKET}.s3.amazonaws.com/{relative_path}"

    import requests
    from tqdm import tqdm

    print(f"Downloading {url} to {local_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(local_path, "wb") as f, tqdm(total=total_size, unit="iB", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))