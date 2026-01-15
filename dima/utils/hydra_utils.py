from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import os

def setup_config(config_path):
    # Reset Hydra to avoid conflicts if already initialized
    GlobalHydra.instance().clear()
    
    # Ensure path is absolute for initialize_config_dir
    abs_config_dir = os.path.abspath(config_path)
    
    # Use initialize_config_dir instead of initialize
    # This allows absolute paths (essential for installed packages)
    initialize_config_dir(config_dir=abs_config_dir, version_base=None)
    
    # Load the configuration
    cfg = compose(config_name="config")
    return cfg