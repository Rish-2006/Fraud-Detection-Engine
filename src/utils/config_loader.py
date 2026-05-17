# src/utils/config_loader.py
import yaml
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load YAML configuration file.
    Returns dict of all project settings.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f'Config loaded from {config_path}')
    return config

# Test it:
if __name__ == '__main__':
    cfg = load_config()
    print('Project:', cfg['project']['name'])
    print('Features:', cfg['data']['features'])