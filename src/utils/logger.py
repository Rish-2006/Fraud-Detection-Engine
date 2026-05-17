# src/utils/logger.py
import logging
import os
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger.
    Every module calls this instead of using print().
    
    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info('Model training started')
    """
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # avoid duplicate handlers
        logger.setLevel(logging.INFO)
        
        # Format: timestamp | level | module | message
        fmt = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Console handler (prints to terminal)
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
        
        # File handler (writes to logs/app.log)
        file_h = logging.FileHandler('logs/app.log')
        file_h.setFormatter(fmt)
        logger.addHandler(file_h)
    
    return logger