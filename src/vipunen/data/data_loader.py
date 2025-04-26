"""
Data loader module for the Vipunen project.

This module provides functions to load data from files, with support for
different file formats and fallback to dummy data generation.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Union

from ..config.config_loader import get_config
from ..utils.file_utils_config import get_file_utils
from FileUtils.core.base import StorageError

logger = logging.getLogger(__name__)

def load_data(file_path: Optional[Union[str, Path]] = None, use_dummy: bool = False) -> pd.DataFrame:
    """
    Load data from a file or create dummy data if requested.
    
    Args:
        file_path: Path to the data file. If None, uses the path from config.
        use_dummy: Whether to use dummy data instead of loading from file
        
    Returns:
        pd.DataFrame: Loaded or generated data
    """
    config = get_config()
    
    if file_path is None:
        file_path = config['paths']['data']
    
    if use_dummy:
        logger.info("Using dummy dataset for demonstration purposes")
        from .dummy_generator import create_dummy_dataset
        return create_dummy_dataset()
    
    logger.info(f"Loading data from {file_path}")
    file_utils = get_file_utils()
    path_obj = Path(file_path)
    file_name = path_obj.name
    
    try:
        # Try with semicolon separator for CSV files
        if path_obj.suffix.lower() == '.csv':
            return file_utils.load_single_file(file_name, input_type="raw", sep=';')
        else:
            # For non-CSV files, let FileUtils auto-detect the format
            return file_utils.load_single_file(file_name, input_type="raw")
    except Exception as e:
        logger.warning(f"Failed to load with semicolon separator: {e}")
        # Try with comma separator if semicolon fails
        try:
            return file_utils.load_single_file(file_name, input_type="raw", sep=',')
        except (FileNotFoundError, StorageError) as e:
            logger.error(f"Could not find or load the data file at {file_path}: {e}")
            logger.info("Creating a dummy dataset for demonstration purposes")
            from .dummy_generator import create_dummy_dataset
            return create_dummy_dataset()

def ensure_data_directory(file_path: str) -> str:
    """
    Ensure the file path includes the data directory.
    If the path starts with 'raw/', prepend 'data/' to it.
    
    Args:
        file_path: Original file path
        
    Returns:
        str: Corrected file path
    """
    if file_path.startswith("raw/"):
        return f"data/{file_path}"
    return file_path 