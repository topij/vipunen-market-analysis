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
    Handles paths potentially containing subdirectories relative to the 'raw' data dir.

    Args:
        file_path: Path to the data file, relative to the project root 
                   (e.g., 'data/raw/subdir/my_file.csv'). 
                   If None, uses the path from config.
        use_dummy: Whether to use dummy data instead of loading from file

    Returns:
        pd.DataFrame: Loaded or generated data
    """
    config = get_config()

    if file_path is None:
        file_path = config['paths'].get('data') # Get default if needed
        if file_path is None:
            logger.error("Data file path not provided and not found in config.")
            return pd.DataFrame() # Return empty if no path

    if use_dummy:
        logger.info("Using dummy dataset for demonstration purposes")
        from .dummy_generator import create_dummy_dataset
        return create_dummy_dataset()

    # --- Only get FileUtils instance if we are actually loading data ---
    file_utils = get_file_utils()
    logger.info(f"Attempting to load data from {file_path}")
    
    path_obj = Path(file_path)
    
    # Determine the input_type and the relative file path for FileUtils
    # Assumes file_path is like 'data/raw/subdir/file.csv' or 'raw/subdir/file.csv'
    # FileUtils.load_single_file likely expects the relative path (e.g., subdir/file.csv)
    # as the first positional argument (file_path)
    input_type = "raw" # Defaulting to raw, adjust if other types are needed
    relative_file_path_str = None
    
    try:
        # Find the index of the input_type directory in the path parts
        parts = path_obj.parts
        type_index = -1
        # Find the last occurrence of input_type
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == input_type:
                type_index = i
                break

        if type_index != -1:
             # Join parts after the input_type directory to form the relative path
            relative_parts = parts[type_index + 1:]
            if not relative_parts:
                logger.error(f"Path '{file_path}' seems to end with the input type '{input_type}', cannot determine relative file path.")
                return pd.DataFrame()
            relative_file_path_str = str(Path(*relative_parts)) # e.g., 'subdir/file.csv' or 'file.csv'
        else:
            # If input_type is not in the path, assume the whole path is relative
            logger.warning(f"Input type '{input_type}' not found in path '{file_path}'. Assuming path is relative to '{input_type}'.")
            if not parts:
                 logger.error(f"Cannot process empty file path: '{file_path}'")
                 return pd.DataFrame()
            relative_file_path_str = str(path_obj) # Use the original path as relative

    except Exception as e:
         logger.error(f"Error processing path '{file_path}': {e}")
         return pd.DataFrame()

    if not relative_file_path_str:
        logger.error(f"Could not determine relative file path for FileUtils from '{file_path}'")
        return pd.DataFrame()
        
    logger.info(f"Calling FileUtils.load_single_file(file_path='{relative_file_path_str}', input_type='{input_type}')")

    try:
        # Reverted: Pass the relative path as the first argument (likely named file_path)
        # FileUtils handles formats. Pass specific pandas kwargs if needed (like sep).
        if path_obj.suffix.lower() == '.csv':
            # Try semicolon first for CSV
            try:
                logger.debug("Attempting load with sep=';'")
                return file_utils.load_single_file(relative_file_path_str, input_type=input_type, sep=';')
            except Exception as e_semi:
                logger.warning(f"Failed loading CSV with semicolon sep: {e_semi}. Trying comma.")
                # Fallback to comma for CSV
                try:
                    logger.debug("Attempting load with sep=',''")
                    return file_utils.load_single_file(relative_file_path_str, input_type=input_type, sep=',')
                except Exception as e_comma:
                     logger.warning(f"Failed loading CSV with comma sep: {e_comma}. Trying default FileUtils load.")
                     # Fallback to default load if comma also fails
                     return file_utils.load_single_file(relative_file_path_str, input_type=input_type)
        else:
            # For non-CSV files, let FileUtils auto-detect the format
            logger.debug(f"Attempting load with auto-detection for {path_obj.suffix}")
            return file_utils.load_single_file(relative_file_path_str, input_type=input_type)

    except StorageError as e:
        logger.error(f"FileUtils StorageError loading {file_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on load failure
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on load failure 