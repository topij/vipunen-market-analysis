"""
Data loading utilities for education market analysis.
"""

import pandas as pd
import os
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from enum import Enum

from src.vipunen.utils.file_handler import VipunenFileHandler
from FileUtils import OutputFileType

# Define InputFileType enum (not available in FileUtils)
class InputFileType(Enum):
    """Supported input file types."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"
    YAML = "yaml"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the file handler
file_handler = VipunenFileHandler()

def ensure_data_directory(file_path: Union[str, Path]) -> str:
    """
    Ensure the file path includes the data directory.
    If the path starts with 'raw/' and doesn't already have 'data/' prefix, prepend 'data/' to it.
    
    Args:
        file_path: Original file path
        
    Returns:
        str: Updated file path
    """
    if isinstance(file_path, str) and file_path.startswith("raw/") and not file_path.startswith("data/"):
        return f"data/{file_path}"
    return str(file_path)

def load_data(file_path: Union[str, Path], shorten_names: bool = False) -> pd.DataFrame:
    """
    Load data from a file with appropriate error handling.
    
    Args:
        file_path: Path to the data file
        shorten_names: Whether to shorten qualification names
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If the file does not exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file cannot be parsed
    """
    try:
        logger.info(f"Loading data from {file_path}")
        
        # Get just the filename without directory structure
        path_obj = Path(file_path)
        file_name = path_obj.name
        
        # Determine file type based on extension
        file_suffix = path_obj.suffix.lower()
        
        # Use the file handler to load the data
        # For CSV files, try both separators
        if file_suffix == ".csv":
            try:
                return file_handler.load_data(
                    file_name, 
                    input_type="raw",
                    sep=';'
                )
            except Exception:
                # Try with comma separator if semicolon fails
                return file_handler.load_data(
                    file_name, 
                    input_type="raw",
                    sep=','
                )
        elif file_suffix in [".xlsx", ".xls"]:
            return file_handler.load_data(
                file_name, 
                input_type="raw"
            )
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Error loading data file: File not found: {file_path}")
    except pd.errors.EmptyDataError:
        logger.error(f"Empty file: {file_path}")
        raise pd.errors.EmptyDataError(f"Error loading data file: Empty file: {file_path}")
    except pd.errors.ParserError as e:
        logger.error(f"Parser error for {file_path}: {e}")
        raise pd.errors.ParserError(f"Error parsing data file: {e}")
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise Exception(f"Error loading data file: {e}")

def create_output_directory(institution_name: str) -> Path:
    """
    Create a directory for output files.
    
    Args:
        institution_name: Short name for the institution
        
    Returns:
        Path: Path to the created directory
    """
    # Use the file handler to create the directory
    return file_handler.create_output_directory(institution_name, base_dir="reports") 