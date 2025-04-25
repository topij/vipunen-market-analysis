"""
Data loading utilities for education market analysis.
"""

import pandas as pd
import os
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_data_directory(file_path: Union[str, Path]) -> str:
    """
    Ensure the file path includes the data directory.
    If the path starts with 'raw/', prepend 'data/' to it.
    
    Args:
        file_path: Original file path
        
    Returns:
        str: Updated file path
    """
    if isinstance(file_path, str) and file_path.startswith("raw/"):
        return f"data/{file_path}"
    return str(file_path)

def load_data(file_path: Union[str, Path], shorten_names: bool = False) -> pd.DataFrame:
    """
    Load data from a CSV file with appropriate error handling.
    
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
        # Ensure the file path includes the data directory if needed
        file_path = ensure_data_directory(file_path)
        
        logger.info(f"Loading data from {file_path}")
        
        # Handle relative paths
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)
        
        # Load the data
        if file_path.endswith(".csv"):
            try:
                df = pd.read_csv(file_path, sep=';')
            except Exception as e:
                # Try with comma separator if semicolon fails
                df = pd.read_csv(file_path, sep=',')
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
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
    # Create a directory name based on the institution name
    dir_name = f"education_market_{institution_name.lower()}"
    
    # Set up the path
    base_dir = Path("data/reports")
    output_dir = base_dir / dir_name
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory at {output_dir}")
    
    return output_dir 