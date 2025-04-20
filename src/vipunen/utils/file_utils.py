"""File handling utilities for the Vipunen project."""
from pathlib import Path
import pandas as pd
from typing import Union, Optional
import logging
import shutil
from vipunen.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

def load_data(file_path: Union[str, Path], directory: Optional[Path] = None) -> pd.DataFrame:
    """Load data from a file (CSV or Excel).
    
    Args:
        file_path: The path to the file to load
        directory: Optional directory where the file is located
        
    Returns:
        The loaded data as a pandas DataFrame
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file type is not supported or the file is empty
    """
    if directory:
        full_file_path = directory / file_path
    else:
        full_file_path = Path(file_path)
        
    if not full_file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found.")
        
    # Determine file type
    file_extension = full_file_path.suffix.lower()
    
    try:
        if file_extension == ".xlsx":
            data = pd.read_excel(full_file_path)
        elif file_extension == ".csv":
            data = pd.read_csv(full_file_path, sep=";")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
        if data.empty:
            raise ValueError(f"The file {file_path} is empty.")
            
        return data
        
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {str(e)}")

def save_data(
    data: pd.DataFrame,
    file_path: Union[str, Path],
    directory: Optional[Path] = None,
    **kwargs
) -> None:
    """Save data to a file (CSV or Excel).
    
    Args:
        data: The data to save
        file_path: The path where to save the file
        directory: Optional directory where to save the file
        **kwargs: Additional arguments to pass to pandas to_csv or to_excel
    """
    if directory:
        full_file_path = directory / file_path
    else:
        full_file_path = Path(file_path)
        
    # Ensure directory exists
    full_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine file type
    file_extension = full_file_path.suffix.lower()
    
    try:
        if file_extension == ".xlsx":
            data.to_excel(full_file_path, **kwargs)
        elif file_extension == ".csv":
            data.to_csv(full_file_path, sep=";", **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
    except Exception as e:
        raise ValueError(f"Error saving file {file_path}: {str(e)}")

def backup_file(filepath: Path, backup_dir: Optional[Path] = None) -> Path:
    """Create a backup of a file.
    
    Args:
        filepath: Path to the file to backup
        backup_dir: Optional directory for backup. If None, uses same directory.
        
    Returns:
        Path to the backup file
    """
    try:
        if not filepath.exists():
            logger.warning(f"File {filepath} does not exist, skipping backup")
            return filepath
            
        backup_dir = backup_dir or filepath.parent
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_path = backup_dir / f"{filepath.stem}_backup{filepath.suffix}"
        shutil.copy2(filepath, backup_path)
        logger.info(f"Created backup at {backup_path}")
        
        return backup_path
        
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        raise

def ensure_data_directories():
    """Ensure all required data directories exist."""
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def get_latest_file(directory: Path, pattern: str = "*.csv") -> Optional[Path]:
    """Get the most recently modified file matching a pattern.
    
    Args:
        directory: Directory to search in
        pattern: File pattern to match
        
    Returns:
        Path to the latest file or None if no files found
    """
    try:
        files = list(directory.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda x: x.stat().st_mtime)
        
    except Exception as e:
        logger.error(f"Error finding latest file: {e}")
        return None 