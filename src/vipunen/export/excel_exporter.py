"""
Excel exporter module for the Vipunen project.

This module provides functions to export data to Excel files.
"""
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

from ..config.config_loader import get_config
from ..utils.file_utils_config import get_file_utils
from FileUtils import OutputFileType
from FileUtils.core.base import StorageError

logger = logging.getLogger(__name__)

def export_to_excel(
    data_dict: Dict[str, pd.DataFrame], 
    file_name: str, 
    output_dir: Optional[Union[str, Path]] = None,
    include_timestamp: bool = True,
    **kwargs
) -> Optional[Path]:
    """
    Export multiple DataFrames to Excel using FileUtils, handling subdirectories.
    
    Args:
        data_dict: Dictionary mapping sheet names to DataFrames
        file_name: Base name for the output file (without extension or subdirs)
        output_dir: Optional directory path *relative* to the project's data directory 
                    (e.g., 'reports/my_subdir' or 'processed/another_run'). 
                    If None, defaults to 'reports'.
        include_timestamp: Whether to include a timestamp in the filename
        **kwargs: Additional arguments for FileUtils.save_data_to_storage
        
    Returns:
        Path: Path to the saved Excel file or None if export failed
    """
    # Get config and file utils instance
    config = get_config()
    file_utils = get_file_utils()

    # Filter out empty DataFrames
    filtered_data = {
        sheet_name: df for sheet_name, df in data_dict.items() 
        if isinstance(df, pd.DataFrame) and not df.empty
    }
    
    if not filtered_data:
        logger.warning("No data to export to Excel")
        return None
    
    base_file_name = file_name # Keep original filename base
    
    # Add timestamp if requested
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_file_name = f"{base_file_name}_{timestamp}"
    
    # Determine output_type and sub_path based on output_dir
    sub_path = None
    if output_dir:
        # Assume output_dir is relative to the project's data directory root
        # e.g., 'reports/subdir1' or 'processed/subdir2' or just 'reports'
        output_path = Path(output_dir)
        
        # Check if path has more than one part (e.g., 'reports/subdir')
        if len(output_path.parts) > 1:
            output_type = output_path.parts[0] # e.g., 'reports'
            # Join the remaining parts to form the sub_path
            sub_path = str(Path(*output_path.parts[1:])) # e.g., 'subdir1' or 'subdir1/subdir2'
        elif len(output_path.parts) == 1:
            # Only the output_type was specified (e.g., 'reports')
            output_type = output_path.parts[0]
            sub_path = None # No sub-directory
        else:
            # Handle unexpected case (e.g., empty path string?) - default
            logger.warning(f"Could not parse output_dir '{output_dir}'. Defaulting output type.")
            output_type = config['paths'].get('output', 'reports')
            sub_path = None
            
    else:
        # Default to 'reports' directory with no subdirectories if output_dir is None
        output_type = config['paths'].get('output', 'reports')
        sub_path = None

    # Clean data before saving
    for sheet_name, df in filtered_data.items():
        if not df.empty:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Use FileUtils with explicit sub_path argument
    try:
        log_path = f"{output_type}/{sub_path}/{base_file_name}.xlsx" if sub_path else f"{output_type}/{base_file_name}.xlsx"
        logger.info(f"Saving Excel file to {log_path}")
        
        # Use the dedicated sub_path parameter
        path_result = file_utils.save_data_to_storage(
            data=filtered_data,
            file_name=base_file_name,          # Just the base filename
            output_type=output_type,           # Base output type (e.g., 'reports')
            output_filetype=OutputFileType.XLSX,
            sub_path=sub_path,                 # Pass sub-directory path here
            include_timestamp=False,           # Timestamp already added to base_file_name
            index=False,
            **kwargs
        )
        
        # Extract the actual path from the result (FileUtils might return dict/tuple)
        if isinstance(path_result, tuple) and path_result:
            if isinstance(path_result[0], dict):
                excel_path_str = next(iter(path_result[0].values()))
            else:
                excel_path_str = path_result[0]
        elif isinstance(path_result, dict):
            excel_path_str = next(iter(path_result.values()))
        else:
            excel_path_str = path_result
            
        if excel_path_str:
            logger.info(f"Exported Excel file to {excel_path_str}")
            return Path(excel_path_str)
        else:
            logger.error("FileUtils returned an empty path result.")
            return None

    except StorageError as e:
        logger.error(f"FileUtils StorageError exporting Excel file: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error exporting Excel file using FileUtils: {e}")
        return None 