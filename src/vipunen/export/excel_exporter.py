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

logger = logging.getLogger(__name__)

def export_to_excel(
    data_dict: Dict[str, pd.DataFrame], 
    file_name: str, 
    output_dir: Optional[Union[str, Path]] = None,
    include_timestamp: bool = True,
    **kwargs
) -> Optional[Path]:
    """
    Export multiple DataFrames to Excel.
    
    Args:
        data_dict: Dictionary mapping sheet names to DataFrames
        file_name: Base name for the output file
        output_dir: Directory to save the file in
        include_timestamp: Whether to include a timestamp in the filename
        **kwargs: Additional arguments for Excel export
        
    Returns:
        Path: Path to the saved Excel file or None if export failed
    """
    # Get config
    config = get_config()
    
    # Filter out empty DataFrames
    filtered_data = {
        sheet_name: df for sheet_name, df in data_dict.items() 
        if isinstance(df, pd.DataFrame) and not df.empty
    }
    
    if not filtered_data:
        logger.warning("No data to export to Excel")
        return None
    
    # Add timestamp if requested
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{file_name}_{timestamp}"
    
    # Ensure filename has .xlsx extension
    if not file_name.endswith('.xlsx'):
        file_name = f"{file_name}.xlsx"
    
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = config['paths'].get('output', 'data/reports')
    
    # Try direct pandas export first
    try:
        # Create full path
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        excel_path = output_dir / file_name
        
        # Use pd.ExcelWriter directly
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for sheet_name, df in filtered_data.items():
                if not df.empty:
                    # Clean data - replace infinities with NaN
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Exported Excel file to {excel_path}")
        return excel_path
        
    except Exception as e:
        logger.error(f"Error exporting Excel file to specific directory: {e}")
        logger.info("Falling back to FileUtils export method")
    
    # Use FileUtils as backup
    try:
        file_utils = get_file_utils()
        path_result = file_utils.save_data_to_storage(
            data=filtered_data,
            file_name=file_name.replace('.xlsx', ''),  # Remove extension as FileUtils adds it
            output_type="reports",
            output_filetype=OutputFileType.XLSX,
            index=False,
            **kwargs
        )
        
        # Extract the actual path from the result
        if isinstance(path_result, tuple) and path_result:
            if isinstance(path_result[0], dict):
                excel_path = next(iter(path_result[0].values()))
            else:
                excel_path = path_result[0]
        elif isinstance(path_result, dict):
            excel_path = next(iter(path_result.values()))
        else:
            excel_path = path_result
            
        logger.info(f"Exported Excel file to {excel_path}")
        return Path(excel_path)
        
    except Exception as e:
        logger.error(f"Error exporting Excel file: {e}")
        return None 