"""
File handling utilities using FileUtils package.

This module provides a singleton wrapper around FileUtils for standardized 
file operations throughout the Vipunen project.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import pandas as pd
from FileUtils import FileUtils, OutputFileType

# Define InputFileType enum (not available in FileUtils)
class InputFileType(Enum):
    """Supported input file types."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"
    YAML = "yaml"

# Configure logging
logger = logging.getLogger(__name__)

class VipunenFileHandler:
    """Singleton wrapper for FileUtils in the Vipunen project."""
    
    _instance = None
    
    def __new__(cls, project_root: Optional[Union[str, Path]] = None):
        if cls._instance is None:
            cls._instance = super(VipunenFileHandler, cls).__new__(cls)
            cls._instance._initialize(project_root)
        return cls._instance
    
    def _initialize(self, project_root: Optional[Union[str, Path]] = None):
        """Initialize the file handler with project configuration."""
        # Set default project root if not provided
        if project_root is None:
            # Try to guess the project root (current directory or parent of src)
            current_path = Path.cwd()
            if (current_path / "src" / "vipunen").exists():
                project_root = current_path
            else:
                # Assume we're in a subdirectory of the project
                project_root = current_path.parents[2]  # Go up to project root
        
        # Create FileUtils instance with standard Vipunen directory structure
        self.file_utils = FileUtils(
            project_root=project_root,
            config={
                "directory_structure": {
                    "data": [
                        "raw",
                        "processed", 
                        "interim",
                        "reports"
                    ],
                    "src": ["vipunen"],
                    "notebooks": []
                },
                "include_timestamp": True,
                "logging_level": "INFO"
            }
        )
        logger.info(f"Initialized VipunenFileHandler with project root: {project_root}")
    
    def load_data(
        self, 
        file_path: Union[str, Path], 
        input_type: str = "raw",
        file_type: Optional[InputFileType] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a file with automatic format detection and error handling.
        
        Args:
            file_path: Path to the data file, relative to the input directory
            input_type: Type of input directory (raw, processed, etc.)
            file_type: Optional explicit file type
            **kwargs: Additional arguments to pass to the pandas read function
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Determine file type if not explicitly provided
            if file_type is None:
                suffix = Path(file_path).suffix.lower()
                if suffix in ['.csv']:
                    file_type = InputFileType.CSV
                elif suffix in ['.xlsx', '.xls']:
                    file_type = InputFileType.EXCEL
                elif suffix in ['.json']:
                    file_type = InputFileType.JSON
                else:
                    raise ValueError(f"Unsupported file format: {suffix}")
            
            # Load the data using FileUtils
            logger.info(f"Loading {file_type.value} data from {file_path} in {input_type} directory")
            data = self.file_utils.load_single_file(
                file_path, 
                input_type=input_type,
                file_type=file_type,
                **kwargs
            )
            
            logger.info(f"Loaded {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def save_data(
        self,
        data: pd.DataFrame,
        file_name: str,
        output_type: str = "processed",
        output_filetype: OutputFileType = OutputFileType.CSV,
        include_timestamp: bool = True,
        **kwargs
    ) -> Path:
        """
        Save data to a file with metadata.
        
        Args:
            data: DataFrame to save
            file_name: Base name for the output file
            output_type: Type of output directory (processed, reports, etc.)
            output_filetype: Output file format
            include_timestamp: Whether to include a timestamp in the filename
            **kwargs: Additional arguments to pass to the pandas write function
            
        Returns:
            Path: Path to the saved file
        """
        try:
            logger.info(f"Saving data to {output_type}/{file_name}.{output_filetype.value}")
            
            # Save with metadata to track data lineage - convert enum to string value
            metadata, path = self.file_utils.save_with_metadata(
                {"data": data},
                output_filetype=output_filetype.value,  # Use string value
                output_type=output_type,
                file_name=file_name,
                include_timestamp=include_timestamp,
                **kwargs
            )
            
            logger.info(f"Saved {len(data)} rows to {path}")
            return Path(path)
            
        except Exception as e:
            logger.error(f"Error saving {file_name}: {e}")
            raise
    
    def export_to_excel(
        self,
        data_dict: Dict[str, pd.DataFrame],
        file_name: str,
        output_type: str = "reports",
        include_timestamp: bool = True,
        **kwargs
    ) -> Path:
        """
        Export multiple DataFrames to a single Excel file.
        
        Args:
            data_dict: Dictionary mapping sheet names to DataFrames
            file_name: Base name for the output file
            output_type: Type of output directory
            include_timestamp: Whether to include a timestamp in the filename
            **kwargs: Additional arguments to pass to ExcelWriter
            
        Returns:
            Path: Path to the saved Excel file
        """
        try:
            logger.info(f"Exporting {len(data_dict)} sheets to Excel file {file_name}")
            
            # Get the output directory path
            base_path = self.file_utils.get_data_path(output_type)
            
            # Generate the filename with timestamp if requested
            if include_timestamp:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                full_file_name = f"{file_name}_{timestamp}.xlsx"
            else:
                full_file_name = f"{file_name}.xlsx"
            
            # Create the full file path
            file_path = base_path / full_file_name
            
            # Use pandas ExcelWriter directly
            with pd.ExcelWriter(file_path, engine='openpyxl', **kwargs) as writer:
                for sheet_name, df in data_dict.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"Exported Excel file to {file_path}")
            return Path(file_path)
            
        except Exception as e:
            logger.error(f"Error exporting Excel file {file_name}: {e}")
            raise
    
    def create_output_directory(
        self, 
        institution_name: str,
        base_dir: str = "reports"
    ) -> Path:
        """
        Create a directory for output files.
        
        Args:
            institution_name: Short name for the institution
            base_dir: Base directory type
            
        Returns:
            Path: Path to the created directory
        """
        # Create a directory name based on the institution name
        dir_name = f"education_market_{institution_name.lower()}"
        
        # Get the full path
        output_dir = self.file_utils.get_data_path(base_dir) / dir_name
        
        # Create the directory using FileUtils
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory at {output_dir}")
        
        return output_dir 