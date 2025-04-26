"""
File handling utilities using FileUtils package.

This module provides a singleton wrapper around FileUtils for standardized 
file operations throughout the Vipunen project.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from FileUtils import FileUtils, OutputFileType
from FileUtils.core.base import StorageError

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
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a file with automatic format detection and error handling.
        
        Args:
            file_path: Path to the data file, relative to the input directory
            input_type: Type of input directory (raw, processed, etc.)
            **kwargs: Additional arguments to pass to the pandas read function
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # FileUtils automatically detects the file type from the extension
            logger.info(f"Loading data from {file_path} in {input_type} directory")
            data = self.file_utils.load_single_file(
                file_path, 
                input_type=input_type,
                **kwargs
            )
            
            logger.info(f"Loaded {len(data)} rows")
            return data
            
        except StorageError as e:
            logger.error(f"Storage error loading {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def load_excel_sheets(
        self,
        file_path: Union[str, Path],
        input_type: str = "raw",
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all sheets from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            input_type: Type of input directory (raw, processed, etc.)
            **kwargs: Additional arguments to pass to pandas
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of sheet names to DataFrames
        """
        try:
            logger.info(f"Loading Excel sheets from {file_path}")
            sheets = self.file_utils.load_excel_sheets(
                file_path,
                input_type=input_type,
                **kwargs
            )
            
            logger.info(f"Loaded {len(sheets)} sheets from Excel file")
            return sheets
            
        except StorageError as e:
            logger.error(f"Storage error loading Excel sheets from {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading Excel sheets from {file_path}: {e}")
            raise
    
    def save_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        file_name: str,
        output_type: str = "processed",
        output_filetype: Union[OutputFileType, str] = OutputFileType.CSV,
        include_timestamp: bool = True,
        **kwargs
    ) -> Union[Path, str]:
        """
        Save data to a file.
        
        Args:
            data: DataFrame or dictionary of DataFrames to save
            file_name: Base name for the output file
            output_type: Type of output directory (processed, reports, etc.)
            output_filetype: Output file format
            include_timestamp: Whether to include a timestamp in the filename
            **kwargs: Additional arguments to pass to the pandas write function
            
        Returns:
            Path or str: Path to the saved file
        """
        try:
            logger.info(f"Saving data to {output_type}/{file_name}")
            
            # FileUtils can handle both single DataFrames and dictionaries of DataFrames
            path = self.file_utils.save_data_to_storage(
                data=data,
                file_name=file_name,
                output_type=output_type,
                output_filetype=output_filetype,
                include_timestamp=include_timestamp,
                **kwargs
            )
            
            if isinstance(data, pd.DataFrame):
                logger.info(f"Saved {len(data)} rows to {path}")
            else:
                logger.info(f"Saved {len(data)} sheets to {path}")
                
            logger.info(f"Raw save_data_to_storage result: {path}")
                
            # Handle different return types from save_data_to_storage
            if isinstance(path, dict):
                # For multi-sheet Excel files, FileUtils might return a dict
                # Return the first file path in the dictionary
                first_path = next(iter(path.values()))
                return Path(first_path)
            elif isinstance(path, tuple):
                # If it returns a tuple, extract the first element as the path
                if isinstance(path[0], dict):
                    # If the first element is a dict, extract the first path
                    first_path = next(iter(path[0].values()))
                    return Path(first_path)
                else:
                    return Path(path[0])
            else:
                # Otherwise, assume it's a string path
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
    ) -> Union[Path, str]:
        """
        Export multiple DataFrames to a single Excel file.
        
        This is a convenience wrapper around save_data that specifically targets Excel files.
        
        Args:
            data_dict: Dictionary mapping sheet names to DataFrames
            file_name: Base name for the output file
            output_type: Type of output directory
            include_timestamp: Whether to include a timestamp in the filename
            **kwargs: Additional arguments to pass to ExcelWriter
            
        Returns:
            Path or str: Path to the saved Excel file
        """
        return self.save_data(
            data=data_dict,
            file_name=file_name,
            output_type=output_type,
            output_filetype=OutputFileType.XLSX,
            include_timestamp=include_timestamp,
            **kwargs
        )
    
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
        
        # Create the directory
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory at {output_dir}")
        
        return output_dir 