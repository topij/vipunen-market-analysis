"""
FileUtils Configuration Module

This module provides a singleton pattern for the FileUtils instance to ensure
consistent configuration across the application.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from FileUtils import FileUtils

# Configure logging
logger = logging.getLogger(__name__)

# Singleton instance of FileUtils
_file_utils_instance = None

def get_file_utils(project_root: Optional[str] = None) -> FileUtils:
    """
    Get or create the FileUtils singleton instance with project-specific configuration.
    
    Args:
        project_root: Optional explicit path to the project root. If not provided,
                     will attempt to determine automatically.
    
    Returns:
        FileUtils: Configured FileUtils instance
    """
    global _file_utils_instance
    
    if _file_utils_instance is None:
        logger.info("Initializing FileUtils package")
        
        # Determine the project root
        if project_root is None:
            # Try to determine the project root
            current_dir = Path(os.path.abspath(os.path.dirname(__file__)))
            
            # Look for typical project root indicators
            root_path = None
            for parent in [current_dir] + list(current_dir.parents):
                # Check for common project root indicators
                if (parent / ".git").exists() or (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
                    root_path = parent
                    break
                
                # Also check for data directory which is common in this project
                if (parent / "data").exists() and (parent / "data" / "raw").exists():
                    root_path = parent
                    break
            
            if root_path is None:
                # Fallback to a reasonable default
                logger.warning("Could not determine project root, using current working directory")
                root_path = Path(os.getcwd())
            
            project_root = str(root_path)
        
        logger.info(f"Using project root: {project_root}")
        
        # Initialize FileUtils with direct configuration
        _file_utils_instance = FileUtils(
            project_root=project_root,
            logging_level=logging.WARNING,
            config_override={
                "include_timestamp": True,
                "csv_delimiter": ";",
                "encoding": "utf-8"
            }
        )
        
        logger.info("FileUtils initialized successfully")
    
    return _file_utils_instance 