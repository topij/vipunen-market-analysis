# FileUtils Integration Guide

This document describes how the Vipunen project integrates with the FileUtils package for standardized file operations.

## Overview

The FileUtils package provides standardized utilities for file operations in data science projects, including:

- Loading data from various formats
- Saving data with standardized metadata
- Managing project directory structure
- Handling file paths consistently

## Integration Architecture

### VipunenFileHandler

The `VipunenFileHandler` class in `src/vipunen/utils/file_handler.py` serves as a singleton wrapper around FileUtils, ensuring consistent file operations throughout the Vipunen project.

```python
class VipunenFileHandler:
    """Singleton wrapper for FileUtils in the Vipunen project."""
    
    _instance = None
    
    def __new__(cls, project_root: Optional[Union[str, Path]] = None):
        if cls._instance is None:
            cls._instance = super(VipunenFileHandler, cls).__new__(cls)
            cls._instance._initialize(project_root)
        return cls._instance
```

Key features include:

1. **Singleton Pattern**: Ensures only one instance is created throughout the application.
2. **Configuration Management**: Maintains standard directory structure and file formats.
3. **Error Handling**: Provides robust error handling for file operations.
4. **Standardized API**: Wraps FileUtils functionality in domain-specific methods.

## Usage

### Loading Data

```python
from src.vipunen.utils.file_handler import VipunenFileHandler
from src.vipunen.data.data_loader import load_data

# Method 1: Using the data loader
df = load_data("amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv")

# Method 2: Using the file handler directly
file_handler = VipunenFileHandler()
df = file_handler.load_data(
    "amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv", 
    input_type="raw",
    file_type=InputFileType.CSV,
    sep=";"
)
```

### Saving Data

```python
# Save a single DataFrame
file_handler = VipunenFileHandler()
output_path = file_handler.save_data(
    data=df,
    file_name="processed_data",
    output_type="processed",
    include_timestamp=True
)
```

### Exporting to Excel

```python
# Export multiple DataFrames to Excel
excel_data = {
    "Total Volumes": total_volumes,
    "Volumes by Qualification": volumes_long_df,
    "Provider's Market": providers_market
}

file_handler = VipunenFileHandler()
excel_path = file_handler.export_to_excel(
    data_dict=excel_data,
    file_name="market_analysis",
    output_type="reports",
    include_timestamp=True
)
```

### Creating Output Directories

```python
# Create a specialized output directory
output_dir = file_handler.create_output_directory(
    institution_name="RI",
    base_dir="reports"
)
```

## Directory Structure

FileUtils enforces a standard directory structure for the project:

```
project_root/
├── data/
│   ├── raw/          # Raw data files
│   ├── processed/    # Processed data files
│   ├── interim/      # Intermediate data files
│   └── reports/      # Generated reports and outputs
├── src/
│   └── vipunen/      # Package source code
└── notebooks/        # Jupyter notebooks
```

## Implementation Notes

### InputFileType Enum

Since FileUtils doesn't provide an input file type enum, we define our own in `src/vipunen/utils/file_handler.py`:

```python
class InputFileType(Enum):
    """Supported input file types."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"
    YAML = "yaml"
```

### Export Implementation

For Excel exports, the VipunenFileHandler uses pandas directly instead of FileUtils' `save_with_metadata` to avoid compatibility issues with certain versions of FileUtils:

```python
def export_to_excel(self, data_dict, file_name, output_type="reports", include_timestamp=True, **kwargs):
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
    
    return Path(file_path)
```

## Troubleshooting

### Common Issues

#### 1. Path Resolution Issues

When you encounter duplicate directory paths like:

```
/path/to/data/raw/data/raw/filename.csv
```

Verify that:
- The path passed to `load_data` uses the expected format
- The `ensure_data_directory` function properly handles existing directory prefixes

#### 2. Excel Export Errors

If you see errors like `'str' object has no attribute 'value'`, check if:
- You're using the correct `OutputFileType` enum (from FileUtils)
- You've properly passed enum values as strings when needed

## Version Compatibility

This integration has been tested with FileUtils v0.6.1.

For newer versions of FileUtils, the integration pattern should remain valid, but check for API changes in:
- Enum handling for file types
- Path resolution logic
- The `save_with_metadata` method signature 