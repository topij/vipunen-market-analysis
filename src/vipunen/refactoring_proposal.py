"""
Refactoring Proposal for Vipunen Project

This document outlines proposed refactoring steps to make the codebase more modular and DRY.
"""

#------------------------------------------------------------------------------
# CURRENT ISSUES
#------------------------------------------------------------------------------

"""
Based on the code analysis, here are the main issues in the current architecture:

1. Duplication between analyze_education_market.py and src/vipunen/analysis/education_market.py
   - The main script contains logic that duplicates what's in the module
   - The EducationMarketAnalyzer class in the module is not being used in the main script

2. run_analysis.py duplicates argument parsing logic from analyze_education_market.py
   - Both files have similar argument parsing code with almost identical parameters

3. Large monolithic functions in analyze_education_market.py
   - The main() function is over 500 lines long with many steps all in one place
   - create_dummy_dataset is a large standalone function

4. Direct file I/O mixed with analysis logic
   - Loading data, creating directories, and exporting results are mixed with analysis code

5. FileUtils usage is inconsistent
   - Sometimes using it directly, sometimes falling back to pandas 
   - Error handling is duplicated in multiple places

6. Visualization code is called directly from the main analysis script
   - This creates tight coupling between analysis and visualization

7. Hardcoded values scattered throughout the code
   - Default institution names, column names, etc. should be centralized
"""

#------------------------------------------------------------------------------
# PROPOSED ARCHITECTURE
#------------------------------------------------------------------------------

"""
Proposed directory structure:

vipunen/
├── __init__.py
├── cli/                      # Command-line interface module
│   ├── __init__.py
│   ├── analyze_cli.py        # Clean CLI implementation
│   └── argument_parser.py    # Centralized argument parsing logic
├── config/                   # Configuration module
│   ├── __init__.py
│   ├── constants.py          # Common constants and defaults
│   └── settings.py           # Settings management
├── data/                     # Data handling module
│   ├── __init__.py
│   ├── data_loader.py        # Centralized data loading
│   ├── data_processor.py     # Data cleaning and preparation
│   └── dummy_generator.py    # Move dummy dataset generation here
├── analysis/                 # Analysis module
│   ├── __init__.py
│   ├── education_market.py   # EducationMarketAnalyzer class
│   ├── market_share.py       # Market share analysis
│   └── qualification.py      # Qualification analysis
├── visualization/            # Visualization module
│   ├── __init__.py
│   └── [existing files]      # Keep existing visualization modules
├── export/                   # Export module
│   ├── __init__.py
│   └── excel_exporter.py     # Centralized Excel export logic
└── utils/                    # Utility functions
    ├── __init__.py
    ├── file_utils_config.py  # FileUtils configuration (existing)
    └── path_helpers.py       # Path manipulation utilities
"""

#------------------------------------------------------------------------------
# IMPLEMENTATION STEPS
#------------------------------------------------------------------------------

"""
1. Create a central configuration module
   - Move all hardcoded values to constants.py
   - Create settings.py to manage runtime configuration

2. Refactor the main analyze_education_market.py script
   - Break down the main() function into smaller, focused functions
   - Move the dummy dataset generation to data/dummy_generator.py
   - Use the EducationMarketAnalyzer class from the module

3. Create a dedicated CLI module
   - Move argument parsing to cli/argument_parser.py
   - Create a clean cli/analyze_cli.py that imports from the modules

4. Standardize FileUtils usage
   - Create a dedicated data_loader.py to centralize file loading
   - Create a dedicated excel_exporter.py for saving results

5. Improve separation of concerns
   - Remove direct visualization calls from analysis code
   - Create a workflow orchestrator class that composes the different steps

6. Clean up run_analysis.py
   - Make it a thin wrapper that calls the cli module
   - Remove duplicated argument parsing logic
"""

#------------------------------------------------------------------------------
# CODE EXAMPLES
#------------------------------------------------------------------------------

# Example 1: Configuration module

"""
# vipunen/config/constants.py

# Column names
YEAR_COL = 'tilastovuosi'
DEGREE_TYPE_COL = 'tutkintotyyppi'
QUALIFICATION_COL = 'tutkinto'
PROVIDER_COL = 'koulutuksenJarjestaja'
SUBCONTRACTOR_COL = 'hankintakoulutuksenJarjestaja'
VOLUME_COL = 'nettoopiskelijamaaraLkm'

# Qualification types to filter
QUALIFICATION_TYPES = ["Ammattitutkinnot", "Erikoisammattitutkinnot"]

# Default institution
DEFAULT_INSTITUTION = "Rastor-instituutti"
DEFAULT_SHORT_NAME = "RI"
DEFAULT_VARIANTS = [
    "Rastor-instituutti ry", 
    "Rastor-instituutti", 
    "RASTOR OY",
    "Rastor Oy"
]

# File paths
DEFAULT_DATA_PATH = "data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv"
DEFAULT_OUTPUT_DIR = "data/reports"

# Excel export column mappings
EXCEL_COLUMN_MAPPING = {
    'tilastovuosi': 'Year',
    'tutkinto': 'Qualification',
    'provider': 'Provider',
    # ... other mappings
}
"""

# Example 2: Data loader module

"""
# vipunen/data/data_loader.py

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Union

from ..utils.file_utils_config import get_file_utils
from FileUtils.core.base import StorageError

logger = logging.getLogger(__name__)

def load_data(file_path: Union[str, Path], use_dummy: bool = False) -> pd.DataFrame:
    \"\"\"
    Load data from a file or create dummy data if requested.
    
    Args:
        file_path: Path to the data file
        use_dummy: Whether to use dummy data instead of loading from file
        
    Returns:
        pd.DataFrame: Loaded or generated data
    \"\"\"
    if use_dummy:
        logger.info("Using dummy dataset for demonstration purposes")
        from .dummy_generator import create_dummy_dataset
        return create_dummy_dataset()
    
    logger.info(f"Loading data from {file_path}")
    file_utils = get_file_utils()
    path_obj = Path(file_path)
    file_name = path_obj.name
    
    try:
        # Try with semicolon separator for CSV files
        if path_obj.suffix.lower() == '.csv':
            return file_utils.load_single_file(file_name, input_type="raw", sep=';')
        else:
            # For non-CSV files, let FileUtils auto-detect the format
            return file_utils.load_single_file(file_name, input_type="raw")
    except Exception:
        # Try with comma separator if semicolon fails
        try:
            return file_utils.load_single_file(file_name, input_type="raw", sep=',')
        except (FileNotFoundError, StorageError) as e:
            logger.error(f"Could not find or load the data file at {file_path}: {e}")
            logger.info("Creating a dummy dataset for demonstration purposes")
            from .dummy_generator import create_dummy_dataset
            return create_dummy_dataset()
"""

# Example 3: Excel export module

"""
# vipunen/export/excel_exporter.py

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

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
    \"\"\"
    Export multiple DataFrames to Excel.
    
    Args:
        data_dict: Dictionary mapping sheet names to DataFrames
        file_name: Base name for the output file
        output_dir: Directory to save the file in
        include_timestamp: Whether to include a timestamp in the filename
        **kwargs: Additional arguments for Excel export
        
    Returns:
        Path: Path to the saved Excel file or None if export failed
    \"\"\"
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
    
    # Try direct pandas export first if output_dir is specified
    if output_dir:
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
            # Fall through to FileUtils method
    
    # Use FileUtils as backup or if no output_dir specified
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
"""

# Example 4: Clean CLI implementation

"""
# vipunen/cli/analyze_cli.py

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..config.constants import (
    DEFAULT_DATA_PATH, DEFAULT_INSTITUTION, DEFAULT_SHORT_NAME, DEFAULT_VARIANTS
)
from ..cli.argument_parser import parse_arguments
from ..data.data_loader import load_data
from ..data.data_processor import clean_and_prepare_data
from ..analysis.education_market import EducationMarketAnalyzer
from ..export.excel_exporter import export_to_excel

logger = logging.getLogger(__name__)

def run_analysis(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    \"\"\"
    Run the education market analysis workflow.
    
    Args:
        args: Dictionary of arguments, defaults to command-line arguments if None
        
    Returns:
        Dict[str, Any]: Dictionary with analysis results
    \"\"\"
    # Parse arguments if not provided
    if args is None:
        args = vars(parse_arguments())
    
    # Load data
    data_file = args.get('data_file', DEFAULT_DATA_PATH)
    use_dummy = args.get('use_dummy', False)
    raw_data = load_data(data_file, use_dummy)
    
    # Get institution information
    institution_name = args.get('institution', DEFAULT_INSTITUTION)
    institution_short_name = args.get('short_name', DEFAULT_SHORT_NAME)
    
    # Set variants
    if args.get('variants'):
        institution_variants = list(args.get('variants', []))
        if institution_name not in institution_variants:
            institution_variants.append(institution_name)
    else:
        institution_variants = DEFAULT_VARIANTS
    
    # Clean and prepare data
    df_clean = clean_and_prepare_data(
        raw_data, 
        institution_names=institution_variants,
        merge_qualifications=True,
        shorten_names=True
    )
    
    # Apply filters if requested
    filter_qual_types = args.get('filter_qual_types', False)
    filter_by_inst_quals = args.get('filter_by_inst_quals', False)
    
    # Create analyzer
    analyzer = EducationMarketAnalyzer(
        data=df_clean,
        institution_names=institution_variants,
        filter_degree_types=filter_qual_types,
        filter_qualifications=None  # Will be calculated if filter_by_inst_quals is True
    )
    
    # Set up qualification filtering if needed
    if filter_by_inst_quals:
        logger.info("Filtering for qualifications offered by the institution")
        # Get qualifications where the institution is active
        latest_year = df_clean["tilastovuosi"].max()
        previous_year = latest_year - 1
        analyzer.reference_year = latest_year
        
        # Will apply filtering internally based on reference_year
    
    # Run analyses
    total_volumes = analyzer.analyze_total_volume()
    volumes_by_qual = analyzer.analyze_volumes_by_qualification()
    market_shares = analyzer.analyze_institution_roles()
    qualification_growth = analyzer.analyze_qualification_growth()
    cagr_analysis = analyzer.calculate_qualification_cagr()
    
    # Prepare Excel data
    excel_data = {
        "Total Volumes": total_volumes,
        "Volumes by Qualification": volumes_by_qual,
        "Provider's Market": market_shares,
        "CAGR Analysis": cagr_analysis
    }
    
    # Export to Excel
    output_dir = args.get('output_dir')
    if output_dir:
        output_dir = Path(output_dir) / f"education_market_{institution_short_name.lower()}"
    else:
        output_dir = Path("data/reports") / f"education_market_{institution_short_name.lower()}"
    
    excel_path = export_to_excel(
        data_dict=excel_data,
        file_name=f"{institution_short_name.lower()}_market_analysis",
        output_dir=output_dir,
        include_timestamp=True
    )
    
    # Return results
    return {
        "total_volumes": total_volumes,
        "volumes_by_qualification": volumes_by_qual,
        "market_shares": market_shares,
        "qualification_cagr": cagr_analysis,
        "excel_path": excel_path
    }

def main():
    \"\"\"Main entry point for the CLI.\"\"\"
    try:
        results = run_analysis()
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
"""

#------------------------------------------------------------------------------
# MIGRATION PLAN
#------------------------------------------------------------------------------

"""
Recommended migration approach:

1. Phase 1: Create the new module structure
   - Set up the directory structure as proposed
   - Move existing code to appropriate locations with minimal changes

2. Phase 2: Incremental refactoring
   - Start with the configuration and utility modules
   - Then address the data loading and processing modules
   - Next implement the clean export functionality
   - Finally refactor the analysis code to use the EducationMarketAnalyzer

3. Phase 3: Improve the CLI
   - Create the new clean CLI implementation
   - Update run_analysis.py to use the new CLI

4. Phase 4: Testing
   - Add tests for each module to ensure functionality is preserved
   - Verify that the refactored code produces identical results to the original

5. Phase 5: Documentation
   - Add proper docstrings to all modules and functions
   - Create a README explaining the new architecture
""" 