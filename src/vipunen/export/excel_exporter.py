"""
Excel export utilities for education market analysis.
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExcelExporter:
    """
    Handles exporting analysis results to Excel files.
    """
    
    def __init__(self, output_dir: Union[str, Path], prefix: str = "analysis"):
        """
        Initialize the Excel exporter.
        
        Args:
            output_dir: Directory to save Excel files
            prefix: Prefix for file names
        """
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _format_numeric_sheet(self, df: pd.DataFrame, 
                            int_columns: Optional[list] = None, 
                            decimal_columns: Optional[dict] = None) -> pd.DataFrame:
        """
        Format numeric columns in a DataFrame for Excel export.
        
        Args:
            df: DataFrame to format
            int_columns: List of columns to format as integers
            decimal_columns: Dict of columns to format with specific decimal places (column -> decimal places)
            
        Returns:
            pd.DataFrame: Formatted DataFrame
        """
        formatted_df = df.copy()
        
        # Format integer columns
        if int_columns:
            for col in int_columns:
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].fillna(0).astype(int)
        
        # Format decimal columns
        if decimal_columns:
            for col, decimals in decimal_columns.items():
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].round(decimals)
        
        return formatted_df
    
    def prepare_total_volumes_sheet(self, volumes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the total volumes DataFrame for Excel export.
        
        Args:
            volumes_df: DataFrame with total volumes
            
        Returns:
            pd.DataFrame: Formatted DataFrame
        """
        if volumes_df.empty:
            return pd.DataFrame()
        
        # Create a copy with index reset
        sheet_data = volumes_df.reset_index(drop=True).copy()
        
        # Format numeric columns
        int_columns = ['järjestäjänä', 'hankintana', 'Yhteensä']
        decimal_columns = {'järjestäjä_osuus (%)': 2}
        
        return self._format_numeric_sheet(sheet_data, int_columns, decimal_columns)
    
    def prepare_volumes_by_role_sheet(self, qualification_volumes: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the volumes by role DataFrame for Excel export.
        
        Args:
            qualification_volumes: DataFrame with volumes by qualification
            
        Returns:
            pd.DataFrame: Formatted DataFrame
        """
        if qualification_volumes.empty:
            return pd.DataFrame()
        
        # Prepare a new DataFrame for the long format table
        volumes_by_qual_role = []
        
        # Get the years from column names
        years = sorted(list(set([int(col.split('_')[0]) for col in qualification_volumes.columns if '_' in col])))
        
        # Process each qualification
        for _, row in qualification_volumes.iterrows():
            qualification = row['tutkinto']
            
            # For each year, get provider and subcontractor amounts
            for year in years:
                provider_col = f"{year}_järjestäjänä"
                subcontractor_col = f"{year}_hankintana"
                
                # Check if the columns exist for this year
                if provider_col in qualification_volumes.columns and subcontractor_col in qualification_volumes.columns:
                    provider_amount = row.get(provider_col, 0)
                    subcontractor_amount = row.get(subcontractor_col, 0)
                    
                    # Only add rows where at least one amount is greater than 0
                    if provider_amount > 0 or subcontractor_amount > 0:
                        volumes_by_qual_role.append({
                            'Year': year,
                            'Qualification': qualification,
                            'Provider Amount': int(provider_amount),
                            'Subcontractor Amount': int(subcontractor_amount),
                            'Total': int(provider_amount + subcontractor_amount),
                            'Provider Role %': round((provider_amount / (provider_amount + subcontractor_amount) * 100), 2) if (provider_amount + subcontractor_amount) > 0 else 0
                        })
        
        # Convert to DataFrame
        volumes_by_role = pd.DataFrame(volumes_by_qual_role)
        
        # Sort by year then qualification
        if not volumes_by_role.empty:
            volumes_by_role = volumes_by_role.sort_values(['Year', 'Qualification'])
        
        return volumes_by_role
    
    def prepare_cagr_sheet(self, cagr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the CAGR DataFrame for Excel export.
        
        Args:
            cagr_df: DataFrame with CAGR values
            
        Returns:
            pd.DataFrame: Formatted DataFrame
        """
        if cagr_df.empty:
            return pd.DataFrame()
        
        # Format numeric columns
        decimal_columns = {'CAGR': 2}
        
        return self._format_numeric_sheet(cagr_df, decimal_columns=decimal_columns)
    
    def export_to_excel(self, data_dict: Dict[str, pd.DataFrame]) -> Path:
        """
        Export multiple DataFrames to a single Excel file.
        
        Args:
            data_dict: Dictionary mapping sheet names to DataFrames
            
        Returns:
            Path: Path to the saved Excel file
        """
        # Generate a timestamp for the file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = self.output_dir / f"{self.prefix}_market_analysis_{timestamp}.xlsx"
        
        try:
            # Create Excel writer
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for sheet_name, sheet_data in data_dict.items():
                    if not isinstance(sheet_data, pd.DataFrame) or sheet_data.empty:
                        logger.warning(f"Skipping empty sheet: {sheet_name}")
                        continue
                    
                    # Make sure we're dealing with a DataFrame with a standard index
                    try:
                        # Reset index if it's not a default RangeIndex
                        if not isinstance(sheet_data.index, pd.RangeIndex):
                            sheet_data = sheet_data.reset_index()
                        
                        # Export to Excel
                        sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
                        logger.info(f"Added sheet '{sheet_name}' with {len(sheet_data)} rows")
                    except Exception as e:
                        logger.error(f"Error exporting sheet '{sheet_name}': {e}")
                        try:
                            # Fallback: Create a simpler DataFrame
                            safe_df = pd.DataFrame(sheet_data.values)
                            if hasattr(sheet_data, 'columns'):
                                safe_df.columns = sheet_data.columns
                            safe_df.to_excel(writer, sheet_name=sheet_name, index=False)
                            logger.info(f"Added sheet '{sheet_name}' using fallback method")
                        except Exception as e2:
                            logger.error(f"Failed to export sheet '{sheet_name}' with fallback: {e2}")
            
            logger.info(f"Exported data to {excel_path}")
            return excel_path
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            raise 