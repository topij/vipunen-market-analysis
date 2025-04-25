"""
Data processing utilities for education market analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def replace_values_in_dataframe(df: pd.DataFrame, replacement_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Replace specific values in a DataFrame based on a dictionary mapping.
    
    Args:
        df: DataFrame to modify
        replacement_dict: Dictionary mapping values to replace with their replacements
        
    Returns:
        pd.DataFrame: Modified DataFrame
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Iterate through the replacement dictionary
    for column, replacements in replacement_dict.items():
        if column in df_copy.columns:
            for old_value, new_value in replacements.items():
                # Replace values
                if old_value == 'Tieto puuttuu' and new_value is None:
                    # For "Tieto puuttuu" replacements, log the change
                    mask = df_copy[column] == old_value
                    df_copy.loc[mask, column] = np.nan
                    logger.info(f"Replaced 'Tieto puuttuu' with NaN in {column} column")
                else:
                    # For other replacements
                    mask = df_copy[column] == old_value
                    df_copy.loc[mask, column] = new_value
    
    return df_copy

def shorten_qualification_names(df: pd.DataFrame, column: str = 'tutkinto') -> pd.DataFrame:
    """
    Shorten qualification names by replacing common suffixes with abbreviations.
    
    Args:
        df: DataFrame with qualification names
        column: Column name containing qualification names
        
    Returns:
        pd.DataFrame: DataFrame with shortened qualification names
    """
    if column not in df.columns:
        logger.warning(f"Column {column} not found, no name shortening applied")
        return df
    
    df_copy = df.copy()
    
    # List of replacements
    replacements = [
        ('erikoisammattitutkinto', 'EAT'),
        ('ammattitutkinto', 'AT'),
        ('perustutkinto', 'PT')
    ]
    
    # Apply replacements
    for old, new in replacements:
        df_copy[column] = df_copy[column].str.replace(f' {old}$', f' {new}', regex=True)
    
    logger.info(f"Shortened qualification names (erikoisammattitutkinto → EAT, ammattitutkinto → AT)")
    return df_copy

def merge_qualification_variants(df: pd.DataFrame, column: str = 'tutkinto', 
                                mapping_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Merge different name variants of the same qualification.
    
    Args:
        df: DataFrame with qualification names
        column: Column name containing qualification names
        mapping_dict: Dictionary mapping old names to new names
        
    Returns:
        pd.DataFrame: DataFrame with merged qualification names
    """
    if column not in df.columns:
        logger.warning(f"Column {column} not found, no name merging applied")
        return df
    
    df_copy = df.copy()
    
    # Default mapping if none provided
    if mapping_dict is None:
        mapping_dict = {
            'Yrittäjän ammattitutkinto': 'Yrittäjyyden ammattitutkinto'
        }
    
    # Apply each mapping
    for old_name, new_name in mapping_dict.items():
        # Count rows before the change
        old_count_before = len(df_copy[df_copy[column] == old_name])
        new_count_before = len(df_copy[df_copy[column] == new_name])
        
        # Replace the old name with the new name
        df_copy.loc[df_copy[column] == old_name, column] = new_name
        
        # Count rows after the change
        old_count_after = len(df_copy[df_copy[column] == old_name])
        new_count_after = len(df_copy[df_copy[column] == new_name])
        
        # Log the change
        if old_count_before > 0:
            logger.info(f"Merged {old_name} into {new_name}:")
            logger.info(f"  - Before: {old_count_before} rows with old name, {new_count_before} rows with new name")
            logger.info(f"  - After: {old_count_after} rows with old name, {new_count_after} rows with new name")
    
    return df_copy

def clean_and_prepare_data(df: pd.DataFrame, institution_names: Optional[List[str]] = None,
                         merge_qualifications: bool = True, shorten_names: bool = False) -> pd.DataFrame:
    """
    Clean and prepare data for analysis.
    
    Args:
        df: Raw DataFrame
        institution_names: List of institution name variants to check
        merge_qualifications: Whether to merge qualification name variants
        shorten_names: Whether to shorten qualification names
        
    Returns:
        pd.DataFrame: Cleaned and prepared DataFrame
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Replace "Tieto puuttuu" with NaN for better handling
    replacements = {
        'hankintakoulutuksenJarjestaja': {'Tieto puuttuu': None}
    }
    df_copy = replace_values_in_dataframe(df_copy, replacements)
    
    # Merge qualification name variants if requested
    if merge_qualifications:
        df_copy = merge_qualification_variants(df_copy)
    
    # Shorten qualification names if requested
    if shorten_names:
        df_copy = shorten_qualification_names(df_copy)
    
    # Convert year to integer if it's not already
    if 'tilastovuosi' in df_copy.columns and df_copy['tilastovuosi'].dtype != 'int64':
        df_copy['tilastovuosi'] = pd.to_numeric(df_copy['tilastovuosi'], errors='coerce').fillna(0).astype(int)
    
    # Validate institution names
    if institution_names:
        # Check if any of the institution names exist in the data
        found_as_provider = df_copy['koulutuksenJarjestaja'].isin(institution_names).any()
        found_as_subcontractor = df_copy['hankintakoulutuksenJarjestaja'].isin(institution_names).any()
        
        if not (found_as_provider or found_as_subcontractor):
            logger.warning(f"Institution names {institution_names} not found in the data")
    
    return df_copy 