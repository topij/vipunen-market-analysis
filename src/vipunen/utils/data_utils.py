"""Data utility functions for the Vipunen project."""
import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by converting to lowercase and replacing spaces with underscores.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame with cleaned column names
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def convert_to_numeric(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Convert specified columns to numeric type.
    
    Args:
        df: DataFrame to convert
        columns: List of columns to convert. If None, converts all numeric columns.
        
    Returns:
        DataFrame with converted columns
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
        
    for col in columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            logger.warning(f"Could not convert column {col} to numeric: {e}")
            
    return df

def handle_missing_values(df: pd.DataFrame, 
                        method: str = 'drop',
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Handle missing values in the DataFrame.
    
    Args:
        df: DataFrame to process
        method: Method to handle missing values ('drop', 'fill', 'interpolate')
        columns: List of columns to process. If None, processes all columns.
        
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns
        
    if method == 'drop':
        df = df.dropna(subset=columns)
    elif method == 'fill':
        df[columns] = df[columns].fillna(0)
    elif method == 'interpolate':
        df[columns] = df[columns].interpolate()
    else:
        raise ValueError(f"Invalid method: {method}")
        
    return df

def filter_by_year(df: pd.DataFrame, 
                  year_col: str,
                  start_year: Optional[int] = None,
                  end_year: Optional[int] = None) -> pd.DataFrame:
    """Filter DataFrame by year range.
    
    Args:
        df: DataFrame to filter
        year_col: Name of the year column
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        
    Returns:
        Filtered DataFrame
    """
    if start_year is not None:
        df = df[df[year_col] >= start_year]
    if end_year is not None:
        df = df[df[year_col] <= end_year]
        
    return df

def aggregate_data(df: pd.DataFrame, group_cols: List[str], value_cols: List[str], agg_func: Union[str, List[str]] = 'sum') -> pd.DataFrame:
    """
    Aggregate data by specified columns.
    
    Args:
        df: Input DataFrame
        group_cols: List of columns to group by
        value_cols: List of columns to aggregate
        agg_func: Aggregation function(s) to apply
        
    Returns:
        Aggregated DataFrame
    """
    try:
        # Convert single aggregation function to list
        if isinstance(agg_func, str):
            agg_func = [agg_func]
            
        # Create aggregation dictionary
        agg_dict = {col: agg_func for col in value_cols}
        
        # Perform aggregation
        aggregated = df.groupby(group_cols).agg(agg_dict)
        
        # Flatten MultiIndex columns
        aggregated.columns = [f"{col[0]}_{col[1]}" for col in aggregated.columns]
        
        return aggregated.reset_index()
        
    except Exception as e:
        logger.error(f"Error aggregating data: {str(e)}")
        raise 