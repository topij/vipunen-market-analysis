"""Data processing module for Vipunen data."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def shorten_qualification_name(name: str, max_length: int = 50) -> str:
    """Shorten qualification name while preserving key information.
    
    Args:
        name: Original qualification name
        max_length: Maximum length for the shortened name
        
    Returns:
        Shortened qualification name
    """
    try:
        # Remove common suffixes and prefixes
        name = re.sub(r'\s*\([^)]*\)', '', name)  # Remove parentheses and their contents
        name = re.sub(r'\s*,.*$', '', name)  # Remove everything after comma
        
        # If still too long, truncate and add ellipsis
        if len(name) > max_length:
            name = name[:max_length-3] + '...'
            
        return name.strip()
        
    except Exception as e:
        logger.error(f"Error shortening qualification name: {e}")
        return name

def clean_data(
    df: pd.DataFrame,
    required_columns: List[str],
    numeric_columns: Optional[List[str]] = None,
    date_columns: Optional[List[str]] = None,
    drop_duplicates: bool = True,
    drop_na: bool = True
) -> pd.DataFrame:
    """Clean and validate the input data.
    
    Args:
        df: Input DataFrame
        required_columns: List of columns that must be present
        numeric_columns: List of columns to convert to numeric
        date_columns: List of columns to convert to datetime
        drop_duplicates: Whether to drop duplicate rows
        drop_na: Whether to drop rows with NA values in required columns
        
    Returns:
        Cleaned DataFrame
    """
    try:
        # Validate required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        result = df.copy()
        
        # Convert numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors='coerce')
                    
        # Convert date columns
        if date_columns:
            for col in date_columns:
                if col in result.columns:
                    result[col] = pd.to_datetime(result[col], errors='coerce')
                    
        # Drop duplicates if requested
        if drop_duplicates:
            result = result.drop_duplicates()
            
        # Drop NA values in required columns if requested
        if drop_na:
            result = result.dropna(subset=required_columns)
            
        return result
        
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise

def filter_by_year(
    df: pd.DataFrame,
    year_col: str = 'tilastovuosi',
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    include_latest_n_years: Optional[int] = None
) -> pd.DataFrame:
    """Filter data by year range.
    
    Args:
        df: Input DataFrame
        year_col: Name of the year column
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        include_latest_n_years: Include only the latest N years
        
    Returns:
        Filtered DataFrame
    """
    try:
        if year_col not in df.columns:
            raise ValueError(f"Year column '{year_col}' not found in DataFrame")
            
        result = df.copy()
        
        # Convert year column to numeric if it isn't already
        result[year_col] = pd.to_numeric(result[year_col], errors='coerce')
        
        # Get min and max years from data
        min_year = result[year_col].min()
        max_year = result[year_col].max()
        
        # If include_latest_n_years is specified, it overrides start_year
        if include_latest_n_years is not None:
            start_year = max_year - include_latest_n_years + 1
            
        # Apply year filters
        if start_year is not None:
            result = result[result[year_col] >= start_year]
        if end_year is not None:
            result = result[result[year_col] <= end_year]
            
        if len(result) == 0:
            logger.warning(
                f"No data found for year range {start_year or min_year} - {end_year or max_year}"
            )
            
        return result
        
    except Exception as e:
        logger.error(f"Error filtering by year: {e}")
        raise

def process_data(
    df: pd.DataFrame,
    required_columns: List[str],
    numeric_columns: Optional[List[str]] = None,
    date_columns: Optional[List[str]] = None,
    year_col: str = 'tilastovuosi',
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    include_latest_n_years: Optional[int] = None,
    qualification_name_col: Optional[str] = None,
    max_name_length: int = 50
) -> pd.DataFrame:
    """Process data by applying cleaning, filtering, and name shortening.
    
    Args:
        df: Input DataFrame
        required_columns: List of columns that must be present
        numeric_columns: List of columns to convert to numeric
        date_columns: List of columns to convert to datetime
        year_col: Name of the year column
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        include_latest_n_years: Include only the latest N years
        qualification_name_col: Column containing qualification names to shorten
        max_name_length: Maximum length for shortened qualification names
        
    Returns:
        Processed DataFrame
    """
    try:
        # Clean data
        result = clean_data(
            df=df,
            required_columns=required_columns,
            numeric_columns=numeric_columns,
            date_columns=date_columns
        )
        
        # Filter by year
        result = filter_by_year(
            df=result,
            year_col=year_col,
            start_year=start_year,
            end_year=end_year,
            include_latest_n_years=include_latest_n_years
        )
        
        # Shorten qualification names if specified
        if qualification_name_col and qualification_name_col in result.columns:
            result[qualification_name_col] = result[qualification_name_col].apply(
                lambda x: shorten_qualification_name(x, max_name_length)
            )
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise 