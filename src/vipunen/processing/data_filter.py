"""Data filtering module for Vipunen data processing."""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
import re

logger = logging.getLogger(__name__)

def filter_degree_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to include only vocational and specialist vocational degrees.
    
    Args:
        df: Input DataFrame with raw data
        
    Returns:
        DataFrame filtered to include only relevant degree types
    """
    try:
        # Define valid degree types
        valid_types = [
            'ammattitutkinto',
            'erikoisammattitutkinto'
        ]
        
        # Filter data
        filtered = df[df['tutkintotyyppi'].isin(valid_types)].copy()
        
        logger.info(f"Filtered data to {len(filtered)} rows with valid degree types")
        return filtered
        
    except Exception as e:
        logger.error(f"Error filtering degree types: {str(e)}")
        raise

def standardize_provider_name(name: str) -> str:
    """
    Standardize provider name by removing common variations.
    
    Args:
        name: Provider name to standardize
        
    Returns:
        Standardized provider name
    """
    try:
        if pd.isna(name):
            return name
            
        # Convert to lowercase and remove extra spaces
        name = str(name).lower().strip()
        
        # Remove common suffixes and prefixes
        name = re.sub(r'\s+(oy|ab|ry|oyj|ky|oy ab)$', '', name)
        name = re.sub(r'^(oy|ab|ry|oyj|ky|oy ab)\s+', '', name)
        
        # Replace hyphens with spaces
        name = name.replace('-', ' ')
        
        # Remove special characters
        name = re.sub(r'[^\w\s]', '', name)
        
        # Remove extra spaces
        name = ' '.join(name.split())
        
        return name
        
    except Exception as e:
        logger.error(f"Error standardizing provider name '{name}': {str(e)}")
        return name

def standardize_provider_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize provider names in both main provider and subcontractor columns.
    
    Args:
        df: Input DataFrame with provider names
        
    Returns:
        DataFrame with standardized provider names
    """
    try:
        df = df.copy()
        
        # Standardize main provider names
        df['koulutuksenJarjestaja'] = df['koulutuksenJarjestaja'].apply(standardize_provider_name)
        
        # Standardize subcontractor names
        df['hankintakoulutuksenJarjestaja'] = df['hankintakoulutuksenJarjestaja'].apply(standardize_provider_name)
        
        logger.info("Provider names standardized")
        return df
        
    except Exception as e:
        logger.error(f"Error standardizing provider names: {str(e)}")
        raise

def filter_by_provider(df: pd.DataFrame, provider: str, 
                      provider_variations: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filter data for a specific provider, including variations of the provider name.
    
    Args:
        df: Input DataFrame with standardized provider names
        provider: Main provider name to filter for
        provider_variations: Optional list of provider name variations
        
    Returns:
        DataFrame filtered for the specified provider
    """
    try:
        # Standardize the main provider name
        provider = standardize_provider_name(provider)
        
        # Create list of provider variations
        variations = [provider]
        if provider_variations:
            variations.extend([standardize_provider_name(v) for v in provider_variations])
        
        # Standardize provider names in the DataFrame
        df = df.copy()
        df['koulutuksenJarjestaja_std'] = df['koulutuksenJarjestaja'].apply(standardize_provider_name)
        df['hankintakoulutuksenJarjestaja_std'] = df['hankintakoulutuksenJarjestaja'].apply(standardize_provider_name)
        
        # Filter data
        filtered = df[
            (df['koulutuksenJarjestaja_std'].isin(variations)) |
            (df['hankintakoulutuksenJarjestaja_std'].isin(variations))
        ].copy()
        
        # Drop temporary columns
        filtered = filtered.drop(['koulutuksenJarjestaja_std', 'hankintakoulutuksenJarjestaja_std'], axis=1)
        
        logger.info(f"Filtered data to {len(filtered)} rows for provider: {provider}")
        return filtered
        
    except Exception as e:
        logger.error(f"Error filtering by provider: {str(e)}")
        raise

def filter_data(df: pd.DataFrame, provider: str, 
               provider_variations: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Apply all data filtering operations.
    
    Args:
        df: Input DataFrame with raw data
        provider: Main provider name to filter for
        provider_variations: Optional list of provider name variations
        
    Returns:
        DataFrame with all filtering operations applied
    """
    try:
        # Apply all filtering functions
        df = filter_degree_types(df)
        df = standardize_provider_names(df)
        df = filter_by_provider(df, provider, provider_variations)
        
        logger.info("Data filtering completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Error in data filtering: {str(e)}")
        raise 