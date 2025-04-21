"""Data cleaning module for Vipunen data processing."""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize column names in the DataFrame.
    
    Args:
        df: Input DataFrame with raw column names
        
    Returns:
        DataFrame with cleaned column names
    """
    try:
        # Create a mapping of old to new column names
        column_mapping = {
            'tilastovuosi': 'year',
            'suorituksenTyyppi': 'completion_type',
            'tutkintotyyppi': 'degree_type',
            'tutkinto': 'degree',
            'koulutuksenJarjestaja': 'main_provider',
            'hankintakoulutuksenJarjestaja': 'subcontractor',
            'hankintakoulutusKyllaEi': 'has_subcontractor',
            'koodiTutkinto': 'degree_code',
            'koodiKoulutuksenJarjestaja': 'main_provider_code',
            'koodiHankintakoulutuksenJarjestaja': 'subcontractor_code',
            'uudetOpiskelijatLkm': 'new_students',
            'opiskelijatLkm': 'total_students',
            'tutkinnonSuorittaneetLkm': 'graduates',
            'nettoopiskelijamaaraLkm': 'net_students',
            'tietojoukkoPaivitettyPvm': 'dataset_updated'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        logger.info("Column names cleaned and standardized")
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning column names: {str(e)}")
        raise

def clean_degree_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize degree types.
    
    Args:
        df: Input DataFrame with degree type information
        
    Returns:
        DataFrame with cleaned degree types
    """
    try:
        # Define valid degree types
        valid_degree_types = {
            'Ammatilliset perustutkinnot': 'basic_vocational',
            'Erikoisammattitutkinnot': 'specialist_vocational',
            'Ammattitutkinnot': 'vocational',
            'Muu ammatillinen koulutus': 'other_vocational',
            'VALMA': 'valma',
            'TELMA': 'telma',
            'TUVA': 'tuva'
        }
        
        # Map degree types
        df['degree_type'] = df['degree_type'].map(valid_degree_types)
        
        # Filter out invalid degree types
        df = df[df['degree_type'].notna()]
        
        logger.info("Degree types cleaned and standardized")
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning degree types: {str(e)}")
        raise

def clean_provider_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize provider names.
    
    Args:
        df: Input DataFrame with provider names
        
    Returns:
        DataFrame with cleaned provider names
    """
    try:
        # Remove extra whitespace
        df['main_provider'] = df['main_provider'].str.strip()
        df['subcontractor'] = df['subcontractor'].str.strip()
        
        # Replace empty strings with NaN
        df['subcontractor'] = df['subcontractor'].replace('', np.nan)
        
        logger.info("Provider names cleaned and standardized")
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning provider names: {str(e)}")
        raise

def clean_student_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize student number columns.
    
    Args:
        df: Input DataFrame with student numbers
        
    Returns:
        DataFrame with cleaned student numbers
    """
    try:
        # Convert to numeric, replacing non-numeric values with NaN
        numeric_columns = ['new_students', 'total_students', 'graduates', 'net_students']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Replace negative values with NaN
        for col in numeric_columns:
            df[col] = df[col].where(df[col] >= 0, np.nan)
        
        logger.info("Student numbers cleaned and standardized")
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning student numbers: {str(e)}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform all data cleaning operations.
    
    Args:
        df: Input DataFrame with raw data
        
    Returns:
        DataFrame with cleaned data
    """
    try:
        # Apply all cleaning functions
        df = clean_column_names(df)
        df = clean_degree_types(df)
        df = clean_provider_names(df)
        df = clean_student_numbers(df)
        
        logger.info("Data cleaning completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Error in data cleaning: {str(e)}")
        raise 