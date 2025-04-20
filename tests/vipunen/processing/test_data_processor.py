"""Tests for the data processing module."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from vipunen.processing.data_processor import (
    shorten_qualification_name,
    clean_data,
    filter_by_year,
    process_data
)

def test_shorten_qualification_name():
    """Test qualification name shortening."""
    # Test removing parentheses
    assert shorten_qualification_name(
        "Autoalan perustutkinto (Helsinki)"
    ) == "Autoalan perustutkinto"
    
    # Test removing content after comma
    assert shorten_qualification_name(
        "Autoalan perustutkinto, ajoneuvoasentaja"
    ) == "Autoalan perustutkinto"
    
    # Test truncation
    long_name = "This is a very long qualification name that should be truncated"
    shortened = shorten_qualification_name(long_name, max_length=30)
    assert len(shortened) <= 30
    assert shortened.endswith('...')
    
    # Test handling empty string
    assert shorten_qualification_name("") == ""
    
    # Test handling None
    assert shorten_qualification_name(None) == None

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'tilastovuosi': ['2020', '2021', '2022', '2020', '2021'],
        'tutkinto': [
            'Autoalan perustutkinto (Helsinki)',
            'Liiketalouden perustutkinto, merkonomi',
            'Data-analytiikan erikoistutkinto',
            'Autoalan perustutkinto (Helsinki)',  # Duplicate
            'Invalid Year'
        ],
        'opiskelijamaara': ['100', '200', '300', '100', 'invalid'],
        'paivays': ['2020-01-01', '2021-01-01', '2022-01-01', '2020-01-01', 'invalid']
    }
    return pd.DataFrame(data)

def test_clean_data(sample_data):
    """Test data cleaning functionality."""
    # Test basic cleaning
    result = clean_data(
        df=sample_data,
        required_columns=['tilastovuosi', 'tutkinto'],
        numeric_columns=['tilastovuosi', 'opiskelijamaara'],
        date_columns=['paivays']
    )
    
    # Check required columns
    assert all(col in result.columns for col in ['tilastovuosi', 'tutkinto'])
    
    # Check numeric conversion
    assert result['tilastovuosi'].dtype in ['int64', 'float64']
    assert result['opiskelijamaara'].dtype in ['int64', 'float64']
    
    # Check date conversion
    assert pd.api.types.is_datetime64_any_dtype(result['paivays'])
    
    # Check duplicate removal
    assert len(result) < len(sample_data)
    
    # Test missing columns
    with pytest.raises(ValueError):
        clean_data(
            df=sample_data,
            required_columns=['non_existent_column']
        )

def test_filter_by_year(sample_data):
    """Test year-based filtering."""
    # Clean data first
    clean_df = clean_data(
        df=sample_data,
        required_columns=['tilastovuosi'],
        numeric_columns=['tilastovuosi']
    )
    
    # Test start_year filter
    result = filter_by_year(clean_df, start_year=2021)
    assert all(year >= 2021 for year in result['tilastovuosi'])
    
    # Test end_year filter
    result = filter_by_year(clean_df, end_year=2021)
    assert all(year <= 2021 for year in result['tilastovuosi'])
    
    # Test include_latest_n_years
    result = filter_by_year(clean_df, include_latest_n_years=2)
    assert len(result['tilastovuosi'].unique()) == 2
    assert max(result['tilastovuosi']) == 2022
    
    # Test invalid year column
    with pytest.raises(ValueError):
        filter_by_year(clean_df, year_col='non_existent_column')

def test_process_data(sample_data):
    """Test complete data processing pipeline."""
    result = process_data(
        df=sample_data,
        required_columns=['tilastovuosi', 'tutkinto'],
        numeric_columns=['tilastovuosi', 'opiskelijamaara'],
        date_columns=['paivays'],
        start_year=2021,
        qualification_name_col='tutkinto',
        max_name_length=30
    )
    
    # Check that all processing steps were applied
    assert all(year >= 2021 for year in result['tilastovuosi'])
    assert all(len(name) <= 30 for name in result['tutkinto'])
    assert result['opiskelijamaara'].dtype in ['int64', 'float64']
    assert pd.api.types.is_datetime64_any_dtype(result['paivays'])
    assert len(result) < len(sample_data)  # Duplicates removed 