"""Tests for data utility functions."""
import pytest
import pandas as pd
import numpy as np
from vipunen.utils.data_utils import (
    clean_column_names,
    convert_to_numeric,
    handle_missing_values,
    filter_by_year,
    aggregate_data
)

@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'Year': [2020, 2021, 2022],
        'Student Count': [100, 150, 200],
        'Provider Name': ['A', 'B', 'C'],
        'Numeric String': ['10', '20', '30'],
        'Mixed Values': ['10', '20', 'invalid']
    })

def test_clean_column_names(sample_data):
    """Test cleaning column names."""
    cleaned = clean_column_names(sample_data)
    assert all(col == col.lower().replace(' ', '_') for col in cleaned.columns)

def test_convert_to_numeric(sample_data):
    """Test converting columns to numeric."""
    # Test with specific columns
    converted = convert_to_numeric(sample_data, columns=['Numeric String'])
    assert pd.api.types.is_numeric_dtype(converted['Numeric String'])
    
    # Test with mixed values
    converted = convert_to_numeric(sample_data, columns=['Mixed Values'])
    assert pd.api.types.is_numeric_dtype(converted['Mixed Values'])
    assert converted['Mixed Values'].isna().sum() == 1

def test_handle_missing_values():
    """Test handling missing values."""
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, 5, np.nan]
    })
    
    # Test drop method
    dropped = handle_missing_values(df, method='drop')
    assert dropped.shape[0] == 1
    
    # Test fill method
    filled = handle_missing_values(df, method='fill')
    assert filled.isna().sum().sum() == 0
    
    # Test interpolate method
    interpolated = handle_missing_values(df, method='interpolate')
    assert interpolated.isna().sum().sum() == 0

def test_filter_by_year(sample_data):
    """Test filtering by year range."""
    # Test start year
    filtered = filter_by_year(sample_data, 'Year', start_year=2021)
    assert filtered['Year'].min() == 2021
    
    # Test end year
    filtered = filter_by_year(sample_data, 'Year', end_year=2021)
    assert filtered['Year'].max() == 2021
    
    # Test both
    filtered = filter_by_year(sample_data, 'Year', start_year=2021, end_year=2021)
    assert len(filtered) == 1
    assert filtered['Year'].iloc[0] == 2021

def test_aggregate_data(sample_data):
    """Test data aggregation."""
    # Test simple sum
    aggregated = aggregate_data(
        sample_data,
        group_cols=['Provider Name'],
        value_cols=['Student Count']
    )
    assert len(aggregated) == 3
    
    # Test multiple aggregation functions
    aggregated = aggregate_data(
        sample_data,
        group_cols=['Provider Name'],
        value_cols=['Student Count'],
        agg_func=['sum', 'mean']
    )
    assert 'Student Count_sum' in aggregated.columns
    assert 'Student Count_mean' in aggregated.columns 