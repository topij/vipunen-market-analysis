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
    """Create sample data for testing."""
    data = {
        'Column Name': [1, 2, 3],
        'Another Column': ['4', '5', 'invalid'],
        'Missing Data': [1.0, np.nan, 3.0],
        'Year': [2020, 2021, 2022],
        'Value': [100, 200, 300]
    }
    return pd.DataFrame(data)

def test_clean_column_names(sample_data):
    """Test column name cleaning."""
    result = clean_column_names(sample_data)
    expected_columns = {
        'column_name',
        'another_column',
        'missing_data',
        'year',
        'value'
    }
    assert set(result.columns) == expected_columns

def test_convert_to_numeric(sample_data):
    """Test numeric conversion."""
    result = convert_to_numeric(sample_data, ['Another Column'])
    assert pd.api.types.is_numeric_dtype(result['Another Column'])
    assert pd.isna(result.loc[2, 'Another Column'])  # 'invalid' should be converted to NaN
    assert result.loc[0, 'Another Column'] == 4.0

def test_handle_missing_values(sample_data):
    """Test missing value handling."""
    # Test drop method
    result_drop = handle_missing_values(sample_data.copy(), method='drop')
    assert len(result_drop) == 2  # One row had NaN
    
    # Test fill method
    result_fill = handle_missing_values(sample_data.copy(), method='fill')
    assert result_fill['Missing Data'].isna().sum() == 0
    assert result_fill.loc[1, 'Missing Data'] == 0
    
    # Test interpolate method
    result_interp = handle_missing_values(sample_data.copy(), method='interpolate')
    assert result_interp['Missing Data'].isna().sum() == 0
    assert result_interp.loc[1, 'Missing Data'] == 2.0  # Should be interpolated between 1 and 3

def test_filter_by_year(sample_data):
    """Test year filtering."""
    result = filter_by_year(
        df=sample_data,
        year_col='Year',
        start_year=2021,
        end_year=2022
    )
    assert len(result) == 2
    assert result['Year'].min() == 2021
    assert result['Year'].max() == 2022

def test_aggregate_data():
    """Test data aggregation."""
    data = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'subgroup': ['X', 'Y', 'X', 'Y'],
        'value1': [1, 2, 3, 4],
        'value2': [10, 20, 30, 40]
    })
    
    result = aggregate_data(
        df=data,
        group_cols=['group'],
        value_cols=['value1', 'value2'],
        agg_func=['sum', 'mean']
    )
    
    # Check structure
    expected_columns = {
        'group',
        'value1_sum', 'value1_mean',
        'value2_sum', 'value2_mean'
    }
    assert set(result.columns) == expected_columns
    
    # Check values
    group_a = result[result['group'] == 'A'].iloc[0]
    assert group_a['value1_sum'] == 3  # 1 + 2
    assert group_a['value1_mean'] == 1.5  # (1 + 2) / 2
    assert group_a['value2_sum'] == 30  # 10 + 20
    assert group_a['value2_mean'] == 15  # (10 + 20) / 2 