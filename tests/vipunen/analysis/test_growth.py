"""Tests for the growth analysis module."""
import pytest
import pandas as pd
import numpy as np
from vipunen.analysis.growth import (
    calculate_cagr,
    calculate_yoy_growth,
    calculate_multiple_yoy_growth
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'tilastovuosi': [2018, 2019, 2020, 2021, 2022] * 2,
        'tutkinto': ['A'] * 5 + ['B'] * 5,
        'nettoopiskelijamaaraLkm_as_jarjestaja': [
            100, 110, 121, 133, 146,  # Growth for A
            200, 190, 180, 170, 160   # Decline for B
        ]
    }
    return pd.DataFrame(data)

def test_calculate_cagr(sample_data):
    """Test CAGR calculation."""
    result = calculate_cagr(
        df=sample_data,
        groupby_columns=['tutkinto'],
        value_column='nettoopiskelijamaaraLkm_as_jarjestaja'
    )
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert 'tutkinto' in result.columns
    assert 'CAGR (%)' in result.columns
    
    # Check values
    a_cagr = result[result['tutkinto'] == 'A']['CAGR (%)'].iloc[0]
    b_cagr = result[result['tutkinto'] == 'B']['CAGR (%)'].iloc[0]
    
    # A should have positive growth
    assert a_cagr > 0
    # B should have negative growth
    assert b_cagr < 0

def test_calculate_yoy_growth(sample_data):
    """Test YoY growth calculation."""
    result = calculate_yoy_growth(
        df=sample_data,
        groupby_col='tutkinto',
        target_col='nettoopiskelijamaaraLkm_as_jarjestaja',
        output_col='kasvu',
        end_year=2022,
        time_window=3
    )
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert 'tutkinto' in result.columns
    assert 'kasvu' in result.columns
    assert 'kasvu_trendi' in result.columns
    
    # Check values
    a_growth = result[result['tutkinto'] == 'A']['kasvu'].iloc[0]
    b_growth = result[result['tutkinto'] == 'B']['kasvu'].iloc[0]
    
    # A should have positive growth
    assert a_growth > 0
    # B should have negative growth
    assert b_growth < 0

def test_calculate_multiple_yoy_growth(sample_data):
    """Test multiple YoY growth calculation."""
    variables_dict = {
        'nettoopiskelijamaaraLkm_as_jarjestaja': 'opiskelijoiden_maaran_kasvu'
    }
    
    result = calculate_multiple_yoy_growth(
        df=sample_data,
        variables_dict=variables_dict,
        time_window=3,
        groupby_col='tutkinto'
    )
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert 'tutkinto' in result.columns
    assert 'opiskelijoiden_maaran_kasvu' in result.columns
    assert 'opiskelijoiden_maaran_kasvu_trendi' in result.columns
    
    # Check values
    a_growth = result[result['tutkinto'] == 'A']['opiskelijoiden_maaran_kasvu'].iloc[0]
    b_growth = result[result['tutkinto'] == 'B']['opiskelijoiden_maaran_kasvu'].iloc[0]
    
    # A should have positive growth
    assert a_growth > 0
    # B should have negative growth
    assert b_growth < 0

def test_cagr_edge_cases():
    """Test CAGR calculation with edge cases."""
    # Test with zero values
    zero_data = pd.DataFrame({
        'tilastovuosi': [2018, 2019, 2020],
        'tutkinto': ['A'] * 3,
        'nettoopiskelijamaaraLkm_as_jarjestaja': [0, 0, 0]
    })
    
    result = calculate_cagr(
        df=zero_data,
        groupby_columns=['tutkinto'],
        value_column='nettoopiskelijamaaraLkm_as_jarjestaja'
    )
    assert result['CAGR (%)'].iloc[0] == 0
    
    # Test with single value
    single_data = pd.DataFrame({
        'tilastovuosi': [2018],
        'tutkinto': ['A'],
        'nettoopiskelijamaaraLkm_as_jarjestaja': [100]
    })
    
    result = calculate_cagr(
        df=single_data,
        groupby_columns=['tutkinto'],
        value_column='nettoopiskelijamaaraLkm_as_jarjestaja'
    )
    assert result['CAGR (%)'].iloc[0] == 0 