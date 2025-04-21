"""Tests for the data aggregator module."""
import pytest
import pandas as pd
import numpy as np
from vipunen.processing.data_aggregator import (
    aggregate_by_provider,
    calculate_market_shares,
    calculate_growth_rates,
    aggregate_market_data
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'year': [2020, 2020, 2021, 2021, 2022, 2022] * 2,
        'degree': ['A', 'B', 'A', 'B', 'A', 'B'] * 2,
        'main_provider': ['Provider1', 'Provider1', 'Provider1', 'Provider1', 'Provider1', 'Provider1'] + 
                        ['Provider2', 'Provider2', 'Provider2', 'Provider2', 'Provider2', 'Provider2'],
        'subcontractor': ['Provider2', 'Provider2', 'Provider2', 'Provider2', 'Provider2', 'Provider2'] + 
                        ['Provider1', 'Provider1', 'Provider1', 'Provider1', 'Provider1', 'Provider1'],
        'net_students': [100, 200, 150, 250, 200, 300] * 2
    }
    return pd.DataFrame(data)

def test_aggregate_by_provider(sample_data):
    """Test provider data aggregation."""
    result = aggregate_by_provider(sample_data, 'Provider1')
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['year', 'degree', 'net_students', 'is_main_provider', 'is_subcontractor'])
    
    # Check aggregation
    assert len(result) == 6  # 3 years * 2 degrees
    assert result['net_students'].sum() == 2400  # Total students for Provider1 (counting both roles)
    
    # Check role indicators
    assert result['is_main_provider'].any()
    assert result['is_subcontractor'].any()

def test_calculate_market_shares(sample_data):
    """Test market share calculations."""
    result = calculate_market_shares(sample_data, 'Provider1')
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['year', 'degree', 'net_students', 'total_market', 'market_share'])
    
    # Check calculations
    assert result['market_share'].between(0, 100).all()
    assert result['total_market'].sum() == 2400  # Total market size
    
    # Check specific market share
    provider1_share = result[
        (result['year'] == 2020) & 
        (result['degree'] == 'A')
    ]['market_share'].iloc[0]
    assert abs(provider1_share - 50.0) < 0.01  # 100/200 = 50%

def test_calculate_growth_rates(sample_data):
    """Test growth rate calculations."""
    yoy_growth, cagr = calculate_growth_rates(sample_data)
    
    # Check YoY growth
    assert isinstance(yoy_growth, pd.DataFrame)
    assert 'yoy_growth' in yoy_growth.columns
    
    # Check CAGR
    assert isinstance(cagr, pd.DataFrame)
    assert 'cagr' in cagr.columns
    assert len(cagr) == 2  # One CAGR per degree
    
    # Check specific growth rates
    degree_a_growth = yoy_growth[
        (yoy_growth['degree'] == 'A') & 
        (yoy_growth['year'] == 2021)
    ]['yoy_growth'].iloc[0]
    assert abs(degree_a_growth - 50.0) < 0.1  # (150-100)/100 = 50%

def test_aggregate_market_data(sample_data):
    """Test comprehensive market data aggregation."""
    results = aggregate_market_data(sample_data, 'Provider1')
    
    # Check result structure
    assert isinstance(results, dict)
    assert all(key in results for key in ['market_shares', 'yoy_growth', 'cagr', 'provider_data'])
    
    # Check each component
    assert isinstance(results['market_shares'], pd.DataFrame)
    assert isinstance(results['yoy_growth'], pd.DataFrame)
    assert isinstance(results['cagr'], pd.DataFrame)
    assert isinstance(results['provider_data'], pd.DataFrame)
    
    # Check data consistency
    assert len(results['market_shares']) == len(results['yoy_growth'])
    assert len(results['cagr']) == 2  # One CAGR per degree 