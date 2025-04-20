"""Tests for the market analysis module."""
import pytest
import pandas as pd
import numpy as np
from vipunen.analysis.market import (
    calculate_market_shares,
    calculate_provider_counts,
    calculate_growth_trends,
    analyze_market
)

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    data = {
        'tilastovuosi': [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022],
        'tutkinto': ['A', 'A', 'B', 'A', 'A', 'B', 'A', 'A', 'B'],
        'koulutuksenJarjestaja': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z'],
        'nettoopiskelijamaaraLkm': [100, 200, 300, 150, 250, 350, 200, 300, 400]
    }
    return pd.DataFrame(data)

def test_calculate_market_shares(sample_market_data):
    """Test market share calculation."""
    result = calculate_market_shares(
        df=sample_market_data,
        group_cols=['tilastovuosi', 'tutkinto'],
        value_col='nettoopiskelijamaaraLkm',
        provider_col='koulutuksenJarjestaja'
    )
    
    # Check structure
    assert all(col in result.columns for col in [
        'tilastovuosi', 'tutkinto', 'koulutuksenJarjestaja',
        'nettoopiskelijamaaraLkm', 'market_total', 'market_share'
    ])
    
    # Check calculations for 2020, tutkinto A
    year_2020_a = result[
        (result['tilastovuosi'] == 2020) & 
        (result['tutkinto'] == 'A')
    ]
    assert len(year_2020_a) == 2  # Two providers
    assert year_2020_a['market_total'].iloc[0] == 300  # 100 + 200
    assert year_2020_a['market_share'].round(2).tolist() == [33.33, 66.67]

def test_calculate_provider_counts(sample_market_data):
    """Test provider count calculation."""
    result = calculate_provider_counts(
        df=sample_market_data,
        group_cols=['tilastovuosi', 'tutkinto'],
        provider_col='koulutuksenJarjestaja'
    )
    
    # Check structure
    assert all(col in result.columns for col in [
        'tilastovuosi', 'tutkinto', 'provider_count'
    ])
    
    # Check counts
    assert result['provider_count'].tolist() == [2, 1, 2, 1, 2, 1]

def test_calculate_growth_trends(sample_market_data):
    """Test growth trend calculation."""
    result = calculate_growth_trends(
        df=sample_market_data,
        group_cols=['tutkinto'],
        value_col='nettoopiskelijamaaraLkm',
        year_col='tilastovuosi'
    )

    # Check structure
    assert all(col in result.columns for col in [
        'tutkinto', 'cagr', 'start_year', 'end_year'
    ])
    
    # Check values
    assert len(result) == 2  # One row per tutkinto
    assert result['cagr'].notna().all()
    assert result['start_year'].min() == 2020
    assert result['end_year'].max() == 2022

def test_analyze_market(sample_market_data):
    """Test comprehensive market analysis."""
    result = analyze_market(
        df=sample_market_data,
        group_cols=['tutkinto'],
        value_col='nettoopiskelijamaaraLkm',
        provider_col='koulutuksenJarjestaja',
        year_col='tilastovuosi'
    )

    # Check all components are present
    assert all(key in result for key in [
        'market_shares', 'provider_counts', 'growth_trends'
    ])

    # Check market shares
    assert len(result['market_shares']) == 3  # One row per unique tutkinto-provider combination
    assert all(col in result['market_shares'].columns for col in [
        'tutkinto', 'koulutuksenJarjestaja', 'nettoopiskelijamaaraLkm',
        'market_total', 'market_share'
    ])

    # Check provider counts
    assert len(result['provider_counts']) == 2  # One row per tutkinto
    assert all(col in result['provider_counts'].columns for col in [
        'tutkinto', 'provider_count'
    ])

    # Check growth trends
    assert len(result['growth_trends']) == 2  # One row per tutkinto
    assert all(col in result['growth_trends'].columns for col in [
        'tutkinto', 'cagr', 'start_year', 'end_year'
    ])

def test_market_shares_with_provider_list(sample_market_data):
    """Test market share calculation with specific providers."""
    result = calculate_market_shares(
        df=sample_market_data,
        group_cols=['tilastovuosi', 'tutkinto'],
        value_col='nettoopiskelijamaaraLkm',
        provider_col='koulutuksenJarjestaja',
        provider_list=['X', 'Y']
    )
    
    # Check only specified providers are included
    assert set(result['koulutuksenJarjestaja'].unique()) == {'X', 'Y'}
    
    # Check calculations for 2020, tutkinto A
    year_2020_a = result[
        (result['tilastovuosi'] == 2020) & 
        (result['tutkinto'] == 'A')
    ]
    assert year_2020_a['market_share'].round(2).tolist() == [33.33, 66.67]

def test_growth_trends_edge_cases():
    """Test growth trend calculation with edge cases."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    result = calculate_growth_trends(
        df=empty_df,
        group_cols=['tutkinto'],
        value_col='nettoopiskelijamaaraLkm',
        year_col='tilastovuosi'
    )
    assert result.empty
    assert all(col in result.columns for col in [
        'tutkinto', 'cagr', 'start_year', 'end_year'
    ]) 