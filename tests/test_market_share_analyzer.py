"""
Tests for the market_share_analyzer module.
"""
import pytest
import pandas as pd
import numpy as np
from src.vipunen.analysis.market_share_analyzer import (
    calculate_market_shares,
    calculate_market_share_changes,
    calculate_total_volumes
)


def test_calculate_market_shares(sample_market_data):
    """Test the calculate_market_shares function."""
    # Test with a single provider name
    provider_names = ['Provider X']
    result = calculate_market_shares(sample_market_data, provider_names)
    
    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that required columns are present
    required_columns = [
        'tilastovuosi', 'tutkinto', 'provider', 'volume_as_provider',
        'volume_as_subcontractor', 'total_volume', 'qualification_market_volume',
        'market_share', 'provider_count', 'subcontractor_count', 'is_target_provider'
    ]
    assert all(col in result.columns for col in required_columns)
    
    # Check that 'is_target_provider' is correctly set
    provider_x_rows = result[result['provider'] == 'Provider X']
    assert all(provider_x_rows['is_target_provider'])
    
    provider_y_rows = result[result['provider'] == 'Provider Y']
    assert not any(provider_y_rows['is_target_provider'])
    
    # Test with multiple provider names
    provider_names = ['Provider X', 'Provider Y']
    result = calculate_market_shares(sample_market_data, provider_names)
    
    # Check that 'is_target_provider' is correctly set for both providers
    provider_x_rows = result[result['provider'] == 'Provider X']
    assert all(provider_x_rows['is_target_provider'])
    
    provider_y_rows = result[result['provider'] == 'Provider Y']
    assert all(provider_y_rows['is_target_provider'])
    
    provider_z_rows = result[result['provider'] == 'Provider Z']
    assert not any(provider_z_rows['is_target_provider'])
    
    # Test with custom column names
    result = calculate_market_shares(
        sample_market_data,
        provider_names,
        year_col='tilastovuosi',
        qual_col='tutkinto',
        provider_col='koulutuksenJarjestaja',
        subcontractor_col='hankintakoulutuksenJarjestaja',
        value_col='nettoopiskelijamaaraLkm'
    )
    
    # Check that the result has the expected shape
    assert len(result) > 0
    
    # Check that market shares sum to 100% for each qualification and year
    for (year, qual), group in result.groupby(['tilastovuosi', 'tutkinto']):
        total_market_share = group['market_share'].sum()
        assert abs(total_market_share - 100) < 0.01, f"Market shares don't sum to 100% for {year}, {qual}"
    
    # Check that Provider X's volume includes their role as both provider and subcontractor
    provider_x_2020_a = result[
        (result['provider'] == 'Provider X') &
        (result['tilastovuosi'] == 2020) &
        (result['tutkinto'] == 'Tutkinto A')
    ].iloc[0]
    
    # In the sample data, Provider X has 100 as provider and should have some as subcontractor
    assert provider_x_2020_a['volume_as_provider'] == 100
    assert provider_x_2020_a['volume_as_subcontractor'] > 0


def test_calculate_market_share_changes(sample_market_data):
    """Test the calculate_market_share_changes function."""
    # First, calculate market shares
    provider_names = ['Provider X', 'Provider Y']
    market_shares = calculate_market_shares(sample_market_data, provider_names)
    
    # Now calculate changes between 2021 and 2020
    result = calculate_market_share_changes(market_shares, current_year=2021, previous_year=2020)
    
    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that required columns are present
    required_columns = [
        'tutkinto', 'provider', 'current_share', 'previous_share',
        'market_share_change', 'market_share_change_percent',
        'volume_change', 'volume_change_percent',
        'current_year', 'previous_year', 'gainer_rank'
    ]
    assert all(col in result.columns for col in required_columns)
    
    # Check that the years are correctly set
    assert all(result['current_year'] == 2021)
    assert all(result['previous_year'] == 2020)
    
    # Check calculations for a specific provider
    provider_x_a = result[
        (result['provider'] == 'Provider X') &
        (result['tutkinto'] == 'Tutkinto A')
    ]
    
    if not provider_x_a.empty:
        provider_x_a = provider_x_a.iloc[0]
        
        # Check that the market share change is correctly calculated
        expected_change = provider_x_a['current_share'] - provider_x_a['previous_share']
        assert abs(provider_x_a['market_share_change'] - expected_change) < 0.01
        
        # Check that the volume change is correctly calculated
        expected_volume_change = provider_x_a['current_volume'] - provider_x_a['previous_volume']
        assert provider_x_a['volume_change'] == expected_volume_change
    
    # Test with default previous_year (should be current_year - 1)
    result_default = calculate_market_share_changes(market_shares, current_year=2022)
    
    # Check that the years are correctly set
    assert all(result_default['current_year'] == 2022)
    assert all(result_default['previous_year'] == 2021)
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    result_empty = calculate_market_share_changes(empty_df, current_year=2021, previous_year=2020)
    
    # Check that the result is an empty DataFrame
    assert result_empty.empty
    
    # Test with missing years
    # Create a market share DataFrame with only one year
    single_year_df = market_shares[market_shares['tilastovuosi'] == 2020].copy()
    result_missing = calculate_market_share_changes(single_year_df, current_year=2021, previous_year=2020)
    
    # Check that the result is an empty DataFrame
    assert result_missing.empty


def test_calculate_total_volumes(sample_market_data):
    """Test the calculate_total_volumes function."""
    # Test with a single provider name
    provider_names = ['Provider X']
    result = calculate_total_volumes(sample_market_data, provider_names)
    
    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that required columns are present
    required_columns = [
        'tilastovuosi', 'järjestäjänä', 'hankintana', 'Yhteensä', 'järjestäjä_osuus (%)'
    ]
    assert all(col in result.columns for col in required_columns)
    
    # Check that there is one row per year
    assert len(result) == len(sample_market_data['tilastovuosi'].unique())
    
    # Check calculations for a specific year
    year_2020 = result[result['tilastovuosi'] == 2020].iloc[0]
    
    # Provider X in 2020 has 100 as provider and should have some as subcontractor
    # from sample_market_data
    assert year_2020['järjestäjänä'] == 100
    assert year_2020['hankintana'] > 0
    assert year_2020['Yhteensä'] == year_2020['järjestäjänä'] + year_2020['hankintana']
    
    # Check that järjestäjä_osuus is correctly calculated
    expected_percentage = year_2020['järjestäjänä'] / year_2020['Yhteensä'] * 100
    assert abs(year_2020['järjestäjä_osuus (%)'] - expected_percentage) < 0.01
    
    # Test with multiple provider names
    provider_names = ['Provider X', 'Provider Y']
    result_multi = calculate_total_volumes(sample_market_data, provider_names)
    
    # Volume should be higher with two providers
    assert result_multi.iloc[0]['Yhteensä'] > result.iloc[0]['Yhteensä']
    
    # Test with custom column names
    result_custom = calculate_total_volumes(
        sample_market_data,
        provider_names,
        year_col='tilastovuosi',
        provider_col='koulutuksenJarjestaja',
        subcontractor_col='hankintakoulutuksenJarjestaja',
        value_col='nettoopiskelijamaaraLkm'
    )
    
    # Should be identical to result_multi
    assert result_custom.equals(result_multi)
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    result_empty = calculate_total_volumes(empty_df, provider_names)
    
    # Should return an empty DataFrame with correct columns
    assert result_empty.empty
    assert all(col in result_empty.columns for col in required_columns if col in result_empty.columns) 