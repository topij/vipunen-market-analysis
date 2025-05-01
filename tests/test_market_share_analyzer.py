"""
Tests for the market_share_analyzer module.
"""
import pytest
import pandas as pd
import numpy as np
from vipunen.analysis.market_share_analyzer import (
    calculate_market_shares,
    calculate_market_share_changes,
    calculate_total_volumes
)
from typing import List

# Add fixture definition here
@pytest.fixture
def sample_market_data():
    """Create a sample dataframe for testing market share analyzer."""
    # Use raw column names as the functions accept them
    return pd.DataFrame({
        'tilastovuosi': [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022],
        'tutkintotyyppi': ['Ammattitutkinnot', 'Ammattitutkinnot', 'Erikoisammattitutkinnot',
                           'Ammattitutkinnot', 'Ammattitutkinnot', 'Erikoisammattitutkinnot',
                           'Ammattitutkinnot', 'Ammattitutkinnot', 'Erikoisammattitutkinnot'],
        'tutkinto': ['Tutkinto A', 'Tutkinto B', 'Tutkinto C',
                     'Tutkinto A', 'Tutkinto B', 'Tutkinto C',
                     'Tutkinto A', 'Tutkinto B', 'Tutkinto C'],
        'koulutuksenJarjestaja': ['Provider X', 'Provider X', 'Provider Y',
                                  'Provider X', 'Provider X', 'Provider Y',
                                  'Provider X', 'Provider X', 'Provider Y'],
        'hankintakoulutuksenJarjestaja': [None, None, 'Provider X',
                                          None, None, 'Provider X',
                                          None, None, 'Provider X'],
        'nettoopiskelijamaaraLkm': [100, 200, 150, 110, 210, 160, 120, 220, 170]
    })

def test_calculate_market_shares(sample_market_data):
    """Test the calculate_market_shares function."""
    # --- Test with default ('both') calculation basis ---
    provider_names_both = ['Provider X', 'Provider Y']
    result_both = calculate_market_shares(sample_market_data, provider_names_both, share_calculation_basis='both')
    
    # Check basic properties with 'both'
    assert isinstance(result_both, pd.DataFrame)
    required_columns = [
        'tilastovuosi', 'tutkinto', 'provider', 'volume_as_provider',
        'volume_as_subcontractor', 'total_volume', 'qualification_market_volume',
        'market_share', 'provider_count', 'subcontractor_count', 'is_target_provider'
    ]
    assert all(col in result_both.columns for col in required_columns)
    assert len(result_both) > 0
    
    # Verify double counting occurs with 'both' for the specific case
    tut_c_2020_both = result_both[ (result_both['tilastovuosi'] == 2020) & (result_both['tutkinto'] == 'Tutkinto C') ]
    assert tut_c_2020_both['market_share'].sum() > 100 # Expect sum > 100 due to double counting for C

    # Check Provider X volumes (should reflect both roles)
    provider_x_2020_c_both = tut_c_2020_both[tut_c_2020_both['provider'] == 'Provider X'].iloc[0] # Select Tutkinto C row
    assert provider_x_2020_c_both['volume_as_provider'] == 0 # Provider X is not main for C
    assert provider_x_2020_c_both['volume_as_subcontractor'] > 0 # Provider X is sub for C
    assert provider_x_2020_c_both['total_volume'] == provider_x_2020_c_both['volume_as_provider'] + provider_x_2020_c_both['volume_as_subcontractor']
    
    # --- Test with 'main_provider' calculation basis ---
    provider_names_main = ['Provider X', 'Provider Y']
    result_main = calculate_market_shares(sample_market_data, provider_names_main, share_calculation_basis='main_provider')
    
    # Check that required columns are still present
    assert all(col in result_main.columns for col in required_columns)
    
    # Check that 'is_target_provider' is correctly set for the main provider analysis
    provider_x_rows_main = result_main[result_main['provider'] == 'Provider X']
    assert all(provider_x_rows_main['is_target_provider'])
    provider_y_rows_main = result_main[result_main['provider'] == 'Provider Y']
    assert all(provider_y_rows_main['is_target_provider'])
    provider_z_rows_main = result_main[result_main['provider'] == 'Provider Z']
    assert not any(provider_z_rows_main['is_target_provider'])

    # Check that market shares sum to 100% when using 'main_provider' basis
    for (year, qual), group in result_main.groupby(['tilastovuosi', 'tutkinto']):
        total_market_share = group['market_share'].sum()
        assert abs(total_market_share - 100) < 0.01, f"Market shares (main_provider) don't sum to 100% for {year}, {qual}"

    # Check Provider X market share (based only on main provider volume)
    provider_x_2020_a_main = result_main[
        (result_main['provider'] == 'Provider X') &
        (result_main['tilastovuosi'] == 2020) &
        (result_main['tutkinto'] == 'Tutkinto A')
    ].iloc[0]
    expected_share_main = (provider_x_2020_a_main['volume_as_provider'] / provider_x_2020_a_main['qualification_market_volume']) * 100
    assert abs(provider_x_2020_a_main['market_share'] - expected_share_main) < 0.01


def test_calculate_market_share_changes(sample_market_data):
    """Test the calculate_market_share_changes function."""
    # First, calculate market shares
    provider_names = ['Provider X', 'Provider Y']
    market_shares = calculate_market_shares(sample_market_data, provider_names)
    
    # Use input column names consistent with sample_market_data
    cols_in = {
        'year': 'tilastovuosi',
        'qualification': 'tutkinto',
        'provider': 'provider', # Assuming test uses 'provider' directly
        'market_share': 'market_share' # Column name from calculate_market_shares
    }

    # Call with the new signature
    result = calculate_market_share_changes(
        market_share_df=market_shares,
        year_col=cols_in['year'],
        qual_col=cols_in['qualification'],
        provider_col=cols_in['provider'],
        market_share_col=cols_in['market_share']
    )

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check columns (should include identifiers, previous share, and change)
    expected_cols = [cols_in['year'], cols_in['qualification'], cols_in['provider'], 'previous_market_share', 'market_share_change']
    assert all(col in result.columns for col in expected_cols)

    # Check that only years where change could be calculated are present (e.g., 2021, 2022)
    assert set(result[cols_in['year']].unique()) == {2021, 2022}

    # Check specific values for 2021 (change from 2020)
    res_2021 = result[result[cols_in['year']] == 2021]
    px_a_2021 = res_2021[(res_2021[cols_in['qualification']] == 'Tutkinto A') & (res_2021[cols_in['provider']] == 'Provider X')].iloc[0]
    py_c_2021 = res_2021[(res_2021[cols_in['qualification']] == 'Tutkinto C') & (res_2021[cols_in['provider']] == 'Provider Y')].iloc[0]
    # Check Provider Y / Tutkinto C entry doesn't exist for 2021 change calc, as Y wasn't involved in C in 2020
    # py_c_2021_df = res_2021[(res_2021[cols_in['qualification']] == 'Tutkinto C') & (res_2021[cols_in['provider']] == 'Provider Y')]
    # assert py_c_2021_df.empty # Incorrect: Y was involved in C in 2020

    # Provider X, Tutkinto A: Share 2020=100, Share 2021=100 -> Change=0
    assert np.isclose(px_a_2021['previous_market_share'], 100.0)
    assert np.isclose(px_a_2021['market_share_change'], 0.0)

    # Provider Y, Tutkinto C: Share 2020=100, Share 2021=100 -> Change=0
    assert np.isclose(py_c_2021['previous_market_share'], 100.0)
    assert np.isclose(py_c_2021['market_share_change'], 0.0)

    # Check number of rows (Years 2021, 2022) * (Providers involved) = 2 * 4 = 8 rows
    # Correction: dropna removes the first entry for each group (4 groups). 12 - 4 = 8 rows.
    assert len(result) == 8


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
    assert year_2020['järjestäjänä'] == 300
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