"""Tests for the market analysis module."""
import pytest
import pandas as pd
import numpy as np
from vipunen.analysis.market import (
    calculate_market_shares,
    calculate_provider_counts,
    calculate_growth_trends,
    analyze_market,
    calculate_provider_rankings,
    analyze_provider_roles,
    track_market_shares,
    filter_qualification_types,
    calculate_total_students,
    get_provider_qualifications,
    analyze_qualification_volumes,
    calculate_year_over_year_changes,
    analyze_market_share_changes,
    calculate_market_concentration
)

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    data = {
        'tilastovuosi': [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022],
        'tutkinto': ['A', 'A', 'B', 'A', 'A', 'B', 'A', 'A', 'B'],
        'tutkintotyyppi': ['ammattitutkinto', 'ammattitutkinto', 'erikoisammattitutkinto',
                          'ammattitutkinto', 'ammattitutkinto', 'erikoisammattitutkinto',
                          'ammattitutkinto', 'ammattitutkinto', 'erikoisammattitutkinto'],
        'koulutuksenJarjestaja': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z'],
        'hankintakoulutuksenJarjestaja': ['Y', 'X', 'X', 'Y', 'X', 'X', 'Y', 'X', 'X'],
        'nettoopiskelijamaaraLkm': [100, 200, 300, 150, 250, 350, 200, 300, 400]
    }
    return pd.DataFrame(data)

def test_calculate_market_shares(sample_market_data):
    """Test market share calculations."""
    # Calculate market shares
    result = calculate_market_shares(sample_market_data)
    
    # Check structure
    assert all(col in result.columns for col in [
        'provider', 'volume', 'market_share', 'rank'
    ])
    
    # Check calculations for provider X and Y
    provider_x = result[result['provider'] == 'X'].iloc[0]
    provider_y = result[result['provider'] == 'Y'].iloc[0]
    
    # Check volumes
    assert provider_x['volume'] == 450  # 100 + 150 + 200
    assert provider_y['volume'] == 750  # 200 + 250 + 300
    
    # Check market shares sum to 100%
    assert abs(result['market_share'].sum() - 100) < 0.1
    
    # Check rankings
    assert provider_y['rank'] == 1  # Y has higher volume
    assert provider_x['rank'] == 2  # X has lower volume

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
        year_col='tilastovuosi',
        value_col='nettoopiskelijamaaraLkm',
        group_cols=['tutkinto']
    )

    # Check all components are present
    assert all(key in result for key in [
        'market_shares', 'provider_counts', 'growth_trends'
    ])

    # Check market shares
    market_shares = result['market_shares']
    assert len(market_shares) == 3  # X, Y, and Z
    assert all(col in market_shares.columns for col in [
        'provider', 'volume', 'market_share', 'rank'
    ])

    # Check provider counts
    assert len(result['provider_counts']) == 2  # One row per tutkinto
    assert 'provider_count' in result['provider_counts'].columns

    # Check growth trends
    assert len(result['growth_trends']) == 2  # One row per tutkinto
    assert 'cagr' in result['growth_trends'].columns

def test_market_shares_with_provider_list(sample_market_data):
    """Test market share calculation with specific providers."""
    result = calculate_market_shares(
        df=sample_market_data,
        group_cols=['tilastovuosi', 'tutkinto'],
        value_col='nettoopiskelijamaaraLkm',
        provider_list=['X', 'Y']
    )
    
    # Check only specified providers are included
    assert set(result['provider'].unique()) == {'X', 'Y'}
    
    # Check calculations for 2020, tutkinto A
    year_2020_a = result[
        (result['tilastovuosi'] == 2020) & 
        (result['tutkinto'] == 'A')
    ]
    assert year_2020_a['market_share'].round(2).tolist() == [66.67, 33.33]

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

def test_calculate_market_shares():
    """Test market share calculations."""
    # Create sample data
    data = {
        'tilastovuosi': [2020, 2020, 2020, 2021, 2021, 2021],
        'tutkinto': ['A', 'A', 'B', 'A', 'A', 'B'],
        'koulutuksenJarjestaja': ['X', 'Y', 'X', 'X', 'Y', 'X'],
        'nettoopiskelijamaaraLkm': [100, 200, 150, 120, 180, 160]
    }
    df = pd.DataFrame(data)
    
    # Calculate market shares
    result = calculate_market_shares(
        df=df,
        group_cols=['tilastovuosi', 'tutkinto'],
        value_col='nettoopiskelijamaaraLkm'
    )
    
    # Verify results
    assert len(result) == 6
    assert 'market_share' in result.columns
    
    # Check that each group sums to 100%
    group_sums = result.groupby(['tilastovuosi', 'tutkinto'])['market_share'].sum()
    for group_sum in group_sums:
        assert abs(group_sum - 100.0) < 0.01, f"Group sum {group_sum} is not 100%"
    
    # Check specific values for 2020, tutkinto A
    year_2020_a = result[
        (result['tilastovuosi'] == 2020) & 
        (result['tutkinto'] == 'A')
    ].sort_values('market_share', ascending=True)  # Sort by market share to ensure consistent order
    assert len(year_2020_a) == 2
    assert year_2020_a['market_share'].round(2).tolist() == [33.33, 66.67]

def test_calculate_growth_trends():
    """Test growth trend calculations."""
    # Create sample data
    data = {
        'tilastovuosi': [2020, 2021, 2022, 2020, 2021, 2022],
        'tutkinto': ['A', 'A', 'A', 'B', 'B', 'B'],
        'nettoopiskelijamaaraLkm': [100, 150, 200, 200, 180, 160]
    }
    df = pd.DataFrame(data)
    
    # Calculate growth trends
    result = calculate_growth_trends(
        df=df,
        year_col='tilastovuosi',
        value_col='nettoopiskelijamaaraLkm',
        group_cols=['tutkinto']
    )
    
    # Verify results
    assert len(result) == 2
    assert all(col in result.columns for col in ['tutkinto', 'cagr', 'start_year', 'end_year'])
    
    # Check CAGR calculations
    tutkinto_a = result[result['tutkinto'] == 'A'].iloc[0]
    assert round(tutkinto_a['cagr'], 2) == 41.42  # CAGR for tutkinto A
    
    tutkinto_b = result[result['tutkinto'] == 'B'].iloc[0]
    assert round(tutkinto_b['cagr'], 2) == -10.56  # CAGR for tutkinto B

def test_calculate_provider_counts():
    """Test provider count calculations."""
    # Create sample data
    data = {
        'tilastovuosi': [2020, 2020, 2020, 2021, 2021, 2021],
        'tutkinto': ['A', 'A', 'B', 'A', 'A', 'B'],
        'koulutuksenJarjestaja': ['X', 'Y', 'X', 'X', 'Y', 'Z']
    }
    df = pd.DataFrame(data)
    
    # Calculate provider counts
    result = calculate_provider_counts(
        df=df,
        group_cols=['tilastovuosi', 'tutkinto']
    )
    
    # Verify results
    assert len(result) == 4
    assert 'provider_count' in result.columns
    
    # Check specific counts
    year_2020_a = result[
        (result['tilastovuosi'] == 2020) & 
        (result['tutkinto'] == 'A')
    ]
    assert year_2020_a['provider_count'].iloc[0] == 2
    
    year_2021_b = result[
        (result['tilastovuosi'] == 2021) & 
        (result['tutkinto'] == 'B')
    ]
    assert year_2021_b['provider_count'].iloc[0] == 1

def test_analyze_market():
    """Test comprehensive market analysis."""
    # Create sample data
    data = {
        'tilastovuosi': [2020, 2020, 2020, 2021, 2021, 2021],
        'tutkinto': ['A', 'A', 'B', 'A', 'A', 'B'],
        'koulutuksenJarjestaja': ['X', 'Y', 'X', 'X', 'Y', 'X'],
        'nettoopiskelijamaaraLkm': [100, 200, 150, 120, 180, 160]
    }
    df = pd.DataFrame(data)
    
    # Perform market analysis
    result = analyze_market(
        df=df,
        year_col='tilastovuosi',
        value_col='nettoopiskelijamaaraLkm',
        group_cols=['tutkinto']
    )
    
    # Verify results
    assert len(result) == 3
    assert all(key in result for key in ['market_shares', 'growth_trends', 'provider_counts'])
    
    # Check market shares
    assert len(result['market_shares']) == 6
    assert 'market_share' in result['market_shares'].columns
    
    # Check growth trends
    assert len(result['growth_trends']) == 2
    assert all(col in result['growth_trends'].columns for col in ['tutkinto', 'cagr'])
    
    # Check provider counts
    assert len(result['provider_counts']) == 4
    assert 'provider_count' in result['provider_counts'].columns

def test_growth_trends_edge_cases():
    """Test edge cases in growth trend calculations."""
    # Test empty DataFrame
    empty_df = pd.DataFrame(columns=['tilastovuosi', 'tutkinto', 'nettoopiskelijamaaraLkm'])
    result = calculate_growth_trends(
        df=empty_df,
        year_col='tilastovuosi',
        value_col='nettoopiskelijamaaraLkm',
        group_cols=['tutkinto']
    )
    assert result.empty
    
    # Test single year data
    single_year_data = {
        'tilastovuosi': [2020],
        'tutkinto': ['A'],
        'nettoopiskelijamaaraLkm': [100]
    }
    df = pd.DataFrame(single_year_data)
    result = calculate_growth_trends(
        df=df,
        year_col='tilastovuosi',
        value_col='nettoopiskelijamaaraLkm',
        group_cols=['tutkinto']
    )
    assert len(result) == 1
    assert pd.isna(result['cagr'].iloc[0])
    
    # Test zero values
    zero_data = {
        'tilastovuosi': [2020, 2021],
        'tutkinto': ['A', 'A'],
        'nettoopiskelijamaaraLkm': [0, 100]
    }
    df = pd.DataFrame(zero_data)
    result = calculate_growth_trends(
        df=df,
        year_col='tilastovuosi',
        value_col='nettoopiskelijamaaraLkm',
        group_cols=['tutkinto']
    )
    assert len(result) == 1
    assert pd.isna(result['cagr'].iloc[0])

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'tilastovuosi': [2020, 2020, 2021, 2021, 2022, 2022] * 4,
        'tutkinto': ['A', 'B', 'A', 'B', 'A', 'B'] * 4,
        'koulutuksenJarjestaja': ['Provider1', 'Provider1', 'Provider1', 'Provider1', 'Provider1', 'Provider1'] +
                                ['Provider2', 'Provider2', 'Provider2', 'Provider2', 'Provider2', 'Provider2'] +
                                ['Provider3', 'Provider3', 'Provider3', 'Provider3', 'Provider3', 'Provider3'] +
                                ['Provider4', 'Provider4', 'Provider4', 'Provider4', 'Provider4', 'Provider4'],
        'hankintakoulutuksenJarjestaja': ['Provider2', 'Provider2', 'Provider2', 'Provider2', 'Provider2', 'Provider2'] +
                                        ['Provider1', 'Provider1', 'Provider1', 'Provider1', 'Provider1', 'Provider1'] +
                                        ['Provider4', 'Provider4', 'Provider4', 'Provider4', 'Provider4', 'Provider4'] +
                                        ['Provider3', 'Provider3', 'Provider3', 'Provider3', 'Provider3', 'Provider3'],
        'nettoopiskelijamaaraLkm': [100, 200, 150, 250, 200, 300] * 4
    }
    return pd.DataFrame(data)

def test_calculate_provider_rankings(sample_market_data):
    """Test calculating provider rankings."""
    result = calculate_provider_rankings(sample_market_data)
    
    # Check structure
    assert all(col in result.columns for col in [
        'tilastovuosi', 'tutkinto', 'provider', 'provider_volume',
        'total_market_volume', 'market_share', 'rank'
    ])
    
    # Check rankings for 2020, tutkinto A
    year_2020_a = result[
        (result['tilastovuosi'] == 2020) & 
        (result['tutkinto'] == 'A')
    ]
    assert len(year_2020_a) == 2  # Two providers
    assert year_2020_a['rank'].tolist() == [1, 2]  # Y has higher volume (200) than X (100)

def test_analyze_provider_roles(sample_market_data):
    """Test analyzing provider roles."""
    result = analyze_provider_roles(sample_market_data, 'X')
    
    # Check structure
    assert all(col in result.columns for col in [
        'tutkinto', 'main_provider_volume', 'subcontractor_volume',
        'total_volume', 'main_provider_percentage', 'subcontractor_percentage'
    ])
    
    # Check calculations for tutkinto A
    qual_a = result[result['tutkinto'] == 'A'].iloc[0]
    assert qual_a['main_provider_volume'] == 450  # 100 + 150 + 200
    assert qual_a['subcontractor_volume'] == 600  # 200 + 250 + 300
    assert qual_a['total_volume'] == 1050  # 450 + 600
    assert abs(qual_a['main_provider_percentage'] - 42.86) < 0.01  # 450/1050 * 100
    assert abs(qual_a['subcontractor_percentage'] - 57.14) < 0.01  # 600/1050 * 100

def test_track_market_shares(sample_market_data):
    """Test tracking market shares over time."""
    result = track_market_shares(sample_market_data, 'X')
    
    # Check structure
    assert all(col in result.columns for col in [
        'tilastovuosi', 'tutkinto', 'provider_volume',
        'total_market_volume', 'market_share'
    ])
    
    # Check calculations for 2020, tutkinto A
    year_2020_a = result[
        (result['tilastovuosi'] == 2020) & 
        (result['tutkinto'] == 'A')
    ].iloc[0]
    assert year_2020_a['provider_volume'] == 300  # 100 (main) + 200 (sub)
    assert year_2020_a['total_market_volume'] == 900  # Sum of all volumes
    assert abs(year_2020_a['market_share'] - 33.33) < 0.01  # 300/900 * 100

def test_filter_qualification_types(sample_market_data):
    """Test filtering of qualification types."""
    result = filter_qualification_types(sample_market_data)
    
    # Check that all rows have valid qualification types
    assert all(result['tutkintotyyppi'].isin(['ammattitutkinto', 'erikoisammattitutkinto']))
    
    # Check that no rows were lost
    assert len(result) == len(sample_market_data)

def test_calculate_total_students(sample_market_data):
    """Test calculation of total students for a provider."""
    result = calculate_total_students(sample_market_data, 'X')
    
    # Check structure of result
    assert all(key in result for key in [
        'total_students',
        'main_provider_students',
        'subcontractor_students',
        'main_provider_percentage',
        'subcontractor_percentage'
    ])
    
    # Check calculations
    assert result['total_students'] == 450  # 100 + 150 + 200
    assert result['main_provider_students'] == 450
    assert result['subcontractor_students'] == 0
    assert result['main_provider_percentage'] == 100.0
    assert result['subcontractor_percentage'] == 0.0

def test_calculate_total_students_with_subcontracting(sample_market_data):
    """Test calculation of total students for a provider that acts as both main provider and subcontractor."""
    # Add subcontractor data
    sample_market_data['hankintakoulutuksenJarjestaja'] = ['Y', 'X', 'X', 'Y', 'X', 'X', 'Y', 'X', 'X']
    
    result = calculate_total_students(sample_market_data, 'Y')
    
    # Check calculations
    assert result['total_students'] == 1200  # 200 + 250 + 300 (as main provider) + 300 + 350 + 400 (as subcontractor)
    assert result['main_provider_students'] == 750  # 200 + 250 + 300
    assert result['subcontractor_students'] == 450  # 300 + 350 + 400
    assert abs(result['main_provider_percentage'] - 62.5) < 0.01  # 750/1200 * 100
    assert abs(result['subcontractor_percentage'] - 37.5) < 0.01  # 450/1200 * 100 

def test_get_provider_qualifications(sample_market_data):
    """Test getting qualifications offered by a provider."""
    # Add subcontractor data
    sample_market_data['hankintakoulutuksenJarjestaja'] = ['Y', 'X', 'X', 'Y', 'X', 'X', 'Y', 'X', 'X']
    
    # Test for provider X
    quals_x = get_provider_qualifications(sample_market_data, 'X')
    assert quals_x == {'A', 'B'}  # X offers both as main provider and subcontractor
    
    # Test for provider Y
    quals_y = get_provider_qualifications(sample_market_data, 'Y')
    assert quals_y == {'A'}  # Y only offers A
    
    # Test for provider Z
    quals_z = get_provider_qualifications(sample_market_data, 'Z')
    assert quals_z == {'B'}  # Z only offers B

def test_analyze_qualification_volumes(sample_market_data):
    """Test analyzing qualification volumes for a provider."""
    result = analyze_qualification_volumes(sample_market_data, 'X', 2020)
    
    # Check structure
    assert all(col in result.columns for col in [
        'tutkinto', 'main_provider_volume', 'subcontractor_volume',
        'total_volume', 'main_provider_percentage', 'subcontractor_percentage'
    ])
    
    # Check calculations for tutkinto A
    qual_a = result[result['tutkinto'] == 'A'].iloc[0]
    assert qual_a['main_provider_volume'] == 100  # Direct volume
    assert qual_a['subcontractor_volume'] == 200  # Subcontracted volume
    assert qual_a['total_volume'] == 300  # Total volume
    assert abs(qual_a['main_provider_percentage'] - 33.33) < 0.1  # ~33.33%
    assert abs(qual_a['subcontractor_percentage'] - 66.67) < 0.1  # ~66.67%

def test_calculate_year_over_year_changes(sample_market_data):
    """Test calculating year-over-year changes."""
    result = calculate_year_over_year_changes(sample_market_data, 2020, 2021)
    
    # Check structure
    assert all(col in result.columns for col in [
        'provider', 'volume_start', 'volume_end', 'volume_change',
        'volume_change_pct', 'market_share_start', 'market_share_end',
        'market_share_change'
    ])
    
    # Check calculations for provider X
    provider_x = result[result['provider'] == 'X'].iloc[0]
    assert provider_x['volume_start'] == 300  # 100 (main) + 200 (sub)
    assert provider_x['volume_end'] == 400  # 150 (main) + 250 (sub)
    assert provider_x['volume_change'] == 100
    assert abs(provider_x['volume_change_pct'] - 33.33) < 0.1  # ~33.33%
    assert provider_x['market_share_change'] > 0  # Should have gained market share

def test_analyze_market_share_changes(sample_market_data):
    """Test analyzing market share changes between years."""
    result = analyze_market_share_changes(sample_market_data, 2020, 2021)
    
    # Check structure
    expected_columns = [
        'provider', 'market_share_start', 'market_share_end',
        'volume_start', 'volume_end', 'volume_change', 'volume_change_pct',
        'market_share_change', 'market_share_change_pct'
    ]
    assert all(col in result.columns for col in expected_columns)
    
    # Check specific provider results
    provider_x = result[result['provider'] == 'X'].iloc[0]
    provider_y = result[result['provider'] == 'Y'].iloc[0]
    
    # Provider X should have positive market share change
    assert provider_x['market_share_change'] > 0
    # Provider Y should have negative market share change
    assert provider_y['market_share_change'] < 0 

def test_calculate_market_concentration(sample_market_data):
    """Test calculation of market concentration metrics."""
    # Calculate market concentration
    result = calculate_market_concentration(
        df=sample_market_data,
        group_cols=['tilastovuosi'],
        value_col='nettoopiskelijamaaraLkm',
        provider_col='koulutuksenJarjestaja',
        subcontractor_col='hankintakoulutuksenJarjestaja'
    )
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'tilastovuosi', 'provider_count', 'cr1', 'cr3', 'cr5', 'hhi'}
    
    # Check specific year
    year_2020 = result[result['tilastovuosi'] == 2020].iloc[0]
    
    # Verify metrics
    assert year_2020['provider_count'] > 0
    assert 0 <= year_2020['cr1'] <= 100  # Top provider share
    assert year_2020['cr1'] <= year_2020['cr3'] <= 100  # Top 3 share
    assert year_2020['cr3'] <= year_2020['cr5'] <= 100  # Top 5 share
    assert 0 <= year_2020['hhi'] <= 10000  # HHI is between 0 and 10000
    
    # Test without grouping
    result_no_group = calculate_market_concentration(
        df=sample_market_data,
        value_col='nettoopiskelijamaaraLkm',
        provider_col='koulutuksenJarjestaja',
        subcontractor_col='hankintakoulutuksenJarjestaja'
    )
    
    # Check structure without grouping
    assert isinstance(result_no_group, pd.DataFrame)
    assert len(result_no_group) == 1
    assert set(result_no_group.columns) == {'provider_count', 'cr1', 'cr3', 'cr5', 'hhi'} 