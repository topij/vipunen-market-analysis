"""
Tests for the MarketAnalyzer class.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.vipunen.analysis.market_analyzer import MarketAnalyzer


@pytest.fixture
def sample_data():
    """Create a sample dataframe for testing."""
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


@pytest.fixture
def market_analyzer(sample_data):
    """Create a MarketAnalyzer instance with sample data."""
    return MarketAnalyzer(sample_data)


def test_init(sample_data):
    """Test the initialization of the MarketAnalyzer."""
    analyzer = MarketAnalyzer(sample_data)
    
    # Check that data is stored correctly
    pd.testing.assert_frame_equal(analyzer.data, sample_data)
    
    # Check that min_year and max_year are set correctly
    assert analyzer.min_year == 2020
    assert analyzer.max_year == 2022


def test_calculate_total_volumes(market_analyzer):
    """Test calculation of total volumes."""
    total_volumes = market_analyzer.calculate_total_volumes()
    
    # Check that the result is a DataFrame
    assert isinstance(total_volumes, pd.DataFrame)
    
    # Check that we have a row for each year
    assert set(total_volumes.index) == {2020, 2021, 2022}
    
    # Check that the volumes are calculated correctly
    assert total_volumes.loc[2020, 'total_volume'] == 450  # 100 + 200 + 150
    assert total_volumes.loc[2021, 'total_volume'] == 480  # 110 + 210 + 160
    assert total_volumes.loc[2022, 'total_volume'] == 510  # 120 + 220 + 170


def test_calculate_volumes_by_qualification(market_analyzer):
    """Test calculation of volumes by qualification type."""
    volumes_by_qual = market_analyzer.calculate_volumes_by_qualification()
    
    # Check that the result is a DataFrame
    assert isinstance(volumes_by_qual, pd.DataFrame)
    
    # Check that we have rows for each qualification type and year
    expected_index = pd.MultiIndex.from_product([
        ['Ammattitutkinnot', 'Erikoisammattitutkinnot'],
        [2020, 2021, 2022]
    ], names=['tutkintotyyppi', 'tilastovuosi'])
    
    assert volumes_by_qual.index.equals(expected_index) or set(map(tuple, volumes_by_qual.index)) == set(map(tuple, expected_index))
    
    # Check that the volumes are calculated correctly
    # Ammattitutkinnot: 2020: 100 + 200 = 300, 2021: 110 + 210 = 320, 2022: 120 + 220 = 340
    # Erikoisammattitutkinnot: 2020: 150, 2021: 160, 2022: 170
    ammatti_2020 = volumes_by_qual.loc[('Ammattitutkinnot', 2020), 'volume']
    ammatti_2021 = volumes_by_qual.loc[('Ammattitutkinnot', 2021), 'volume']
    ammatti_2022 = volumes_by_qual.loc[('Ammattitutkinnot', 2022), 'volume']
    
    erikois_2020 = volumes_by_qual.loc[('Erikoisammattitutkinnot', 2020), 'volume']
    erikois_2021 = volumes_by_qual.loc[('Erikoisammattitutkinnot', 2021), 'volume']
    erikois_2022 = volumes_by_qual.loc[('Erikoisammattitutkinnot', 2022), 'volume']
    
    assert ammatti_2020 == 300
    assert ammatti_2021 == 320
    assert ammatti_2022 == 340
    
    assert erikois_2020 == 150
    assert erikois_2021 == 160
    assert erikois_2022 == 170


def test_calculate_market_shares(market_analyzer):
    """Test calculation of market shares."""
    market_shares = market_analyzer.calculate_market_shares()
    
    # Check that the result is a DataFrame
    assert isinstance(market_shares, pd.DataFrame)
    
    # Check that we have rows for each provider and year
    assert len(market_shares) == 6  # 2 providers × 3 years
    
    # Check that market shares sum to 1 for each year
    for year in [2020, 2021, 2022]:
        year_shares = market_shares[market_shares['tilastovuosi'] == year]['market_share']
        assert np.isclose(year_shares.sum(), 1.0)
    
    # Provider X's market share: (100 + 200) / 450 = 300/450 = 0.6667 for 2020
    provider_x_2020 = market_shares[(market_shares['koulutuksenJarjestaja'] == 'Provider X') & 
                                    (market_shares['tilastovuosi'] == 2020)]['market_share'].values[0]
    assert np.isclose(provider_x_2020, 300/450)


def test_calculate_qualification_growth(market_analyzer):
    """Test calculation of qualification growth."""
    qual_growth = market_analyzer.calculate_qualification_growth()
    
    # Check that the result is a DataFrame
    assert isinstance(qual_growth, pd.DataFrame)
    
    # Check that growth is calculated correctly
    # For Ammattitutkinnot: (340 - 300) / 300 = 0.1333
    # For Erikoisammattitutkinnot: (170 - 150) / 150 = 0.1333
    ammatti_growth = qual_growth.loc['Ammattitutkinnot', 'growth_rate']
    erikois_growth = qual_growth.loc['Erikoisammattitutkinnot', 'growth_rate']
    
    assert np.isclose(ammatti_growth, (340 - 300) / 300)
    assert np.isclose(erikois_growth, (170 - 150) / 150)


def test_calculate_qualification_cagr(market_analyzer):
    """Test calculation of qualification CAGR."""
    qual_cagr = market_analyzer.calculate_qualification_cagr()
    
    # Check that the result is a DataFrame
    assert isinstance(qual_cagr, pd.DataFrame)
    
    # Check that CAGR is calculated correctly
    # For Ammattitutkinnot: (340/300)^(1/2) - 1 = 0.0644
    # For Erikoisammattitutkinnot: (170/150)^(1/2) - 1 = 0.0644
    ammatti_cagr = qual_cagr.loc['Ammattitutkinnot', 'cagr']
    erikois_cagr = qual_cagr.loc['Erikoisammattitutkinnot', 'cagr']
    
    assert np.isclose(ammatti_cagr, (340/300)**(1/2) - 1)
    assert np.isclose(erikois_cagr, (170/150)**(1/2) - 1)


def test_get_all_results(market_analyzer):
    """Test getting all analysis results."""
    results = market_analyzer.get_all_results()
    
    # Check that the result is a dictionary
    assert isinstance(results, dict)
    
    # Check that all expected keys are present
    expected_keys = [
        'total_volumes', 
        'volumes_by_qualification',
        'market_shares',
        'qualification_growth',
        'qualification_cagr'
    ]
    for key in expected_keys:
        assert key in results
    
    # Check that all values are DataFrames
    for key, value in results.items():
        assert isinstance(value, pd.DataFrame)


def test_empty_data():
    """Test initialization with empty data."""
    empty_df = pd.DataFrame({
        'tilastovuosi': [],
        'tutkintotyyppi': [],
        'tutkinto': [],
        'koulutuksenJarjestaja': [],
        'hankintakoulutuksenJarjestaja': [],
        'nettoopiskelijamaaraLkm': []
    })
    
    # Initialize with empty data
    analyzer = MarketAnalyzer(empty_df)
    
    # Check that min_year and max_year are None
    assert analyzer.min_year is None
    assert analyzer.max_year is None
    
    # Check that all calculations return empty DataFrames
    assert analyzer.calculate_total_volumes().empty
    assert analyzer.calculate_volumes_by_qualification().empty
    assert analyzer.calculate_market_shares().empty
    assert analyzer.calculate_qualification_growth().empty
    assert analyzer.calculate_qualification_cagr().empty
    
    # Check that get_all_results returns a dict with empty DataFrames
    results = analyzer.get_all_results()
    for key, value in results.items():
        assert value.empty


def test_data_with_single_year():
    """Test with data from a single year."""
    single_year_df = pd.DataFrame({
        'tilastovuosi': [2020, 2020, 2020],
        'tutkintotyyppi': ['Ammattitutkinnot', 'Ammattitutkinnot', 'Erikoisammattitutkinnot'],
        'tutkinto': ['Tutkinto A', 'Tutkinto B', 'Tutkinto C'],
        'koulutuksenJarjestaja': ['Provider X', 'Provider X', 'Provider Y'],
        'hankintakoulutuksenJarjestaja': [None, None, 'Provider X'],
        'nettoopiskelijamaaraLkm': [100, 200, 150]
    })
    
    analyzer = MarketAnalyzer(single_year_df)
    
    # Check that min_year and max_year are both 2020
    assert analyzer.min_year == 2020
    assert analyzer.max_year == 2020
    
    # Check that total volumes and volumes by qualification work
    assert not analyzer.calculate_total_volumes().empty
    assert not analyzer.calculate_volumes_by_qualification().empty
    assert not analyzer.calculate_market_shares().empty
    
    # Growth and CAGR should be empty since they require multiple years
    assert analyzer.calculate_qualification_growth().empty
    assert analyzer.calculate_qualification_cagr().empty 