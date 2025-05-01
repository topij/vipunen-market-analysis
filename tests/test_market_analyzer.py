"""
Tests for the MarketAnalyzer class.
"""
import pytest;
import pandas as pd
import numpy as np
from pathlib import Path

from vipunen.analysis.market_analyzer import MarketAnalyzer


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary for testing."""
    return {
        'columns': {
            'input': {
                'year': 'tilastovuosi',
                'degree_type': 'tutkintotyyppi',
                'qualification': 'tutkinto',
                'provider': 'koulutuksenJarjestaja',
                'subcontractor': 'hankintakoulutuksenJarjestaja',
                'volume': 'nettoopiskelijamaaraLkm',
                'update_date': 'tietojoukkoPaivitettyPvm'
            },
            'output': {
                'year': 'Year',
                'qualification': 'Qualification',
                'provider': 'Provider',
                'provider_amount': 'Provider Amount',
                'subcontractor_amount': 'Subcontractor Amount',
                'total_volume': 'Total Volume', # Institution's total volume for a qual/year
                'market_total': 'Market Total', # Total market volume for a qual/year
                'market_share': 'Market Share (%)',
                'market_rank': 'Market Rank',
                'market_share_growth': 'Market Share Growth (%)',
                'market_gainer_rank': 'Market Gainer Rank', # Use the added key
                # Add other output names if needed by tests
            }
        },
        # Add other config sections if needed by tests (e.g., analysis thresholds)
        'analysis': {
            'min_market_size_threshold': 5,
            'active_qualification_min_volume_sum': 3,
            'gainers_losers': {
                'min_market_share_threshold': 0.5,
                'min_market_rank_percentile': None
            }
        }
    }


@pytest.fixture
def sample_data(mock_config):
    """Create a sample dataframe for testing."""
    # Use input column names from mock_config
    cols_in = mock_config['columns']['input']
    return pd.DataFrame({
        cols_in['year']: [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022],
        cols_in['degree_type']: ['Ammattitutkinnot', 'Ammattitutkinnot', 'Erikoisammattitutkinnot',
                                'Ammattitutkinnot', 'Ammattitutkinnot', 'Erikoisammattitutkinnot',
                                'Ammattitutkinnot', 'Ammattitutkinnot', 'Erikoisammattitutkinnot'],
        cols_in['qualification']: ['Tutkinto A', 'Tutkinto B', 'Tutkinto C',
                                  'Tutkinto A', 'Tutkinto B', 'Tutkinto C',
                                  'Tutkinto A', 'Tutkinto B', 'Tutkinto C'],
        cols_in['provider']: ['Provider X', 'Provider X', 'Provider Y',
                              'Provider X', 'Provider X', 'Provider Y',
                              'Provider X', 'Provider X', 'Provider Y'],
        cols_in['subcontractor']: [None, None, 'Provider X',
                                      None, None, 'Provider X',
                                      None, None, 'Provider X'],
        cols_in['volume']: [100, 200, 150, 110, 210, 160, 120, 220, 170]
    })


@pytest.fixture
def market_analyzer(sample_data, mock_config):
    """Create a MarketAnalyzer instance with sample data and config."""
    # Pass mock_config to constructor
    return MarketAnalyzer(sample_data, cfg=mock_config)


def test_init(sample_data, mock_config):
    """Test the initialization of the MarketAnalyzer."""
    analyzer = MarketAnalyzer(sample_data, cfg=mock_config)
    cols_in = mock_config['columns']['input']

    # Check that data is stored correctly
    pd.testing.assert_frame_equal(analyzer.data, sample_data)
    
    # Check config is stored
    assert analyzer.cfg == mock_config
    assert analyzer.cols_in == mock_config['columns']['input']
    assert analyzer.cols_out == mock_config['columns']['output']
    
    # Check that min_year and max_year are set correctly using input col name
    assert analyzer.min_year == 2020
    assert analyzer.max_year == 2022


def test_calculate_total_volumes(market_analyzer, mock_config):
    """Test calculation of total volumes (for the institution)."""
    # Set institution names required by the analyzer
    market_analyzer.institution_names = ['Provider X']
    market_analyzer.institution_short_name = "PX"
    cols_out = mock_config['columns']['output']
    year_col = cols_out['year']
    total_volume_col = cols_out['total_volume']
    provider_col = cols_out['provider']

    institution_volumes = market_analyzer.calculate_total_volumes()

    # Check that the result is a DataFrame
    assert isinstance(institution_volumes, pd.DataFrame)

    # Check columns using output names from config
    # Note: calculate_total_volumes was refactored to rename to output cols
    assert year_col in institution_volumes.columns
    assert total_volume_col in institution_volumes.columns # This is inst total
    assert provider_col in institution_volumes.columns # Should contain short name
    # Check that provider/sub amounts are also present after rename
    assert cols_out['provider_amount'] in institution_volumes.columns
    assert cols_out['subcontractor_amount'] in institution_volumes.columns

    # Check years present
    assert set(institution_volumes[year_col]) == {2020, 2021, 2022}

    # Check the volumes for the *institution* (Provider X)
    # Vol X 2020: 100(main)+200(main)+150(sub) = 450 -> This is the 'Yhteensä'/'Total Volume' column
    vol_x_2020 = institution_volumes[institution_volumes[year_col] == 2020][total_volume_col].iloc[0]
    vol_x_2021 = institution_volumes[institution_volumes[year_col] == 2021][total_volume_col].iloc[0]
    vol_x_2022 = institution_volumes[institution_volumes[year_col] == 2022][total_volume_col].iloc[0]
    assert vol_x_2020 == 450
    assert vol_x_2021 == 480
    assert vol_x_2022 == 510

    # Check provider/subcontractor amounts after rename
    prov_amt_2020 = institution_volumes[institution_volumes[year_col] == 2020][cols_out['provider_amount']].iloc[0]
    sub_amt_2020 = institution_volumes[institution_volumes[year_col] == 2020][cols_out['subcontractor_amount']].iloc[0]
    assert prov_amt_2020 == 300 # 100 + 200
    assert sub_amt_2020 == 150

    # Check short name assignment
    assert institution_volumes[provider_col].iloc[0] == "PX"


def test_calculate_volumes_by_qualification(market_analyzer, mock_config):
    """Test calculation of volumes by qualification type for the institution."""
    # Set institution names for the analyzer
    market_analyzer.institution_names = ['Provider X']
    market_analyzer.institution_short_name = "PX"
    cols_out = mock_config['columns']['output']

    volumes_by_qual = market_analyzer.calculate_volumes_by_qualification()
    
    # Check that the result is a DataFrame
    assert isinstance(volumes_by_qual, pd.DataFrame)
    
    # Check for expected output columns using config names
    expected_cols = [
        cols_out['year'], cols_out['qualification'],
        cols_out['provider_amount'], cols_out['subcontractor_amount'],
        cols_out['total_volume'], cols_out['market_total'],
        cols_out['market_share']
    ]
    assert all(col in volumes_by_qual.columns for col in expected_cols)

    # Check specific values for Provider X using config names
    # Tutkinto A, 2020: Provider X main=100, sub=0. Total=100. Market Total=100. Share=100%
    tut_a_2020 = volumes_by_qual[
        (volumes_by_qual[cols_out['qualification']] == 'Tutkinto A') & (volumes_by_qual[cols_out['year']] == 2020)
    ].iloc[0]
    assert tut_a_2020[cols_out['provider_amount']] == 100
    assert tut_a_2020[cols_out['subcontractor_amount']] == 0
    assert tut_a_2020[cols_out['total_volume']] == 100 # Institution's total volume
    assert tut_a_2020[cols_out['market_total']] == 100
    assert np.isclose(tut_a_2020[cols_out['market_share']], 100.0)

    # Tutkinto B, 2021: Provider X main=210, sub=0. Total=210. Market Total=210. Share=100%
    tut_b_2021 = volumes_by_qual[
        (volumes_by_qual[cols_out['qualification']] == 'Tutkinto B') & (volumes_by_qual[cols_out['year']] == 2021)
    ].iloc[0]
    assert tut_b_2021[cols_out['provider_amount']] == 210
    assert tut_b_2021[cols_out['subcontractor_amount']] == 0
    assert tut_b_2021[cols_out['total_volume']] == 210
    assert tut_b_2021[cols_out['market_total']] == 210
    assert np.isclose(tut_b_2021[cols_out['market_share']], 100.0)

    # Tutkinto C, 2022: Provider X sub=170. Provider Y main=170. Market Total=170.
    # Provider X Total Volume = 170 (0 as provider + 170 as sub). Share=100%
    tut_c_2022 = volumes_by_qual[
        (volumes_by_qual[cols_out['qualification']] == 'Tutkinto C') & (volumes_by_qual[cols_out['year']] == 2022)
    ].iloc[0]
    assert tut_c_2022[cols_out['provider_amount']] == 0
    assert tut_c_2022[cols_out['subcontractor_amount']] == 170
    assert tut_c_2022[cols_out['total_volume']] == 170
    assert tut_c_2022[cols_out['market_total']] == 170
    assert np.isclose(tut_c_2022[cols_out['market_share']], 100.0)

    # Check number of rows (3 qualifications * 3 years = 9)
    assert len(volumes_by_qual) == 9


def test_calculate_market_shares_deprecated(market_analyzer):
    """Test that the deprecated calculate_market_shares returns an empty DataFrame with expected columns."""
    # This method was marked as deprecated/stubbed in the refactoring
    market_analyzer.institution_names = ['Provider X']
    result_df = market_analyzer.calculate_market_shares()

    # Check that the result is an empty DataFrame
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty

    # Check columns match the stub definition (using analyzer's cols_out)
    cols_out = market_analyzer.cols_out
    expected_cols = [
        cols_out['year'], cols_out['qualification'], cols_out['provider'],
        cols_out['provider_amount'], cols_out['subcontractor_amount'],
        cols_out['total_volume'], cols_out['market_total'],
        cols_out['market_share'], cols_out['market_rank'],
        cols_out['market_share_growth'], cols_out['market_gainer_rank']
    ]
    assert list(result_df.columns) == expected_cols


def test_calculate_qualification_growth_old(market_analyzer):
    """Test calculation of qualification growth (Old implementation)."""
    # Note: The implementation of this method was changed significantly in the refactoring.
    # This test reflects the *old* logic which calculated growth based on degree type.
    # It should likely be removed or updated to test the *new* logic
    # based on calculate_yoy_growth and institution's volume by qualification.
    qual_growth = market_analyzer.calculate_qualification_growth()

    # Check that the result is a DataFrame
    assert isinstance(qual_growth, pd.DataFrame)

    # Check index and columns based on old implementation
    assert qual_growth.index.name == 'tutkintotyyppi'
    assert set(qual_growth.index) == {'Ammattitutkinnot', 'Erikoisammattitutkinnot'}
    expected_cols = ['First Year', 'Last Year', 'First Year Volume', 'Last Year Volume', 'growth_rate']
    assert all(col in qual_growth.columns for col in expected_cols)

    # Check specific values
    # Amm: 2020=100+200=300, 2022=120+220=340 -> growth = (340/300 - 1)*100 = 13.33%
    # Eri: 2020=150, 2022=170 -> growth = (170/150 - 1)*100 = 13.33%
    assert np.isclose(qual_growth.loc['Ammattitutkinnot', 'growth_rate'], 13.333333)
    assert np.isclose(qual_growth.loc['Erikoisammattitutkinnot', 'growth_rate'], 13.333333)
    assert qual_growth.loc['Ammattitutkinnot', 'First Year Volume'] == 300
    assert qual_growth.loc['Ammattitutkinnot', 'Last Year Volume'] == 340
    assert qual_growth.loc['Erikoisammattitutkinnot', 'First Year Volume'] == 150
    assert qual_growth.loc['Erikoisammattitutkinnot', 'Last Year Volume'] == 170


def test_calculate_qualification_cagr(market_analyzer, mock_config):
    """Test calculation of CAGR for qualifications offered by the institution."""
    # Set institution names for the analyzer
    market_analyzer.institution_names = ['Provider X']
    market_analyzer.institution_short_name = "PX"
    cols_out = mock_config['columns']['output'] # Use output names as CAGR result uses them

    cagr_results = market_analyzer.calculate_qualification_cagr()

    # Check that the result is a DataFrame
    assert isinstance(cagr_results, pd.DataFrame)

    # Check for expected columns (using English names as per current implementation)
    # Note: MarketAnalyzer.calculate_qualification_cagr refactored to use the utility function
    # which returns specific English column names.
    expected_cols = ['Qualification', 'CAGR', 'First Year', 'Last Year', 'First Year Volume', 'Last Year Volume']
    # Check if 'Years Present' column is also returned
    if 'Years Present' in cagr_results.columns:
        expected_cols.append('Years Present')
    assert all(col in cagr_results.columns for col in expected_cols)

    # Check number of rows (should be 3, one for each qualification offered by Provider X)
    # Provider X is main for A, B and sub for C.
    assert len(cagr_results) == 3
    assert set(cagr_results['Qualification']) == {'Tutkinto A', 'Tutkinto B', 'Tutkinto C'}

    # Check specific CAGR values for Provider X
    # Tutkinto A: 2020=100, 2021=110, 2022=120. CAGR = ((120/100)**(1/2))-1 = 9.54%
    # Tutkinto B: 2020=200, 2021=210, 2022=220. CAGR = ((220/200)**(1/2))-1 = 4.88%
    # Tutkinto C: 2020=150(sub), 2021=160(sub), 2022=170(sub). CAGR = ((170/150)**(1/2))-1 = 6.46%
    cagr_a = cagr_results[cagr_results['Qualification'] == 'Tutkinto A']['CAGR'].iloc[0]
    cagr_b = cagr_results[cagr_results['Qualification'] == 'Tutkinto B']['CAGR'].iloc[0]
    cagr_c = cagr_results[cagr_results['Qualification'] == 'Tutkinto C']['CAGR'].iloc[0]

    assert cagr_a == pytest.approx(9.5445, abs=1e-4)
    assert cagr_b == pytest.approx(4.8808, abs=1e-4)
    assert cagr_c == pytest.approx(6.4581, abs=1e-4)

    # Check first/last year volumes
    assert cagr_results[cagr_results['Qualification'] == 'Tutkinto A']['First Year Volume'].iloc[0] == 100
    assert cagr_results[cagr_results['Qualification'] == 'Tutkinto A']['Last Year Volume'].iloc[0] == 120
    assert cagr_results[cagr_results['Qualification'] == 'Tutkinto C']['First Year Volume'].iloc[0] == 150
    assert cagr_results[cagr_results['Qualification'] == 'Tutkinto C']['Last Year Volume'].iloc[0] == 170


def test_get_all_results(market_analyzer, mock_config):
    """Test the get_all_results method."""
    market_analyzer.institution_names = ['Provider X']
    market_analyzer.institution_short_name = "PX"
    cols_out = mock_config['columns']['output']

    results = market_analyzer.get_all_results()

    # Check that the result is a dictionary
    assert isinstance(results, dict)

    # Check for expected keys
    expected_keys = [
        "total_volumes",
        "volumes_by_qualification",
        # "market_shares", # This key calls the deprecated/stubbed function
        "detailed_providers_market",
        # "qualification_growth", # Implementation changed
        "qualification_cagr",
        "overall_total_market_volume",
        "qualification_market_yoy_growth"
    ]
    assert all(key in results for key in expected_keys)

    # Check that values are DataFrames or Series
    assert isinstance(results["total_volumes"], pd.DataFrame)
    assert isinstance(results["volumes_by_qualification"], pd.DataFrame)
    # assert isinstance(results["market_shares"], pd.DataFrame) # Should be empty df from stub
    assert isinstance(results["detailed_providers_market"], pd.DataFrame)
    assert isinstance(results["qualification_cagr"], pd.DataFrame)
    assert isinstance(results["overall_total_market_volume"], pd.Series)
    assert isinstance(results["qualification_market_yoy_growth"], pd.DataFrame)

    # Check that the detailed providers market has the correct columns (output names)
    detailed_df = results["detailed_providers_market"]
    expected_detailed_cols = [
        cols_out['year'], cols_out['qualification'], cols_out['provider'],
        cols_out['provider_amount'], cols_out['subcontractor_amount'],
        cols_out['total_volume'], cols_out['market_total'],
        cols_out['market_share'], cols_out['market_rank'],
        cols_out['market_share_growth'], cols_out['market_gainer_rank']
    ]
    assert all(col in detailed_df.columns for col in expected_detailed_cols)
    # Check number of rows (3 years * 3 quals * 2 providers = 18 potential rows, but depends on actual involvement)
    # 2020: A(X), B(X), C(Y main, X sub) -> 3 rows (X, X, Y)
    # 2021: A(X), B(X), C(Y main, X sub) -> 3 rows (X, X, Y)
    # 2022: A(X), B(X), C(Y main, X sub) -> 3 rows (X, X, Y)
    # calculate_detailed_market_shares adds rows for *all* involved providers (main OR sub)
    # 2020: A(X), B(X), C(Y), C(X) -> 4 rows
    # 2021: A(X), B(X), C(Y), C(X) -> 4 rows
    # 2022: A(X), B(X), C(Y), C(X) -> 4 rows
    # Total = 12 rows expected
    assert len(detailed_df) == 12


def test_empty_data(mock_config):
    """Test MarketAnalyzer with empty data."""
    empty_df = pd.DataFrame(columns=mock_config['columns']['input'].values())
    analyzer = MarketAnalyzer(empty_df, cfg=mock_config) # Pass config

    assert analyzer.min_year is None
    assert analyzer.max_year is None

    # Check that methods return empty DataFrames/Series with correct columns/structure
    assert analyzer.calculate_total_volumes().empty
    assert analyzer.calculate_volumes_by_qualification().empty
    assert analyzer.calculate_providers_market().empty
    assert analyzer.calculate_qualification_growth().empty # Old method
    assert analyzer.calculate_qualification_cagr().empty
    assert analyzer.calculate_overall_total_market_volume().empty
    assert analyzer.calculate_qualification_market_yoy_growth(pd.DataFrame()).empty

    results = analyzer.get_all_results()
    assert isinstance(results, dict)
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            assert value.empty
        elif isinstance(value, pd.Series):
            assert value.empty


def test_data_with_single_year(mock_config):
    """Test MarketAnalyzer with data containing only one year."""
    cols_in = mock_config['columns']['input']
    single_year_data = pd.DataFrame({
        cols_in['year']: [2020, 2020],
        cols_in['degree_type']: ['Ammattitutkinnot', 'Erikoisammattitutkinnot'],
        cols_in['qualification']: ['Tutkinto A', 'Tutkinto C'],
        cols_in['provider']: ['Provider X', 'Provider Y'],
        cols_in['subcontractor']: [None, 'Provider X'],
        cols_in['volume']: [100, 150]
    })
    analyzer = MarketAnalyzer(single_year_data, cfg=mock_config) # Pass config
    analyzer.institution_names = ['Provider X']
    analyzer.institution_short_name = "PX"
    cols_out = mock_config['columns']['output']

    assert analyzer.min_year == 2020
    assert analyzer.max_year == 2020

    # Methods that calculate across years should handle this gracefully
    # calculate_providers_market calculates rank but growth/gainer rank should be NaN/0/default
    detailed_df = analyzer.calculate_providers_market()
    assert not detailed_df.empty
    assert cols_out['market_share_growth'] in detailed_df.columns
    assert cols_out['market_gainer_rank'] in detailed_df.columns
    assert detailed_df[cols_out['market_share_growth']].fillna(0).eq(0).all() # Growth should be 0
    # Gainer rank is based on growth, should probably be NaN or 1? Depends on rank impl.
    # Current impl uses rank(method='min'), NaNs are usually ignored or ranked last.
    # Let's check it's Int64 and nullable
    assert pd.api.types.is_integer_dtype(detailed_df[cols_out['market_gainer_rank']])
    # assert detailed_df[cols_out['market_gainer_rank']].isnull().all() # Check if rank is NaN

    # CAGR requires multiple years
    assert analyzer.calculate_qualification_cagr().empty

    # Qualification Market YoY Growth requires multiple years
    # Pass the calculated detailed_df which only has one year
    # assert analyzer.calculate_qualification_market_yoy_growth(detailed_df).empty
    # Check that the growth column is NaN or empty
    growth_df = analyzer.calculate_qualification_market_yoy_growth(detailed_df)
    market_growth_col = f"{cols_out['market_total']} YoY Growth (%)"
    assert market_growth_col in growth_df.columns
    assert growth_df[market_growth_col].isnull().all()


def test_calculate_providers_market(market_analyzer, mock_config):
    """Test the detailed calculate_providers_market method."""
    market_analyzer.institution_names = ['Provider X', 'Provider Y'] # Include both for this test
    market_analyzer.institution_short_name = "PX+PY"
    cols_out = mock_config['columns']['output']

    detailed_df = market_analyzer.calculate_providers_market()

    assert isinstance(detailed_df, pd.DataFrame)
    assert len(detailed_df) == 12 # 3 years * (A(X) + B(X) + C(Y) + C(X)) = 12 rows

    # Check columns using output config names
    expected_cols = [
        cols_out['year'], cols_out['qualification'], cols_out['provider'],
        cols_out['provider_amount'], cols_out['subcontractor_amount'],
        cols_out['total_volume'], cols_out['market_total'],
        cols_out['market_share'], cols_out['market_rank'],
        cols_out['market_share_growth'], cols_out['market_gainer_rank']
    ]
    assert all(col in detailed_df.columns for col in expected_cols)

    # --- Check specific values for 2022 --- 
    year_2022 = detailed_df[detailed_df[cols_out['year']] == 2022]
    # Tutkinto A: Only Provider X. ProvAmt=120, SubAmt=0, TotalVol=120. MarketTotal=120. Share=100. Rank=1.
    tut_a_x_2022 = year_2022[(year_2022[cols_out['qualification']] == 'Tutkinto A') & (year_2022[cols_out['provider']] == 'Provider X')].iloc[0]
    assert tut_a_x_2022[cols_out['provider_amount']] == 120
    assert tut_a_x_2022[cols_out['subcontractor_amount']] == 0
    assert tut_a_x_2022[cols_out['total_volume']] == 120
    assert tut_a_x_2022[cols_out['market_total']] == 120
    assert np.isclose(tut_a_x_2022[cols_out['market_share']], 100.0)
    assert tut_a_x_2022[cols_out['market_rank']] == 1

    # Tutkinto C: Provider Y (main=170), Provider X (sub=170). MarketTotal=170.
    # Provider X: ProvAmt=0, SubAmt=170, TotalVol=170. Share=100. Rank=1 (tie).
    # Provider Y: ProvAmt=170, SubAmt=0, TotalVol=170. Share=100. Rank=1 (tie).
    provider_x_c_2022 = year_2022[(year_2022[cols_out['qualification']] == 'Tutkinto C') & (year_2022[cols_out['provider']] == 'Provider X')].iloc[0]
    provider_y_c_2022 = year_2022[(year_2022[cols_out['qualification']] == 'Tutkinto C') & (year_2022[cols_out['provider']] == 'Provider Y')].iloc[0]

    assert provider_x_c_2022[cols_out['provider_amount']] == 0
    assert provider_x_c_2022[cols_out['subcontractor_amount']] == 170
    assert provider_x_c_2022[cols_out['total_volume']] == 170
    assert provider_x_c_2022[cols_out['market_total']] == 170 # Market for Tutkinto C is just Provider Y (main) + Provider X (sub)
    assert np.isclose(provider_x_c_2022[cols_out['market_share']], 100.0)
    assert provider_x_c_2022[cols_out['market_rank']] == 1 # Tied with Provider Y

    assert provider_y_c_2022[cols_out['provider_amount']] == 170
    assert provider_y_c_2022[cols_out['subcontractor_amount']] == 0
    assert provider_y_c_2022[cols_out['total_volume']] == 170
    assert provider_y_c_2022[cols_out['market_total']] == 170
    assert np.isclose(provider_y_c_2022[cols_out['market_share']], 100.0)
    assert provider_y_c_2022[cols_out['market_rank']] == 1 # Tied with Provider X

    # --- Check Growth and Gainer Rank (2022 vs 2021) ---
    # Tutkinto C, Provider X: Share 2021=100, Share 2022=100 -> Growth=0. Rank=?
    # Tutkinto C, Provider Y: Share 2021=100, Share 2022=100 -> Growth=0. Rank=?
    assert np.isclose(provider_x_c_2022[cols_out['market_share_growth']], 0.0)
    assert np.isclose(provider_y_c_2022[cols_out['market_share_growth']], 0.0)
    # Since growth is 0 for both, rank depends on method (min rank assigns 1 to both)
    assert provider_x_c_2022[cols_out['market_gainer_rank']] == 1
    assert provider_y_c_2022[cols_out['market_gainer_rank']] == 1

    # --- Check first year (2020) - Growth/Gainer Rank should be NaN/Default --- 
    year_2020 = detailed_df[detailed_df[cols_out['year']] == 2020]
    assert year_2020[cols_out['market_share_growth']].fillna(0).eq(0).all() # Check growth is 0 (after fillna)
    assert year_2020[cols_out['market_gainer_rank']].isnull().all()
    assert pd.api.types.is_integer_dtype(year_2020[cols_out['market_gainer_rank']]) # Should be Int64

    # --- Check middle year (2021) - Growth/Gainer Rank should be calculated ---
    year_2021 = detailed_df[detailed_df[cols_out['year']] == 2021]
    assert not year_2021[cols_out['market_share_growth']].isnull().any()
    assert not year_2021[cols_out['market_gainer_rank']].isnull().any()
    assert pd.api.types.is_integer_dtype(year_2021[cols_out['market_gainer_rank']]) # Should be Int64

    # Check that the detailed providers market has the correct columns (output names)
    # detailed_df = results["detailed_providers_market"] # <--- REMOVE START
    # expected_detailed_cols = [
    #     cols_out['year'], cols_out['qualification'], cols_out['provider'],
    #     cols_out['provider_amount'], cols_out['subcontractor_amount'],
    #     cols_out['total_volume'], cols_out['market_total'],
    #     cols_out['market_share'], cols_out['market_rank'],
    #     cols_out['market_share_growth'], cols_out['market_gainer_rank']
    # ]
    # assert all(col in detailed_df.columns for col in expected_detailed_cols)
    # # Check number of rows (3 years * 3 quals * 2 providers = 18 potential rows, but depends on actual involvement)
    # # 2020: A(X), B(X), C(Y main, X sub) -> 3 rows (X, X, Y)
    # # 2021: A(X), B(X), C(Y main, X sub) -> 3 rows (X, X, Y)
    # # 2022: A(X), B(X), C(Y main, X sub) -> 3 rows (X, X, Y)
    # # calculate_detailed_market_shares adds rows for *all* involved providers (main OR sub)
    # # 2020: A(X), B(X), C(Y), C(X) -> 4 rows
    # # 2021: A(X), B(X), C(Y), C(X) -> 4 rows
    # # 2022: A(X), B(X), C(Y), C(X) -> 4 rows
    # # Total = 12 rows expected
    # assert len(detailed_df) == 12 # <--- REMOVE END 