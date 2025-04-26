"""
Pytest configuration and fixtures for the vipunen-project test suite.
"""
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Define constants for testing
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_OUTPUT_DIR = Path(__file__).parent / "output"


@pytest.fixture(scope="session", autouse=True)
def setup_test_dirs():
    """Create test directories needed for testing."""
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Cleanup function to run after tests complete
    yield
    
    # Uncomment if you want to clean up test output after tests
    # import shutil
    # shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)


@pytest.fixture
def sample_education_data():
    """Create a sample DataFrame for education data testing."""
    return pd.DataFrame({
        'tilastovuosi': [2020, 2020, 2021, 2021, 2022, 2022],
        'koulutuksenJarjestaja': ['Provider A', 'Provider B', 'Provider A', 'Provider B', 'Provider A', 'Provider B'],
        'hankintakoulutuksenJarjestaja': [None, 'Provider A', None, 'Provider A', None, 'Provider A'],
        'tutkintotyyppi': ['Ammattitutkinnot', 'Erikoisammattitutkinnot', 'Ammattitutkinnot', 
                          'Erikoisammattitutkinnot', 'Ammattitutkinnot', 'Erikoisammattitutkinnot'],
        'tutkinto': ['Liiketoiminnan ammattitutkinto', 'Johtamisen erikoisammattitutkinto', 
                    'Liiketoiminnan ammattitutkinto', 'Johtamisen erikoisammattitutkinto', 
                    'Liiketoiminnan ammattitutkinto', 'Johtamisen erikoisammattitutkinto'],
        'nettoopiskelijamaaraLkm': [100, 150, 120, 140, 130, 145]
    })


@pytest.fixture
def sample_market_data():
    """Create sample market data with multiple years, qualifications, and providers."""
    data = {
        'tilastovuosi': [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022],
        'tutkinto': ['Tutkinto A', 'Tutkinto A', 'Tutkinto B', 'Tutkinto A', 'Tutkinto A', 'Tutkinto B', 'Tutkinto A', 'Tutkinto A', 'Tutkinto B'],
        'tutkintotyyppi': ['Ammattitutkinnot', 'Ammattitutkinnot', 'Erikoisammattitutkinnot',
                          'Ammattitutkinnot', 'Ammattitutkinnot', 'Erikoisammattitutkinnot',
                          'Ammattitutkinnot', 'Ammattitutkinnot', 'Erikoisammattitutkinnot'],
        'koulutuksenJarjestaja': ['Provider X', 'Provider Y', 'Provider Z', 'Provider X', 'Provider Y', 'Provider Z', 'Provider X', 'Provider Y', 'Provider Z'],
        'hankintakoulutuksenJarjestaja': ['Provider Y', 'Provider X', None, 'Provider Y', 'Provider X', None, 'Provider Y', 'Provider X', None],
        'nettoopiskelijamaaraLkm': [100, 200, 300, 150, 250, 350, 200, 300, 400]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_volume_data():
    """Create sample volume data for visualization testing."""
    return pd.DataFrame({
        'tilastovuosi': [2020, 2021, 2022],
        'järjestäjänä': [250, 300, 350],
        'hankintana': [150, 180, 200],
        'Yhteensä': [400, 480, 550],
        'järjestäjä_osuus (%)': [62.5, 62.5, 63.6]
    })


@pytest.fixture
def sample_volumes_by_qualification():
    """Create sample qualification volume data for visualization testing."""
    data = []
    years = [2020, 2021, 2022]
    qualifications = ['Liiketoiminnan AT', 'Johtamisen EAT', 'Myynnin AT', 'Yrittäjyyden AT']
    
    for year in years:
        for qual in qualifications:
            # Values increase slightly each year
            factor = 1 + (year - 2020) * 0.1
            provider_vol = int(100 * factor if qual == 'Liiketoiminnan AT' else 50 * factor)
            subcontractor_vol = int(50 * factor if qual == 'Johtamisen EAT' else 25 * factor)
            total_vol = provider_vol + subcontractor_vol
            market_vol = int(total_vol * 5)  # Total market is 5x the institution's volume
            
            data.append({
                'tilastovuosi': year,
                'tutkinto': qual,
                'tutkintotyyppi': 'Ammattitutkinnot' if 'AT' in qual else 'Erikoisammattitutkinnot',
                f'{year}_järjestäjänä': provider_vol,
                f'{year}_hankintana': subcontractor_vol,
                f'{year}_yhteensä': total_vol,
                f'{year}_market_total': market_vol,
                f'{year}_market_share': (total_vol / market_vol * 100)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_file_utils():
    """Create a mock FileUtils instance."""
    with patch('src.vipunen.utils.file_utils_config.get_file_utils') as mock_get_file_utils:
        mock_file_utils = MagicMock()
        mock_get_file_utils.return_value = mock_file_utils
        
        # Mock the load_single_file method
        mock_file_utils.load_single_file.return_value = pd.DataFrame({
            'tilastovuosi': [2020, 2021, 2022],
            'koulutuksenJarjestaja': ['Provider A', 'Provider A', 'Provider A'],
            'nettoopiskelijamaaraLkm': [100, 120, 130]
        })
        
        # Mock the save_data_to_storage method
        mock_file_utils.save_data_to_storage.return_value = Path('mock/output/test.xlsx')
        
        yield mock_file_utils 