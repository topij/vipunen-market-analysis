d"""Tests for the data filtering module."""
import pytest
import pandas as pd
import numpy as np
from vipunen.processing.data_filter import (
    filter_degree_types,
    standardize_provider_name,
    standardize_provider_names,
    filter_by_provider,
    filter_data
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'tilastovuosi': [2020, 2020, 2021, 2021, 2022, 2022] * 2,
        'tutkintotyyppi': ['ammattitutkinto', 'erikoisammattitutkinto', 'ammattitutkinto', 
                          'erikoisammattitutkinto', 'ammattitutkinto', 'erikoisammattitutkinto'] * 2,
        'tutkinto': ['A', 'B', 'A', 'B', 'A', 'B'] * 2,
        'koulutuksenJarjestaja': ['Provider1 Oy', 'Provider1 Oy', 'Provider1 Oy', 
                                'Provider1 Oy', 'Provider1 Oy', 'Provider1 Oy'] + 
                               ['Provider2 Ab', 'Provider2 Ab', 'Provider2 Ab', 
                                'Provider2 Ab', 'Provider2 Ab', 'Provider2 Ab'],
        'hankintakoulutuksenJarjestaja': ['Provider2 Ab', 'Provider2 Ab', 'Provider2 Ab', 
                                        'Provider2 Ab', 'Provider2 Ab', 'Provider2 Ab'] + 
                                       ['Provider1 Oy', 'Provider1 Oy', 'Provider1 Oy', 
                                        'Provider1 Oy', 'Provider1 Oy', 'Provider1 Oy'],
        'nettoopiskelijamaaraLkm': [100, 200, 150, 250, 200, 300] * 2
    }
    return pd.DataFrame(data)

def test_filter_degree_types(sample_data):
    """Test filtering of degree types."""
    result = filter_degree_types(sample_data)
    
    # Check that only valid degree types are included
    assert len(result) == len(sample_data)  # All rows should be valid
    assert set(result['tutkintotyyppi'].unique()) == {'ammattitutkinto', 'erikoisammattitutkinto'}

def test_standardize_provider_name():
    """Test provider name standardization."""
    # Test common variations
    assert standardize_provider_name('Provider1 Oy') == 'provider1'
    assert standardize_provider_name('Provider1 Ab') == 'provider1'
    assert standardize_provider_name('Provider1 Ry') == 'provider1'
    assert standardize_provider_name('Provider1 Oyj') == 'provider1'
    
    # Test with special characters
    assert standardize_provider_name('Provider-1 Oy') == 'provider 1'
    assert standardize_provider_name('Provider 1 Oy') == 'provider 1'
    
    # Test with empty or invalid values
    assert pd.isna(standardize_provider_name(np.nan))
    assert standardize_provider_name('') == ''

def test_standardize_provider_names(sample_data):
    """Test standardization of provider names in DataFrame."""
    result = standardize_provider_names(sample_data)
    
    # Check that provider names are standardized
    assert all(result['koulutuksenJarjestaja'].str.lower() == result['koulutuksenJarjestaja'])
    assert all(result['hankintakoulutuksenJarjestaja'].str.lower() == result['hankintakoulutuksenJarjestaja'])
    
    # Check that suffixes are removed
    assert not any(result['koulutuksenJarjestaja'].str.endswith((' oy', ' ab', ' ry')))
    assert not any(result['hankintakoulutuksenJarjestaja'].str.endswith((' oy', ' ab', ' ry')))

def test_filter_by_provider(sample_data):
    """Test filtering by provider."""
    # Test with main provider name
    result = filter_by_provider(sample_data, 'Provider1')
    assert len(result) == 12  # All rows should be included
    
    # Test with provider variations
    variations = ['Provider1 Oy', 'Provider1 Ab']
    result = filter_by_provider(sample_data, 'Provider1', variations)
    assert len(result) == 12  # All rows should be included
    
    # Test with non-matching provider
    result = filter_by_provider(sample_data, 'NonExistentProvider')
    assert len(result) == 0

def test_filter_data(sample_data):
    """Test comprehensive data filtering."""
    # Test with main provider name
    result = filter_data(sample_data, 'Provider1')
    assert len(result) == 12  # All rows should be included
    
    # Test with provider variations
    variations = ['Provider1 Oy', 'Provider1 Ab']
    result = filter_data(sample_data, 'Provider1', variations)
    assert len(result) == 12  # All rows should be included
    
    # Check that degree types are filtered
    assert set(result['tutkintotyyppi'].unique()) == {'ammattitutkinto', 'erikoisammattitutkinto'}
    
    # Check that provider names are standardized
    assert not any(result['koulutuksenJarjestaja'].str.endswith((' oy', ' ab', ' ry')))
    assert not any(result['hankintakoulutuksenJarjestaja'].str.endswith((' oy', ' ab', ' ry'))) 