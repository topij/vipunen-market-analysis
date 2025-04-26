"""
Tests for data loading and preparation functions.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from src.vipunen.data.loader import load_data
from src.vipunen.data.preparation import clean_and_prepare_data


@pytest.fixture
def sample_raw_data():
    """Create a sample dataframe with raw data."""
    return pd.DataFrame({
        'tilastovuosi': [2020, 2021, 2022] * 4,
        'tutkintotyyppi': ['Ammattitutkinnot'] * 6 + ['Erikoisammattitutkinnot'] * 6,
        'tutkinto': ['Tutkinto A'] * 3 + ['Tutkinto B'] * 3 + ['Tutkinto C'] * 3 + ['Tutkinto D'] * 3,
        'koulutuksenJarjestaja': ['Provider X'] * 6 + ['Provider Y'] * 6,
        'hankintakoulutuksenJarjestaja': [None] * 6 + ['Provider X'] * 6,
        'nettoopiskelijamaaraLkm': [100, 110, 120, 200, 210, 220, 150, 160, 170, 250, 260, 270]
    })


@patch('pandas.read_csv')
def test_load_data_from_csv(mock_read_csv, sample_raw_data):
    """Test loading data from a CSV file."""
    # Configure mock to return sample data
    mock_read_csv.return_value = sample_raw_data
    
    # Call the function with a test file path
    file_path = 'test_data.csv'
    result = load_data(file_path=file_path)
    
    # Check that read_csv was called with the correct file path
    mock_read_csv.assert_called_once_with(file_path)
    
    # Verify result is the expected dataframe
    pd.testing.assert_frame_equal(result, sample_raw_data)


@patch('pandas.read_csv')
def test_load_data_with_dummy_data(mock_read_csv):
    """Test loading dummy data."""
    # Call the function with use_dummy=True
    result = load_data(file_path='doesnt_matter.csv', use_dummy=True)
    
    # Verify that read_csv was not called
    mock_read_csv.assert_not_called()
    
    # Verify result is a DataFrame with expected structure
    assert isinstance(result, pd.DataFrame)
    assert 'tilastovuosi' in result.columns
    assert 'tutkintotyyppi' in result.columns
    assert 'tutkinto' in result.columns
    assert 'koulutuksenJarjestaja' in result.columns
    assert 'hankintakoulutuksenJarjestaja' in result.columns
    assert 'nettoopiskelijamaaraLkm' in result.columns
    assert not result.empty


def test_clean_and_prepare_data(sample_raw_data):
    """Test data cleaning and preparation."""
    # Call the function with sample data
    result = clean_and_prepare_data(sample_raw_data)
    
    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that we have the same number of rows
    assert len(result) == len(sample_raw_data)
    
    # Check that all required columns exist
    required_columns = [
        'tilastovuosi', 'tutkintotyyppi', 'tutkinto', 
        'koulutuksenJarjestaja', 'hankintakoulutuksenJarjestaja',
        'nettoopiskelijamaaraLkm'
    ]
    for col in required_columns:
        assert col in result.columns
    
    # Check that no null values exist in important columns
    for col in ['tilastovuosi', 'tutkintotyyppi', 'tutkinto', 'nettoopiskelijamaaraLkm']:
        assert result[col].isnull().sum() == 0
    
    # Check that tilastovuosi is numeric
    assert pd.api.types.is_numeric_dtype(result['tilastovuosi'])
    
    # Check that nettoopiskelijamaaraLkm is numeric
    assert pd.api.types.is_numeric_dtype(result['nettoopiskelijamaaraLkm'])


def test_clean_and_prepare_data_with_missing_values():
    """Test data cleaning with missing values."""
    # Create data with some missing values
    data_with_missing = pd.DataFrame({
        'tilastovuosi': [2020, 2021, None, 2022],
        'tutkintotyyppi': ['Ammattitutkinnot', None, 'Ammattitutkinnot', 'Erikoisammattitutkinnot'],
        'tutkinto': ['Tutkinto A', 'Tutkinto B', None, 'Tutkinto D'],
        'koulutuksenJarjestaja': ['Provider X', 'Provider X', 'Provider Y', None],
        'hankintakoulutuksenJarjestaja': [None, None, 'Provider X', 'Provider X'],
        'nettoopiskelijamaaraLkm': [100, None, 150, 250]
    })
    
    # Call the function
    result = clean_and_prepare_data(data_with_missing)
    
    # Check that rows with missing values in critical columns were dropped
    assert len(result) < len(data_with_missing)
    assert result['tilastovuosi'].isnull().sum() == 0
    assert result['tutkintotyyppi'].isnull().sum() == 0
    assert result['tutkinto'].isnull().sum() == 0
    assert result['nettoopiskelijamaaraLkm'].isnull().sum() == 0


@patch('pandas.read_csv')
def test_load_data_file_not_found(mock_read_csv):
    """Test load_data with file not found error."""
    # Configure mock to raise FileNotFoundError
    mock_read_csv.side_effect = FileNotFoundError("File not found")
    
    # Call the function and expect it to raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_data(file_path='nonexistent_file.csv')


@patch('pandas.read_csv')
def test_load_data_with_invalid_file(mock_read_csv):
    """Test load_data with invalid file."""
    # Configure mock to raise an exception
    mock_read_csv.side_effect = Exception("Invalid file")
    
    # Call the function and expect it to raise Exception
    with pytest.raises(Exception):
        load_data(file_path='invalid_file.csv') 