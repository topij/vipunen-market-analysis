"""
Tests for the data_processor module.
"""
import pytest
import pandas as pd
import numpy as np
from vipunen.data.data_processor import (
    clean_and_prepare_data,
    shorten_qualification_names,
    merge_qualification_variants,
    replace_values_in_dataframe
)


def test_clean_and_prepare_data(sample_education_data):
    """Test the clean_and_prepare_data function."""
    # Test with default parameters
    result = clean_and_prepare_data(sample_education_data)
    
    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that the shape is the same as the input
    assert result.shape == sample_education_data.shape
    
    # Test with institution names
    institution_names = ['Provider A']
    result = clean_and_prepare_data(sample_education_data, institution_names=institution_names)
    
    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Test with all options enabled
    result = clean_and_prepare_data(
        sample_education_data,
        institution_names=['Provider A'],
        merge_qualifications=True,
        shorten_names=True
    )
    
    # Check that qualifications have been shortened
    assert 'AT' in result['tutkinto'].iloc[0] or 'EAT' in result['tutkinto'].iloc[0]


def test_shorten_qualification_names(sample_education_data):
    """Test the shorten_qualification_names function."""
    # Test shortening qualification names
    result = shorten_qualification_names(sample_education_data)
    
    # Check that 'ammattitutkinto' has been replaced with 'AT'
    assert 'Liiketoiminnan AT' in result['tutkinto'].values
    
    # Check that 'erikoisammattitutkinto' has been replaced with 'EAT'
    assert 'Johtamisen EAT' in result['tutkinto'].values
    
    # Test with a different column
    # First create a copy of the data with a different column name
    df_with_custom_col = sample_education_data.rename(columns={'tutkinto': 'custom_column'})
    
    # Now test with the custom column
    result = shorten_qualification_names(df_with_custom_col, column='custom_column')
    
    # Check that qualification names have been shortened in the custom column
    assert 'Liiketoiminnan AT' in result['custom_column'].values
    
    # Test with a column that doesn't exist
    result = shorten_qualification_names(sample_education_data, column='nonexistent_column')
    
    # Check that the result is the same as the input (no changes)
    assert result.equals(sample_education_data)


def test_merge_qualification_variants():
    """Test the merge_qualification_variants function."""
    # Create sample data with qualification variants
    data = pd.DataFrame({
        'tutkinto': [
            'Yrittäjän ammattitutkinto',
            'Yrittäjyyden ammattitutkinto',
            'Some other qualification'
        ]
    })
    
    # Test with default mapping
    result = merge_qualification_variants(data)
    
    # Check that 'Yrittäjän ammattitutkinto' has been replaced
    assert 'Yrittäjän ammattitutkinto' not in result['tutkinto'].values
    assert 'Yrittäjyyden ammattitutkinto' in result['tutkinto'].values
    assert len(result[result['tutkinto'] == 'Yrittäjyyden ammattitutkinto']) == 2
    
    # Test with custom mapping
    custom_mapping = {
        'Some other qualification': 'Custom Qualification'
    }
    
    result = merge_qualification_variants(data, mapping_dict=custom_mapping)
    
    # Check that 'Some other qualification' has been replaced
    assert 'Some other qualification' not in result['tutkinto'].values
    assert 'Custom Qualification' in result['tutkinto'].values
    
    # Test with a column that doesn't exist
    result = merge_qualification_variants(data, column='nonexistent_column')
    
    # Check that the result is the same as the input (no changes)
    assert result.equals(data)


def test_replace_values_in_dataframe():
    """Test the replace_values_in_dataframe function."""
    # Create sample data
    data = pd.DataFrame({
        'col1': ['A', 'B', 'Tieto puuttuu'],
        'col2': [1, 2, 3]
    })
    
    # Test replacing values
    replacements = {
        'col1': {'A': 'X', 'B': 'Y', 'Tieto puuttuu': None}
    }
    
    result = replace_values_in_dataframe(data, replacements)
    
    # Check that values have been replaced
    assert 'A' not in result['col1'].values
    assert 'B' not in result['col1'].values
    assert 'X' in result['col1'].values
    assert 'Y' in result['col1'].values
    
    # Check that 'Tieto puuttuu' has been replaced with NaN
    assert result['col1'].isna().sum() == 1
    
    # Test with a column that doesn't exist
    replacements = {
        'nonexistent_column': {'A': 'X'}
    }
    
    result = replace_values_in_dataframe(data, replacements)
    
    # Check that the result is the same as the input (no changes)
    assert result.equals(data) 