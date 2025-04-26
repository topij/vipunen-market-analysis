"""
Tests for the analyze_education_market.py script.
"""
import pytest
import pandas as pd
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import analyze_education_market
from tests.conftest import TEST_DATA_DIR, TEST_OUTPUT_DIR


@pytest.fixture
def mock_export_to_excel():
    """Mock the export_to_excel function."""
    with patch('analyze_education_market.export_to_excel') as mock_export:
        mock_export.return_value = Path('mock/output/test.xlsx')
        yield mock_export


@pytest.fixture
def mock_plot_functions():
    """Mock all plot functions."""
    with patch('analyze_education_market.plot_total_volumes') as mock_total_volumes, \
         patch('analyze_education_market.plot_top_qualifications') as mock_top_quals, \
         patch('analyze_education_market.plot_market_share_heatmap') as mock_heatmap, \
         patch('analyze_education_market.plot_qualification_market_shares') as mock_qual_shares, \
         patch('analyze_education_market.plot_qualification_growth') as mock_growth, \
         patch('analyze_education_market.plot_qualification_time_series') as mock_time_series:
         
        # Set return values
        mock_total_volumes.return_value = 'mock/plots/total_volumes.png'
        mock_top_quals.return_value = 'mock/plots/top_qualifications.png'
        mock_heatmap.return_value = 'mock/plots/market_share_heatmap.png'
        mock_qual_shares.return_value = 'mock/plots/qualification_market_shares.png'
        mock_growth.return_value = 'mock/plots/qualification_growth.png'
        mock_time_series.return_value = 'mock/plots/qualification_time_series.png'
        
        yield {
            'total_volumes': mock_total_volumes,
            'top_qualifications': mock_top_quals,
            'market_share_heatmap': mock_heatmap,
            'qualification_market_shares': mock_qual_shares,
            'qualification_growth': mock_growth,
            'qualification_time_series': mock_time_series
        }


def test_parse_arguments():
    """Test the argument parsing function."""
    with patch('sys.argv', ['analyze_education_market.py']):
        args = analyze_education_market.parse_arguments()
        
        # Check default values
        assert args.data_file == "data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv"
        assert args.institution == "Rastor-instituutti ry"
        assert args.short_name == "RI"
        assert args.variants == []
        assert args.output_dir is None
        assert not args.use_dummy
        assert not args.filter_qual_types
        assert not args.filter_by_inst_quals
    
    # Test with custom arguments
    with patch('sys.argv', [
        'analyze_education_market.py',
        '--data-file', 'custom/path.csv',
        '--institution', 'Test Institution',
        '--short-name', 'Test',
        '--variant', 'Test Variant 1',
        '--variant', 'Test Variant 2',
        '--output-dir', 'custom/output',
        '--use-dummy',
        '--filter-qual-types',
        '--filter-by-institution-quals'
    ]):
        args = analyze_education_market.parse_arguments()
        
        # Check custom values
        assert args.data_file == "custom/path.csv"
        assert args.institution == "Test Institution"
        assert args.short_name == "Test"
        assert args.variants == ['Test Variant 1', 'Test Variant 2']
        assert args.output_dir == "custom/output"
        assert args.use_dummy
        assert args.filter_qual_types
        assert args.filter_by_inst_quals


def test_ensure_data_directory():
    """Test the ensure_data_directory function."""
    # Test with path starting with "raw/"
    path = "raw/test.csv"
    result = analyze_education_market.ensure_data_directory(path)
    assert result == "data/raw/test.csv"
    
    # Test with path not starting with "raw/"
    path = "data/raw/test.csv"
    result = analyze_education_market.ensure_data_directory(path)
    assert result == "data/raw/test.csv"
    
    # Test with absolute path
    path = "/absolute/path/test.csv"
    result = analyze_education_market.ensure_data_directory(path)
    assert result == "/absolute/path/test.csv"


@patch('analyze_education_market.file_utils')
def test_export_to_excel(mock_file_utils):
    """Test the export_to_excel function."""
    # Create mock data
    data_dict = {
        'Sheet1': pd.DataFrame({'A': [1, 2, 3]}),
        'Sheet2': pd.DataFrame({'B': [4, 5, 6]})
    }
    
    # Mock the save_data_to_storage method
    mock_file_utils.save_data_to_storage.return_value = 'mock/output/test.xlsx'
    
    # Call the function
    result = analyze_education_market.export_to_excel(
        data_dict=data_dict,
        file_name='test',
        output_type='reports'
    )
    
    # Check that file_utils.save_data_to_storage was called with the right arguments
    mock_file_utils.save_data_to_storage.assert_called_once()
    call_args = mock_file_utils.save_data_to_storage.call_args[1]
    assert call_args['file_name'] == 'test'
    assert call_args['output_type'] == 'reports'
    assert call_args['output_filetype'] == analyze_education_market.OutputFileType.XLSX
    assert call_args['index'] is False
    
    # Check that the result is the expected path
    assert result == Path('mock/output/test.xlsx')


@patch('analyze_education_market.file_utils')
def test_export_to_excel_with_empty_data(mock_file_utils):
    """Test export_to_excel with empty data."""
    # Create mock data with an empty DataFrame
    data_dict = {
        'Sheet1': pd.DataFrame(),
        'Sheet2': pd.DataFrame({'B': [4, 5, 6]})
    }
    
    # Call the function
    result = analyze_education_market.export_to_excel(
        data_dict=data_dict,
        file_name='test',
        output_type='reports'
    )
    
    # Check that file_utils.save_data_to_storage was called with only non-empty data
    mock_file_utils.save_data_to_storage.assert_called_once()
    call_args = mock_file_utils.save_data_to_storage.call_args[1]
    assert 'Sheet1' not in call_args['data']
    assert 'Sheet2' in call_args['data']


@patch('analyze_education_market.file_utils')
def test_export_to_excel_error_handling(mock_file_utils):
    """Test export_to_excel error handling."""
    # Create mock data
    data_dict = {
        'Sheet1': pd.DataFrame({'A': [1, 2, 3]})
    }
    
    # Make the save_data_to_storage method raise an exception
    mock_file_utils.save_data_to_storage.side_effect = Exception("Test error")
    
    # Call the function
    with pytest.raises(Exception):
        analyze_education_market.export_to_excel(
            data_dict=data_dict,
            file_name='test',
            output_type='reports'
        )


def test_create_dummy_dataset():
    """Test the create_dummy_dataset function."""
    # Call the function
    result = analyze_education_market.create_dummy_dataset()
    
    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that the required columns are present
    required_columns = [
        'tilastovuosi', 'tutkintotyyppi', 'tutkinto',
        'koulutuksenJarjestaja', 'hankintakoulutuksenJarjestaja',
        'nettoopiskelijamaaraLkm'
    ]
    assert all(col in result.columns for col in required_columns)
    
    # Check that there are reasonable number of rows
    # For the years 2017-2024, 5 qualifications, 6 providers -> 8*5*6 = 240 rows
    assert len(result) > 100


@patch('analyze_education_market.logger')
@patch('analyze_education_market.file_utils.load_single_file')
@patch('analyze_education_market.file_utils.create_directory')
@patch('analyze_education_market.file_utils.save_data_to_storage')
@patch('analyze_education_market.pd.ExcelWriter')
@patch('analyze_education_market.pd.DataFrame.to_excel')
def test_main_function(
    mock_to_excel,
    mock_excel_writer,
    mock_save_data,
    mock_create_dir,
    mock_load_file,
    mock_logger,
    sample_education_data,
    mock_plot_functions
):
    """Test the main function with mocked dependencies."""
    # Configure mocks
    mock_create_dir.return_value = Path('mock/data/reports')
    mock_load_file.return_value = sample_education_data
    mock_save_data.return_value = Path('mock/output/test.xlsx')
    
    # Mock command line arguments
    with patch('sys.argv', [
        'analyze_education_market.py',
        '--use-dummy',  # Use dummy data to avoid file loading
        '--output-dir', str(TEST_OUTPUT_DIR)
    ]):
        # Call the main function
        result = analyze_education_market.main()
        
        # Check that the result is a dictionary
        assert isinstance(result, dict)
        
        # Check that all expected keys are present
        expected_keys = [
            'total_volumes',
            'volumes_by_qualification',
            'volumes_long',
            'market_shares',
            'qualification_cagr',
            'excel_path'
        ]
        assert all(key in result for key in expected_keys)
        
        # Check that all plot functions were called
        for mock_plot in mock_plot_functions.values():
            mock_plot.assert_called_once() 