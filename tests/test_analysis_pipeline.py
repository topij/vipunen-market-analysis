"""
Tests for the full education market analysis pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.vipunen.cli.analyze_cli import run_analysis


@pytest.fixture
def mock_dependencies():
    """Mock all dependencies for the analysis pipeline."""
    with patch('src.vipunen.cli.analyze_cli.load_data') as mock_load_data, \
         patch('src.vipunen.cli.analyze_cli.clean_and_prepare_data') as mock_clean_data, \
         patch('src.vipunen.cli.analyze_cli.MarketAnalyzer') as mock_analyzer_class, \
         patch('src.vipunen.cli.analyze_cli.export_to_excel') as mock_export, \
         patch('src.vipunen.cli.analyze_cli.get_config') as mock_get_config:
        
        # Create sample data
        sample_data = pd.DataFrame({
            'tilastovuosi': [2020, 2021, 2022] * 2,
            'tutkintotyyppi': ['Ammattitutkinnot', 'Ammattitutkinnot', 'Ammattitutkinnot',
                              'Erikoisammattitutkinnot', 'Erikoisammattitutkinnot', 'Erikoisammattitutkinnot'],
            'tutkinto': ['Tutkinto A', 'Tutkinto A', 'Tutkinto A', 'Tutkinto B', 'Tutkinto B', 'Tutkinto B'],
            'koulutuksenJarjestaja': ['Provider X', 'Provider X', 'Provider X', 'Provider Y', 'Provider Y', 'Provider Y'],
            'hankintakoulutuksenJarjestaja': [None, None, None, 'Provider X', 'Provider X', 'Provider X'],
            'nettoopiskelijamaaraLkm': [100, 110, 120, 200, 220, 240]
        })
        
        # Set up mock returns
        mock_load_data.return_value = sample_data
        mock_clean_data.return_value = sample_data
        
        # Create mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Mock analyze method to return test results
        mock_analyzer.analyze.return_value = {
            'total_volumes': pd.DataFrame({
                'tilastovuosi': [2020, 2021, 2022],
                'järjestäjänä': [300, 330, 360],
                'hankintana': [200, 220, 240],
                'Yhteensä': [500, 550, 600],
                'järjestäjä_osuus (%)': [60.0, 60.0, 60.0]
            }),
            'volumes_by_qualification': pd.DataFrame({
                'tilastovuosi': [2020, 2020, 2021, 2021, 2022, 2022],
                'tutkinto': ['Tutkinto A', 'Tutkinto B', 'Tutkinto A', 'Tutkinto B', 'Tutkinto A', 'Tutkinto B'],
                'Provider Amount': [100, 200, 110, 220, 120, 240],
                'Subcontractor Amount': [50, 100, 55, 110, 60, 120],
                'Total Volume': [150, 300, 165, 330, 180, 360],
                'Market Total': [1000, 2000, 1100, 2200, 1200, 2400],
                'Market Share (%)': [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
            }),
            'market_shares': pd.DataFrame({
                'tilastovuosi': [2020, 2020, 2021, 2021, 2022, 2022],
                'tutkinto': ['Tutkinto A', 'Tutkinto B', 'Tutkinto A', 'Tutkinto B', 'Tutkinto A', 'Tutkinto B'],
                'provider': ['Provider X', 'Provider X', 'Provider X', 'Provider X', 'Provider X', 'Provider X'],
                'market_share': [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
            }),
            'qualification_growth': pd.DataFrame({
                'tutkinto': ['Tutkinto A', 'Tutkinto B'],
                'growth_percent': [20.0, 20.0],
                'growth_trend': ['Growing', 'Growing']
            }),
            'qualification_cagr': pd.DataFrame({
                'tutkinto': ['Tutkinto A', 'Tutkinto B'],
                'CAGR': [9.5, 9.5],
                'First Year': [2020, 2020],
                'Last Year': [2022, 2022]
            })
        }
        
        # Mock export_to_excel to return a test path
        mock_export.return_value = Path("test_output.xlsx")
        
        # Mock configuration
        mock_get_config.return_value = {
            'institutions': {
                'default': {
                    'name': 'Provider X',
                    'short_name': 'PX',
                    'variants': ['Provider X', 'Provider X Inc']
                }
            },
            'paths': {
                'data': 'data/raw/test_data.csv',
                'output': 'data/reports'
            },
            'qualification_types': ['Ammattitutkinnot', 'Erikoisammattitutkinnot']
        }
        
        yield {
            'mock_load_data': mock_load_data,
            'mock_clean_data': mock_clean_data,
            'mock_analyzer': mock_analyzer,
            'mock_analyzer_class': mock_analyzer_class,
            'mock_export': mock_export,
            'mock_get_config': mock_get_config,
            'sample_data': sample_data
        }


def test_run_analysis_with_default_args(mock_dependencies):
    """Test running the analysis with default arguments."""
    # Run the analysis
    results = run_analysis()
    
    # Verify that all dependencies were called correctly
    mock_dependencies['mock_load_data'].assert_called_once()
    mock_dependencies['mock_clean_data'].assert_called_once()
    mock_dependencies['mock_analyzer_class'].assert_called_once()
    mock_dependencies['mock_analyzer'].analyze.assert_called_once()
    mock_dependencies['mock_export'].assert_called_once()
    
    # Verify that results contain all expected keys
    expected_keys = [
        'total_volumes', 'volumes_by_qualification', "detailed_providers_market",
        "qualification_cagr", 'excel_path'
    ]
    for key in expected_keys:
        assert key in results, f"Missing result key: {key}"
    
    # Verify that the excel_path is correct
    assert results['excel_path'] == Path("test_output.xlsx")


def test_run_analysis_with_custom_args(mock_dependencies):
    """Test running the analysis with custom arguments."""
    # Define custom arguments
    custom_args = {
        'institution': 'Custom Provider',
        'short_name': 'CP',
        'variants': ['Custom Provider', 'Custom Provider Inc'],
        'filter_qual_types': True,
        'filter_by_inst_quals': True,
        'use_dummy': True,
        'output_dir': 'custom/output/dir'
    }
    
    # Run the analysis with custom arguments
    results = run_analysis(custom_args)
    
    # Verify that the load_data function was called with use_dummy=True
    mock_dependencies['mock_load_data'].assert_called_once_with(file_path=mock_dependencies['mock_get_config'].return_value['paths']['data'], use_dummy=True)
    
    # Verify MarketAnalyzer was initialized correctly
    mock_dependencies['mock_analyzer_class'].assert_called_once()
    call_args, call_kwargs = mock_dependencies['mock_analyzer_class'].call_args
    
    # Check that it was called with only the 'data' keyword argument
    assert list(call_kwargs.keys()) == ['data'] 
    
    # Check the 'data' argument passed to MarketAnalyzer 
    assert 'data' in call_kwargs
    assert isinstance(call_kwargs['data'], pd.DataFrame)
    # In this specific test case, filtering makes the data empty
    assert call_kwargs['data'].empty
    
    # Verify that export_to_excel was called with the correct file name
    mock_dependencies['mock_export'].assert_called_once()
    args, kwargs = mock_dependencies['mock_export'].call_args
    assert 'file_name' in kwargs
    assert kwargs['file_name'] == 'cp_market_analysis'
    assert 'output_dir' in kwargs


def test_run_analysis_error_handling(mock_dependencies):
    """Test error handling in the analysis pipeline."""
    # Make MarketAnalyzer raise an exception
    mock_dependencies['mock_analyzer_class'].side_effect = Exception("Test error")
    
    # Run the analysis and expect it to complete but return empty results
    with patch('src.vipunen.cli.analyze_cli.logger') as mock_logger:
        results = run_analysis()
        
        # Verify that the error was logged
        mock_logger.error.assert_called()
        
        # Verify that empty DataFrame are returned for all data keys
        data_keys = [
            'total_volumes', 'volumes_by_qualification', "detailed_providers_market",
            "qualification_cagr"
        ]
        for key in data_keys:
            assert key in results
            assert isinstance(results[key], pd.DataFrame)
            assert results[key].empty 