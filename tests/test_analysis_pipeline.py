"""
Tests for the full education market analysis pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the refactored workflow function
from src.vipunen.cli.analyze_cli import run_analysis_workflow


@pytest.fixture
def mock_dependencies():
    """Mock all dependencies for the analysis pipeline."""
    # Add mocks for visualization components
    with patch('src.vipunen.cli.analyze_cli.load_data') as mock_load_data, \
         patch('src.vipunen.cli.analyze_cli.clean_and_prepare_data') as mock_clean_data, \
         patch('src.vipunen.cli.analyze_cli.MarketAnalyzer') as mock_analyzer_class, \
         patch('src.vipunen.cli.analyze_cli.export_analysis_results') as mock_export, \
         patch('src.vipunen.cli.analyze_cli.get_config') as mock_get_config, \
         patch('src.vipunen.cli.analyze_cli.EducationVisualizer') as mock_visualizer_class, \
         patch('src.vipunen.cli.analyze_cli.prepare_analysis_data') as mock_prepare_data, \
         patch('src.vipunen.cli.analyze_cli.perform_market_analysis') as mock_perform_analysis, \
         patch('src.vipunen.cli.analyze_cli.generate_visualizations') as mock_generate_visualizations: # Mock the whole function

        # --- Mock Configuration ---
        mock_config = {
            'institutions': {
                'default': {
                    'name': 'Provider X',
                    'short_name': 'PX',
                    'variants': ['Provider X', 'Provider X Inc']
                },
                'Custom Provider': {
                    'name': 'Custom Provider Full Name',
                    'short_name': 'CP',
                    'variants': ['Custom Provider', 'Custom Provider Inc']
                }
            },
            'paths': {
                'data': 'data/raw/test_data.csv',
                'output': 'data/reports'
            },
            'qualification_types': ['Ammattitutkinnot', 'Erikoisammattitutkinnot'],
            'columns': {
                'input': {
                    'provider': 'koulutuksenJarjestaja',
                    'subcontractor': 'hankintakoulutuksenJarjestaja',
                    'qualification': 'tutkinto',
                    'year': 'tilastovuosi',
                    'volume': 'nettoopiskelijamaaraLkm',
                    'degree_type': 'tutkintotyyppi',
                    'update_date': 'tietojoukkoPaivitettyPvm' # Added for caption testing
                },
                'output': {
                    'year': 'Year',
                    'qualification': 'Qualification',
                    'provider': 'Provider',
                    'provider_amount': 'Provider Amount',
                    'subcontractor_amount': 'Subcontractor Amount',
                    'total_volume': 'Total Volume',
                    'market_total': 'Market Total',
                    'market_share': 'Market Share (%)',
                    'market_rank': 'Market Rank',
                    'market_share_growth': 'Market Share Growth (%)',
                    'market_gainer_rank': 'Market Gainer Rank'
                    # Add other output columns if needed by tests
                }
            },
            'excel': { # Added for export test
                'sheets': [
                    {'name': 'Sheet1'}, {'name': 'Sheet2'},
                    {'name': 'Sheet3'}, {'name': 'Sheet4'}
                ]
            }
        }
        mock_get_config.return_value = mock_config

        # --- Mock Data Preparation ---
        sample_data = pd.DataFrame({
            'tilastovuosi': [2020, 2021, 2022] * 2,
            'tutkintotyyppi': ['Ammattitutkinnot']*3 + ['Erikoisammattitutkinnot']*3,
            'tutkinto': ['Tutkinto A']*3 + ['Tutkinto B']*3,
            'koulutuksenJarjestaja': ['Provider X']*3 + ['Provider Y']*3,
            'hankintakoulutuksenJarjestaja': [None]*3 + ['Provider X']*3,
            'nettoopiskelijamaaraLkm': [100, 110, 120, 200, 220, 240],
            'tietojoukkoPaivitettyPvm': ['2023-01-01']*6 # Added for date extraction
        })
        # Mock prepare_analysis_data return value
        mock_prepare_data.return_value = (
            sample_data, # df_clean
            'default', # institution_key
            ['Provider X', 'Provider X Inc'], # institution_variants
            'PX', # institution_short_name
            '01.01.2023', # data_update_date_str
            False # filter_qual_types
        )

        # --- Mock Analysis ---
        # Create mock analyzer instance (to be returned by perform_market_analysis)
        mock_analyzer = MagicMock()
        mock_analyzer.min_year = 2020
        mock_analyzer.max_year = 2022
        mock_analyzer.institution_short_name = 'PX'
        mock_analyzer.institution_names = ['Provider X', 'Provider X Inc']

        # Mock analysis results dictionary returned by MarketAnalyzer.analyze()
        # Update this to reflect the actual keys returned by the current MarketAnalyzer
        mock_analysis_results = {
            'total_volumes': pd.DataFrame({'Year': [2020, 2021, 2022], 'Total Volume': [500, 550, 600]}),
            'volumes_by_qualification': pd.DataFrame({'Qualification': ['A', 'B'], 'Total Volume': [150, 300]}),
            'detailed_providers_market': pd.DataFrame({'Provider': ['PX'], 'Market Share (%)': [15.0]}),
            'qualification_cagr': pd.DataFrame({'Qualification': ['A', 'B'], 'CAGR': [9.5, 9.5]}),
            'overall_total_market_volume': pd.Series([1000, 1100, 1200], index=[2020, 2021, 2022]),
            'qualification_market_yoy_growth': pd.DataFrame({'Qualification': ['A'], 'Market Total YoY Growth (%)': [10.0]}),
            'provider_counts_by_year': pd.DataFrame({'Year': [2020], 'Unique_Providers_Count': [5]}),
            'bcg_data': pd.DataFrame({'Qualification': ['A'], 'Market Growth (%)': [10.0], 'Relative Market Share': [0.5], 'Institution Volume': [150]})
        }
        # Mock perform_market_analysis return value
        mock_perform_analysis.return_value = (mock_analysis_results, mock_analyzer)

        # --- Mock Export ---
        mock_export.return_value = "test_output.xlsx" # Return path as string

        # --- Mock Visualization ---
        mock_visualizer = MagicMock()
        mock_visualizer_class.return_value = mock_visualizer

        # Yield all mocks needed by tests
        yield {
            'mock_get_config': mock_get_config,
            'mock_prepare_data': mock_prepare_data,
            'mock_perform_analysis': mock_perform_analysis,
            'mock_export': mock_export,
            'mock_visualizer': mock_visualizer,
            'mock_visualizer_class': mock_visualizer_class,
            'mock_generate_visualizations': mock_generate_visualizations,
            'mock_analyzer': mock_analyzer, # Pass the instance for potential checks
            'mock_analysis_results': mock_analysis_results # Pass results for checking
        }

# Rename test functions for clarity
def test_run_analysis_workflow_with_default_args(mock_dependencies):
    """Test running the analysis workflow with default arguments."""
    # Run the workflow with empty args (defaults handled internally or by prepare_analysis_data mock)
    results = run_analysis_workflow({})

    # Verify that the core workflow steps were called
    mock_dependencies['mock_get_config'].assert_called_once()
    mock_dependencies['mock_prepare_data'].assert_called_once()
    mock_dependencies['mock_perform_analysis'].assert_called_once()
    mock_dependencies['mock_export'].assert_called_once()
    mock_dependencies['mock_generate_visualizations'].assert_called_once() # Check visualization call

    # Verify the structure of the returned dictionary
    assert 'analysis_results' in results
    assert 'excel_path' in results

    # Verify that analysis_results contain all expected keys from the mock
    expected_keys = mock_dependencies['mock_analysis_results'].keys()
    assert results['analysis_results'].keys() == expected_keys

    # Verify that the excel_path is correct
    assert results['excel_path'] == "test_output.xlsx"

def test_run_analysis_workflow_with_custom_args(mock_dependencies):
    """Test running the analysis workflow with custom arguments."""
    # Define custom arguments
    custom_args = {
        'institution': 'Custom Provider',
        'short_name': 'CP',
        'variants': ['Custom Provider', 'Custom Provider Inc'],
        'filter_qual_types': True,
        #'filter_by_inst_quals': True, # This argument is removed
        'use_dummy': True, # This is handled by prepare_analysis_data mock now
        'output_dir': 'custom/output/dir',
        'plot_format': 'png' # Added for viz test
    }

    # Update mock return values for custom args scenario if needed
    # Example: Change prepare_analysis_data mock return for 'Custom Provider'
    mock_dependencies['mock_prepare_data'].return_value = (
        pd.DataFrame({'tutkinto': ['C'], 'nettoopiskelijamaaraLkm': [50]}), # Sample filtered df
        'Custom Provider', # institution_key
        ['Custom Provider', 'Custom Provider Inc'], # institution_variants
        'CP', # institution_short_name
        '01.01.2023', # data_update_date_str
        True # filter_qual_types
    )
    # Update perform_market_analysis mock return for custom provider
    mock_analyzer_cp = MagicMock()
    mock_analyzer_cp.institution_short_name = 'CP'
    mock_analysis_results_cp = {'detailed_providers_market': pd.DataFrame({'Provider': ['CP']})}
    mock_dependencies['mock_perform_analysis'].return_value = (mock_analysis_results_cp, mock_analyzer_cp)

    # Run the workflow with custom arguments
    results = run_analysis_workflow(custom_args)

    # Verify prepare_analysis_data was called with custom args
    mock_dependencies['mock_prepare_data'].assert_called_once_with(
        mock_dependencies['mock_get_config'].return_value, custom_args
    )

    # Verify perform_market_analysis was called correctly
    mock_dependencies['mock_perform_analysis'].assert_called_once()
    call_args, call_kwargs = mock_dependencies['mock_perform_analysis'].call_args
    assert call_args[0].equals(mock_dependencies['mock_prepare_data'].return_value[0]) # df_clean
    assert call_args[1] == mock_dependencies['mock_get_config'].return_value # config
    assert call_args[2] == ['Custom Provider', 'Custom Provider Inc'] # institution_variants
    assert call_args[3] == 'CP' # institution_short_name
    assert call_args[4] == True # filter_qual_types

    # Verify that export_analysis_results was called with the correct institution short name and output path
    mock_dependencies['mock_export'].assert_called_once()
    call_args_export, call_kwargs_export = mock_dependencies['mock_export'].call_args
    assert call_args_export[2] == 'CP' # institution_short_name
    assert call_args_export[3] == 'custom/output/dir' # base_output_path

    # Verify Visualizer was initialized with custom path and format
    mock_dependencies['mock_visualizer_class'].assert_called_once()
    call_args_viz_init, call_kwargs_viz_init = mock_dependencies['mock_visualizer_class'].call_args
    assert call_kwargs_viz_init['output_dir'] == Path('custom/output/dir/education_market_cp')
    assert call_kwargs_viz_init['output_format'] == 'png'
    assert call_kwargs_viz_init['institution_short_name'] == 'CP'

    # Verify generate_visualizations was called
    mock_dependencies['mock_generate_visualizations'].assert_called_once()


def test_run_analysis_workflow_error_handling(mock_dependencies):
    """Test error handling in the analysis pipeline workflow."""
    # Make perform_market_analysis raise an exception
    mock_dependencies['mock_perform_analysis'].side_effect = Exception("Test analysis error")

    # Run the workflow and expect it to complete but return indication of failure
    with patch('src.vipunen.cli.analyze_cli.logger') as mock_logger:
        results = run_analysis_workflow({})

        # Verify that the error was logged (inside perform_market_analysis mock)
        # Note: Depending on where the error is caught, the log might be in perform_market_analysis
        # Check if the workflow log indicates skipping subsequent steps
        # This depends on the exact implementation of error handling in run_analysis_workflow
        # Let's check if export and visualization were NOT called
        mock_dependencies['mock_export'].assert_not_called()
        mock_dependencies['mock_generate_visualizations'].assert_not_called()

        # Verify that the results dictionary indicates failure or partial results
        # The current implementation returns empty/dummy results if analysis fails
        assert 'analysis_results' in results
        assert 'excel_path' in results
        assert results['excel_path'] is None # Expect None if export skipped due to error

        # Check if analysis_results contains expected keys, even if empty (depends on error handling)
        # perform_market_analysis mock raises error, so its return value isn't used.
        # run_analysis_workflow should catch this and return the default empty dict structure from perform_market_analysis
        # Let's assume it returns empty dataframes as defined in perform_market_analysis error handling
        expected_keys = [
            "total_volumes", "volumes_by_qualification",
            "detailed_providers_market", "qualification_cagr"
            # Note: The other keys (bcg_data etc.) might not be present if analysis failed early
        ]
        assert isinstance(results['analysis_results'], dict)
        # We might not get all keys if the error happened early in analyze()
        # Check if the main expected one is empty
        assert results['analysis_results'].get("detailed_providers_market", pd.DataFrame()).empty 