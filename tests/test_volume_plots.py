"""
Tests for the volume_plots module.
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from vipunen.visualization.volume_plots import (
    save_plot,
    plot_total_volumes,
    plot_top_qualifications
)
from tests.conftest import TEST_OUTPUT_DIR


@pytest.fixture
def mock_plt_savefig():
    """Mock plt.savefig to avoid file operations during tests."""
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        yield mock_savefig


@pytest.fixture
def mock_plt_close():
    """Mock plt.close to avoid closing figures during tests."""
    with patch('matplotlib.pyplot.close') as mock_close:
        yield mock_close


def test_save_plot(mock_plt_savefig, mock_plt_close):
    """Test the save_plot function."""
    # Create a simple figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    # Test with explicit file path
    file_path = TEST_OUTPUT_DIR / "test_plot.png"
    result = save_plot(fig, file_path=file_path)
    
    # Check that savefig was called with the correct arguments
    mock_plt_savefig.assert_called_once()
    mock_plt_close.assert_called_once_with(fig)
    assert result == str(file_path)
    
    # Reset mocks
    mock_plt_savefig.reset_mock()
    mock_plt_close.reset_mock()
    
    # Test with plot_name and output_dir
    output_dir = TEST_OUTPUT_DIR
    plot_name = "test_plot_2"
    result = save_plot(fig, plot_name=plot_name, output_dir=output_dir)
    
    # Check that savefig was called with the correct arguments
    mock_plt_savefig.assert_called_once()
    mock_plt_close.assert_called_once_with(fig)
    assert result == str(output_dir / f"{plot_name}.png")
    
    # Reset mocks
    mock_plt_savefig.reset_mock()
    mock_plt_close.reset_mock()
    
    # Test with default values
    with patch('os.makedirs') as mock_makedirs:
        result = save_plot(fig)
        
        # Check that savefig was called
        mock_plt_savefig.assert_called_once()
        mock_plt_close.assert_called_once_with(fig)
        mock_makedirs.assert_called_once()
        assert "plots/plot.png" in result


def test_plot_total_volumes(sample_volume_data, mock_plt_savefig, mock_plt_close):
    """Test the plot_total_volumes function."""
    # Test with default parameters
    result = plot_total_volumes(
        volumes_df=sample_volume_data,
        institution_short_name="TestInstitution",
        output_dir=TEST_OUTPUT_DIR
    )
    
    # Check that a figure was created and saved
    mock_plt_savefig.assert_called_once()
    mock_plt_close.assert_called_once()
    assert result is not None
    assert "testinstitution_total_volumes" in result
    
    # Reset mocks
    mock_plt_savefig.reset_mock()
    mock_plt_close.reset_mock()
    
    # Test with custom output path
    output_path = "custom_path"
    result = plot_total_volumes(
        volumes_df=sample_volume_data,
        output_path=output_path,
        institution_short_name="TestInstitution",
        output_dir=TEST_OUTPUT_DIR
    )
    
    # Check that a figure was created and saved with the custom path
    mock_plt_savefig.assert_called_once()
    mock_plt_close.assert_called_once()
    assert result is not None
    assert output_path in result
    
    # Test with incomplete data
    incomplete_data = sample_volume_data.drop(columns=['järjestäjänä'])
    result = plot_total_volumes(
        volumes_df=incomplete_data,
        institution_short_name="TestInstitution",
        output_dir=TEST_OUTPUT_DIR
    )
    
    # No figure should be created for incomplete data
    assert mock_plt_savefig.call_count == 1  # Still 1 from previous call
    assert mock_plt_close.call_count == 1  # Still 1 from previous call
    assert result is None


def test_plot_top_qualifications(sample_volumes_by_qualification, mock_plt_savefig, mock_plt_close):
    """Test the plot_top_qualifications function."""
    # Test with default parameters
    result = plot_top_qualifications(
        volumes_by_qual=sample_volumes_by_qualification,
        institution_short_name="TestInstitution",
        output_dir=TEST_OUTPUT_DIR
    )
    
    # Check that a figure was created and saved
    mock_plt_savefig.assert_called_once()
    mock_plt_close.assert_called_once()
    assert result is not None
    assert "testinstitution_top_qualifications" in result
    
    # Reset mocks
    mock_plt_savefig.reset_mock()
    mock_plt_close.reset_mock()
    
    # Test with custom output path and parameters
    output_path = "custom_qual_path"
    result = plot_top_qualifications(
        volumes_by_qual=sample_volumes_by_qualification,
        output_path=output_path,
        year=2022,  # Specific year
        top_n=3,    # Top 3 instead of default 5
        institution_short_name="TestInstitution",
        output_dir=TEST_OUTPUT_DIR
    )
    
    # Check that a figure was created and saved with the custom path
    mock_plt_savefig.assert_called_once()
    mock_plt_close.assert_called_once()
    assert result is not None
    assert output_path in result
    
    # Test with incomplete data - here we'll drop all qualification data
    incomplete_data = pd.DataFrame({
        'tilastovuosi': [2020, 2021, 2022],
        'järjestäjänä': [100, 120, 130]
    })
    
    result = plot_top_qualifications(
        volumes_by_qual=incomplete_data,
        institution_short_name="TestInstitution",
        output_dir=TEST_OUTPUT_DIR
    )
    
    # No figure should be created for incomplete data
    assert mock_plt_savefig.call_count == 1  # Still 1 from previous call
    assert mock_plt_close.call_count == 1  # Still 1 from previous call
    assert result is None 