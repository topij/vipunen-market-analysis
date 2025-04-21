"""Tests for the visualization module."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import plotly.graph_objects as go
from vipunen.visualization.plotter import (
    plot_market_shares,
    plot_growth_trends,
    plot_provider_counts,
    create_market_analysis_plots,
    save_plot
)

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    data = {
        'tilastovuosi': [2020, 2020, 2020, 2021, 2021, 2021],
        'tutkinto': ['A', 'A', 'B', 'A', 'A', 'B'],
        'kouluttaja': ['X', 'Y', 'X', 'X', 'Y', 'X'],
        'market_share': [33.33, 66.67, 100.0, 40.0, 60.0, 100.0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_growth_data():
    """Create sample growth data for testing."""
    data = {
        'tutkinto': ['A', 'B', 'C'],
        'cagr': [10.5, -5.2, 15.8]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_provider_data():
    """Create sample provider data for testing."""
    data = {
        'tilastovuosi': [2020, 2020, 2021, 2021],
        'tutkinto': ['A', 'B', 'A', 'B'],
        'provider_count': [2, 1, 2, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_market_analysis():
    """Create sample market analysis data for testing."""
    return {
        'market_shares': pd.DataFrame({
            'tilastovuosi': [2020, 2020, 2021, 2021],
            'tutkinto': ['A', 'B', 'A', 'B'],
            'kouluttaja': ['X', 'Y', 'X', 'Y'],
            'market_share': [40.0, 60.0, 45.0, 55.0]
        }),
        'growth_trends': pd.DataFrame({
            'tutkinto': ['A', 'B'],
            'cagr': [10.5, 15.2]
        }),
        'provider_counts': pd.DataFrame({
            'tilastovuosi': [2020, 2021],
            'provider_count': [2, 2]
        })
    }

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_plot_market_shares(sample_market_data):
    """Test market share plot creation."""
    fig = plot_market_shares(
        df=sample_market_data,
        group_cols=['tilastovuosi', 'tutkinto', 'kouluttaja'],
        value_col='market_share'
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Market Shares"
    assert len(fig.data) == 1  # One treemap trace

def test_plot_growth_trends(sample_growth_data):
    """Test growth trends plot creation."""
    fig = plot_growth_trends(
        df=sample_growth_data,
        group_col='tutkinto',
        cagr_col='cagr'
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Growth Trends"
    assert len(fig.data) == 1  # One bar trace
    assert fig.data[0].orientation == 'h'  # Horizontal bars

def test_plot_provider_counts(sample_provider_data):
    """Test provider counts plot creation."""
    fig = plot_provider_counts(
        df=sample_provider_data,
        group_cols=['tilastovuosi', 'tutkinto'],
        count_col='provider_count'
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Provider Counts"
    assert len(fig.data) == 2  # Two bar traces (one per year)
    assert fig.layout.barmode == 'group'

def test_plot_market_shares_empty_data():
    """Test market share plot with empty data."""
    empty_df = pd.DataFrame(columns=['tilastovuosi', 'tutkinto', 'kouluttaja', 'market_share'])
    fig = plot_market_shares(
        df=empty_df,
        group_cols=['tilastovuosi', 'tutkinto', 'kouluttaja'],
        value_col='market_share'
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # Empty treemap trace

def test_plot_growth_trends_empty_data():
    """Test growth trends plot with empty data."""
    empty_df = pd.DataFrame(columns=['tutkinto', 'cagr'])
    fig = plot_growth_trends(
        df=empty_df,
        group_col='tutkinto',
        cagr_col='cagr'
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # Empty bar trace
    assert fig.data[0].orientation == 'h'  # Horizontal bars
    assert fig.layout.title.text == "Growth Trends"
    assert fig.layout.xaxis.title.text == "CAGR (%)"
    assert fig.layout.yaxis.title.text == "tutkinto"
    assert len(fig.data[0].x) == 0  # No data points
    assert len(fig.data[0].y) == 0  # No data points

def test_plot_provider_counts_empty_data():
    """Test provider counts plot with empty data."""
    empty_df = pd.DataFrame(columns=['tilastovuosi', 'tutkinto', 'provider_count'])
    fig = plot_provider_counts(
        df=empty_df,
        group_cols=['tilastovuosi', 'tutkinto'],
        count_col='provider_count'
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0  # No bar traces

def test_save_plot(tmp_path, sample_growth_data):
    """Test saving a plot to file."""
    fig = plot_growth_trends(
        df=sample_growth_data,
        group_col='tutkinto',
        cagr_col='cagr'
    )
    
    output_file = tmp_path / "test_plot.png"
    save_plot(fig, str(output_file))
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_create_market_analysis_plots(sample_market_analysis, tmp_path):
    """Test creation of all market analysis plots."""
    create_market_analysis_plots(
        market_analysis=sample_market_analysis,
        output_dir=str(tmp_path)
    )
    
    # Check that all plot files were created
    assert (tmp_path / "market_shares.png").exists()
    assert (tmp_path / "growth_trends.png").exists()
    assert (tmp_path / "provider_counts.png").exists()
    
    # Check that the files are not empty
    assert (tmp_path / "market_shares.png").stat().st_size > 0
    assert (tmp_path / "growth_trends.png").stat().st_size > 0
    assert (tmp_path / "provider_counts.png").stat().st_size > 0

def test_plot_market_shares_invalid_data():
    """Test market share plotting with invalid data."""
    invalid_data = pd.DataFrame({
        'year': [2020],
        'provider': ['A'],
        'share': ['invalid']  # Non-numeric value
    })
    
    with pytest.raises(Exception):
        plot_market_shares(
            df=invalid_data,
            year_col='year',
            provider_col='provider',
            share_col='share'
        ) 