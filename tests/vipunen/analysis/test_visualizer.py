"""Tests for the visualization module."""
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from vipunen.analysis.visualizer import (
    create_color_palette,
    wrap_text,
    plot_market_shares,
    plot_growth_trends,
    plot_provider_counts
)

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    data = {
        'tilastovuosi': [2022] * 6,
        'tutkinto': ['A', 'A', 'A', 'B', 'B', 'B'],
        'koulutuksenJarjestaja': ['X', 'Y', 'Z', 'X', 'Y', 'Z'],
        'market_share': [10, 20, 30, 15, 25, 35]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_growth_data():
    """Create sample growth data for testing."""
    data = {
        'tutkinto': ['A', 'B', 'C', 'D'],
        'kasvu': [10.5, -5.2, 15.3, -2.1],
        'kasvu_trendi': ['Kasvussa', 'Laskussa', 'Kasvussa', 'Laskussa']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_provider_data():
    """Create sample provider count data for testing."""
    data = {
        'tilastovuosi': [2018, 2019, 2020, 2021, 2022],
        'provider_count': [10, 12, 11, 13, 15]
    }
    return pd.DataFrame(data)

def test_create_color_palette():
    """Test color palette creation."""
    base_colors = ['red', 'blue', 'green']
    n_colors = 5
    
    palette = create_color_palette(base_colors, n_colors)
    
    assert isinstance(palette, list)
    assert len(palette) == n_colors
    assert palette[:3] == base_colors
    assert palette[3:] == base_colors[:2]  # Should cycle through colors

def test_wrap_text():
    """Test text wrapping."""
    # Test short text (no wrapping needed)
    short_text = "Short text"
    assert wrap_text(short_text) == short_text
    
    # Test long text
    long_text = "This is a very long text that needs to be wrapped properly"
    wrapped = wrap_text(long_text, wrap_limit=20)
    
    assert isinstance(wrapped, str)
    assert max(len(line) for line in wrapped.split('\n')) <= 20
    assert ''.join(wrapped.split()) == ''.join(long_text.split())  # Content should be same

def test_plot_market_shares(sample_market_data):
    """Test market share plotting."""
    # Test basic plot
    fig = plot_market_shares(
        df=sample_market_data,
        x_col='tutkinto',
        y_col='market_share'
    )
    assert isinstance(fig, plt.Figure)
    plt.close()
    
    # Test with hue
    fig = plot_market_shares(
        df=sample_market_data,
        x_col='tutkinto',
        y_col='market_share',
        hue_col='koulutuksenJarjestaja'
    )
    assert isinstance(fig, plt.Figure)
    plt.close()
    
    # Test with custom labels
    fig = plot_market_shares(
        df=sample_market_data,
        x_col='tutkinto',
        y_col='market_share',
        title='Custom Title',
        x_label='X Label',
        y_label='Y Label'
    )
    assert isinstance(fig, plt.Figure)
    plt.close()

def test_plot_growth_trends(sample_growth_data):
    """Test growth trends plotting."""
    fig = plot_growth_trends(
        df=sample_growth_data,
        x_col='tutkinto',
        y_col='kasvu',
        trend_col='kasvu_trendi'
    )
    assert isinstance(fig, plt.Figure)
    plt.close()
    
    # Test with custom labels
    fig = plot_growth_trends(
        df=sample_growth_data,
        x_col='tutkinto',
        y_col='kasvu',
        trend_col='kasvu_trendi',
        title='Custom Title',
        x_label='X Label',
        y_label='Y Label'
    )
    assert isinstance(fig, plt.Figure)
    plt.close()

def test_plot_provider_counts(sample_provider_data):
    """Test provider counts plotting."""
    fig = plot_provider_counts(
        df=sample_provider_data,
        x_col='tilastovuosi',
        y_col='provider_count'
    )
    assert isinstance(fig, plt.Figure)
    plt.close()
    
    # Test with custom labels
    fig = plot_provider_counts(
        df=sample_provider_data,
        x_col='tilastovuosi',
        y_col='provider_count',
        title='Custom Title',
        x_label='X Label',
        y_label='Y Label'
    )
    assert isinstance(fig, plt.Figure)
    plt.close()

def test_plot_saving(tmp_path, sample_market_data, sample_growth_data, sample_provider_data):
    """Test plot saving functionality."""
    save_path = tmp_path / "test_plot.png"

    # Test market share plot saving
    plot_market_shares(
        df=sample_market_data,
        x_col='tutkinto',
        y_col='market_share',
        save_path=save_path
    )
    assert save_path.exists()
    plt.close()

    # Test growth trends plot saving
    save_path = tmp_path / "test_growth.png"
    plot_growth_trends(
        df=sample_growth_data,
        x_col='tutkinto',
        y_col='kasvu',
        trend_col='kasvu_trendi',
        save_path=save_path
    )
    assert save_path.exists()
    plt.close()

    # Test provider counts plot saving
    save_path = tmp_path / "test_providers.png"
    plot_provider_counts(
        df=sample_provider_data,
        x_col='tilastovuosi',
        y_col='provider_count',
        save_path=save_path
    )
    assert save_path.exists()
    plt.close() 