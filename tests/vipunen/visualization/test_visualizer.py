"""Tests for the visualization module."""
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from vipunen.visualization.visualizer import VipunenVisualizer, BRAND_COLORS

@pytest.fixture
def visualizer():
    """Create a VipunenVisualizer instance for testing."""
    return VipunenVisualizer()

@pytest.fixture
def sample_market_data():
    """Create sample market share data for testing."""
    data = {
        'tilastovuosi': [2020, 2020, 2020, 2021, 2021, 2021],
        'koulutuksenJarjestaja': ['A', 'B', 'C', 'A', 'B', 'C'],
        'market_share': [40, 35, 25, 45, 30, 25]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_trend_data():
    """Create sample trend data for testing."""
    data = {
        'tilastovuosi': [2020, 2021, 2022] * 2,
        'tutkinto': ['A'] * 3 + ['B'] * 3,
        'nettoopiskelijamaaraLkm': [100, 150, 200, 300, 250, 200]
    }
    return pd.DataFrame(data)

def test_create_color_palette(visualizer):
    """Test color palette creation."""
    # Test with fewer colors than base palette
    colors = visualizer.create_color_palette(3)
    assert len(colors) == 3
    assert all(isinstance(c, str) for c in colors)
    assert all(c.startswith('#') for c in colors)
    
    # Test with more colors than base palette
    colors = visualizer.create_color_palette(6)
    assert len(colors) == 6
    assert all(isinstance(c, str) for c in colors)
    assert all(c.startswith('#') for c in colors)

def test_wrap_text(visualizer):
    """Test text wrapping functionality."""
    text = "This is a long text that should be wrapped at some point"
    wrapped = visualizer.wrap_text(text, width=20)
    
    # Check that text was wrapped
    assert '\n' in wrapped
    # Check that no line is longer than width
    assert all(len(line) <= 20 for line in wrapped.split('\n'))

def test_plot_market_shares(visualizer, sample_market_data):
    """Test market share plotting."""
    fig = visualizer.plot_market_shares(
        df=sample_market_data,
        title="Test Market Shares"
    )
    
    # Check that figure was created
    assert isinstance(fig, plt.Figure)
    
    # Check plot elements
    ax = fig.axes[0]
    assert ax.get_title() == "Test Market Shares"
    assert ax.get_xlabel() == "Year"
    assert ax.get_ylabel() == "Market Share (%)"
    
    # Check legend
    assert ax.get_legend() is not None
    
    plt.close(fig)

def test_plot_growth_trends(visualizer, sample_trend_data):
    """Test growth trend plotting."""
    # Test without grouping
    fig = visualizer.plot_growth_trends(
        df=sample_trend_data,
        title="Test Growth Trends"
    )
    
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "Test Growth Trends"
    plt.close(fig)
    
    # Test with grouping
    fig = visualizer.plot_growth_trends(
        df=sample_trend_data,
        group_col='tutkinto',
        title="Test Growth Trends by Group"
    )
    
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert ax.get_legend() is not None
    assert len(ax.get_lines()) == 2  # Two groups
    plt.close(fig)

def test_save_plot(visualizer, sample_market_data, tmp_path):
    """Test plot saving functionality."""
    # Create a plot
    fig = visualizer.plot_market_shares(sample_market_data)
    
    # Save plot in temporary directory
    saved_files = visualizer.save_plot(
        fig=fig,
        filename="test_plot",
        directory=str(tmp_path),
        formats=['png', 'pdf']
    )
    
    # Check that files were saved
    assert len(saved_files) == 2
    assert all(isinstance(p, Path) for p in saved_files)
    assert all(p.exists() for p in saved_files)
    assert any(p.suffix == '.png' for p in saved_files)
    assert any(p.suffix == '.pdf' for p in saved_files)
    
    plt.close(fig)

def test_custom_colors():
    """Test visualizer with custom colors."""
    custom_colors = {
        'primary': '#FF0000',
        'secondary': '#00FF00',
        'accent1': '#0000FF',
        'accent2': '#FFFF00',
        'neutral': '#999999',
        'background': '#FFFFFF',
        'text': '#000000'
    }
    
    viz = VipunenVisualizer(colors=custom_colors)
    assert viz.colors == custom_colors
    
    # Test that default colors weren't modified
    assert BRAND_COLORS['primary'] == '#007AC9' 