"""Visualization module for Vipunen data analysis."""
import logging
from typing import List, Optional, Dict, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Define brand colors
BRAND_COLORS = {
    'primary': '#003580',  # Dark blue
    'secondary': '#00A0E1',  # Light blue
    'accent': '#FFB81C',  # Yellow
    'success': '#00A651',  # Green
    'warning': '#F9A01B',  # Orange
    'danger': '#E30613',  # Red
    'gray': '#6C6C6C'  # Gray
}

def set_plot_style() -> None:
    """Set the default plot style using brand colors."""
    try:
        # Use seaborn's default style
        sns.set_style("whitegrid")
        
        # Configure matplotlib rcParams
        plt.rcParams.update({
            'figure.facecolor': BRAND_COLORS['background'],
            'axes.facecolor': BRAND_COLORS['background'],
            'axes.edgecolor': BRAND_COLORS['text'],
            'axes.labelcolor': BRAND_COLORS['text'],
            'text.color': BRAND_COLORS['text'],
            'xtick.color': BRAND_COLORS['text'],
            'ytick.color': BRAND_COLORS['text'],
            'grid.color': '#e0e0e0',
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif'],
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    except Exception as e:
        logger.error(f"Error setting plot style: {str(e)}")
        raise

def plot_market_shares(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    title: str = "Market Shares",
    color_scale: Optional[List[str]] = None,
    width: int = 800,
    height: int = 600
) -> go.Figure:
    """
    Create a treemap visualization of market shares.
    
    Args:
        df: DataFrame containing market share data
        group_cols: Columns to use for hierarchical grouping
        value_col: Column containing the values to visualize
        title: Title for the plot
        color_scale: Optional custom color scale
        width: Plot width in pixels
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    try:
        # Create treemap
        fig = px.treemap(
            df,
            path=group_cols,
            values=value_col,
            title=title,
            width=width,
            height=height
        )
        
        # Update colors if custom scale provided
        if color_scale:
            fig.update_traces(marker_colors=color_scale)
        else:
            fig.update_traces(marker_colors=list(BRAND_COLORS.values()))
        
        # Update layout
        fig.update_layout(
            margin=dict(t=50, l=25, r=25, b=25),
            font=dict(family="Arial", size=12)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating market shares plot: {str(e)}")
        raise

def plot_growth_trends(
    df: pd.DataFrame,
    group_col: str,
    cagr_col: str,
    title: str = "Growth Trends",
    width: int = 800,
    height: int = 600
) -> go.Figure:
    """
    Create a bar chart visualization of growth trends.
    
    Args:
        df: DataFrame containing growth trend data
        group_col: Column containing group names
        cagr_col: Column containing CAGR values
        title: Title for the plot
        width: Plot width in pixels
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    try:
        # Sort by CAGR
        df_sorted = df.sort_values(cagr_col, ascending=True)
        
        # Create bar chart
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(
            go.Bar(
                x=df_sorted[cagr_col],
                y=df_sorted[group_col],
                orientation='h',
                marker_color=BRAND_COLORS['primary'],
                text=df_sorted[cagr_col].round(1).astype(str) + '%',
                textposition='auto'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="CAGR (%)",
            yaxis_title=group_col,
            width=width,
            height=height,
            margin=dict(t=50, l=25, r=25, b=25),
            font=dict(family="Arial", size=12)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating growth trends plot: {str(e)}")
        raise

def plot_provider_counts(
    df: pd.DataFrame,
    group_cols: List[str],
    count_col: str,
    title: str = "Provider Counts",
    width: int = 800,
    height: int = 600
) -> go.Figure:
    """
    Create a bar chart visualization of provider counts.
    
    Args:
        df: DataFrame containing provider count data
        group_cols: Columns to group by
        count_col: Column containing provider counts
        title: Title for the plot
        width: Plot width in pixels
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    try:
        # Create bar chart
        fig = go.Figure()
        
        # Add bars for each group
        for i, group in enumerate(df[group_cols[0]].unique()):
            group_data = df[df[group_cols[0]] == group]
            fig.add_trace(
                go.Bar(
                    x=[str(group)],
                    y=[group_data[count_col].iloc[0]],
                    name=str(group),
                    marker_color=list(BRAND_COLORS.values())[i % len(BRAND_COLORS)]
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=group_cols[0],
            yaxis_title=count_col,
            width=width,
            height=height,
            barmode='group',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating provider counts plot: {str(e)}")
        raise

def save_plot(
    fig: go.Figure,
    filename: str,
    format: str = 'png',
    width: Optional[int] = None,
    height: Optional[int] = None
) -> None:
    """
    Save a plot to a file.
    
    Args:
        fig: Plotly Figure object
        filename: Output filename
        format: Output format (png, jpg, pdf, etc.)
        width: Optional custom width
        height: Optional custom height
    """
    try:
        fig.write_image(
            filename,
            format=format,
            width=width,
            height=height
        )
        logger.info(f"Plot saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving plot: {str(e)}")
        raise

def create_market_analysis_plots(
    market_analysis: Dict[str, pd.DataFrame],
    year_col: str = 'tilastovuosi',
    provider_col: str = 'koulutuksenJarjestaja',
    share_col: str = 'market_share',
    group_col: str = 'tutkinto',
    cagr_col: str = 'cagr',
    count_col: str = 'provider_count'
) -> Dict[str, go.Figure]:
    """
    Create visualizations for market analysis results.
    
    Args:
        market_analysis: Dictionary containing market analysis results
        year_col: Column containing year values
        provider_col: Column containing provider names
        share_col: Column containing market share values
        group_col: Column containing group names
        cagr_col: Column containing CAGR values
        count_col: Column containing provider counts
        
    Returns:
        Dictionary containing Plotly figures for each visualization
    """
    try:
        plots = {}
        
        # Market shares plot
        if 'market_shares' in market_analysis:
            plots['market_shares'] = plot_market_shares(
                df=market_analysis['market_shares'],
                group_cols=[year_col, group_col, provider_col],
                value_col=share_col,
                title="Market Shares by Year and Provider"
            )
        
        # Growth trends plot
        if 'growth_trends' in market_analysis:
            plots['growth_trends'] = plot_growth_trends(
                df=market_analysis['growth_trends'],
                group_col=group_col,
                cagr_col=cagr_col,
                title="Growth Trends by Provider"
            )
        
        # Provider counts plot
        if 'provider_counts' in market_analysis:
            plots['provider_counts'] = plot_provider_counts(
                df=market_analysis['provider_counts'],
                group_cols=[year_col, group_col],
                count_col=count_col,
                title="Provider Counts by Year"
            )
        
        return plots
        
    except Exception as e:
        logger.error(f"Error creating market analysis plots: {str(e)}")
        raise 