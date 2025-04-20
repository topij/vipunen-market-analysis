"""Visualization functions for the Vipunen project."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from vipunen.config import BRAND_COLORS, SLIDE_HEIGHT_INCHES, SLIDE_WIDTH_INCHES

def create_color_palette(base_colors: List[str], n: int) -> List[str]:
    """Create a color palette by cycling through base colors.
    
    Args:
        base_colors: List of base colors
        n: Number of colors to generate
        
    Returns:
        List of colors
    """
    from itertools import cycle
    color_cycler = cycle(base_colors)
    return [next(color_cycler) for _ in range(n)]

def wrap_text(text: str, wrap_limit: int = 18) -> str:
    """Wrap text to fit within a specified limit.
    
    Args:
        text: Text to wrap
        wrap_limit: Maximum length of each line
        
    Returns:
        Wrapped text
    """
    words = text.split()
    lines = []
    remaining_text = text

    while len(remaining_text) > wrap_limit:
        split_at = remaining_text.rfind(" ", 0, wrap_limit)
        if split_at == -1:
            split_at = wrap_limit
        lines.append(remaining_text[:split_at])
        remaining_text = remaining_text[split_at:].strip()

    lines.append(remaining_text)
    return "\n".join(lines)

def plot_market_shares(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None,
    title: str = "Market Shares",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize: Tuple[float, float] = (SLIDE_WIDTH_INCHES, SLIDE_HEIGHT_INCHES),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Create a market share plot.
    
    Args:
        df: DataFrame containing the data
        x_col: Column to use for x-axis
        y_col: Column to use for y-axis
        hue_col: Optional column to use for color grouping
        title: Plot title
        x_label: Optional x-axis label
        y_label: Optional y-axis label
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    if hue_col:
        n_colors = len(df[hue_col].unique())
        palette = create_color_palette(BRAND_COLORS, n_colors)
        sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, palette=palette)
    else:
        sns.barplot(data=df, x=x_col, y=y_col, color=BRAND_COLORS[0])
    
    plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_growth_trends(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    trend_col: str,
    title: str = "Growth Trends",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize: Tuple[float, float] = (SLIDE_WIDTH_INCHES, SLIDE_HEIGHT_INCHES),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Create a growth trends plot.
    
    Args:
        df: DataFrame containing the data
        x_col: Column to use for x-axis
        y_col: Column to use for y-axis
        trend_col: Column containing trend information
        title: Plot title
        x_label: Optional x-axis label
        y_label: Optional y-axis label
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Create color mapping for trends
    trend_colors = {
        'Kasvussa': BRAND_COLORS[0],  # green
        'Laskussa': BRAND_COLORS[4]   # red
    }
    
    # Plot bars with trend colors
    bars = plt.bar(
        df[x_col],
        df[y_col],
        color=[trend_colors[trend] for trend in df[trend_col]]
    )
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.1f}%',
            ha='center',
            va='bottom'
        )
    
    plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_provider_counts(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Provider Counts",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize: Tuple[float, float] = (SLIDE_WIDTH_INCHES, SLIDE_HEIGHT_INCHES),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Create a provider counts plot.
    
    Args:
        df: DataFrame containing the data
        x_col: Column to use for x-axis
        y_col: Column to use for y-axis
        title: Plot title
        x_label: Optional x-axis label
        y_label: Optional y-axis label
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Create line plot
    plt.plot(df[x_col], df[y_col], marker='o', color=BRAND_COLORS[1])
    
    # Add value labels
    for x, y in zip(df[x_col], df[y_col]):
        plt.text(x, y, f'{y:.0f}', ha='center', va='bottom')
    
    plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf() 