"""Visualization module for Vipunen data."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import logging
from pathlib import Path
import textwrap

logger = logging.getLogger(__name__)

# Default brand colors
BRAND_COLORS = {
    'primary': '#007AC9',      # Main brand color
    'secondary': '#00A97A',    # Secondary brand color
    'accent1': '#FFB800',      # Accent color 1
    'accent2': '#E31E24',      # Accent color 2
    'neutral': '#666666',      # Neutral gray
    'background': '#FFFFFF',   # Background color
    'text': '#333333'          # Text color
}

class VipunenVisualizer:
    """Class for creating and styling visualizations."""
    
    def __init__(
        self,
        colors: Optional[Dict[str, str]] = None,
        style: str = 'whitegrid',
        figure_size: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        font_family: str = 'Arial'
    ):
        """Initialize the visualizer with custom settings.
        
        Args:
            colors: Custom color palette. If None, uses default BRAND_COLORS
            style: Seaborn style name
            figure_size: Default figure size (width, height)
            dpi: Resolution for saved figures
            font_family: Font family for text elements
        """
        self.colors = colors or BRAND_COLORS.copy()
        self.style = style
        self.figure_size = figure_size
        self.dpi = dpi
        self.font_family = font_family
        
        # Apply style settings
        self._apply_style()
        
    def _apply_style(self):
        """Apply style settings to matplotlib."""
        # Set seaborn style
        sns.set_style(self.style)
        
        # Set matplotlib params
        plt.rcParams.update({
            'figure.figsize': self.figure_size,
            'figure.dpi': self.dpi,
            'font.family': self.font_family,
            'axes.labelcolor': self.colors['text'],
            'text.color': self.colors['text'],
            'axes.edgecolor': self.colors['neutral'],
            'axes.grid': True,
            'grid.color': '#EEEEEE'
        })
        
    def create_color_palette(self, n_colors: int) -> List[str]:
        """Create a color palette with specified number of colors.
        
        Args:
            n_colors: Number of colors needed
            
        Returns:
            List of hex color codes
        """
        base_colors = [
            self.colors['primary'],
            self.colors['secondary'],
            self.colors['accent1'],
            self.colors['accent2']
        ]
        
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        
        # If more colors needed, create additional shades
        return sns.color_palette(base_colors, n_colors=n_colors).as_hex()
        
    def wrap_text(self, text: str, width: int = 30) -> str:
        """Wrap text to specified width.
        
        Args:
            text: Text to wrap
            width: Maximum line width
            
        Returns:
            Wrapped text
        """
        return '\n'.join(textwrap.wrap(text, width=width))
        
    def plot_market_shares(
        self,
        df: pd.DataFrame,
        year_col: str = 'tilastovuosi',
        provider_col: str = 'koulutuksenJarjestaja',
        share_col: str = 'market_share',
        title: str = 'Market Shares by Provider',
        wrap_width: int = 30,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """Create a stacked bar plot of market shares.
        
        Args:
            df: DataFrame with market share data
            year_col: Column containing years
            provider_col: Column containing provider names
            share_col: Column containing market share values
            title: Plot title
            wrap_width: Width for wrapping provider names
            figsize: Optional custom figure size
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize or self.figure_size)
        
        # Create color palette
        n_providers = df[provider_col].nunique()
        colors = self.create_color_palette(n_providers)
        
        # Prepare data
        pivot_data = df.pivot(
            index=year_col,
            columns=provider_col,
            values=share_col
        )
        
        # Create stacked bar plot
        ax = pivot_data.plot(
            kind='bar',
            stacked=True,
            color=colors,
            width=0.8
        )
        
        # Style the plot
        plt.title(title, pad=20, color=self.colors['text'])
        plt.xlabel('Year', labelpad=10)
        plt.ylabel('Market Share (%)', labelpad=10)
        
        # Wrap legend labels
        handles, labels = ax.get_legend_handles_labels()
        wrapped_labels = [self.wrap_text(label, wrap_width) for label in labels]
        plt.legend(
            handles,
            wrapped_labels,
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        
        plt.tight_layout()
        return plt.gcf()
        
    def plot_growth_trends(
        self,
        df: pd.DataFrame,
        year_col: str = 'tilastovuosi',
        value_col: str = 'nettoopiskelijamaaraLkm',
        group_col: Optional[str] = None,
        title: str = 'Growth Trends',
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """Create a line plot of growth trends.
        
        Args:
            df: DataFrame with trend data
            year_col: Column containing years
            value_col: Column containing values
            group_col: Optional column for grouping
            title: Plot title
            figsize: Optional custom figure size
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize or self.figure_size)
        
        if group_col:
            # Create color palette for groups
            n_groups = df[group_col].nunique()
            colors = self.create_color_palette(n_groups)
            
            # Plot each group
            for (name, group), color in zip(df.groupby(group_col), colors):
                plt.plot(
                    group[year_col],
                    group[value_col],
                    marker='o',
                    label=self.wrap_text(str(name)),
                    color=color
                )
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Single line plot
            plt.plot(
                df[year_col],
                df[value_col],
                marker='o',
                color=self.colors['primary']
            )
            
        plt.title(title, pad=20, color=self.colors['text'])
        plt.xlabel('Year', labelpad=10)
        plt.ylabel('Value', labelpad=10)
        
        plt.grid(True, color='#EEEEEE')
        plt.tight_layout()
        return plt.gcf()
        
    def save_plot(
        self,
        fig: plt.Figure,
        filename: str,
        directory: str = 'plots',
        formats: List[str] = ['png', 'pdf'],
        dpi: Optional[int] = None
    ) -> List[Path]:
        """Save plot in specified formats.
        
        Args:
            fig: Matplotlib figure to save
            filename: Base filename without extension
            directory: Directory to save in
            formats: List of formats to save as
            dpi: Optional custom DPI for raster formats
            
        Returns:
            List of paths where files were saved
        """
        try:
            # Create directory if it doesn't exist
            save_dir = Path(directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            for fmt in formats:
                save_path = save_dir / f"{filename}.{fmt}"
                fig.savefig(
                    save_path,
                    dpi=dpi or self.dpi,
                    bbox_inches='tight',
                    format=fmt
                )
                saved_files.append(save_path)
                logger.info(f"Saved plot as {save_path}")
                
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            raise
            
    def close_all(self):
        """Close all open matplotlib figures."""
        plt.close('all') 