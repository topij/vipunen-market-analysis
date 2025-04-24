import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import squarify  # For treemap plots

# Color palette definition (using the brand colors from the style sheets)
COLOR_PALETTES = {
    "roles": {
        "järjestäjänä": "#7dc35a",  # Green
        "hankintana": "#006e82",    # Blue
    },
    "growth": {
        "positive": "#7dc35a",  # Green
        "negative": "#e86b3d",  # Orange
    },
    "main": [
        "#7dc35a",  # Green
        "#006e82",  # Blue
        "#674da1",  # Purple
        "#7eafbb",  # Light blue
        "#e86b3d",  # Orange
        "#00b5e2",  # Cyan
        "#00263a",  # Dark blue
    ],
    "heatmap": {
        "low": "#f2f2f2",       # Light gray 
        "mid": "#7eafbb",       # Light blue
        "high": "#006e82",      # Dark blue
    },
    "market_share": "YlGnBu"    # Yellow-Green-Blue colormap for market share visualization
}

# Text constants for captions and labels
TEXT_CONSTANTS = {
    "data_source": "Vipunen API, tiedot päivitetty {date}.",
    "market_note": "Markkinaosuus määritetty {year} tutkintojen mukaan.",
    "current_year_note": "Huom. vuosi {year} on kesken."
}

class EducationVisualizer:
    """Base class for education market visualizations"""
    
    def __init__(self, style="default", style_dir=None, output_dir=None):
        """
        Initialize the visualizer with style settings
        
        Args:
            style: Name of the style to use - 'default', 'overview', or 'edge'
            style_dir: Directory containing matplotlib style files
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        
        # Configure default plot style
        plt.style.use('default')  # Reset to default style
        
        # Apply custom style if available
        if style_dir:
            style_dir = Path(style_dir)
            if style == "overview" and (style_dir / "ri_multipage_overview.mplstyle").exists():
                plt.style.use(str(style_dir / "ri_multipage_overview.mplstyle"))
            elif style == "edge" and (style_dir / "ri_right_edge_plot_style.mplstyle").exists():
                plt.style.use(str(style_dir / "ri_right_edge_plot_style.mplstyle"))
        
        # Set additional styling
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 12
        
        # Initialize data attributes
        self.current_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    def apply_common_formatting(self, fig, ax, title, caption=None, x_label=None, y_label=None):
        """
        Apply common formatting to a matplotlib figure and axis
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axis
            title: Title for the plot
            caption: Optional caption text to display below the plot
            x_label: Optional x-axis label
            y_label: Optional y-axis label
        """
        # Set title with proper styling
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, loc='left')
        
        # Apply axis labels if provided
        if x_label:
            ax.set_xlabel(x_label, fontsize=12)
        if y_label:
            ax.set_ylabel(y_label, fontsize=12)
        
        # Add caption if provided
        if caption:
            fig.text(0.01, 0.01, caption, fontsize=10, color='#555555')
        
        # Add subtle grid lines
        ax.grid(axis='y', linestyle='-', alpha=0.2)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def save_visualization(self, fig, filename, dpi=300):
        """
        Save visualization to file
        
        Args:
            fig: Matplotlib figure to save
            filename: Name of the file to save (without extension)
            dpi: Resolution for saving the figure
        """
        if self.output_dir:
            output_dir = Path(self.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            filepath = output_dir / f"{filename}.png"
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Visualization saved to {filepath}")
            return str(filepath)
        else:
            print("No output directory specified. Visualization not saved.")
            return None
    
    def create_stacked_bar_chart(self, data, x_col, y_cols, colors, labels, title, 
                                caption=None, show_totals=True, show_percentages=True,
                                filename=None, figsize=(12, 8)):
        """
        Create a stacked bar chart for comparing categories
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis categories
            y_cols: List of column names for stacked values
            colors: List of colors for each stack
            labels: List of labels for each stack
            title: Title for the chart
            caption: Optional caption
            show_totals: Whether to show total values on top of each bar
            show_percentages: Whether to show percentage in the middle of the primary bar
            filename: Optional filename to save the visualization
            figsize: Figure size tuple
            
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert x values to strings for categorical display
        x_values = data[x_col].astype(str)
        
        # Create stacked bars
        bottom = np.zeros(len(data))
        bars = []
        
        for i, col in enumerate(y_cols):
            bar = ax.bar(x_values, data[col], bottom=bottom, 
                        label=labels[i], color=colors[i])
            bars.append(bar)
            bottom += data[col].values
        
        # Add labels for total values on top of bars
        if show_totals and 'Yhteensä' in data.columns:
            for i, value in enumerate(data['Yhteensä']):
                ax.text(i, value + (value * 0.02), f"{value:.1f}", 
                        ha='center', va='bottom', fontsize=10)
        
        # Add percentage labels in the middle of the primary bar
        if show_percentages and 'järjestäjä_osuus (%)' in data.columns:
            for i, (perc, total) in enumerate(zip(data['järjestäjä_osuus (%)'], data['Yhteensä'])):
                ax.text(i, total/2, f"{perc:.1f}%", 
                        ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        
        # Apply common formatting
        self.apply_common_formatting(fig, ax, title, caption)
        
        # Add legend
        ax.legend(labels=labels)
        
        # Save if filename provided
        if filename:
            self.save_visualization(fig, filename)
        
        return fig, ax
    
    def create_area_chart(self, data, x_col, y_cols, colors, labels, title, 
                         caption=None, stacked=True, filename=None, figsize=(16, 10)):
        """
        Create an area chart for time series data
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis (time)
            y_cols: List of column names for y values
            colors: List of colors for each area
            labels: List of labels for each area
            title: Title for the chart
            caption: Optional caption
            stacked: Whether to create a stacked area chart
            filename: Optional filename to save the visualization
            figsize: Figure size tuple
            
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert x values to strings for categorical display
        x_values = data[x_col].astype(str)
        
        if stacked:
            ax.stackplot(x_values, *[data[col] for col in y_cols], 
                        labels=labels, colors=colors, alpha=0.8)
        else:
            for i, col in enumerate(y_cols):
                ax.fill_between(x_values, data[col], alpha=0.4, color=colors[i], label=labels[i])
                ax.plot(x_values, data[col], color=colors[i], linewidth=2)
        
        # Apply common formatting
        self.apply_common_formatting(fig, ax, title, caption)
        
        # Improve x-axis formatting
        plt.xticks(rotation=0)
        
        # Add subtle grid lines for readability
        ax.grid(True, linestyle='-', alpha=0.1)
        
        # Add legend
        ax.legend(loc='upper left', frameon=False)
        
        # Add y-axis gridlines
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        
        # Save if filename provided
        if filename:
            self.save_visualization(fig, filename)
        
        return fig, ax
    
    def create_line_chart(self, data, x_col, y_cols, colors, labels, title,
                         caption=None, markers=True, filename=None, figsize=(16, 10)):
        """
        Create a line chart for time series comparisons
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis (time)
            y_cols: List of column names for y values
            colors: List of colors for each line
            labels: List of labels for each line
            title: Title for the chart
            caption: Optional caption
            markers: Whether to show markers on data points
            filename: Optional filename to save the visualization
            figsize: Figure size tuple
            
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert x values to strings for categorical display if needed
        x_values = data[x_col] if pd.api.types.is_numeric_dtype(data[x_col]) else data[x_col].astype(str)
        
        # Line styles for differentiation
        line_styles = ['-', '--', ':', '-.']
        marker_styles = ['o', 's', '^', 'D', 'v']
        
        # Create lines
        for i, col in enumerate(y_cols):
            line_style = line_styles[i % len(line_styles)]
            marker_style = marker_styles[i % len(marker_styles)] if markers else None
            ax.plot(x_values, data[col], color=colors[i], linestyle=line_style,
                   marker=marker_style, markersize=8, linewidth=2.5,
                   label=labels[i])
        
        # Apply common formatting
        self.apply_common_formatting(fig, ax, title, caption)
        
        # Improve x-axis formatting
        if not pd.api.types.is_numeric_dtype(data[x_col]):
            plt.xticks(rotation=0)
        
        # Add legend
        ax.legend(loc='best', frameon=True, framealpha=0.9)
        
        # Add y-axis gridlines
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        
        # Save if filename provided
        if filename:
            self.save_visualization(fig, filename)
        
        return fig, ax
    
    def create_heatmap(self, data, title, caption=None, cmap=None, annot=True,
                      filename=None, figsize=(14, 10), fmt=".2f"):
        """
        Create a heatmap for tabular data
        
        Args:
            data: DataFrame containing the data
            title: Title for the chart
            caption: Optional caption
            cmap: Colormap to use (defaults to blue gradient)
            annot: Whether to annotate cells with values
            filename: Optional filename to save the visualization
            figsize: Figure size tuple
            fmt: Format string for annotations
            
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colormap if not provided
        if cmap is None:
            cmap = LinearSegmentedColormap.from_list(
                "custom_blue",
                [COLOR_PALETTES["heatmap"]["low"], 
                 COLOR_PALETTES["heatmap"]["mid"], 
                 COLOR_PALETTES["heatmap"]["high"]],
                N=100
            )
        
        # Create heatmap
        sns.heatmap(data, ax=ax, cmap=cmap, annot=annot, fmt=fmt,
                   linewidths=0.5, linecolor='white',
                   cbar=False)
        
        # Apply common formatting (special case for heatmap)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, loc='left')
        
        # Add caption if provided
        if caption:
            fig.text(0.01, 0.01, caption, fontsize=10, color='#555555')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            self.save_visualization(fig, filename)
        
        return fig, ax
    
    def create_horizontal_bar_chart(self, data, x_col, y_col, color_col=None, title=None,
                                  caption=None, show_values=True, sort_by=None,
                                  filename=None, figsize=(12, 8)):
        """
        Create a horizontal bar chart
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis values (lengths of bars)
            y_col: Column name for y-axis categories
            color_col: Column name to determine bar colors (optional)
            title: Title for the chart
            caption: Optional caption
            show_values: Whether to show values at end of bars
            sort_by: Column to sort by (default: x_col)
            filename: Optional filename to save the visualization
            figsize: Figure size tuple
            
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort data if requested
        if sort_by:
            data = data.sort_values(by=sort_by)
        else:
            data = data.sort_values(by=x_col)
        
        # Determine colors
        if color_col and color_col in data.columns:
            # Color based on a column value (e.g., positive/negative)
            colors = [COLOR_PALETTES["growth"]["positive"] if val >= 0 else 
                      COLOR_PALETTES["growth"]["negative"] for val in data[color_col]]
        else:
            # Use a single color
            colors = COLOR_PALETTES["main"][0]
        
        # Create horizontal bars
        bars = ax.barh(data[y_col], data[x_col], color=colors)
        
        # Add value labels at the end of bars
        if show_values:
            for i, bar in enumerate(bars):
                value = data[x_col].iloc[i]
                text_color = 'black'
                ax.text(
                    value + (abs(value) * 0.02),  # Slight offset
                    bar.get_y() + bar.get_height()/2,
                    f"{value:.1f}",
                    va='center',
                    ha='left' if value >= 0 else 'right',
                    color=text_color,
                    fontsize=10
                )
        
        # Apply common formatting
        self.apply_common_formatting(fig, ax, title, caption)
        
        # Add a vertical line at x=0 if we have positive and negative values
        if data[x_col].min() < 0 and data[x_col].max() > 0:
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Save if filename provided
        if filename:
            self.save_visualization(fig, filename)
        
        return fig, ax
    
    def create_treemap(self, data, value_col, label_col, color_col=None, title=None,
                      caption=None, filename=None, figsize=(16, 10)):
        """
        Create a treemap visualization
        
        Args:
            data: DataFrame containing the data
            value_col: Column name for size values
            label_col: Column name for segment labels
            color_col: Column name for color values (optional)
            title: Title for the chart
            caption: Optional caption
            filename: Optional filename to save the visualization
            figsize: Figure size tuple
            
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for treemap
        sizes = data[value_col].values
        labels = data[label_col].values
        
        # Normalize sizes for better visualization
        total_size = sum(sizes)
        norm_sizes = [size / total_size * 100 for size in sizes]
        
        # Determine colors
        if color_col and color_col in data.columns:
            # Colors based on provided column
            color_values = data[color_col]
            norm = plt.Normalize(color_values.min(), color_values.max())
            colors = [plt.cm.viridis(norm(val)) for val in color_values]
        else:
            # Use palette colors in sequence
            colors = [COLOR_PALETTES["main"][i % len(COLOR_PALETTES["main"])] 
                     for i in range(len(sizes))]
        
        # Create treemap
        squarify.plot(
            sizes=norm_sizes,
            label=[f"{labels[i]}\n({norm_sizes[i]:.1f}%)" for i in range(len(labels))],
            color=colors,
            alpha=0.8,
            ax=ax
        )
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        
        # Apply title and caption
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, loc='left')
        
        # Add caption if provided
        if caption:
            fig.text(0.01, 0.01, caption, fontsize=10, color='#555555')
        
        # Save if filename provided
        if filename:
            self.save_visualization(fig, filename)
        
        return fig, ax 