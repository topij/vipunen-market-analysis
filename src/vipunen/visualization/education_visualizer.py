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
import textwrap

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
    
    def _wrap_text(self, text, width):
        """Wrap text to a specified width."""
        return '\n'.join(textwrap.wrap(text, width=width))

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
        else:
            ax.set_xlabel(None)  # Ensure no x-label if not provided
        
        if y_label:
            ax.set_ylabel(y_label, fontsize=12)
        else:
            ax.set_ylabel(None)  # Ensure no y-label if not provided
        
        # Add caption if provided
        if caption:
            # Position caption slightly lower and more to the left
            fig.text(0.05, 0.01, caption, fontsize=9, color='#555555', ha='left')
        
        # Add subtle horizontal grid lines
        ax.grid(axis='y', linestyle='-', alpha=0.5)  # Increased alpha for visibility
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Make bottom and left spines lighter
        ax.spines['bottom'].set_color('#cccccc')
        ax.spines['left'].set_color('#cccccc')
        
        # Remove x-axis tick lines, but keep labels
        ax.tick_params(axis='x', length=0)
        # Set y-axis ticks color
        ax.tick_params(axis='y', colors='#555555')
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
    
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
                         caption=None, stacked=True, filename=None, figsize=(10, 6)):
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
                        labels=labels, colors=colors, alpha=1.0)
        else:
            for i, col in enumerate(y_cols):
                ax.fill_between(x_values, data[col], alpha=0.4, color=colors[i], label=labels[i])
                ax.plot(x_values, data[col], color=colors[i], linewidth=2)
        
        # Apply common formatting
        self.apply_common_formatting(fig, ax, title, caption, x_label=None, y_label="Netto-opiskelijamäärä")
        
        # Add legend to bottom left with frame
        ax.legend(loc='lower left', frameon=True)
        
        # Save if filename provided
        if filename:
            self.save_visualization(fig, filename)
        
        return fig, ax
    
    def create_line_chart(self, data, x_col, y_cols, colors, labels, title,
                         caption=None, markers=True, filename=None, figsize=(10, 6)):
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
            ax.plot(x_values, data[col], color=colors[i % len(colors)], linestyle=line_style,
                   marker=marker_style, markersize=8, linewidth=2.5,
                   label=labels[i])
        
        # Apply common formatting
        self.apply_common_formatting(fig, ax, title, caption, x_label=None, y_label="Markkinaosuus (%)")
        
        # Add legend
        ax.legend(loc='upper left', frameon=True, framealpha=0.9)
        
        # Save if filename provided
        if filename:
            self.save_visualization(fig, filename)
        
        return fig, ax
    
    def create_heatmap(self, data, title, caption=None, cmap='Purples', annot=True,
                      filename=None, figsize=(10, 8), fmt=".2f", wrap_width=25):
        """
        Create a heatmap for tabular data

        Args:
            data: DataFrame containing the data (Index=Qualifications, Columns=Years)
            title: Title for the chart
            caption: Optional caption
            cmap: Colormap to use
            annot: Whether to annotate cells with values
            filename: Optional filename to save the visualization
            figsize: Figure size tuple
            fmt: Format string for annotations
            wrap_width: Width to wrap y-axis labels

        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(data, ax=ax, cmap=cmap, annot=annot, fmt=fmt,
                   linewidths=0.5, linecolor='white',
                   cbar=False)

        # Apply common formatting (partially)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, loc='left')

        # Add caption if provided
        if caption:
            fig.text(0.05, 0.01, caption, fontsize=9, color='#555555', ha='left')

        # Customize axes specifically for heatmap
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.tick_params(axis='y', length=0, rotation=0) # Remove y-ticks, ensure horizontal labels
        ax.tick_params(axis='x', length=0, rotation=0) # Remove x-ticks, ensure horizontal labels

        # Wrap y-axis labels
        ax.set_yticklabels([self._wrap_text(label.get_text(), wrap_width) for label in ax.get_yticklabels()],
                           fontsize=10, va='center')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)

        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent caption overlap

        # Save if filename provided
        if filename:
            self.save_visualization(fig, filename)

        return fig, ax

    def create_heatmap_with_marginals(
        self,
        heatmap_data, # Pivot table (Index=Qualification, Columns=Year, Values=Market Share %)
        top_data,     # Series/DataFrame (Index=Year, Values=Total Volume)
        right_data,   # Series/DataFrame (Index=Qualification, Values=Latest Year Volume)
        title,
        top_title,
        right_title,
        caption=None,
        cmap='Purples',
        heatmap_fmt=".2f",
        line_color=COLOR_PALETTES["main"][0],
        bar_palette='Blues',
        filename=None,
        figsize=(12, 10), # Adjusted figsize
        wrap_width=25,
        heatmap_annot=True
    ):
        """
        Create a heatmap with marginal line and bar plots.

        Args:
            heatmap_data: Pivot table for the main heatmap.
            top_data: Data for the top line plot (total volume over years).
            right_data: Data for the right bar plot (volume per qualification in latest year).
            title: Main title for the figure.
            top_title: Title for the top line plot.
            right_title: Title (x-axis label) for the right bar plot.
            caption: Optional caption for the figure.
            cmap: Colormap for the heatmap.
            heatmap_fmt: Format string for heatmap annotations.
            line_color: Color for the top line plot.
            bar_palette: Palette name or list of colors for the right bar plot.
            filename: Optional filename to save the visualization.
            figsize: Figure size tuple.
            wrap_width: Width to wrap heatmap y-axis labels.
            heatmap_annot: Whether to annotate heatmap cells.

        Returns:
            fig: Matplotlib figure object.
        """

        # Ensure index/column alignment for plotting
        heatmap_data = heatmap_data.sort_index(ascending=True)
        right_data = right_data.reindex(heatmap_data.index) # Align bar plot order to heatmap
        top_data = top_data.reindex(heatmap_data.columns) # Align line plot order to heatmap

        fig = plt.figure(figsize=figsize)

        # Define GridSpec
        gs = fig.add_gridspec(
            2, 2, width_ratios=[15, 4], height_ratios=[3, 15],
            wspace=0.05, hspace=0.05
        )

        # Create axes
        ax_top = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top) # Share X with top plot
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main) # Share Y with main plot

        # --- Plot Top Line Plot (Total Volume) ---
        ax_top.plot(top_data.index.astype(str), top_data.values, color=line_color, marker='o')
        ax_top.set_title(top_title, fontsize=10, color='gray', loc='left')
        # Styling
        ax_top.tick_params(axis='x', bottom=False, labelbottom=False) # Hide x-axis labels/ticks
        ax_top.tick_params(axis='y', length=0)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['bottom'].set_visible(False)
        ax_top.spines['left'].set_color('#cccccc')
        ax_top.grid(axis='y', linestyle='-', alpha=0.5)
        ax_top.set_ylabel("Koko markkina", fontsize=9, color='gray')
        ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ','))) # Thousands separator

        # --- Plot Main Heatmap (Market Share %) ---
        sns.heatmap(heatmap_data, ax=ax_main, cmap=cmap, annot=heatmap_annot, fmt=heatmap_fmt,
                    linewidths=0.5, linecolor='white', cbar=False)
        # Styling
        ax_main.tick_params(axis='x', bottom=False, length=0)
        ax_main.tick_params(axis='y', left=False, length=0, rotation=0)
        ax_main.set_xlabel(None)
        ax_main.set_ylabel(None)
        # Wrap y-axis labels
        ax_main.set_yticklabels([self._wrap_text(label.get_text(), wrap_width) for label in ax_main.get_yticklabels()],
                                fontsize=10, va='center')
        ax_main.set_xticklabels(ax_main.get_xticklabels(), fontsize=10)
        # Remove spines
        for spine in ax_main.spines.values():
            spine.set_visible(False)

        # --- Plot Right Bar Plot (Latest Year Volume) ---
        sns.barplot(x=right_data.values, y=right_data.index, ax=ax_right, orient='h', palette=bar_palette)
        # Styling
        ax_right.tick_params(axis='y', left=False, labelleft=False) # Hide y-axis labels/ticks
        ax_right.tick_params(axis='x', length=0)
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['bottom'].set_color('#cccccc')
        ax_right.spines['left'].set_visible(False)
        ax_right.set_xlabel(right_title, fontsize=9, color='gray')
        ax_right.grid(axis='x', linestyle='-', alpha=0.5)
        ax_right.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ','))) # Thousands separator

        # --- Figure Level Formatting ---
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98, ha='left', x=0.05)
        if caption:
            fig.text(0.05, 0.01, caption, fontsize=9, color='#555555', ha='left')

        # Adjust layout slightly if needed
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent caption/title overlap

        # Save if filename provided
        if filename:
            self.save_visualization(fig, filename)

        return fig # Return only the figure object

    def create_horizontal_bar_chart(self, data, x_col, y_col, volume_col=None, color_col=None, title=None,
                                  caption=None, sort_by=None, # Removed show_values
                                  filename=None, figsize=(10, 8), wrap_width=25,
                                  y_label_detail_format="({:.0f})", # Format for detail in y-label
                                  x_label_text="Value"):
        """
        Create a horizontal bar chart

        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis values (lengths of bars, e.g., growth %)
            y_col: Column name for y-axis categories (e.g., qualification name)
            volume_col: Optional column name for detail value (e.g. volume or share), added to y-axis labels.
            color_col: Column name to determine bar colors (optional, uses x_col if None and values can be pos/neg)
            title: Title for the chart
            caption: Optional caption
            sort_by: Column to sort by (default: None, assumes pre-sorted)
            filename: Optional filename to save the visualization
            figsize: Figure size tuple
            wrap_width: Width to wrap y-axis labels.
            y_label_detail_format: Python f-string format specifier for the detail value in y-labels.
            x_label_text: Text for the x-axis label.

        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Use x_col for coloring if color_col is not specified
        if color_col is None:
            color_col = x_col

        # Sort data if requested
        if sort_by:
            data = data.sort_values(by=sort_by)
        # else: # Assuming data is pre-sorted if sort_by is None
        #     data = data.sort_values(by=x_col)

        # Determine colors based on the sign of the value in color_col
        if color_col and color_col in data.columns and pd.api.types.is_numeric_dtype(data[color_col]):
            colors = [COLOR_PALETTES["growth"]["positive"] if val >= 0 else
                      COLOR_PALETTES["growth"]["negative"] for val in data[color_col]]
        else:
            # Fallback to a single color if coloring logic doesn't apply
            colors = [COLOR_PALETTES["main"][0]] * len(data)

        # Prepare y-axis labels with optional volume/detail
        if volume_col and volume_col in data.columns:
            try:
                y_labels_raw = [f"{row[y_col]} {y_label_detail_format.format(row[volume_col])}" for _, row in data.iterrows()]
            except (ValueError, TypeError):
                 # Fallback if formatting fails
                y_labels_raw = [f"{row[y_col]} ({row[volume_col]})" for _, row in data.iterrows()]
        else:
            y_labels_raw = data[y_col].tolist()

        y_labels_wrapped = [self._wrap_text(label, wrap_width) for label in y_labels_raw]

        # Create horizontal bars using numerical indices for y position
        y_pos = np.arange(len(data))
        bars = ax.barh(y_pos, data[x_col], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels_wrapped, fontsize=9)

        # Apply common formatting
        self.apply_common_formatting(fig, ax, title, caption, x_label=x_label_text, y_label=None)

        # Customize for horizontal bar chart
        ax.tick_params(axis='y', length=0) # Remove y-tick marks

        # Add a vertical line at x=0 if we have positive and negative values
        if data[x_col].min() < 0 and data[x_col].max() > 0:
            ax.axvline(x=0, color='#555555', linestyle='-', alpha=0.7, linewidth=1)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent caption/title overlap

        # Save if filename provided
        if filename:
            self.save_visualization(fig, filename)

        return fig, ax
    
    def create_treemap(self, data, value_col, label_col, detail_col=None, title=None,
                      caption=None, filename=None, figsize=(12, 8), wrap_width=20):
        """
        Create a treemap visualization

        Args:
            data: DataFrame containing the data
            value_col: Column name for size values (e.g., total market volume)
            label_col: Column name for segment labels (e.g., qualification name)
            detail_col: Column name for the detail value to display inside segments (e.g., market share %)
            title: Title for the chart
            caption: Optional caption
            filename: Optional filename to save the visualization
            figsize: Figure size tuple
            wrap_width: Width to wrap the label text inside segments.

        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Prepare data for treemap
        sizes = data[value_col].values
        labels = data[label_col].values
        details = data[detail_col].values if detail_col and detail_col in data.columns else sizes # Use size if no detail

        # Normalize sizes for plotting
        norm_sizes = squarify.normalize_sizes(sizes, dx=100, dy=100)

        # Use palette colors in sequence
        colors = [COLOR_PALETTES["main"][i % len(COLOR_PALETTES["main"])]
                 for i in range(len(sizes))]

        # Create treemap rectangles
        rects = squarify.plot(
            sizes=norm_sizes,
            color=colors,
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
            ax=ax,
            label=None # Labels will be added manually
        )

        # Add text labels inside rectangles
        for i, rect in enumerate(rects):
            x, y, dx, dy = rect['x'], rect['y'], rect['dx'], rect['dy']
            label_text = self._wrap_text(labels[i], wrap_width)
            detail_val = details[i]
            # Format detail as percentage if name suggests it, otherwise float
            detail_fmt = "({:.1f}%)" if ('%' in detail_col if detail_col else False) else "({:.1f})"
            full_label = f"{label_text}\n{detail_fmt.format(detail_val)}"

            # Determine text color based on background lightness (simple heuristic)
            bg_color = colors[i % len(colors)]
            r, g, b, _ = to_rgba(bg_color)
            luminance = 0.299*r + 0.587*g + 0.114*b # Calculate perceived luminance
            text_color = 'white' if luminance < 0.5 else 'black'

            # Adjust font size based on rectangle size (heuristic)
            fontsize = max(6, min(10, int(np.sqrt(dx * dy) / 1.5)))

            ax.text(x + dx / 2, y + dy / 2, full_label,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=fontsize, color=text_color, wrap=True)

        # Remove axes and frame
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        # Apply title and caption (using common formatting principles)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, loc='left')
        if caption:
            fig.text(0.05, 0.01, caption, fontsize=9, color='#555555', ha='left')

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent caption/title overlap

        # Save if filename provided
        if filename:
            self.save_visualization(fig, filename)

        return fig, ax 