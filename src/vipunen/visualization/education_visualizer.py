import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import squarify
import textwrap
import warnings # Add import
from matplotlib.backends.backend_pdf import PdfPages # Added import
import plotly.graph_objects as go
import io
import matplotlib.image as mpimg
import logging # Add import
import datetime
from typing import Optional

# Try importing kaleido for static image export, but don't make it a hard requirement
kaleido_installed = False
try:
    import kaleido
    # Optionally check version or perform a basic operation
    kaleido_installed = True
except ImportError:
    print("Warning: kaleido package not found. Plotly static image export (needed for PDF) will not work.")
    print("Please install it: pip install kaleido")

# Set up logger for this module
logger = logging.getLogger(__name__)

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

    def __init__(self, style="default", style_dir=None, output_dir=None, output_format='png', 
                 institution_short_name="unknown", include_timestamp=True):
        """
        Initialize the visualizer with style settings
        
        Args:
            style: Name of the style to use - 'default', 'overview', or 'edge'
            style_dir: Directory containing matplotlib style files
            output_dir: Directory to save visualizations
            output_format: Output format ('png' or 'pdf'). Defaults to 'png'.
            institution_short_name: Short name of the institution being analyzed (for PDF filename).
            include_timestamp: Whether to include a timestamp in the PDF filename.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.output_format = output_format
        self.pdf_pages = None
        self._pdf_has_content = False # Track if PDF has pages added
        self.pdf_path = None # Store the generated PDF path

        if self.output_format == 'pdf' and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
            pdf_filename = f"{institution_short_name.lower()}_visualizations{'_' + timestamp if timestamp else ''}.pdf"
            self.pdf_path = self.output_dir / pdf_filename # Store the path
            self.pdf_pages = PdfPages(self.pdf_path)
            logger.info(f"Initialized PDF output at: {self.pdf_path}") # Use logger
        elif self.output_dir:
             os.makedirs(self.output_dir, exist_ok=True) # Ensure dir exists for PNGs too
        
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
        Save visualization to file (either individual PNG or add to PDF).

        Args:
            fig: Matplotlib figure to save.
            filename: Base name for the file (used for PNG).
            dpi: Resolution for saving the PNG figure.

        Returns:
            str: Path to the saved PNG file if format is 'png'.
            bool: True if figure was added to PDF, False otherwise.
                  Returns None if no output directory is specified.
        """
        standard_pdf_figsize = (16, 9) # Define standard 16:9 size for PDF pages
        
        if not self.output_dir:
            print("No output directory specified. Visualization not saved.")
            return None

        if self.output_format == 'pdf':
            if self.pdf_pages:
                try:
                    # --- Force standard size for PDF pages ---
                    original_size = fig.get_size_inches()
                    fig.set_size_inches(standard_pdf_figsize, forward=True)
                    # --- Save to PDF ---
                    self.pdf_pages.savefig(fig) # Removed bbox_inches='tight' for uniform PDF page size
                    # --- Restore original size (optional, good practice if fig is reused) ---
                    fig.set_size_inches(original_size, forward=True) 
                    
                    self._pdf_has_content = True # Mark that we've added a page
                    print(f"Added figure to PDF: {filename}")
                    return True
                except Exception as e:
                    print(f"Error adding figure to PDF: {e}")
                    # Attempt to restore size even on error
                    try: fig.set_size_inches(original_size, forward=True)
                    except: pass
                    return False
            else:
                print("PDF output format selected, but PdfPages object not initialized.")
                return False
        elif self.output_format == 'png':
            filepath = self.output_dir / f"{filename}.png"
            try:
                # PNGs can keep their original intended size
                fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
                print(f"Visualization saved to {filepath}")
                return str(filepath)
            except Exception as e:
                print(f"Error saving PNG file {filepath}: {e}")
                return None
        else:
            print(f"Unsupported output format: {self.output_format}")
            return None

    def close_pdf(self):
        """Closes the PDF file object if it's open and has content, ensuring it is finalized correctly."""
        if self.output_format == 'pdf' and self.pdf_pages:
            pdf_object_to_close = self.pdf_pages
            self.pdf_pages = None # Prevent further use immediately
            
            pdf_file_path = None
            try:
                # Safely try to get the filename associated with the PdfPages object
                if hasattr(pdf_object_to_close, '_file') and hasattr(pdf_object_to_close._file, 'fh') and pdf_object_to_close._file.fh:
                    pdf_file_path = pdf_object_to_close._file.fh.name
            except Exception as e:
                 print(f"Warning: Could not get PDF filename before closing - {e}")

            try:
                if self._pdf_has_content:
                    print(f"Closing PDF file with content{(f' at {pdf_file_path}' if pdf_file_path else '')}...")
                    pdf_object_to_close.close()
                    print(f"PDF file successfully closed.")
                else:
                    # If no content was added, close the object and attempt to delete the empty file
                    print("Closing empty PDF file...")
                    pdf_object_to_close.close() # Close to release handle
                    if pdf_file_path:
                        try:
                            os.remove(pdf_file_path)
                            print(f"Removed empty PDF file: {pdf_file_path}")
                        except OSError as e:
                            print(f"Warning: Could not remove empty PDF file '{pdf_file_path}': {e}")
                    else:
                        print("Warning: Cannot remove empty PDF file (path unknown).")
                        
            except Exception as e:
                print(f"Error during PDF closing/cleanup: {e}")
            finally:
                 # Ensure the reference is cleared even if errors occurred
                 pdf_object_to_close = None 
        elif self.output_format == 'pdf':
             # This case should ideally not happen if initialized correctly
             print("Warning: close_pdf called but self.pdf_pages was already None.")

    def create_stacked_bar_chart(self, data, x_col, y_cols, colors, labels, title, 
                                caption=None, show_totals=True, show_percentages=True,
                                figsize=(12, 6.75)):
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
            figsize: Figure size tuple (defaults to 16:9 aspect ratio)
            
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
        
        return fig, ax
    
    def create_area_chart(self, data, x_col, y_cols, colors, labels, title,
                         caption=None, stacked=True, figsize=(12, 6.75)):
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
            figsize: Figure size tuple (defaults to 16:9 aspect ratio)
            
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
        
        return fig, ax
    
    def create_line_chart(self, data, x_col, y_cols, colors, labels, title,
                         caption=None, markers=True, figsize=(12, 6.75)):
        """
        Create a line chart for time series comparisons
        
        Args:
            data: DataFrame containing the data (Index=time/category, Columns=series)
            x_col: Index or column name for x-axis (time/category). If passing index, use data.index.
            y_cols: List of column names for y values
            colors: List of colors for each line
            labels: List of labels for each line
            title: Title for the chart
            caption: Optional caption
            markers: Whether to show markers on data points
            figsize: Figure size tuple (defaults to 16:9 aspect ratio)
            
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Use index if x_col is the index, otherwise use the column
        if isinstance(x_col, pd.Index):
            x_values = x_col.astype(str) # Assume index needs string conversion for plotting
        else:
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
        
        return fig, ax
    
    def create_heatmap(self, data, title, caption=None, cmap='Purples', annot=True,
                      figsize=(12, 6.75), fmt=".2f", wrap_width=25):
        """
        Create a heatmap for tabular data

        Args:
            data: DataFrame containing the data (Index=Qualifications, Columns=Years)
            title: Title for the chart
            caption: Optional caption
            cmap: Colormap to use
            annot: Whether to annotate cells with values
            figsize: Figure size tuple (defaults to 16:9 aspect ratio)
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
        line_color=None, # Default to None, pick from palette if needed
        bar_palette='Blues',
        figsize=(15, 8.4375), # Adjusted figsize to ~16:9 (wider)
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
            line_color: Color for the top line plot. Defaults to first color in main palette.
            bar_palette: Palette name or list of colors for the right bar plot.
            figsize: Figure size tuple (defaults to 16:9 aspect ratio).
            wrap_width: Width to wrap heatmap y-axis labels.
            heatmap_annot: Whether to annotate heatmap cells.

        Returns:
            fig: Matplotlib figure object.
        """
        # Default line color if not provided
        if line_color is None:
             line_color=COLOR_PALETTES["main"][0]

        # Ensure index/column alignment for plotting
        heatmap_data = heatmap_data.sort_index(ascending=True)
        right_data = right_data.reindex(heatmap_data.index) # Align bar plot order to heatmap
        top_data = top_data.reindex(heatmap_data.columns) # Align line plot order to heatmap

        fig = plt.figure(figsize=figsize)

        # Define GridSpec - Reverting to previous state
        gs = fig.add_gridspec(
            # 2, 2, width_ratios=[15, 2], height_ratios=[4, 25], # Match notebook/test 
            2, 2, width_ratios=[18, 4], height_ratios=[3, 20], # Reverted state
            # wspace=0.0, hspace=0.0 # Match notebook/test 
            wspace=0.05, hspace=0 # Reverted state
        )

        # Create axes
        ax_top = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top) # Share X with top plot
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main) # Share Y with main plot

        # --- Plot Top Line Plot (Total Volume) ---
        ax_top.plot(top_data.index.astype(str), top_data.values, color=line_color, marker='o')
        ax_top.set_title(top_title, fontsize=10, color='gray', loc='left')
        # Styling
        ax_top.tick_params(axis='x', bottom=False, labelbottom=False) # Hide labels on top plot
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
        # Enable x-ticks and labels at the bottom of the heatmap, remove rotation
        ax_main.tick_params(axis='x', bottom=True, labelbottom=True)
        ax_main.tick_params(axis='y', left=False, length=0, rotation=0)
        ax_main.set_xlabel(None)
        ax_main.set_ylabel(None)
        # Wrap y-axis labels
        ax_main.set_yticklabels([self._wrap_text(label.get_text(), wrap_width) for label in ax_main.get_yticklabels()],
                                fontsize=10, va='center')
        # Remove spines
        for spine in ax_main.spines.values():
            spine.set_visible(False)

        # --- Plot Right Bar Plot (Latest Year Volume) ---
        # Align right_data index ONLY IF right_data is not None
        if right_data is not None:
             right_data = right_data.reindex(heatmap_data.index) # Align bar plot order to heatmap

        # Check if right_data is not None and not empty before proceeding
        if right_data is not None and not right_data.empty:
             sns.barplot(x=right_data.values, y=right_data.index, ax=ax_right, orient='h', palette=bar_palette, hue=right_data.index, legend=False)
             # Styling
             ax_right.tick_params(axis='y', left=False, labelleft=False, length=0) # Hide y ticks/labels
             ax_right.tick_params(axis='x', length=0)
             ax_right.spines['top'].set_visible(False)
             ax_right.spines['right'].set_visible(False)
             ax_right.spines['bottom'].set_color('#cccccc')
             ax_right.spines['left'].set_visible(False)
             ax_right.set_xlabel(right_title, fontsize=9, color='gray')
             ax_right.set_ylabel(None) # Explicitly hide y-axis label
             ax_right.grid(axis='x', linestyle='-', alpha=0.5)
             # Set specific ticks: 0 and max value
             max_val = right_data.values.max()
             if max_val > 0:
                 ax_right.set_xticks([0, max_val])
                 # Keep formatter for potentially large max_val
                 ax_right.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
                 # Adjust xlim to give space for max label
                 ax_right.set_xlim(left=0, right=max_val * 1.05)
             else: # Handle case where max_val is 0 or data is empty
                 ax_right.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
                 
             # Explicitly set ylim to match heatmap after plotting - REMOVED
             # ax_right.set_ylim(ax_main.get_ylim())

        else:
             # Handle case where right_data is empty or None (e.g., hide axis?)
             ax_right.set_visible(False)
             

        # --- Figure Level Formatting ---
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98, ha='left', x=0.05)
        if caption:
            fig.text(0.05, 0.01, caption, fontsize=9, color='#555555', ha='left')

        # Adjust layout slightly if needed - Removed subplots_adjust
        # fig.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.9) # Match notebook/test 

        return fig # Return only the figure object

    def create_horizontal_bar_chart(self, data, x_col, y_col, volume_col=None, color_col=None, title=None,
                                  caption=None, sort_by=None, # Removed show_values
                                  figsize=(12, 6.75), wrap_width=25, # Adjusted figsize
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
            figsize: Figure size tuple (defaults to 16:9 aspect ratio)
            wrap_width: Width to wrap y-axis labels.
            y_label_detail_format: Python f-string format specifier for the detail value in y-labels.
            x_label_text: Text for the x-axis label.

        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        if sort_by:
            data = data.sort_values(sort_by)
            
        fig, ax = plt.subplots(figsize=figsize)

        # Determine bar colors
        if color_col and color_col in data.columns:
            colors = [COLOR_PALETTES["main"][i % len(COLOR_PALETTES["main"])] for i in range(len(data))] # Placeholder logic
        elif data[x_col].dtype in [np.float64, np.int64] and (data[x_col] < 0).any(): # Color pos/neg if numeric
            colors = ['#d62728' if x < 0 else '#2ca02c' for x in data[x_col]] # Red for negative, Green for positive
        else:
            colors = COLOR_PALETTES["main"][0] # Default single color

        # Create bars
        bars = ax.barh(data[y_col], data[x_col], color=colors)

        # Add data labels next to bars (optimized)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            # Determine position based on bar width
            # Use a small fixed offset or relative offset
            offset = 2 # Small fixed offset
            label_x_pos = width + offset if width >= 0 else width - offset
            ha = 'left' if width >= 0 else 'right'
            
            # Get corresponding data for the bar
            row_data = data.iloc[i]
            y_label_text = str(row_data[y_col])
            detail_value = row_data.get(volume_col) # Use .get for safety
            
            # Format the label string
            label_string = self._wrap_text(y_label_text, wrap_width) # Wrap base label
            if volume_col and pd.notna(detail_value):
                try:
                    detail_formatted = y_label_detail_format.format(detail_value)
                    label_string = f"{label_string} {detail_formatted}"
                except (ValueError, TypeError, KeyError):
                     # Fallback if format or column fails
                     label_string = f"{label_string} ({detail_value})"
            
            # Add the combined text label next to the bar
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2.,
                    label_string, va='center', ha=ha, fontsize=9)

        # Apply common formatting (partially, horizontal needs tweaks)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, loc='left')
        ax.set_xlabel(x_label_text, fontsize=12)
        ax.set_ylabel(None) # Y-label handled by tick labels

        # Customize y-axis labels with wrapping and optional detail (REMOVED - Handled by text labels now)
        # def format_ylabel(label_text, detail_value):
        #     wrapped = self._wrap_text(str(label_text), wrap_width)
        #     if pd.notna(detail_value):
        #          return f"{wrapped} {y_label_detail_format.format(detail_value)}"
        #     return wrapped
        #
        # if volume_col and volume_col in data.columns:
        #     y_labels = [format_ylabel(data[y_col].iloc[i], data[volume_col].iloc[i]) for i in range(len(data))]
        # else:
        #     y_labels = [self._wrap_text(str(label), wrap_width) for label in data[y_col]]
        # ax.set_yticklabels(y_labels, fontsize=10, va='center')

        # Grid lines (REMOVED)
        # ax.grid(axis='x', linestyle='-', alpha=0.5)

        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_color('#cccccc')
        # ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_visible(False) # Set invisible
        ax.spines['left'].set_visible(False)   # Set invisible

        # Ticks
        # ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='y', length=0, left=False, labelleft=False) # Remove ticks and labels
        ax.tick_params(axis='x', colors='#555555')
        
        # Add a vertical line at x=0 if the data spans positive and negative
        if data[x_col].min() < 0 < data[x_col].max():
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

        # Adjust layout
        # Need to adjust margins, especially left, to accommodate labels outside the bars
        plt.tight_layout(rect=[0.01, 0.05, 0.90, 0.93]) # Give more space on right, less on left

        # Add caption if provided
        if caption:
            fig.text(0.01, 0.01, caption, fontsize=9, color='#555555', ha='left', va='bottom')

        return fig, ax
    
    def create_treemap(self, data, value_col, label_col, detail_col=None, title=None,
                      caption=None, figsize=(16, 9), wrap_width=20):
        """
        Create a treemap visualization using Matplotlib and Squarify.

        Args:
            data: DataFrame containing the data, sorted appropriately for layout stability.
            value_col: Column name for size values (e.g., total market volume).
            label_col: Column name for segment labels (e.g., qualification name).
            detail_col: Column name for the detail value to display inside segments (e.g., market share %).
            title: Title for the chart.
            caption: Optional caption.
            figsize: Figure size tuple (defaults to 16:9 aspect ratio).
            wrap_width: Width to wrap the label text inside segments.

        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Basic check for empty data before accessing columns
        if data.empty:
            logger.warning(f"Skipping treemap '{title}': Input data is empty.")
            plt.close(fig) # Close the empty figure
            return fig, ax
            
        # Prepare data for treemap
        # Ensure columns exist before accessing
        if value_col not in data.columns or label_col not in data.columns:
             logger.error(f"Missing required columns for treemap '{title}': Need '{value_col}' and '{label_col}'. Found: {data.columns.tolist()}")
             plt.close(fig)
             return fig, ax
             
        sizes = data[value_col].values
        labels = data[label_col].values
        # Use size if no detail_col provided or column doesn't exist
        details = data[detail_col].values if detail_col and detail_col in data.columns else sizes

        # Check for empty data after extracting values
        if len(sizes) == 0:
            logger.warning(f"Skipping treemap '{title}': No data after preparing values/labels.")
            plt.close(fig) # Close the empty figure
            return fig, ax # Return the empty fig/ax

        # Normalize sizes for plotting layout (0-100 range often used)
        try:
            # Ensure sizes are numeric and non-negative before normalizing
            numeric_sizes = pd.to_numeric(sizes, errors='coerce')
            numeric_sizes = numeric_sizes[pd.notna(numeric_sizes) & (numeric_sizes >= 0)]
            if len(numeric_sizes) == 0:
                 raise ValueError("No valid positive sizes for normalization.")
            norm_sizes = squarify.normalize_sizes(numeric_sizes, dx=100, dy=100)
        except ValueError as e:
            logger.error(f"Error normalizing sizes for treemap '{title}': {e}. Original Sizes: {sizes}")
            plt.close(fig)
            return fig, ax

        # Compute rectangle coordinates using squarify
        # Width (dx) and height (dy) of the plotting area
        plot_width = 100
        plot_height = 100
        try:
            # Use only the valid normalized sizes
            rects_coords = squarify.squarify(norm_sizes, 0, 0, plot_width, plot_height)
        except ValueError as e:
            logger.error(f"Error computing squarify layout for treemap '{title}': {e}. Normalized Sizes: {norm_sizes}")
            plt.close(fig)
            return fig, ax

        # Use palette colors in sequence
        colors = [COLOR_PALETTES["main"][i % len(COLOR_PALETTES["main"])]
                 for i in range(len(norm_sizes))] # Match length of norm_sizes

        # Create the plot
        ax.set_xlim(0, plot_width)
        ax.set_ylim(0, plot_height)

        # Draw rectangles and add labels
        # Iterate based on norm_sizes length which matches rects_coords and colors
        valid_indices = data[pd.to_numeric(data[value_col], errors='coerce').notna() & (pd.to_numeric(data[value_col], errors='coerce') >= 0)].index
        if len(valid_indices) != len(rects_coords):
             logger.warning(f"Mismatch between valid data rows ({len(valid_indices)}) and squarify rectangles ({len(rects_coords)}) for treemap '{title}'. Labels might be incorrect.")
             # Attempt to proceed, but labels might be off

        for i, rect_info in enumerate(rects_coords):
            if i >= len(valid_indices):
                 logger.warning(f"Skipping rectangle {i+1} in treemap '{title}' due to index mismatch.")
                 continue # Avoid index error if mismatch occurs
                 
            data_idx = valid_indices[i]
            x, y, dx, dy = rect_info['x'], rect_info['y'], rect_info['dx'], rect_info['dy']
            color = colors[i]

            # Draw rectangle
            ax.add_patch(plt.Rectangle((x, y), dx, dy, facecolor=color, edgecolor='white', linewidth=1))

            # Prepare label text using original data index
            label_text = self._wrap_text(str(data.loc[data_idx, label_col]), wrap_width) # Ensure label is string
            detail_val = data.loc[data_idx, detail_col] if detail_col and detail_col in data.columns else data.loc[data_idx, value_col]
            # Format detail as percentage if name suggests it, otherwise float
            detail_fmt = "({:.1f}%)" if (detail_col and '%' in detail_col) else "({:.1f})"
            try:
                 detail_formatted = detail_fmt.format(detail_val)
            except (ValueError, TypeError):
                 detail_formatted = f"({detail_val})" # Fallback format

            full_label = f"{label_text}\n{detail_formatted}"

            # Determine text color based on background lightness
            try:
                r, g, b, _ = to_rgba(color)
                luminance = 0.299*r + 0.587*g + 0.114*b
                text_color = 'white' if luminance < 0.5 else 'black'
            except Exception: # Catch potential errors in color conversion
                text_color = 'black' # Default text color

            # Adjust font size based on rectangle size
            fontsize = max(6, min(10, int(np.sqrt(dx * dy) / 1.5)))

            # Add text
            ax.text(x + dx / 2, y + dy / 2, full_label,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=fontsize, color=text_color, wrap=True)

        # Remove axes and frame
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        # Apply title and caption (using common formatting principles)
        if title:
            ax.set_title(title, fontsize=18, fontweight='bold', pad=20, loc='left')
        if caption:
            fig.text(0.05, 0.01, caption, fontsize=9, color='#555555', ha='left')

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent caption/title overlap

        return fig, ax 

    def create_volume_and_provider_count_plot(
        self, 
        volume_data: pd.DataFrame, 
        count_data: pd.DataFrame, 
        title: str, 
        volume_title: str,
        count_title: str, 
        # Add parameters for column names
        year_col: str, 
        vol_provider_col: str,
        vol_subcontractor_col: str,
        count_provider_col: str,
        count_subcontractor_col: str,
        caption: Optional[str] = None,
        figsize=(16, 6) # Wider figure to accommodate two plots
    ):
        """
        Creates a figure with two subplots:
        - Left: Stacked bar chart of student volumes (Provider vs Subcontractor).
        - Right: Grouped bar chart of unique provider and subcontractor counts.

        Args:
            volume_data: DataFrame with yearly volume data.
            count_data: DataFrame with yearly unique counts.
            title: Overall title for the figure.
            volume_title: Title for the left (volume) subplot.
            count_title: Title for the right (count) subplot.
            # Add descriptions for column name parameters
            year_col: Name of the column containing the year.
            vol_provider_col: Name of the column for volume as provider.
            vol_subcontractor_col: Name of the column for volume as subcontractor.
            count_provider_col: Name of the column for unique provider count.
            count_subcontractor_col: Name of the column for unique subcontractor count.
            caption: Optional caption for the figure.
            figsize: Figure size tuple.

        Returns:
            matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]: The figure and the two axes objects.
        """
        fig, (ax_vol, ax_count) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]}) # Give more space to volume plot
        
        # --- Left Plot: Volume Stacked Bar --- 
        # Use passed-in column names
        # vol_year_col = 'Year' # Removed
        # vol_provider_col = 'Provider Amount' # Removed
        # vol_subcontractor_col = 'Subcontractor Amount' # Removed
        
        # Use config colors if possible, otherwise defaults
        provider_color = COLOR_PALETTES.get('roles', {}).get('järjestäjänä', '#7dc35a') # Green
        subcontractor_color = COLOR_PALETTES.get('roles', {}).get('hankintana', '#e86b3d') # Orange/Configured blue?
                                                                                           # Using orange to match example image
        
        if not volume_data.empty and all(c in volume_data.columns for c in [year_col, vol_provider_col, vol_subcontractor_col]):
            volume_data = volume_data.sort_values(year_col)
            years_str = volume_data[year_col].astype(str)
            
            # Plot bars using passed-in column names
            ax_vol.bar(years_str, volume_data[vol_provider_col], label='järjestäjänä', color=provider_color)
            ax_vol.bar(years_str, volume_data[vol_subcontractor_col], bottom=volume_data[vol_provider_col], label='hankintana', color=subcontractor_color)
            
            ax_vol.set_ylabel("Netto-opiskelijamäärä")
            ax_vol.legend(loc='center left')
            ax_vol.set_title(volume_title, fontsize=14, loc='left', pad=15) # Subplot title
        else:
            logger.warning("Volume data is empty or missing required columns for the left subplot.")
            ax_vol.set_title("(Volume Data Missing)", fontsize=14, loc='left', pad=15)

        # Apply common formatting (partial)
        ax_vol.spines['top'].set_visible(False)
        ax_vol.spines['right'].set_visible(False)
        ax_vol.spines['bottom'].set_color('#cccccc')
        ax_vol.spines['left'].set_color('#cccccc')
        ax_vol.grid(axis='y', linestyle='-', alpha=0.5)
        ax_vol.tick_params(axis='x', length=0)
        ax_vol.tick_params(axis='y', colors='#555555')
        ax_vol.set_ylim(bottom=0)
        ax_vol.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Ensure integer years if axis allows

        # --- Right Plot: Provider Count Grouped Bar ---
        # Use passed-in column names
        # count_year_col = 'Year' # Removed
        # count_provider_col = 'Unique_Providers_Count' # Removed
        # count_subcontractor_col = 'Unique_Subcontractors_Count' # Removed
        
        # Use the same colors as volume plot for consistency
        provider_count_color = provider_color
        subcontractor_count_color = subcontractor_color

        if not count_data.empty and all(c in count_data.columns for c in [year_col, count_provider_col, count_subcontractor_col]):
            count_data = count_data.sort_values(year_col)
            years_str = count_data[year_col].astype(str)
            x = np.arange(len(years_str))  # the label locations
            width = 0.35  # the width of the bars

            # Use passed-in column names
            rects1 = ax_count.bar(x - width/2, count_data[count_provider_col], width, label='järjestäjiä', color=provider_count_color)
            rects2 = ax_count.bar(x + width/2, count_data[count_subcontractor_col], width, label='hankintoja', color=subcontractor_count_color)

            ax_count.set_ylabel("Kouluttajien määrä")
            ax_count.set_xticks(x)
            ax_count.set_xticklabels(years_str)
            ax_count.legend(loc='upper right')
            ax_count.set_title(count_title, fontsize=14, loc='left', pad=15) # Subplot title
        else:
            logger.warning("Count data is empty or missing required columns for the right subplot.")
            ax_count.set_title("(Count Data Missing)", fontsize=14, loc='left', pad=15)

        # Apply common formatting (partial)
        ax_count.spines['top'].set_visible(False)
        ax_count.spines['right'].set_visible(False)
        ax_count.spines['bottom'].set_color('#cccccc')
        ax_count.spines['left'].set_color('#cccccc')
        ax_count.grid(axis='y', linestyle='-', alpha=0.5)
        ax_count.tick_params(axis='x', length=0)
        ax_count.tick_params(axis='y', colors='#555555')
        ax_count.set_ylim(bottom=0)
        # ax_count.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Already handled by set_xticks
        
        # --- Overall Figure Formatting ---
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98, ha='center') # Centered super title
        if caption:
            # Adjust caption position for two subplots
            fig.text(0.5, 0.01, caption, fontsize=9, color='#555555', ha='center', va='bottom') 
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

        return fig, (ax_vol, ax_count) 

    def create_bcg_matrix(
        self,
        data: pd.DataFrame,
        growth_col: str,
        share_col: str,
        size_col: str,
        label_col: str,
        title: str,
        caption: Optional[str] = None,
        avg_growth: Optional[float] = None,
        share_threshold: float = 1.0,
        max_bubble_size: int = 2000,
        figsize=(12, 12) # BCG often square
    ):
        """
        Creates a BCG Growth-Share Matrix plot (Bubble Chart).

        Args:
            data: DataFrame containing BCG data (qualification, growth, relative share, volume).
            growth_col: Column name for Market Growth Rate (%).
            share_col: Column name for Relative Market Share.
            size_col: Column name for Institution Volume (determines bubble size).
            label_col: Column name for bubble labels (qualification name).
            title: Title for the chart.
            caption: Optional caption for the figure.
            avg_growth: Market growth rate threshold (Y-axis divider). If None, calculates mean from data.
            share_threshold: Relative market share threshold (X-axis divider). Defaults to 1.0.
            max_bubble_size: Maximum size for bubbles in the scatter plot.
            figsize: Figure size tuple.

        Returns:
            matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axes object.
        """
        fig, ax = plt.subplots(figsize=figsize)

        if data.empty:
            logger.warning("BCG data is empty. Cannot create plot.")
            ax.text(0.5, 0.5, "No data available for BCG Matrix", ha='center', va='center')
            return fig, ax

        # Prepare data
        df = data.copy()
        df[growth_col] = pd.to_numeric(df[growth_col], errors='coerce')
        df[share_col] = pd.to_numeric(df[share_col], errors='coerce')
        df[size_col] = pd.to_numeric(df[size_col], errors='coerce')

        # Handle potential infinities in relative share (e.g., from 100% market share)
        # Replace inf with a value slightly larger than the max finite share or a fixed large value
        max_finite_share = df.loc[np.isfinite(df[share_col]), share_col].max()
        inf_replacement = max(max_finite_share * 1.1, share_threshold * 2) if pd.notna(max_finite_share) else share_threshold * 2
        df[share_col] = df[share_col].replace([np.inf], inf_replacement)
        df = df.dropna(subset=[growth_col, share_col, size_col]) # Drop rows with missing essential data
        
        if df.empty:
            logger.warning("BCG data is empty after cleaning. Cannot create plot.")
            ax.text(0.5, 0.5, "No valid data for BCG Matrix after cleaning", ha='center', va='center')
            return fig, ax

        # Determine bubble sizes (scale volume to reasonable plot size)
        min_vol = df[size_col].min()
        max_vol = df[size_col].max()
        if max_vol > min_vol:
             # Linear scaling
             sizes = (df[size_col] - min_vol) / (max_vol - min_vol) * (max_bubble_size - 50) + 50 # Scale between 50 and max_bubble_size
             # Alternative: Area scaling (bubbles proportional to volume area)
             # sizes = df[size_col] / max_vol * max_bubble_size 
        else:
             sizes = [max_bubble_size / 2] * len(df) # Use medium size if all volumes are the same
        sizes = sizes.fillna(50) # Default size for any NaNs left

        # Determine quadrant dividers
        if avg_growth is None:
            avg_growth = df[growth_col].mean()
            logger.info(f"Using calculated average market growth for BCG divider: {avg_growth:.2f}%")
        else:
             logger.info(f"Using provided average market growth for BCG divider: {avg_growth:.2f}%")

        # Create scatter plot (bubble chart)
        scatter = ax.scatter(df[share_col], df[growth_col], s=sizes,
                             c=COLOR_PALETTES['main'][1], # Use a consistent color (e.g., blue)
                             alpha=0.6, edgecolors='w', linewidth=0.5)

        # Add quadrant lines
        ax.axhline(avg_growth, color='grey', linestyle='--', linewidth=1)
        ax.axvline(share_threshold, color='grey', linestyle='--', linewidth=1)

        # Add quadrant labels (Stars, Question Marks, Cash Cows, Dogs)
        plot_xlim = ax.get_xlim()
        plot_ylim = ax.get_ylim()
        ax.text(plot_xlim[1] * 0.95, plot_ylim[1] * 0.95, 'Stars', ha='right', va='top', fontsize=12, color='grey', alpha=0.7)
        ax.text(plot_xlim[0] * 0.95 if plot_xlim[0] != 0 else plot_xlim[1]*0.05, plot_ylim[1] * 0.95, 'Question Marks', ha='left', va='top', fontsize=12, color='grey', alpha=0.7)
        ax.text(plot_xlim[1] * 0.95, plot_ylim[0] * 0.95 if plot_ylim[0] != 0 else plot_ylim[1]*0.05, 'Cash Cows', ha='right', va='bottom', fontsize=12, color='grey', alpha=0.7)
        ax.text(plot_xlim[0] * 0.95 if plot_xlim[0] != 0 else plot_xlim[1]*0.05, plot_ylim[0] * 0.95 if plot_ylim[0] != 0 else plot_ylim[1]*0.05, 'Dogs', ha='left', va='bottom', fontsize=12, color='grey', alpha=0.7)

        # Add labels to bubbles
        for i, row in df.iterrows():
            # Use a small offset for the label
            x_pos = row[share_col]
            y_pos = row[growth_col]
            label = str(row[label_col]) # Ensure label is string
            # Simple offset - might need adjustment based on bubble size or density
            ax.text(x_pos + 0.02, y_pos + 0.02, label, fontsize=7, ha='left', va='bottom', alpha=0.8)

        # Apply common formatting (partially)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, loc='left')
        ax.set_xlabel("Relative Market Share", fontsize=12)
        ax.set_ylabel("Market Growth Rate (%)", fontsize=12)
        
        # Specific formatting for BCG
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#cccccc')
        ax.spines['left'].set_color('#cccccc')
        ax.grid(True, linestyle='-', alpha=0.3)
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', colors='#555555')

        # Invert X-axis (common in BCG)
        ax.invert_xaxis()
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Add caption
        if caption:
            fig.text(0.05, 0.01, caption, fontsize=9, color='#555555', ha='left')
            
        return fig, ax 