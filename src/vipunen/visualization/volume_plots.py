"""
Volume visualization utilities for education market analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from pathlib import Path
from typing import Optional, Union, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_plot(fig: plt.Figure, file_path: Optional[Union[str, Path]] = None, 
             plot_name: Optional[str] = None, output_dir: Optional[Union[str, Path]] = None) -> str:
    """
    Save a matplotlib figure to a file.
    
    Args:
        fig: Figure to save
        file_path: Full path to save the file
        plot_name: Name of the plot (used if file_path is not provided)
        output_dir: Directory to save the file (used if file_path is not provided)
        
    Returns:
        str: Path where the figure was saved
    """
    if file_path is None:
        # Create a file path based on plot_name and output_dir
        if output_dir is None:
            output_dir = "plots"
        
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the file path
        if plot_name is None:
            plot_name = "plot"
        
        file_path = os.path.join(output_dir, f"{plot_name}.png")
    
    # Save the figure
    fig.savefig(file_path, bbox_inches='tight', dpi=300)
    logger.info(f"Figure saved to {file_path}")
    
    # Close the figure to free up memory
    plt.close(fig)
    
    return file_path

def plot_total_volumes(volumes_df: pd.DataFrame, output_path: Optional[str] = None, 
                     institution_short_name: str = "Institution", 
                     output_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
    """
    Create a bar chart showing total volumes over time.
    
    Args:
        volumes_df: DataFrame with volume data by year
        output_path: Path to save the plot
        institution_short_name: Short name for the institution (used in title and file name)
        output_dir: Directory to save the plot
        
    Returns:
        Optional[str]: Path where the plot was saved, or None if no plot was created
    """
    # Check if we have the required columns
    required_columns = ['tilastovuosi', 'järjestäjänä', 'hankintana']
    if not all(col in volumes_df.columns for col in required_columns) or volumes_df.empty:
        logger.warning(f"Missing required columns for plot_total_volumes: {required_columns}")
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract years and ensure they're treated as categorical
    years = volumes_df['tilastovuosi'].astype(str)
    
    # Create the stacked bar chart
    ax.bar(years, volumes_df['järjestäjänä'], label='As Provider', color='#3498db')
    ax.bar(years, volumes_df['hankintana'], bottom=volumes_df['järjestäjänä'], 
          label='As Subcontractor', color='#e74c3c')
    
    # Add the total volume as text on top of each bar
    for i, (_, row) in enumerate(volumes_df.iterrows()):
        total = row['järjestäjänä'] + row['hankintana']
        ax.text(i, total + (total * 0.02), f'{int(total)}', ha='center', va='bottom', fontsize=9)
    
    # Add provider role percentages in the middle of the provider section
    for i, (_, row) in enumerate(volumes_df.iterrows()):
        provider_vol = row['järjestäjänä']
        if provider_vol > 0:
            percentage = row['järjestäjä_osuus (%)']
            y_pos = provider_vol / 2  # Middle of the provider section
            ax.text(i, y_pos, f'{percentage:.1f}%', ha='center', va='center', 
                  fontsize=9, fontweight='bold', color='white')
    
    # Configure the plot
    ax.set_title(f'{institution_short_name} - Student Volume by Year and Role', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Students', fontsize=12)
    ax.legend(loc='upper right')
    
    # Format y-axis to use thousands separator
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        output_path = f"{institution_short_name.lower()}_total_volumes"
    
    return save_plot(fig, plot_name=output_path, output_dir=output_dir)

def plot_top_qualifications(volumes_by_qual: pd.DataFrame, output_path: Optional[str] = None, 
                         year: Optional[int] = None, top_n: int = 5, 
                         institution_short_name: str = "Institution", 
                         output_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
    """
    Create a bar chart showing top qualifications by volume.
    
    Args:
        volumes_by_qual: DataFrame with volume data by qualification
        output_path: Path to save the plot
        year: Year to analyze (will use the most recent if None)
        top_n: Number of top qualifications to show
        institution_short_name: Short name for the institution
        output_dir: Directory to save the plot
        
    Returns:
        Optional[str]: Path where the plot was saved, or None if no plot was created
    """
    # Determine the columns needed based on what's available
    volume_col = 'kouluttaja yhteensä'
    share_col = 'markkinaosuus (%)'
    
    # If the standard columns don't exist, try to find alternatives
    if volume_col not in volumes_by_qual.columns:
        # Look for alternative volume columns like year_role (e.g. 2018_hankintana)
        year_cols = [col for col in volumes_by_qual.columns if ('_järjestäjänä' in col or '_hankintana' in col)]
        if year_cols:
            volume_col = year_cols[0]  # Use the first one as a fallback
            logger.warning(f"Using '{volume_col}' for volume data instead of 'kouluttaja yhteensä'")
        else:
            logger.error("No suitable volume column found for top qualifications plot")
            return None
    
    if 'tutkinto' not in volumes_by_qual.columns:
        logger.error("Qualification column 'tutkinto' not found")
        return None
    
    # Copy the DataFrame to avoid modifying the original
    df = volumes_by_qual.copy()
    
    # If year is provided and year-specific columns exist, use them
    if year is not None and any(str(year) in col for col in df.columns):
        prov_col = f"{year}_järjestäjänä"
        sub_col = f"{year}_hankintana"
        
        if prov_col in df.columns and sub_col in df.columns:
            # Create a year-specific total volume column
            df[volume_col] = df[prov_col] + df[sub_col]
        
    # Calculate dummy market share if needed
    if share_col not in df.columns:
        logger.warning("Calculating dummy market share percentage from volume data")
        total_volume = df[volume_col].sum()
        df[share_col] = (df[volume_col] / total_volume * 100) if total_volume > 0 else 0
    
    # Replace NaN values
    df[volume_col] = df[volume_col].fillna(0)
    df[share_col] = df[share_col].fillna(0)
    
    # Sort by volume and get top N
    top_quals = df.nlargest(top_n, volume_col)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create the bar chart
    sns.barplot(y='tutkinto', x=volume_col, data=top_quals, palette='viridis', ax=ax)
    
    # Add volume and market share as text
    for i, (_, row) in enumerate(top_quals.iterrows()):
        volume = row[volume_col]
        share = row[share_col] if share_col in row else 0
        ax.text(volume + (volume * 0.02), i, 
               f'{int(volume)} ({share:.1f}%)', 
               va='center', fontsize=9)
    
    # Configure the plot
    title_year = f" in {year}" if year is not None else ""
    ax.set_title(f'{institution_short_name} - Top {top_n} Qualifications by Volume{title_year}', fontsize=14)
    ax.set_xlabel('Number of Students', fontsize=12)
    ax.set_ylabel('Qualification', fontsize=12)
    
    # Format x-axis to use thousands separator
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        output_path = f"{institution_short_name.lower()}_top_qualifications"
    
    return save_plot(fig, plot_name=output_path, output_dir=output_dir) 