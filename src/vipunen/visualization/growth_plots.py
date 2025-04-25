"""
Growth visualization utilities for education market analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the save_plot function from volume_plots to avoid duplication
from .volume_plots import save_plot

def plot_qualification_growth(growth_df: pd.DataFrame, 
                          output_path: Optional[str] = None,
                          year: Optional[int] = None, 
                          metric: str = 'nettoopiskelijamaaraLkm_growth',
                          top_n: int = 5, 
                          bottom_n: int = 5,
                          institution_short_name: str = "Institution", 
                          output_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
    """
    Create a bar chart showing top growing and declining qualifications.
    
    Args:
        growth_df: DataFrame with growth data
        output_path: Path to save the plot
        year: Year to analyze
        metric: Column name for the growth metric
        top_n: Number of top growing qualifications to show
        bottom_n: Number of top declining qualifications to show
        institution_short_name: Short name for the institution
        output_dir: Directory to save the plot
        
    Returns:
        Optional[str]: Path where the plot was saved, or None if no plot was created
    """
    # Check if we have the required columns
    required_columns = ['tutkinto', metric]
    if not all(col in growth_df.columns for col in required_columns) or growth_df.empty:
        logger.warning(f"Missing required columns for plot_qualification_growth: {required_columns}")
        return None
    
    # Filter for specific year if provided
    if year is not None and 'tilastovuosi' in growth_df.columns:
        growth_df = growth_df[growth_df['tilastovuosi'] == year]
    
    if growth_df.empty:
        logger.warning(f"No growth data available for year {year}")
        return None
    
    # Handle infinite and NaN values
    growth_df[metric] = growth_df[metric].replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with NaN growth
    filtered_df = growth_df.dropna(subset=[metric])
    
    if filtered_df.empty:
        logger.warning(f"No valid growth data after filtering")
        return None
    
    # Get top growing and declining qualifications
    top_growing = filtered_df.nlargest(top_n, metric)
    top_declining = filtered_df.nsmallest(bottom_n, metric)
    
    # Combine them
    combined = pd.concat([top_growing, top_declining])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the bar chart
    sns.barplot(y='tutkinto', x=metric, data=combined, palette='RdYlGn_r', ax=ax)
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add growth values as text
    for i, (_, row) in enumerate(combined.iterrows()):
        growth = row[metric]
        ax.text(growth + (5 if growth >= 0 else -5), 
              i, 
              f"{growth:.1f}%", 
              ha='left' if growth >= 0 else 'right', 
              va='center', 
              fontsize=9)
    
    # Configure the plot
    year_text = f" in {year}" if year is not None else ""
    ax.set_title(f'{institution_short_name} - Qualification Growth{year_text}', fontsize=14)
    ax.set_xlabel('Growth (%)', fontsize=12)
    ax.set_ylabel('Qualification', fontsize=12)
    
    # Add gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        output_path = f"{institution_short_name.lower()}_qualification_growth"
    
    return save_plot(fig, plot_name=output_path, output_dir=output_dir)

def plot_qualification_time_series(volumes_by_qual: pd.DataFrame, 
                               output_path: Optional[str] = None,
                               qualifications: Optional[List[str]] = None, 
                               top_n: int = 5,
                               metric: str = 'kouluttaja yhteensä',
                               institution_short_name: str = "Institution", 
                               output_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
    """
    Create a line chart showing volume trends over time for top qualifications.
    
    Args:
        volumes_by_qual: DataFrame with volume data by qualification
        output_path: Path to save the plot
        qualifications: List of qualifications to include (if None, will use top N)
        top_n: Number of top qualifications to include (if qualifications is None)
        metric: Column name for the volume metric
        institution_short_name: Short name for the institution
        output_dir: Directory to save the plot
        
    Returns:
        Optional[str]: Path where the plot was saved, or None if no plot was created
    """
    # Check if volumes_by_qual has the required column
    if 'tutkinto' not in volumes_by_qual.columns:
        logger.warning("Required column 'tutkinto' not found")
        return None
    
    # Determine years and metric columns
    if 'tilastovuosi' in volumes_by_qual.columns and metric in volumes_by_qual.columns:
        # Data is already in the right format with years in rows
        time_series_data = volumes_by_qual.copy()
        years = sorted(time_series_data['tilastovuosi'].unique())
    else:
        # Try to extract years from column names (e.g. "2018_järjestäjänä")
        year_cols = []
        for col in volumes_by_qual.columns:
            parts = col.split('_')
            if len(parts) == 2 and parts[0].isdigit():
                year_cols.append(col)
        
        if not year_cols:
            logger.warning("No year columns found in the data")
            return None
        
        # Look for alternative volume metric
        if metric not in volumes_by_qual.columns:
            # Use the sum of provider and subcontractor volumes
            logger.warning(f"Missing required column '{metric}'")
            # Try to use columns like "2018_järjestäjänä" and "2018_hankintana"
            years = sorted(list(set([int(col.split('_')[0]) for col in year_cols])))
            
            # Create a new DataFrame for time series
            time_series_data = []
            
            for _, row in volumes_by_qual.iterrows():
                qual = row['tutkinto']
                
                for year in years:
                    prov_col = f"{year}_järjestäjänä"
                    subcont_col = f"{year}_hankintana"
                    
                    if prov_col in volumes_by_qual.columns and subcont_col in volumes_by_qual.columns:
                        total_vol = row.get(prov_col, 0) + row.get(subcont_col, 0)
                        
                        if total_vol > 0:
                            time_series_data.append({
                                'tutkinto': qual,
                                'tilastovuosi': year,
                                'volume': total_vol
                            })
            
            time_series_data = pd.DataFrame(time_series_data)
            metric = 'volume'  # Use the new volume column
        else:
            logger.error(f"Cannot create time series - insufficient data")
            return None
    
    # If the data is empty after preparation, return None
    if len(time_series_data) == 0:
        logger.warning("No data available for time series plot")
        return None
    
    # Determine qualifications to include
    if qualifications is None:
        # Get the latest year in the data
        if 'tilastovuosi' in time_series_data.columns:
            latest_year = time_series_data['tilastovuosi'].max()
            latest_data = time_series_data[time_series_data['tilastovuosi'] == latest_year]
            
            # Get the top N qualifications by volume
            top_quals = latest_data.nlargest(top_n, metric)['tutkinto'].tolist()
        else:
            # If we can't determine the latest year, just use the total volume across all years
            qualification_totals = time_series_data.groupby('tutkinto')[metric].sum()
            top_quals = qualification_totals.nlargest(top_n).index.tolist()
    else:
        top_quals = qualifications
    
    # Filter data to include only the selected qualifications
    filtered_data = time_series_data[time_series_data['tutkinto'].isin(top_quals)]
    
    if filtered_data.empty:
        logger.warning(f"No data available for selected qualifications")
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot a line for each qualification
    for qual in top_quals:
        qual_data = filtered_data[filtered_data['tutkinto'] == qual]
        
        if not qual_data.empty:
            qual_data = qual_data.sort_values('tilastovuosi')
            ax.plot(qual_data['tilastovuosi'], qual_data[metric], marker='o', linewidth=2, label=qual)
    
    # Configure the plot
    ax.set_title(f'{institution_short_name} - Volume Trends for Top Qualifications', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Students', fontsize=12)
    
    # Format y-axis to use thousands separator
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        output_path = f"{institution_short_name.lower()}_qualification_time_series"
    
    return save_plot(fig, plot_name=output_path, output_dir=output_dir) 