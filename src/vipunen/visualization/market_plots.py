"""
Market visualization utilities for education market analysis.
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

def plot_market_share_heatmap(market_share_df: pd.DataFrame, 
                            output_path: Optional[str] = None,
                            institution_short_name: str = "Institution", 
                            output_dir: Optional[Union[str, Path]] = None,
                            reference_year: Optional[int] = None) -> Optional[str]:
    """
    Create a heatmap showing market shares across qualifications.
    
    Args:
        market_share_df: DataFrame with market share data
        output_path: Path to save the plot
        institution_short_name: Short name for the institution
        output_dir: Directory to save the plot
        reference_year: Year to use for the heatmap (latest if None)
        
    Returns:
        Optional[str]: Path where the plot was saved, or None if no plot was created
    """
    # Check if we have enough data
    required_columns = ['tilastovuosi', 'tutkinto', 'provider', 'market_share']
    if not all(col in market_share_df.columns for col in required_columns) or market_share_df.empty:
        logger.warning(f"Missing required columns for market_share_heatmap: {required_columns}")
        return None
    
    # Copy the DataFrame to avoid modifying the original
    df = market_share_df.copy()
    
    # Filter for target year
    if reference_year is None:
        reference_year = df['tilastovuosi'].max()
    
    year_data = df[df['tilastovuosi'] == reference_year]
    
    if year_data.empty:
        logger.warning(f"No market share data available for year {reference_year}")
        return None
    
    # Get the top qualifications by volume
    top_quals = year_data.groupby('tutkinto')['market_share'].sum().nlargest(10).index.tolist()
    
    # Get top providers across these qualifications
    top_providers = year_data[year_data['tutkinto'].isin(top_quals)].groupby('provider')['market_share'].sum().nlargest(10).index.tolist()
    
    # Filter data to include only top qualifications and providers
    heatmap_data = year_data[
        (year_data['tutkinto'].isin(top_quals)) & 
        (year_data['provider'].isin(top_providers))
    ]
    
    # Create a pivot table for the heatmap
    pivot_data = heatmap_data.pivot_table(
        index='tutkinto',
        columns='provider',
        values='market_share',
        fill_value=0
    )
    
    # Sort by total market share
    pivot_data = pivot_data.loc[pivot_data.sum(axis=1).sort_values(ascending=False).index]
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Generate the heatmap
    sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=.5, ax=ax, cbar_kws={'label': 'Market Share (%)'})
    
    # Configure the plot
    ax.set_title(f'Market Share Heatmap for Top Qualifications in {reference_year}', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        output_path = f"{institution_short_name.lower()}_market_share_heatmap"
    
    return save_plot(fig, plot_name=output_path, output_dir=output_dir)

def plot_qualification_market_shares(volumes_df: pd.DataFrame, 
                                   institution_names: List[str],
                                   output_path: Optional[str] = None, 
                                   top_n: int = 10,
                                   institution_short_name: str = "Institution", 
                                   output_dir: Optional[Union[str, Path]] = None,
                                   reference_year: Optional[int] = None) -> Optional[str]:
    """
    Create a stacked bar chart showing market shares for top qualifications.
    
    Args:
        volumes_df: DataFrame with volume data
        institution_names: List of names for the institution
        output_path: Path to save the plot
        top_n: Number of top qualifications to include
        institution_short_name: Short name for the institution
        output_dir: Directory to save the plot
        reference_year: Year to use for the analysis (latest if None)
        
    Returns:
        Optional[str]: Path where the plot was saved, or None if no plot was created
    """
    # Check if we have enough data
    required_columns = ['tilastovuosi', 'tutkinto', 'koulutuksenJarjestaja', 'nettoopiskelijamaaraLkm']
    if not all(col in volumes_df.columns for col in required_columns) or volumes_df.empty:
        logger.warning(f"Missing required columns for qualification_market_shares: {required_columns}")
        return None
    
    # Copy the DataFrame to avoid modifying the original
    df = volumes_df.copy()
    
    # Filter for target year
    if reference_year is None:
        reference_year = df['tilastovuosi'].max()
    
    year_data = df[df['tilastovuosi'] == reference_year]
    
    if year_data.empty:
        logger.warning(f"No volume data available for year {reference_year}")
        return None
    
    # Calculate market shares
    # 1. Get total volume per qualification
    qual_totals = year_data.groupby('tutkinto')['nettoopiskelijamaaraLkm'].sum().reset_index()
    qual_totals.rename(columns={'nettoopiskelijamaaraLkm': 'total_volume'}, inplace=True)
    
    # 2. Get volumes per qualification and provider
    provider_volumes = year_data.groupby(['tutkinto', 'koulutuksenJarjestaja'])['nettoopiskelijamaaraLkm'].sum().reset_index()
    
    # 3. Merge to calculate market shares
    market_shares = pd.merge(provider_volumes, qual_totals, on='tutkinto')
    market_shares['market_share'] = market_shares['nettoopiskelijamaaraLkm'] / market_shares['total_volume'] * 100
    
    # 4. Flag rows for the target institution
    market_shares['is_target'] = market_shares['koulutuksenJarjestaja'].isin(institution_names)
    
    # 5. Get qualifications where the target institution has a presence
    target_quals = market_shares[market_shares['is_target']]['tutkinto'].unique()
    
    if len(target_quals) == 0:
        logger.warning(f"No qualifications found for institution {institution_names}")
        return None
    
    # 6. Filter to only include these qualifications
    filtered_shares = market_shares[market_shares['tutkinto'].isin(target_quals)]
    
    # 7. Calculate total volume per qualification for the target institution
    target_volumes = filtered_shares[filtered_shares['is_target']].groupby('tutkinto')['nettoopiskelijamaaraLkm'].sum().reset_index()
    target_volumes.rename(columns={'nettoopiskelijamaaraLkm': 'target_volume'}, inplace=True)
    
    # 8. Merge to get target volumes
    qual_summary = pd.merge(qual_totals[qual_totals['tutkinto'].isin(target_quals)], 
                           target_volumes, on='tutkinto', how='left')
    qual_summary['target_volume'] = qual_summary['target_volume'].fillna(0)
    qual_summary['target_share'] = qual_summary['target_volume'] / qual_summary['total_volume'] * 100
    
    # 9. Get top N qualifications by target institution's volume
    top_quals = qual_summary.nlargest(top_n, 'target_volume')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by target share
    top_quals = top_quals.sort_values('target_share')
    
    # Set up the bar positions
    y_pos = np.arange(len(top_quals))
    
    # Create the stacked bar chart
    # First, the target institution's share
    target_bars = ax.barh(y_pos, top_quals['target_share'], height=0.7, color='#3498db', label=institution_short_name)
    
    # Then, the rest of the market
    rest_bars = ax.barh(y_pos, 100 - top_quals['target_share'], height=0.7, left=top_quals['target_share'], 
                      color='#e74c3c', label='Other Providers')
    
    # Add qualification names as y-tick labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_quals['tutkinto'])
    
    # Add market share percentages inside the bars
    for i, (_, row) in enumerate(top_quals.iterrows()):
        # Only show text for bars wide enough
        if row['target_share'] > 5:
            ax.text(row['target_share'] / 2, i, f"{row['target_share']:.1f}%", 
                  ha='center', va='center', color='white', fontweight='bold')
        
        # Add total volume at the end of each bar
        ax.text(101, i, f"Total: {int(row['total_volume'])}", 
              ha='left', va='center', fontsize=9)
    
    # Configure the plot
    ax.set_title(f'{institution_short_name} - Market Share in Top {top_n} Qualifications ({reference_year})', fontsize=14)
    ax.set_xlabel('Market Share (%)', fontsize=12)
    ax.set_xlim(0, 120)  # Leave space for annotations
    ax.legend(loc='lower right')
    
    # Add gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        output_path = f"{institution_short_name.lower()}_qualification_market_shares"
    
    return save_plot(fig, plot_name=output_path, output_dir=output_dir) 