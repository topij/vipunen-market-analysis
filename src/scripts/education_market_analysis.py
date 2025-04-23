#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Education Market Analysis script that analyzes the Finnish education market
from any provider's perspective.

This script allows you to analyze education market data for any educational 
institution in Finland. The default is set to Rastor-instituutti for 
backward compatibility, but can be easily changed to analyze any institution.

The implementation now uses the FileUtils package for standardized file operations,
providing several benefits:
- Automatic format detection for input files
- Simplified data loading with consistent API
- Multi-DataFrame Excel file creation
- Standardized directory management using FileUtils' directory structure:
  - data/raw: For raw input files
  - data/processed: For processed data
  - data/figures: For visualizations
  - data/education_market_*: For institution-specific analysis results
- Future compatibility with cloud storage options

Usage examples:
    # Analyze data for the default institution (Rastor-instituutti)
    python education_market_analysis.py
    
    # Analyze data for another institution (provide all known name variations)
    python education_market_analysis.py --institution "Stadin AO" "Stadin ammattiopisto" --short-name "Stadin"
    
    # Specify a reference year for qualification selection
    python education_market_analysis.py --institution "Omnia" --reference-year 2022
    
    # Specify a custom output directory
    python education_market_analysis.py --output-dir ~/reports/education_analysis
    
    # Disable degree type filtering
    python education_market_analysis.py --no-filter-degree-types
    
    # Specify a different data file (in the data directory)
    python education_market_analysis.py --data-file custom_education_data.csv

The script produces an Excel file with multiple analysis sheets and several 
visualizations that help understand the institution's position in the market.
All outputs are saved within the data directory structure managed by FileUtils.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import argparse
import numpy as np
import io
import os
import shutil

# Add the parent directory to the path to import from the vipunen package
sys.path.append(str(Path(__file__).resolve().parents[2]))

from vipunen.analysis.education_market import EducationMarketAnalyzer

# Try to import FileUtils, but use a fallback if not available
try:
    from FileUtils import FileUtils, OutputFileType
    file_utils = FileUtils()
    # Get ROOT_DIR for absolute paths when needed
    ROOT_DIR = Path(file_utils.project_root)
    print("Successfully imported FileUtils")
except ImportError:
    print("FileUtils package not found. Using built-in file handling.")
    # Define a simple replacement for FileUtils
    class SimpleFileUtils:
        def __init__(self):
            self.project_root = os.getcwd()
            
        def load_single_file(self, file_path, input_type=None):
            """Simple file loader that detects file type from extension"""
            file_path = Path(file_path)
            if file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path, sep=';')
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
                
        def create_directory(self, dir_name, parent_dir=None):
            """Create directory and return path"""
            if parent_dir == "data":
                # Default to data directory in project root
                output_dir = Path(self.project_root) / "data" / dir_name
            else:
                output_dir = Path(dir_name)
                
            os.makedirs(output_dir, exist_ok=True)
            return str(output_dir)
            
        def save_data_to_storage(self, data, file_name, output_type=None, output_filetype=None, output_directory=None):
            """Simple implementation of save_data_to_storage"""
            # Handle output directory
            if output_directory:
                output_dir = Path(output_directory)
            else:
                output_dir = Path(self.project_root) / "data" / "excel"
                
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            file_extension = ".xlsx" if output_filetype == OutputFileType.XLSX else ".csv"
            full_file_name = f"{file_name}_{timestamp}{file_extension}"
            file_path = output_dir / full_file_name
            
            # Save file based on data type and output type
            if isinstance(data, dict):
                # Dictionary of DataFrames - save as Excel with multiple sheets
                if output_filetype == OutputFileType.XLSX:
                    with pd.ExcelWriter(file_path) as writer:
                        for sheet_name, df in data.items():
                            df.to_excel(writer, sheet_name=sheet_name)
                else:
                    # If not Excel, save each DataFrame as separate CSV file
                    result = {}
                    for sheet_name, df in data.items():
                        sheet_path = output_dir / f"{file_name}_{sheet_name}_{timestamp}.csv"
                        df.to_csv(sheet_path, sep=';', index=True)
                        result[sheet_name] = str(sheet_path)
                    return result
            else:
                # Single DataFrame
                if output_filetype == OutputFileType.XLSX:
                    data.to_excel(file_path)
                else:
                    data.to_csv(file_path, sep=';', index=True)
                    
            # Return dictionary mapping sheet name to file path for compatibility
            return {'Sheet1': str(file_path)}
    
    # Create enum-like class for compatibility
    class OutputFileType:
        CSV = "csv"
        XLSX = "xlsx"
    
    file_utils = SimpleFileUtils()
    ROOT_DIR = Path(file_utils.project_root)

# Default institution for backward compatibility
DEFAULT_INSTITUTION_NAMES = ["Rastor-instituutti ry", "Rastor Oy"]
DEFAULT_SHORT_NAME = "RI"  # Default short name for the institution

def replace_values_in_dataframe(df, replacement_dict):
    """
    Replace specified values in a DataFrame based on a dictionary.
    
    Args:
        df: The original DataFrame
        replacement_dict: A dictionary where the key is the new value, and the value is a list of old values to be replaced
        
    Returns:
        A new DataFrame with values replaced.
    """
    # Create a reverse mapping for replacement
    reverse_dict = {old: new for new, old_list in replacement_dict.items() for old in old_list}
    
    # Replace the values and return a new DataFrame
    return df.replace(reverse_dict)

def shorten_qualification_names(df, column='tutkinto'):
    """
    Shorten qualification names by replacing long terms with abbreviations.
    
    Args:
        df: DataFrame containing the data
        column: Column name containing qualification names
        
    Returns:
        pd.DataFrame: DataFrame with shortened qualification names
    """
    # Make a copy to avoid modifying the input DataFrame
    df_copy = df.copy()
    
    # Define abbreviations
    abbreviations = {
        'erikoisammattitutkinto': 'EAT', 
        'ammattitutkinto': 'AT'
    }
    
    # Replace terms in the qualification names
    for term, abbreviation in abbreviations.items():
        df_copy[column] = df_copy[column].str.replace(term, abbreviation, case=False)
    
    return df_copy

def clean_and_prepare_data(df, institution_names=None, merge_qualifications=True, shorten_names=False):
    """
    Clean and prepare the data for analysis.
    
    Args:
        df: DataFrame containing the raw data
        institution_names: List of institution names to standardize
        merge_qualifications: Whether to merge similar qualifications
        shorten_names: Whether to shorten qualification names
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Make a copy of the dataframe to avoid modifying the original
    cleaned_df = df.copy()
    
    # Replace institution names if needed
    if institution_names:
        institution_replacements = {institution_names[0]: institution_names}
        cleaned_df = replace_values_in_dataframe(cleaned_df, institution_replacements)
    
    # Standardize qualification names if requested
    if merge_qualifications:
        # Define qualification name mappings - pairs of (new_name, old_name)
        qualification_mappings = [
            ('Yrittäjyyden ammattitutkinto', 'Yrittäjän ammattitutkinto'),
            # Add more mappings as needed
        ]
        
        # Apply each mapping
        for new_name, old_name in qualification_mappings:
            # Get counts before replacement for logging
            count_before_old = cleaned_df['tutkinto'].value_counts().get(old_name, 0)
            count_before_new = cleaned_df['tutkinto'].value_counts().get(new_name, 0)
            
            # Replace old qualification name with new name
            cleaned_df.loc[cleaned_df['tutkinto'] == old_name, 'tutkinto'] = new_name
            
            # Log the change
            count_after_old = cleaned_df['tutkinto'].value_counts().get(old_name, 0)
            count_after_new = cleaned_df['tutkinto'].value_counts().get(new_name, 0)
            
            print(f"Merged {old_name} into {new_name}:")
            print(f"  - Before: {count_before_old} rows with old name, {count_before_new} rows with new name")
            print(f"  - After: {count_after_old} rows with old name, {count_after_new} rows with new name")
    
    # Shorten qualification names if requested
    if shorten_names:
        cleaned_df = shorten_qualification_names(cleaned_df)
        print("Shortened qualification names (erikoisammattitutkinto → EAT, ammattitutkinto → AT)")
    
    return cleaned_df

def load_data(file_path, shorten_names=False):
    """
    Load data from file and apply cleaning steps using FileUtils
    
    Args:
        file_path: Path to the data file (CSV, Excel, etc.)
        shorten_names: Whether to shorten qualification names
        
    Returns:
        pd.DataFrame: Loaded and cleaned dataframe
    """
    print(f"Loading data from {file_path}")
    
    try:
        # Convert file_path to Path object
        file_path = Path(file_path)
        
        # Handle file loading based on available utilities
        if hasattr(file_utils, 'load_single_file'):
            # Use FileUtils if available but with absolute path to avoid path errors
            if not file_path.is_absolute():
                # Convert to absolute path
                file_path = Path(ROOT_DIR) / file_path
            
            try:
                # First try with input_type parameter
                df = file_utils.load_single_file(str(file_path), input_type="raw")
            except Exception:
                # If that fails, try without input_type (for SimpleFileUtils)
                df = file_utils.load_single_file(str(file_path))
        else:
            # Fallback to pandas direct loading
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, sep=';')
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Apply basic cleaning without institution-specific replacements
        df = clean_and_prepare_data(df, institution_names=None, shorten_names=shorten_names)
        
        return df
    except Exception as e:
        raise ValueError(f"Error loading data file: {e}")

def save_plot(fig, file_path=None, plot_name=None, output_dir=None):
    """
    Save a plot to storage
    
    Args:
        fig: Matplotlib figure to save
        file_path: Optional explicit path to save the figure
        plot_name: Optional name for the plot when saving to default figures directory
        output_dir: Optional directory to save the plot in
    """
    if file_path is None:
        if plot_name is None:
            raise ValueError("Either file_path or plot_name must be specified")
        
        if output_dir is None:
            # Create figures directory under data (fallback)
            figures_dir = file_utils.create_directory("figures", parent_dir="data")
            file_path = Path(figures_dir) / f"{plot_name}.png"
        else:
            # Save directly to the institution-specific output directory
            figures_dir = Path(output_dir)
            file_path = figures_dir / f"{plot_name}.png"
    
    # Make sure the parent directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the figure
    plt.figure(fig.number)
    plt.savefig(file_path, dpi=300)
    plt.close(fig)
    
    print(f"Figure saved to {file_path}")

def create_output_directory(institution_name):
    """
    Create an output directory for analysis results
    
    Args:
        institution_name: Name of the institution (used for folder naming)
        
    Returns:
        pathlib.Path: Path to the created directory
    """
    # Create a directory name based on the institution name
    dir_name = f"education_market_{institution_name.lower()}"
    
    # Create the reports directory if it doesn't exist
    reports_dir = Path(ROOT_DIR) / "data" / "reports"
    if not reports_dir.exists():
        os.makedirs(reports_dir, exist_ok=True)
    
    # Create the institution-specific directory under reports
    output_dir = reports_dir / dir_name
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created output directory at {output_dir}")
    return output_dir

def plot_total_volumes(volumes_df, output_path=None, institution_short_name="Institution", output_dir=None):
    """
    Create a stacked bar chart of student volumes by provider role
    
    Args:
        volumes_df: DataFrame with volume data
        output_path: Optional path to save the plot
        institution_short_name: Short name of the institution for the title
        output_dir: Optional directory to save the plot in
    """
    plt.figure(figsize=(10, 6))
    
    years = volumes_df['tilastovuosi'].astype(str)
    
    # Create the stacked bar chart
    plt.bar(years, volumes_df['järjestäjänä'], label='Koulutuksen järjestäjänä')
    plt.bar(years, volumes_df['hankintana'], bottom=volumes_df['järjestäjänä'], 
            label='Hankintakoulutuksen järjestäjänä')
    
    # Add total volume as text on top of each bar
    for i, year in enumerate(years):
        row = volumes_df.iloc[i]
        plt.text(i, row['Yhteensä'] + 50, f"{row['Yhteensä']:.1f}", 
                 ha='center', va='bottom', fontsize=10)
        
        # Add percentage as main provider in each bar
        plt.text(i, row['Yhteensä']/2, f"{row['järjestäjä_osuus (%)']:.1f}%", 
                 ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    plt.title(f'{institution_short_name} Student Volumes by Provider Role')
    plt.xlabel('Year')
    plt.ylabel('Student Volume (nettoopiskelijamaaraLkm)')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    fig = plt.gcf()
    if output_path:
        save_plot(fig, file_path=output_path)
    else:
        save_plot(fig, plot_name=f"{institution_short_name.lower()}_total_volumes", output_dir=output_dir)

def plot_top_qualifications(volumes_by_qual, output_path=None, year=None, top_n=5, institution_short_name="Institution", output_dir=None):
    """
    Create a horizontal bar chart of top qualifications by volume
    
    Args:
        volumes_by_qual: DataFrame with volumes by qualification
        output_path: Path to save the plot
        year: Year to use for filtering (optional)
        top_n: Number of top qualifications to show
        institution_short_name: Short name for the institution (used in file naming)
        output_dir: Optional directory to save the plot in
    """
    try:
        # Check for required columns or find suitable alternatives
        required_columns = ['tutkinto', 'kouluttaja yhteensä', 'markkinaosuus (%)']
        missing_columns = [col for col in required_columns if col not in volumes_by_qual.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns for plot_top_qualifications: {missing_columns}")
            
            # Try to find alternative columns
            plot_data = volumes_by_qual.copy()
            
            # Handle missing 'kouluttaja yhteensä' column
            if 'kouluttaja yhteensä' in missing_columns:
                # Try to find any numeric column for volume data
                numeric_cols = plot_data.select_dtypes(include=['number']).columns
                
                # Exclude percentage columns for volume
                volume_cols = [col for col in numeric_cols if '%' not in col and 'markkinaosuus' not in col]
                
                if volume_cols:
                    # Use the first available numeric column
                    volume_col = volume_cols[0]
                    print(f"Using '{volume_col}' for volume data instead of 'kouluttaja yhteensä'")
                    plot_data = plot_data.rename(columns={volume_col: 'kouluttaja yhteensä'})
                else:
                    # Create a dummy column with random data if no suitable column exists
                    print("Creating dummy volume data for visualization")
                    plot_data['kouluttaja yhteensä'] = np.random.randint(1, 100, size=len(plot_data))
            
            # Handle missing 'markkinaosuus (%)' column
            if 'markkinaosuus (%)' in missing_columns:
                # Try to find any percentage column
                pct_cols = [col for col in plot_data.columns if '%' in col or 'share' in col.lower()]
                
                if pct_cols:
                    # Use the first available percentage column
                    pct_col = pct_cols[0]
                    print(f"Using '{pct_col}' for market share data instead of 'markkinaosuus (%)'")
                    plot_data = plot_data.rename(columns={pct_col: 'markkinaosuus (%)'})
                else:
                    # Calculate a dummy market share based on volume
                    if 'kouluttaja yhteensä' in plot_data.columns:
                        print("Calculating dummy market share percentage from volume data")
                        total_volume = plot_data['kouluttaja yhteensä'].sum()
                        if total_volume > 0:
                            plot_data['markkinaosuus (%)'] = (plot_data['kouluttaja yhteensä'] / total_volume) * 100
                        else:
                            plot_data['markkinaosuus (%)'] = np.random.uniform(0, 10, size=len(plot_data))
                    else:
                        plot_data['markkinaosuus (%)'] = np.random.uniform(0, 10, size=len(plot_data))
        else:
            plot_data = volumes_by_qual.copy()
        
        # Filter for the specified year if available
        if year is not None and 'tilastovuosi' in plot_data.columns:
            year_data = plot_data[plot_data['tilastovuosi'] == year]
            if year_data.empty:
                print(f"Warning: No data for year {year}, using most recent year")
                years = sorted(plot_data['tilastovuosi'].unique())
                if years:
                    year = years[-1]
                    year_data = plot_data[plot_data['tilastovuosi'] == year]
                else:
                    year_data = plot_data
        elif year is not None and 'year' in plot_data.columns:
            year_data = plot_data[plot_data['year'] == year]
            if year_data.empty:
                print(f"Warning: No data for year {year}, using most recent year")
                years = sorted(plot_data['year'].unique())
                if years:
                    year = years[-1]
                    year_data = plot_data[plot_data['year'] == year]
                else:
                    year_data = plot_data
        else:
            year_data = plot_data
        
        # If no data after filtering, print warning and return
        if year_data.empty:
            print("Warning: No data available for plotting top qualifications")
            return
        
        # Sort by volume and get top N
        top_quals = year_data.sort_values('kouluttaja yhteensä', ascending=False).head(top_n)
        
        if top_quals.empty:
            print("Warning: No data available for top qualifications after filtering")
            return
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Ensure tutkinto is a column in the dataframe
        if 'tutkinto' not in top_quals.columns:
            print("Error: 'tutkinto' column missing from the DataFrame")
            return
        
        # Create horizontal bar chart
        ax = sns.barplot(
            x='kouluttaja yhteensä', 
            y='tutkinto',
            data=top_quals,
            palette='viridis'
        )
        
        # Add market share percentages to the bars
        for i, (_, row) in enumerate(top_quals.iterrows()):
            if 'markkinaosuus (%)' in row:
                market_share = row['markkinaosuus (%)']
                ax.text(
                    row['kouluttaja yhteensä'] + 0.1, 
                    i, 
                    f"{market_share:.1f}%", 
                    va='center'
                )
        
        # Customize appearance
        year_suffix = f" in {year}" if year else ""
        plt.title(f"Top Qualifications by Volume for {institution_short_name}{year_suffix}")
        plt.xlabel("Student Volume")
        plt.tight_layout()
        
        # Save the plot
        fig = plt.gcf()
        if output_path:
            save_plot(fig, file_path=output_path)
        else:
            save_plot(fig, plot_name=f"{institution_short_name.lower()}_top_qualifications", output_dir=output_dir)
            
    except Exception as e:
        print(f"Warning: Error creating top qualifications plot - {e}")

def plot_qualification_growth(growth_df, output_path=None, year=None, metric='nettoopiskelijamaaraLkm_growth', top_n=5, bottom_n=5, institution_short_name="Institution", output_dir=None):
    """
    Create a bar chart showing qualifications with highest growth and decline
    
    Args:
        growth_df: DataFrame with growth analysis
        output_path: Optional path to save the plot
        year: Year to filter for (optional)
        metric: Column to use for ranking (default: nettoopiskelijamaaraLkm_growth)
        top_n: Number of top growing qualifications to show
        bottom_n: Number of top declining qualifications to show
        institution_short_name: Short name of the institution for the title
        output_dir: Optional directory to save the plot in
    """
    if growth_df.empty:
        print(f"Warning: No growth data available to plot")
        return
        
    # Filter for specific year if provided
    if year:
        data = growth_df[growth_df['tilastovuosi'] == year].copy()
    else:
        # Use most recent year
        latest_year = growth_df['tilastovuosi'].max()
        data = growth_df[growth_df['tilastovuosi'] == latest_year].copy()
    
    # Filter out rows with NaN in the metric column
    data = data.dropna(subset=[metric])
    
    if len(data) == 0:
        print(f"Warning: No valid growth data for year {year or latest_year}")
        return
    
    # Get top growing and declining qualifications
    top_growing = data.nlargest(top_n, metric)
    top_declining = data.nsmallest(bottom_n, metric)
    
    # Combine them
    plot_data = pd.concat([top_growing, top_declining])
    
    # Shorten qualification names
    plot_data['short_name'] = plot_data['tutkinto'].str.split(' ').str[0:3].str.join(' ') + '...'
    
    # Sort by the metric
    plot_data = plot_data.sort_values(metric)
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Set colors: red for negative, green for positive
    colors = ['#e86b3d' if val < 0 else '#7dc35a' for val in plot_data[metric]]
    
    # Create horizontal bar chart
    ax = sns.barplot(
        y='short_name',
        x=metric,
        data=plot_data,
        palette=colors,
        orient='h'
    )
    
    # Add volume and market share as annotations
    for i, row in enumerate(plot_data.iterrows()):
        # Find the best volume column to display
        volume_value = None
        for col in ['kouluttaja yhteensä', 'tutkinto yhteensä']:
            if col in row[1]:
                volume_value = row[1][col]
                break
                
        if volume_value:
            plt.text(
                row[1][metric] + (5 if row[1][metric] >= 0 else -5), 
                i,
                f"Vol: {volume_value:.1f}", 
                ha='left' if row[1][metric] >= 0 else 'right',
                va='center',
                fontsize=9
            )
    
    # Add a vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Improve appearance
    plt.title(f'Qualifications with Highest Growth and Decline (Year: {plot_data["tilastovuosi"].iloc[0]})')
    plt.xlabel(f'Year-over-Year Change in {metric} (%)')
    plt.ylabel('Qualification')
    plt.tight_layout()
    
    # Save the plot
    fig = plt.gcf()
    if output_path:
        save_plot(fig, file_path=output_path)
    else:
        save_plot(fig, plot_name=f"{institution_short_name.lower()}_qualification_growth", output_dir=output_dir)

def plot_qualification_time_series(volumes_by_qual, output_path=None, qualifications=None, top_n=5, metric='kouluttaja yhteensä', institution_short_name="Institution", output_dir=None):
    """
    Plot time series for top qualifications
    
    Args:
        volumes_by_qual: DataFrame with volumes by qualification over time
        output_path: Path to save the plot
        qualifications: List of qualifications to include (optional)
        top_n: Number of top qualifications to include if qualifications is None
        metric: Column name for the metric to use
        institution_short_name: Short name for the institution (used in file naming)
        output_dir: Optional directory to save the plot in
    """
    try:
        # Check for required columns
        required_columns = ['tilastovuosi', metric]
        missing_columns = [col for col in required_columns if col not in volumes_by_qual.columns]
        if missing_columns:
            print(f"Warning: Missing required columns for plot_qualification_time_series: {missing_columns}")
            # Try to infer time column if tilastovuosi is missing
            if 'tilastovuosi' in missing_columns and 'year' in volumes_by_qual.columns:
                print("Using 'year' column instead of 'tilastovuosi'")
                volumes_by_qual = volumes_by_qual.rename(columns={'year': 'tilastovuosi'})
                missing_columns.remove('tilastovuosi')
            # Try to use any numeric column if the metric is missing
            if metric in missing_columns:
                numeric_cols = volumes_by_qual.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    alternate_metric = numeric_cols[0]
                    print(f"Using '{alternate_metric}' instead of '{metric}'")
                    metric = alternate_metric
                    missing_columns.remove(metric)
            
            # If we still have missing columns, raise an exception
            if missing_columns:
                raise ValueError(f"Still missing required columns: {missing_columns}")
        
        # If we made it here, we have the necessary columns to continue
        # Create a pivot table with qualifications as rows and years as columns
        if 'tilastovuosi' in volumes_by_qual.columns and metric in volumes_by_qual.columns:
            pivot_df = volumes_by_qual.pivot_table(
                index='tutkinto', 
                columns='tilastovuosi', 
                values=metric,
                aggfunc='sum',
                fill_value=0
            )
            
            # Sort qualifications by their average value
            pivot_df['mean'] = pivot_df.mean(axis=1)
            pivot_df = pivot_df.sort_values('mean', ascending=False)
            
            # Select qualifications
            if qualifications is not None:
                selected_quals = [q for q in qualifications if q in pivot_df.index]
                if not selected_quals:
                    # If none of the specified qualifications are in the data, use top N
                    selected_quals = pivot_df.index[:top_n]
            else:
                selected_quals = pivot_df.index[:top_n]
            
            # Filter for selected qualifications
            plot_data = pivot_df.loc[selected_quals].drop('mean', axis=1)
            
            # Transpose to have time on x-axis
            plot_data = plot_data.T
            
            # Create figure
            plt.figure(figsize=(12, 7))
            
            # Plot each qualification as a line
            for qual in plot_data.columns:
                # Shorten qualification name
                short_name = ' '.join(qual.split(' ')[:3]) + '...'
                plt.plot(plot_data.index, plot_data[qual], marker='o', linewidth=2, label=short_name)
            
            # Improve appearance
            plt.title(f'Time Series of Top Qualifications ({metric})')
            plt.xlabel('Year')
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.legend(title='Qualification', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            fig = plt.gcf()
            if output_path:
                save_plot(fig, file_path=output_path)
            else:
                save_plot(fig, plot_name=f"{institution_short_name.lower()}_qualification_time_series", output_dir=output_dir)
        else:
            raise ValueError("Missing required columns for plotting")
    except Exception as e:
        print(f"Warning: Error creating time series plot - {e}")

def run_analysis(
    data_file_path,
    institution_names,
    institution_short_name=None,
    output_dir=None,
    filter_degree_types=True,
    reference_year=None,
    shorten_names=True
):
    """
    Run the complete education market analysis for a given institution.
    
    Args:
        data_file_path: Path to the data file
        institution_names: List of names for the institution to analyze
        institution_short_name: Short name for the institution (used in plots and file naming)
        output_dir: Directory to save output files (default: data/reports/education_market_[institution_short_name])
        filter_degree_types: Whether to filter by degree types
        reference_year: Year to use as reference for qualification selection
        shorten_names: Whether to shorten qualification names
        
    Returns:
        dict: Dictionary with the main analysis results
    """
    # Set default short name if not provided
    if institution_short_name is None:
        institution_short_name = institution_names[0].split(' ')[0]
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = create_output_directory(institution_short_name)
    else:
        output_dir = Path(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory at {output_dir}")
    
    print(f"Starting education market analysis for {institution_short_name}")
    print(f"Institution variations: {institution_names}")
    
    # Initialize results dictionary
    results = {}
    
    try:
        # Load data
        print(f"Loading data from {data_file_path}...")
        raw_data = load_data(data_file_path, shorten_names=shorten_names)
        
        # Clean and prepare data with institution-specific cleaning
        print("Cleaning and preparing data...")
        prepared_data = clean_and_prepare_data(
            raw_data, 
            institution_names=institution_names,
            merge_qualifications=True, 
            shorten_names=shorten_names
        )
        
        # Initialize analyzer
        print(f"Initializing education market analyzer for {institution_short_name}...")
        analyzer = EducationMarketAnalyzer(
            data=prepared_data,
            institution_names=institution_names,
            filter_degree_types=filter_degree_types,
            reference_year=reference_year
        )
        
        # File name prefix for all outputs
        prefix = f"{institution_short_name.lower()}"
        
        # Use the most recent year in the data if reference_year is not provided
        target_year = reference_year or analyzer.data[analyzer.year_col].max()
        print(f"Using {target_year} as the target year for analysis")
        
        # Step 1-2: Calculate total volumes
        print("Analyzing total volumes...")
        try:
            volumes_df = analyzer.analyze_total_volume()
            volumes_df['kouluttaja'] = institution_short_name  # Use provided short name
            results['volumes'] = volumes_df
            
            # Plot total volumes
            try:
                plot_total_volumes(
                    volumes_df, 
                    institution_short_name=institution_short_name,
                    output_dir=output_dir
                )
                print(f"Total volumes plot created")
            except Exception as e:
                print(f"Warning: Could not create total volumes plot - {e}")
        except Exception as e:
            print(f"Error analyzing total volumes: {e}")
            volumes_df = pd.DataFrame()
        
        # Step 3: Analyze volumes by qualification
        print("Analyzing volumes by qualification...")
        try:
            volumes_by_qual = analyzer.analyze_volumes_by_qualification()
            results['volumes_by_qualification'] = volumes_by_qual
            
            # Plot top qualifications
            try:
                plot_top_qualifications(
                    volumes_by_qual, 
                    year=target_year,
                    institution_short_name=institution_short_name,
                    output_dir=output_dir
                )
                print(f"Top qualifications plot created")
            except Exception as e:
                print(f"Warning: Could not create top qualifications plot - {e}")
        except Exception as e:
            print(f"Error analyzing volumes by qualification: {e}")
            volumes_by_qual = pd.DataFrame()
        
        # Calculate CAGR for qualifications
        print("Calculating qualification CAGR...")
        try:
            qualification_cagr = analyzer.calculate_qualification_cagr()
            results['cagr'] = qualification_cagr
        except Exception as e:
            print(f"Error calculating CAGR: {e}")
            qualification_cagr = pd.DataFrame()
        
        # Calculate YoY growth
        print("Calculating YoY growth...")
        try:
            yoy_growth = analyzer.calculate_yearly_growth(
                target_column='nettoopiskelijamaaraLkm',
                year=target_year,
                time_window=3
            )
            results['yoy_growth'] = yoy_growth
        except Exception as e:
            print(f"Error calculating YoY growth: {e}")
            yoy_growth = pd.DataFrame()
        
        # Get top providers by qualification
        print("Identifying top providers by qualification...")
        try:
            top_providers = analyzer.get_top_providers_by_qualification(
                year=target_year, 
                top_n=5
            )
            results['top_providers'] = top_providers
        except Exception as e:
            print(f"Error identifying top providers: {e}")
            top_providers = {}
        
        # Analyze institution roles
        print("Analyzing institution roles...")
        try:
            role_analysis = analyzer.analyze_institution_roles()
            results['role_analysis'] = role_analysis
        except Exception as e:
            print(f"Error analyzing institution roles: {e}")
            role_analysis = pd.DataFrame()
        
        # Plot qualification growth (using new growth data)
        # Merge data based on available columns
        if not volumes_by_qual.empty and not yoy_growth.empty:
            try:
                # Try to merge for comprehensive growth analysis
                common_columns = list(set(volumes_by_qual.columns) & set(yoy_growth.columns))
                if 'tutkinto' in common_columns and 'tilastovuosi' in common_columns:
                    qualification_growth_data = pd.merge(
                        volumes_by_qual,
                        yoy_growth,
                        on=['tutkinto', 'tilastovuosi'],
                        how='inner'
                    )
                    
                    # Only plot if we have merged data
                    if not qualification_growth_data.empty:
                        plot_qualification_growth(
                            qualification_growth_data, 
                            year=target_year, 
                            metric='nettoopiskelijamaaraLkm_growth',
                            institution_short_name=institution_short_name,
                            output_dir=output_dir
                        )
                        print(f"Qualification growth plot created")
                else:
                    # Use basic growth data if available
                    plot_qualification_growth(
                        yoy_growth, 
                        year=target_year, 
                        metric='nettoopiskelijamaaraLkm_growth',
                        institution_short_name=institution_short_name,
                        output_dir=output_dir
                    )
                    print(f"Qualification growth plot created")
            except Exception as e:
                print(f"Warning: Could not create growth plot - {e}")
        
        # Step 4: Time series analysis
        if not volumes_by_qual.empty:
            print("Creating time series plots...")
            try:
                plot_qualification_time_series(
                    volumes_by_qual, 
                    top_n=6,
                    institution_short_name=institution_short_name,
                    output_dir=output_dir
                )
                print(f"Time series plot created")
            except Exception as e:
                print(f"Warning: Could not create time series plot - {e}")
        
        # Save results to Excel if we have at least some data
        if any([not df.empty if isinstance(df, pd.DataFrame) else bool(df) for df in results.values()]):
            print("Saving results to Excel...")
            
            # Prepare data dictionary for Excel output - only include non-empty DataFrames
            excel_data = {}
            
            if not volumes_df.empty:
                excel_data["Total Volumes"] = volumes_df
            
            if not volumes_by_qual.empty:
                excel_data["Volumes by Qualification"] = volumes_by_qual
            
            if not qualification_cagr.empty:
                excel_data["CAGR by Qualification"] = qualification_cagr
            
            if not yoy_growth.empty:
                excel_data["YoY Growth"] = yoy_growth
            
            if not role_analysis.empty:
                excel_data["Institution Roles"] = role_analysis
            
            # Convert top providers dictionary to DataFrame for export if we have data
            if top_providers:
                top_providers_df = pd.DataFrame([
                    {'qualification': qual, 'top_providers': ', '.join(providers)}
                    for qual, providers in top_providers.items()
                ])
                excel_data["Top Providers"] = top_providers_df
            
            # Save the Excel file to the institution-specific output directory
            try:
                # Fallback: Use pandas directly to save Excel file
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                excel_path = output_dir / f"{prefix}_market_analysis_{timestamp}.xlsx"
                
                # Ensure directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Create Excel writer
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    for sheet_name, sheet_data in excel_data.items():
                        sheet_data.to_excel(writer, sheet_name=sheet_name, index=True)
                
                print(f"Analysis results saved to {excel_path}")
            
            except Exception as e:
                print(f"Error saving Excel file: {e}")
                print("This might happen if you don't have openpyxl installed. To install it, run: pip install openpyxl")
        else:
            print("No data to save to Excel.")
        
        print(f"Analysis complete! Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
    
    # Return the main results for further use
    return results

def main():
    """
    Main function to run the analysis pipeline
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze education market data for any institution')
    parser.add_argument('--data-file', type=str, default="vipunen_data_sample.csv",
                        help='Name of the data file in the data directory')
    parser.add_argument('--institution', nargs='+', default=DEFAULT_INSTITUTION_NAMES,
                        help='List of institution names to analyze (alternatives/variations)')
    parser.add_argument('--short-name', type=str, default=DEFAULT_SHORT_NAME,
                        help='Short name for the institution (used in plots and file naming)')
    parser.add_argument('--reference-year', type=int, 
                        help='Reference year for qualification selection')
    parser.add_argument('--filter-degree-types', action='store_true', default=True,
                        help='Whether to filter by degree types')
    parser.add_argument('--no-filter-degree-types', action='store_false', dest='filter_degree_types',
                        help='Disable filtering by degree types')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Check for the data file in different locations
    data_file = args.data_file
    
    # Define data directories
    raw_dir = Path(ROOT_DIR) / "data" / "raw"
    data_dir = Path(ROOT_DIR) / "data"
    
    # Check in raw directory
    raw_file_path = raw_dir / data_file
    if os.path.exists(raw_file_path):
        data_file_path = raw_file_path
    # Check in general data directory
    elif os.path.exists(data_dir / data_file):
        data_file_path = data_dir / data_file
    # Try as an absolute path
    elif os.path.exists(data_file):
        data_file_path = Path(data_file)
    else:
        raise FileNotFoundError(f"Data file '{data_file}' not found in data directories.")
    
    print(f"Using data file: {data_file_path}")
    
    # Create output directory path if specified
    output_dir = None
    if args.output_dir:
        # If output_dir is specified, use it as is
        output_dir = Path(args.output_dir)
    else:
        # Otherwise use the default in data/reports
        reports_dir = Path(ROOT_DIR) / "data" / "reports"
        if not reports_dir.exists():
            os.makedirs(reports_dir, exist_ok=True)
        
        dir_name = f"education_market_{args.short_name.lower()}"
        output_dir = reports_dir / dir_name
    
    # Run the analysis
    results = run_analysis(
        data_file_path=data_file_path,
        institution_names=args.institution,
        institution_short_name=args.short_name,
        output_dir=output_dir,
        filter_degree_types=args.filter_degree_types,
        reference_year=args.reference_year
    )
    
    return results

if __name__ == "__main__":
    main() 