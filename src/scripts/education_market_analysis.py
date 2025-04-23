#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Education Market Analysis script that analyzes the Finnish education market
from any provider's perspective.

This script allows you to analyze education market data for any educational 
institution in Finland. The default is set to Rastor-instituutti for 
backward compatibility, but can be easily changed to analyze any institution.

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
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import argparse
import numpy as np

# Add the parent directory to the path to import from the vipunen package
sys.path.append(str(Path(__file__).resolve().parents[2]))

from vipunen.analysis.education_market import EducationMarketAnalyzer

# Set paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

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
    Load data from CSV file and apply cleaning steps
    
    Args:
        file_path: Path to the CSV file
        shorten_names: Whether to shorten qualification names
        
    Returns:
        pd.DataFrame: Loaded and cleaned dataframe
    """
    print(f"Loading data from {file_path}")
    
    # Determine file type based on extension
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.csv':
        try:
            # Read CSV file with correct delimiter
            df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    # Apply basic cleaning without institution-specific replacements
    df = clean_and_prepare_data(df, institution_names=None, shorten_names=shorten_names)
    
    return df

def plot_total_volumes(volumes_df, output_path, institution_short_name="Institution"):
    """
    Create a stacked bar chart of student volumes by provider role
    
    Args:
        volumes_df: DataFrame with volume data
        output_path: Path to save the plot
        institution_short_name: Short name of the institution for the title
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
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_top_qualifications(volumes_by_qual, output_path, year=None, top_n=5):
    """
    Create a bar chart of top qualifications by volume
    
    Args:
        volumes_by_qual: DataFrame with volumes by qualification
        output_path: Path to save the plot
        year: Year to filter for (optional)
        top_n: Number of top qualifications to show
    """
    # Filter for specific year if provided
    if year:
        data = volumes_by_qual[volumes_by_qual['tilastovuosi'] == year].copy()
    else:
        # Use most recent year
        latest_year = volumes_by_qual['tilastovuosi'].max()
        data = volumes_by_qual[volumes_by_qual['tilastovuosi'] == latest_year].copy()
    
    # Sort by institution's total volume and get top N
    data = data.sort_values('kouluttaja yhteensä', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    
    # Shorten qualification names for better display
    data['short_name'] = data['tutkinto'].str.split(' ').str[0:3].str.join(' ') + '...'
    
    # Create a stacked bar chart
    ax = sns.barplot(
        x='short_name', 
        y='kouluttaja yhteensä', 
        data=data,
        color=sns.color_palette()[0]
    )
    
    # Add market share as text
    for i, row in enumerate(data.iterrows()):
        plt.text(
            i, row[1]['kouluttaja yhteensä'] + 5, f"{row[1]['markkinaosuus (%)']:.1f}%", 
            ha='center', fontsize=9, fontweight='bold'
        )
    
    plt.title(f'Top {top_n} Qualifications by Volume (Year: {data["tilastovuosi"].iloc[0]})')
    plt.xlabel('Qualification')
    plt.ylabel('Student Volume')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_qualification_growth(growth_df, output_path, year=None, metric='nettoopiskelijamaaraLkm_growth', top_n=5, bottom_n=5):
    """
    Create a bar chart showing qualifications with highest growth and decline
    
    Args:
        growth_df: DataFrame with growth analysis
        output_path: Path to save the plot
        year: Year to filter for (optional)
        metric: Column to use for ranking (default: nettoopiskelijamaaraLkm_growth)
        top_n: Number of top growing qualifications to show
        bottom_n: Number of top declining qualifications to show
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
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_qualification_time_series(volumes_by_qual, output_path, qualifications=None, top_n=5, metric='kouluttaja yhteensä'):
    """
    Create a time series plot for selected qualifications
    
    Args:
        volumes_by_qual: DataFrame with volumes by qualification
        output_path: Path to save the plot
        qualifications: List of qualifications to include (optional)
        top_n: Number of top qualifications to include if qualifications not specified
        metric: Metric to plot and use for selecting top qualifications
    """
    # Create a pivot table with years as columns
    pivot_df = volumes_by_qual.pivot_table(
        index='tutkinto',
        columns='tilastovuosi',
        values=metric
    )
    
    # Calculate the mean volume over all years for each qualification
    pivot_df['mean'] = pivot_df.mean(axis=1)
    
    # If qualifications not specified, use top_n by mean volume
    if qualifications is None:
        selected_quals = pivot_df.nlargest(top_n, 'mean').index.tolist()
    else:
        selected_quals = qualifications
        # Filter for qualifications that exist in the data
        selected_quals = [q for q in selected_quals if q in pivot_df.index]
    
    if not selected_quals:
        print("Warning: No qualifications to plot")
        return
        
    # Filter for selected qualifications and drop the mean column
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
    plt.savefig(output_path, dpi=300)
    plt.close()

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
        output_dir: Directory to save output files (default: output/[institution_short_name])
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
        output_dir = OUTPUT_DIR / f"education_market_{institution_short_name.lower()}"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
                    output_dir / f"{prefix}_total_volumes.png",
                    institution_short_name=institution_short_name
                )
                print(f"Total volumes plot saved to {output_dir / f'{prefix}_total_volumes.png'}")
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
                    output_dir / f"{prefix}_top_qualifications.png", 
                    year=target_year
                )
                print(f"Top qualifications plot saved to {output_dir / f'{prefix}_top_qualifications.png'}")
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
                            output_dir / f"{prefix}_qualification_growth.png",
                            year=target_year, 
                            metric='nettoopiskelijamaaraLkm_growth'
                        )
                        print(f"Qualification growth plot saved to {output_dir / f'{prefix}_qualification_growth.png'}")
                else:
                    # Use basic growth data if available
                    plot_qualification_growth(
                        yoy_growth, 
                        output_dir / f"{prefix}_qualification_growth.png",
                        year=target_year, 
                        metric='nettoopiskelijamaaraLkm_growth'
                    )
                    print(f"Qualification growth plot saved to {output_dir / f'{prefix}_qualification_growth.png'}")
            except Exception as e:
                print(f"Warning: Could not create growth plot - {e}")
        
        # Step 4: Time series analysis
        if not volumes_by_qual.empty:
            print("Creating time series plots...")
            try:
                plot_qualification_time_series(
                    volumes_by_qual, 
                    output_dir / f"{prefix}_qualification_time_series.png",
                    top_n=6
                )
                print(f"Time series plot saved to {output_dir / f'{prefix}_qualification_time_series.png'}")
            except Exception as e:
                print(f"Warning: Could not create time series plot - {e}")
        
        # Save results to Excel if we have at least some data
        if any([not df.empty if isinstance(df, pd.DataFrame) else bool(df) for df in results.values()]):
            print("Saving results to Excel...")
            excel_path = output_dir / f"{prefix}_market_analysis.xlsx"
            try:
                with pd.ExcelWriter(excel_path) as writer:
                    # Save each DataFrame to a separate sheet
                    if not volumes_df.empty:
                        volumes_df.to_excel(writer, sheet_name="Total Volumes", index=False)
                    
                    if not volumes_by_qual.empty:
                        volumes_by_qual.to_excel(writer, sheet_name="Volumes by Qualification", index=False)
                    
                    if not qualification_cagr.empty:
                        qualification_cagr.to_excel(writer, sheet_name="CAGR by Qualification", index=False)
                    
                    if not yoy_growth.empty:
                        yoy_growth.to_excel(writer, sheet_name="YoY Growth", index=False)
                    
                    if not role_analysis.empty:
                        role_analysis.to_excel(writer, sheet_name="Institution Roles", index=False)
                    
                    # Convert top providers dictionary to DataFrame for export if we have data
                    if top_providers:
                        top_providers_df = pd.DataFrame([
                            {'qualification': qual, 'top_providers': ', '.join(providers)}
                            for qual, providers in top_providers.items()
                        ])
                        top_providers_df.to_excel(writer, sheet_name="Top Providers", index=False)
                
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
    
    # Define input path
    data_file_path = DATA_DIR / args.data_file
    
    # Create output directory path if specified
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    
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