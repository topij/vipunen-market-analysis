#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Education Market Analysis script that analyzes the Finnish education market
from one provider's perspective, focusing specifically on Rastor-instituutti.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import argparse

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

# Constants
# Filter data for the institution under analysis (Rastor-instituutti)
RI_INSTITUTIONS_LIST = ["Rastor-instituutti ry", "Rastor Oy"]

def load_data(file_path):
    """
    Load data from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    print(f"Loading data from {file_path}")
    
    # Determine file type based on extension
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.csv':
        # Read CSV file with correct delimiter
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    return df

def plot_total_volumes(ri_volumes, output_path):
    """
    Create a stacked bar chart of student volumes by provider role
    
    Args:
        ri_volumes: DataFrame with volume data
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    years = ri_volumes['tilastovuosi'].astype(str)
    
    # Create the stacked bar chart
    plt.bar(years, ri_volumes['järjestäjänä'], label='Koulutuksen järjestäjänä')
    plt.bar(years, ri_volumes['hankintana'], bottom=ri_volumes['järjestäjänä'], 
            label='Hankintakoulutuksen järjestäjänä')
    
    # Add total volume as text on top of each bar
    for i, year in enumerate(years):
        row = ri_volumes.iloc[i]
        plt.text(i, row['Yhteensä'] + 50, f"{row['Yhteensä']:.1f}", 
                 ha='center', va='bottom', fontsize=10)
        
        # Add percentage as main provider in each bar
        plt.text(i, row['Yhteensä']/2, f"{row['järjestäjä_osuus (%)']:.1f}%", 
                 ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    plt.title('Rastor-instituutti Student Volumes by Provider Role')
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
        volume = row[1]['kouluttaja yhteensä']
        market_share = row[1]['markkinaosuus (%)']
        plt.text(
            i, volume + 5, f"{market_share:.1f}%", 
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

def main():
    """Main function to run the analysis"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze education market data')
    parser.add_argument('--filter-degree-types', action='store_true', 
                        help='Filter data by degree types for all analyses')
    args = parser.parse_args()
    
    # Load the data
    data_file = RAW_DATA_DIR / "amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv"
    df = load_data(data_file)
    
    # Create analyzer instance
    analyzer = EducationMarketAnalyzer(
        data=df,
        institution_names=RI_INSTITUTIONS_LIST,
        filter_degree_types=args.filter_degree_types
    )
    
    # Step 1 & 2: Calculate student volumes and breakdown by provider role
    ri_volumes = analyzer.analyze_total_volume()
    
    # Save the results
    ri_volumes.to_csv(OUTPUT_DIR / "ri_volumes_summary.csv", index=False)
    
    # Print the results
    print("\nSummary of student volumes for Rastor-instituutti:")
    print(ri_volumes.to_string(index=False))
    
    # Create visualization
    plot_total_volumes(ri_volumes, OUTPUT_DIR / "ri_volumes_by_role.png")
    
    # Step 3: Calculate volumes by qualification
    volumes_by_qual = analyzer.analyze_volumes_by_qualification()
    
    # Save the results
    volumes_by_qual.to_csv(OUTPUT_DIR / "ri_volumes_by_qualification.csv", index=False)
    
    # Print summary of qualifications
    print(f"\nFound {len(volumes_by_qual['tutkinto'].unique())} qualifications offered by Rastor-instituutti")
    
    # Create visualization of top qualifications
    plot_top_qualifications(volumes_by_qual, OUTPUT_DIR / "ri_top_qualifications.png")
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    
    return ri_volumes, volumes_by_qual

if __name__ == "__main__":
    main() 