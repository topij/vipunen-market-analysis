#!/usr/bin/env python
"""
Education Market Analysis Workflow Example

This script demonstrates the complete workflow for analyzing the education market
using the refactored vipunen package.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import logging

# Import FileUtils
from FileUtils import OutputFileType

# Import modules from the vipunen package
from src.vipunen.data.data_loader import load_data, create_output_directory
from src.vipunen.data.data_processor import clean_and_prepare_data, shorten_qualification_names
from src.vipunen.analysis.market_share_analyzer import calculate_market_shares, calculate_market_share_changes, calculate_total_volumes
from src.vipunen.analysis.qualification_analyzer import analyze_qualification_growth, calculate_cagr_for_groups
from src.vipunen.visualization.volume_plots import plot_total_volumes, plot_top_qualifications
from src.vipunen.visualization.market_plots import plot_market_share_heatmap, plot_qualification_market_shares
from src.vipunen.visualization.growth_plots import plot_qualification_growth, plot_qualification_time_series
from src.vipunen.export.excel_exporter import ExcelExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Education market analysis workflow")
    
    parser.add_argument(
        "--data-file", "-d",
        dest="data_file",
        help="Path to the data file (CSV)",
        default="data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv"
    )
    
    parser.add_argument(
        "--institution", "-i",
        dest="institution",
        help="Main name of the institution to analyze",
        default="Rastor-instituutti"
    )
    
    parser.add_argument(
        "--short-name", "-s",
        dest="short_name",
        help="Short name for the institution (used in titles and file names)",
        default="Rastor"
    )
    
    parser.add_argument(
        "--variant", "-v",
        dest="variants",
        action="append",
        help="Name variant for the institution (can be specified multiple times)",
        default=[]
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        dest="output_dir",
        help="Base directory for output files (default: data/reports)",
        default=None
    )
    
    parser.add_argument(
        "--use-dummy", "-u",
        dest="use_dummy",
        action="store_true",
        help="Use dummy data instead of loading from file",
        default=False
    )
    
    parser.add_argument(
        "--filter-qual-types",
        dest="filter_qual_types",
        action="store_true",
        help="Filter data to include only ammattitutkinto and erikoisammattitutkinto",
        default=False
    )
    
    parser.add_argument(
        "--filter-by-institution-quals",
        dest="filter_by_inst_quals",
        action="store_true",
        help="Filter data to include only qualifications offered by the institution under analysis during the current and previous year",
        default=False
    )
    
    return parser.parse_args()

def ensure_data_directory(file_path):
    """
    Ensure the file path includes the data directory.
    If the path starts with 'raw/', prepend 'data/' to it.
    """
    if file_path.startswith("raw/"):
        return f"data/{file_path}"
    return file_path

def main():
    """Main function to run the full analysis workflow."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Step 1: Define parameters for the analysis
    data_file_path = ensure_data_directory(args.data_file)
    institution_name = args.institution
    institution_short_name = args.short_name
    
    # Set default variants if none provided
    if not args.variants:
        institution_variants = [
            "Rastor-instituutti ry", 
            "Rastor-instituutti", 
            "RASTOR OY",
            "Rastor Oy"
        ]
    else:
        # Add the main institution name to the variants
        institution_variants = list(args.variants)
        if institution_name not in institution_variants:
            institution_variants.append(institution_name)
    
    # Step 2: Create output directories
    logger.info("Creating output directories")
    base_output_dir = args.output_dir
    output_dir = create_output_directory(institution_short_name)
    if base_output_dir:
        output_dir = Path(base_output_dir) / output_dir.name
        os.makedirs(output_dir, exist_ok=True)
        
    plots_dir = output_dir / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Step 3: Load the raw data
    logger.info(f"Loading raw data from {data_file_path}")
    try:
        if args.use_dummy:
            logger.info("Using dummy dataset for demonstration purposes")
            raw_data = create_dummy_dataset()
        else:
            raw_data = load_data(data_file_path)
        
        logger.info(f"Loaded {len(raw_data)} rows of data")
    except FileNotFoundError:
        logger.error(f"Could not find the data file at {data_file_path}")
        logger.info("Creating a dummy dataset for demonstration purposes")
        # Create a dummy dataset for demonstration
        raw_data = create_dummy_dataset()
    
    # Step 4: Clean and prepare the data
    logger.info("Cleaning and preparing the data")
    df_clean = clean_and_prepare_data(
        raw_data, 
        institution_names=institution_variants,
        merge_qualifications=True,
        shorten_names=True
    )
    
    # Step 5: Apply optional filters
    df_filtered = df_clean.copy()
    
    # Filter for qualification types if requested
    if args.filter_qual_types:
        logger.info("Filtering for ammattitutkinto and erikoisammattitutkinto")
        qualification_types = ["Ammattitutkinnot", "Erikoisammattitutkinnot"]
        df_filtered = df_filtered[df_filtered["tutkintotyyppi"].isin(qualification_types)]
    else:
        logger.info("Using all qualification types")
    
    # Filter for qualifications offered by the institution if requested
    if args.filter_by_inst_quals:
        latest_year = df_filtered["tilastovuosi"].max()
        previous_year = latest_year - 1
        
        # Get data for current and previous year
        recent_years_data = df_filtered[df_filtered["tilastovuosi"].isin([latest_year, previous_year])]
        
        # Find qualifications where the institution is either the provider or subcontractor
        institution_quals = []
        
        # Check as main provider
        provider_quals = recent_years_data[
            recent_years_data["koulutuksenJarjestaja"].isin(institution_variants)
        ]["tutkinto"].unique()
        institution_quals.extend(provider_quals)
        
        # Check as subcontractor
        subcontractor_quals = recent_years_data[
            recent_years_data["hankintakoulutuksenJarjestaja"].isin(institution_variants)
        ]["tutkinto"].unique()
        institution_quals.extend(subcontractor_quals)
        
        # Get unique list
        institution_quals = list(set(institution_quals))
        
        logger.info(f"Filtering for {len(institution_quals)} qualifications offered by {institution_name}")
        df_filtered = df_filtered[df_filtered["tutkinto"].isin(institution_quals)]
    else:
        logger.info("Using all qualifications")
    
    # Step 6: Calculate institution's volumes by qualification
    logger.info(f"Calculating volumes for {institution_name}")
    
    # Calculate total volumes using the updated function
    total_volumes = calculate_total_volumes(
        df=df_filtered,
        provider_names=institution_variants
    )
    
    # Set the kouluttaja column for compatibility with the original implementation
    total_volumes['kouluttaja'] = institution_short_name
    
    # Step 7: Calculate volumes by qualification
    logger.info("Calculating volumes by qualification")
    volumes_by_qual = []
    
    # Process each year
    for year in sorted(df_filtered["tilastovuosi"].unique()):
        year_data = df_filtered[df_filtered["tilastovuosi"] == year]
        
        # Process each qualification
        for qual in sorted(year_data["tutkinto"].unique()):
            qual_data = year_data[year_data["tutkinto"] == qual]
            
            # Calculate volumes for the institution
            provider_vol = qual_data[
                qual_data["koulutuksenJarjestaja"].isin(institution_variants)
            ]["nettoopiskelijamaaraLkm"].sum()
            
            subcontractor_vol = qual_data[
                qual_data["hankintakoulutuksenJarjestaja"].isin(institution_variants)
            ]["nettoopiskelijamaaraLkm"].sum()
            
            # Calculate total market volume for this qualification
            market_volume = qual_data["nettoopiskelijamaaraLkm"].sum()
            
            # Only add the qualification if the institution has some presence
            if provider_vol > 0 or subcontractor_vol > 0:
                volumes_by_qual.append({
                    "tilastovuosi": year,
                    "tutkinto": qual,
                    f"{year}_järjestäjänä": provider_vol,
                    f"{year}_hankintana": subcontractor_vol,
                    f"{year}_yhteensä": provider_vol + subcontractor_vol,
                    f"{year}_market_total": market_volume,
                    f"{year}_market_share": ((provider_vol + subcontractor_vol) / market_volume * 100) if market_volume > 0 else 0
                })
    
    volumes_by_qual_df = pd.DataFrame(volumes_by_qual)
    
    # Convert volumes_by_qual_df to long format for better Excel reporting
    logger.info("Converting qualification volumes to long format")
    volumes_long_format = []
    
    for _, row in volumes_by_qual_df.iterrows():
        year = row["tilastovuosi"]
        qualification = row["tutkinto"]
        provider_amount = row[f"{year}_järjestäjänä"]
        subcontractor_amount = row[f"{year}_hankintana"]
        total_amount = row[f"{year}_yhteensä"]
        market_total = row[f"{year}_market_total"]
        market_share = row[f"{year}_market_share"]
        
        volumes_long_format.append({
            "Year": year,
            "Qualification": qualification,
            "Provider Amount": provider_amount,
            "Subcontractor Amount": subcontractor_amount,
            "Total Amount": total_amount,
            "Market Total": market_total,
            "Market Share (%)": market_share
        })
    
    volumes_long_df = pd.DataFrame(volumes_long_format)
    
    # Step 8: Calculate market shares
    logger.info("Calculating market shares")
    market_shares = calculate_market_shares(df_filtered, institution_variants)
    
    # Step 9: Calculate year-over-year market share changes for all years
    logger.info("Calculating market share changes for all consecutive year pairs")
    all_years = sorted(df_filtered["tilastovuosi"].unique())
    latest_year = all_years[-1] if all_years else None
    market_share_changes_all = []
    
    # Calculate changes for each consecutive year pair
    for i in range(1, len(all_years)):
        current_year = all_years[i]
        previous_year = all_years[i-1]
        
        # Calculate changes for this year pair
        year_changes = calculate_market_share_changes(market_shares, current_year, previous_year)
        
        # Store the year changes
        market_share_changes_all.append(year_changes)
    
    # Combine all year changes
    if market_share_changes_all:
        market_share_changes = pd.concat(market_share_changes_all, ignore_index=True)
    else:
        # If no changes (e.g., only one year in the data), use empty DataFrame
        market_share_changes = pd.DataFrame()
    
    # Step 9.5: Create combined "Provider's Market" dataset
    logger.info("Creating combined Provider's Market dataset")
    
    # Get the list of qualifications offered by the analyzed institution
    institution_quals = df_filtered[
        (df_filtered["koulutuksenJarjestaja"].isin(institution_variants)) | 
        (df_filtered["hankintakoulutuksenJarjestaja"].isin(institution_variants))
    ]["tutkinto"].unique()
    
    # Filter market_shares to only include those qualifications
    providers_market = market_shares[market_shares["tutkinto"].isin(institution_quals)].copy()
    
    # Remove unwanted columns
    if "is_target_provider" in providers_market.columns:
        providers_market = providers_market.drop(columns=["is_target_provider"])
    if "index" in providers_market.columns:
        providers_market = providers_market.drop(columns=["index"])
    
    # Check column names in both DataFrames
    logger.info(f"Market shares columns: {market_shares.columns.tolist()}")
    logger.info(f"Market share changes columns: {market_share_changes.columns.tolist()}")
    
    # Map column names from market_shares to the desired output columns
    shares_column_mapping = {
        'tilastovuosi': 'Year',
        'tutkinto': 'Qualification',
        'provider': 'Provider',  # This will be 'kouluttaja' in the merged result
        'volume_as_provider': 'Provider Amount',
        'volume_as_subcontractor': 'Subcontractor Amount',
        'total_volume': 'Total Volume',
        'qualification_market_volume': 'Market Total',
        'market_share': 'Market Share (%)',
        'provider_count': 'Provider Count',
        'subcontractor_count': 'Subcontractor Count'
    }
    
    # Rename the market_shares columns
    providers_market = providers_market.rename(columns=shares_column_mapping)
    
    # Prepare for merge by creating a copy of the provider column with the name kouluttaja
    providers_market['kouluttaja'] = providers_market['Provider']
    
    # Now try to merge in the YoY data
    # Prepare market_share_changes for merge
    changes_for_merge = market_share_changes.copy()
    
    # Find relevant columns in market_share_changes
    yoy_col = None
    for col in changes_for_merge.columns:
        if 'change' in col.lower() and 'percent' in col.lower():
            yoy_col = col
            break
    
    gainer_rank_col = None
    for col in changes_for_merge.columns:
        if 'rank' in col.lower():
            gainer_rank_col = col
            break
    
    # If we found both columns, we can merge
    if yoy_col and gainer_rank_col:
        # Rename to match our desired output
        col_mapping = {
            'provider': 'kouluttaja',
            'tutkinto': 'Qualification',
            'current_year': 'Year',
            yoy_col: 'Market Share Growth (%)',
            gainer_rank_col: 'market gainer'
        }
        
        # Create an empty DataFrame to store the merged data
        providers_market_with_growth = pd.DataFrame()
        
        # Process each year separately
        for year in providers_market['Year'].unique():
            # Get the providers market data for this year
            year_providers = providers_market[providers_market['Year'] == year].copy()
            
            # Get the market share changes for this year
            year_changes = changes_for_merge[changes_for_merge['current_year'] == year].copy()
            
            if not year_changes.empty:
                # Make sure we have all the columns we need
                required_columns = ['tutkinto', 'provider', 'current_year', yoy_col, gainer_rank_col]
                if all(col in year_changes.columns for col in required_columns):
                    # Keep only the columns we need
                    year_changes = year_changes[required_columns]
                    
                    # Rename columns for the merge
                    year_changes = year_changes.rename(columns=col_mapping)
                    
                    # Merge for this year
                    year_providers = pd.merge(
                        year_providers,
                        year_changes,
                        on=['kouluttaja', 'Qualification', 'Year'],
                        how='left'
                    )
                else:
                    # Missing columns
                    logger.warning(f"Missing required columns for year {year} in market_share_changes")
                    year_providers['Market Share Growth (%)'] = np.nan
                    year_providers['market gainer'] = np.nan
                
                # Append to the result
                providers_market_with_growth = pd.concat([providers_market_with_growth, year_providers], ignore_index=True)
            else:
                # No changes for this year (might be the first year)
                year_providers['Market Share Growth (%)'] = np.nan
                year_providers['market gainer'] = np.nan
                providers_market_with_growth = pd.concat([providers_market_with_growth, year_providers], ignore_index=True)
        
        # Update the main DataFrame
        if not providers_market_with_growth.empty:
            providers_market = providers_market_with_growth
    else:
        # If we can't find the right columns, add empty columns
        providers_market['Market Share Growth (%)'] = np.nan
        providers_market['market gainer'] = np.nan
    
    # Clean up the extra column we added for merging
    if 'kouluttaja' in providers_market.columns:
        providers_market = providers_market.drop(columns=['kouluttaja'])
    
    # Make sure all requested columns exist
    required_columns = [
        "Year", "Qualification", "Provider", "Provider Amount", "Subcontractor Amount",
        "Total Volume", "Market Total", "Market Share (%)", "Market Rank", 
        "Market Share Growth (%)", "market gainer"
    ]
    
    # Ensure market rank exists
    if "Market Rank" not in providers_market.columns:
        # Calculate ranks for each qualification and year
        providers_market["Market Rank"] = np.nan
        for (year, qual), group in providers_market.groupby(["Year", "Qualification"]):
            ranks = group["Market Share (%)"].rank(ascending=False, method="min")
            providers_market.loc[(providers_market["Year"] == year) & 
                             (providers_market["Qualification"] == qual), "Market Rank"] = ranks
    
    for col in required_columns:
        if col not in providers_market.columns:
            providers_market[col] = np.nan
    
    # Sort the dataframe
    providers_market = providers_market.sort_values(
        by=["Year", "Qualification", "Market Rank"]
    ).reset_index(drop=True)
    
    # Reorder columns
    providers_market = providers_market[required_columns]
    
    # Step 10: Analyze qualification growth
    logger.info("Analyzing qualification growth")
    qualification_growth = analyze_qualification_growth(volumes_by_qual_df)
    
    # Step 11: Calculate CAGR for qualifications
    logger.info("Calculating CAGR for qualifications")
    qualification_cagr = calculate_cagr_for_groups(
        volumes_by_qual_df, 
        ["tutkinto"], 
        f"{latest_year}_yhteensä"
    )
    
    # Enhance CAGR Analysis with additional columns
    logger.info("Enhancing CAGR Analysis with additional information")
    
    # Create an improved CAGR dataframe with additional columns
    enhanced_cagr = []
    
    # Group by qualification to get the first and last years, start and end volumes
    for qual, group in volumes_by_qual_df.groupby("tutkinto"):
        years = sorted(group["tilastovuosi"].unique())
        first_year = min(years)
        last_year = max(years)
        
        # Get start and end volumes
        start_volume = group[group["tilastovuosi"] == first_year][f"{first_year}_yhteensä"].sum()
        end_volume = group[group["tilastovuosi"] == last_year][f"{last_year}_yhteensä"].sum()
        
        # Calculate CAGR if we have more than one year and non-zero volumes
        cagr = np.nan
        if len(years) > 1 and start_volume > 0 and end_volume > 0:
            years_between = last_year - first_year
            cagr = (end_volume / start_volume) ** (1 / years_between) - 1
            cagr = cagr * 100  # Convert to percentage
        
        # Create the record
        enhanced_cagr.append({
            "tutkinto": qual,
            "CAGR": cagr,
            "First Year Offered": first_year,
            "Last Year Offered": last_year,
            "Start Volume": start_volume,
            "End Volume": end_volume,
            "Years": len(years),
            "year_range": f"{first_year}-{last_year}"
        })
    
    # Convert to DataFrame and sort
    enhanced_cagr_df = pd.DataFrame(enhanced_cagr)
    enhanced_cagr_df = enhanced_cagr_df.sort_values(by=["tutkinto"]).reset_index(drop=True)
    
    # Step 12: Create visualizations
    logger.info("Creating visualizations")
    
    # Plot total volumes
    plot_total_volumes(
        total_volumes, 
        institution_short_name=institution_short_name, 
        output_dir=plots_dir
    )
    
    # Plot top qualifications
    plot_top_qualifications(
        volumes_by_qual_df, 
        year=latest_year,
        top_n=10,
        institution_short_name=institution_short_name, 
        output_dir=plots_dir
    )
    
    # Plot market share heatmap
    plot_market_share_heatmap(
        market_shares, 
        institution_short_name=institution_short_name, 
        output_dir=plots_dir,
        reference_year=latest_year
    )
    
    # Plot qualification market shares
    plot_qualification_market_shares(
        df_filtered, 
        institution_names=institution_variants,
        institution_short_name=institution_short_name, 
        output_dir=plots_dir,
        reference_year=latest_year
    )
    
    # Plot qualification growth
    plot_qualification_growth(
        qualification_growth, 
        year=latest_year,
        institution_short_name=institution_short_name, 
        output_dir=plots_dir
    )
    
    # Plot qualification time series
    plot_qualification_time_series(
        volumes_by_qual_df, 
        top_n=5,
        institution_short_name=institution_short_name, 
        output_dir=plots_dir
    )
    
    # Step 13: Export to Excel
    logger.info("Exporting results to Excel")
    exporter = ExcelExporter(output_dir, prefix=institution_short_name.lower())
    
    excel_data = {
        "Total Volumes": total_volumes,
        "Volumes by Qualification": volumes_long_df,
        "Provider's Market": providers_market,
        "CAGR Analysis": enhanced_cagr_df
    }
    
    excel_path = exporter.export_to_excel(excel_data)
    logger.info(f"Analysis results exported to {excel_path}")
    logger.info("Analysis complete!")

def create_dummy_dataset():
    """Create a dummy dataset for demonstration purposes."""
    years = [2018, 2019, 2020, 2021, 2022]
    qualification_types = ["Ammattitutkinnot", "Erikoisammattitutkinnot"]
    qualifications = [
        "Johtamisen ja yritysjohtamisen EAT", 
        "Lähiesihenkilötyön AT", 
        "Yrittäjyyden AT",
        "Myynnin ja markkinoinnin AT",
        "Projektipäällikön AT"
    ]
    providers = [
        "Rastor-instituutti ry", 
        "Markkinointi-instituutti", 
        "Business College Helsinki",
        "Tampereen Aikuiskoulutuskeskus",
        "Turun Aikuiskoulutuskeskus"
    ]
    
    data = []
    
    for year in years:
        for qual_type in qualification_types:
            for qual in qualifications:
                for provider in providers:
                    # Main provider role
                    student_count = np.random.randint(10, 200)
                    data.append({
                        "tilastovuosi": year,
                        "tutkintotyyppi": qual_type,
                        "tutkinto": qual,
                        "koulutuksenJarjestaja": provider,
                        "hankintakoulutuksenJarjestaja": "Tieto puuttuu",
                        "hankintakoulutusKyllaEi": False,
                        "nettoopiskelijamaaraLkm": student_count
                    })
                    
                    # Some subcontractor relationships
                    if np.random.random() > 0.7:
                        subcontractor = np.random.choice([p for p in providers if p != provider])
                        student_count_sub = np.random.randint(5, 50)
                        data.append({
                            "tilastovuosi": year,
                            "tutkintotyyppi": qual_type,
                            "tutkinto": qual,
                            "koulutuksenJarjestaja": provider,
                            "hankintakoulutuksenJarjestaja": subcontractor,
                            "hankintakoulutusKyllaEi": True,
                            "nettoopiskelijamaaraLkm": student_count_sub
                        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    main() 