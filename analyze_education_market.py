#!/usr/bin/env python
"""
Education Market Analysis Workflow Example

This script demonstrates the complete workflow for analyzing the education market
using the FileUtils package directly with minimal wrappers.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Import FileUtils and configuration
from FileUtils import FileUtils, OutputFileType
from src.vipunen.utils.file_utils_config import get_file_utils

# Import analysis modules from the vipunen package
from src.vipunen.data.data_processor import clean_and_prepare_data, shorten_qualification_names
from src.vipunen.analysis.market_share_analyzer import calculate_market_shares, calculate_market_share_changes, calculate_total_volumes
from src.vipunen.analysis.qualification_analyzer import analyze_qualification_growth, calculate_cagr_for_groups
from src.vipunen.visualization.volume_plots import plot_total_volumes, plot_top_qualifications
from src.vipunen.visualization.market_plots import plot_market_share_heatmap, plot_qualification_market_shares
from src.vipunen.visualization.growth_plots import plot_qualification_growth, plot_qualification_time_series
from FileUtils.core.base import StorageError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the FileUtils instance
file_utils = get_file_utils()

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
        default="Rastor-instituutti ry"
    )
    
    parser.add_argument(
        "--short-name", "-s",
        dest="short_name",
        help="Short name for the institution (used in titles and file names)",
        default="RI"
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

def export_to_excel(data_dict, file_name, output_type="reports", **kwargs):
    """
    Export multiple DataFrames to Excel using FileUtils directly.
    
    Args:
        data_dict: Dictionary mapping sheet names to DataFrames
        file_name: Base name for the output file
        output_type: Output directory type
        **kwargs: Additional arguments for Excel export
        
    Returns:
        Path: Path to the saved Excel file
    """
    # Filter out empty DataFrames
    filtered_data = {
        sheet_name: df for sheet_name, df in data_dict.items() 
        if isinstance(df, pd.DataFrame) and not df.empty
    }
    
    if not filtered_data:
        logger.warning("No data to export to Excel")
        return None
    
    try:
        # Log what we're exporting
        for sheet_name, df in filtered_data.items():
            logger.info(f"Preparing sheet '{sheet_name}' with {len(df)} rows")
            
            # Reset index if it's not a default RangeIndex
            if not isinstance(df.index, pd.RangeIndex):
                filtered_data[sheet_name] = df.reset_index(drop=True)
            
            # Check for problematic data that might cause Excel export issues
            if df.isnull().values.any():
                logger.warning(f"Sheet '{sheet_name}' contains NULL values which might cause export issues")
            
            # Check for infinite values
            if np.isinf(df.select_dtypes(include=[np.number]).values).any():
                logger.warning(f"Sheet '{sheet_name}' contains infinite values which might cause export issues")
                # Replace infinite values with a large number or NaN
                for col in df.select_dtypes(include=[np.number]).columns:
                    filtered_data[sheet_name][col] = filtered_data[sheet_name][col].replace([np.inf, -np.inf], np.nan)
        
        # Use FileUtils directly to save the Excel file
        path_result = file_utils.save_data_to_storage(
            data=filtered_data,
            file_name=file_name,
            output_type=output_type,
            output_filetype=OutputFileType.XLSX,
            index=False,  # Don't include index in Excel export
            **kwargs
        )
        
        # Extract the actual path from the result (which might be a dict or tuple)
        if isinstance(path_result, dict):
            excel_path = next(iter(path_result.values()))
        elif isinstance(path_result, tuple) and path_result:
            if isinstance(path_result[0], dict):
                excel_path = next(iter(path_result[0].values()))
            else:
                excel_path = path_result[0]
        else:
            excel_path = path_result
        
        logger.info(f"Exported Excel file to {excel_path}")
        return Path(excel_path)
        
    except Exception as e:
        logger.error(f"Error exporting Excel file: {e}")
        # More detailed error information
        import traceback
        logger.error(f"Export error details: {traceback.format_exc()}")
        
        # Try to identify which sheet might be causing issues
        try:
            # Test each sheet individually
            for sheet_name, df in filtered_data.items():
                try:
                    test_export = {sheet_name: df}
                    test_path = file_utils.save_data_to_storage(
                        data=test_export,
                        file_name=f"{file_name}_test_{sheet_name}",
                        output_type=output_type,
                        output_filetype=OutputFileType.XLSX,
                        index=False
                    )
                    logger.info(f"Test export of sheet '{sheet_name}' succeeded")
                except Exception as sheet_error:
                    logger.error(f"Sheet '{sheet_name}' may be causing export issues: {sheet_error}")
        except Exception as test_error:
            logger.error(f"Unable to run sheet tests: {test_error}")
        
        raise

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
    dir_name = f"education_market_{institution_short_name.lower()}"
    
    # Create reports directory under data using FileUtils
    reports_dir = file_utils.create_directory("reports", parent_dir="data")
    
    # Create the output directory under reports using standard Path methods
    output_dir = reports_dir / dir_name
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create plots directory under the output directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Override output directory if specified in args
    if args.output_dir:
        output_dir = Path(args.output_dir) / dir_name
        os.makedirs(output_dir, exist_ok=True)
        plots_dir = output_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
    
    # Step 3: Load the raw data using FileUtils directly
    logger.info(f"Loading raw data from {data_file_path}")
    try:
        if args.use_dummy:
            logger.info("Using dummy dataset for demonstration purposes")
            raw_data = create_dummy_dataset()
        else:
            # Direct use of FileUtils instead of wrapper
            logger.info(f"Loading data from {data_file_path}")
            path_obj = Path(data_file_path)
            file_name = path_obj.name
            
            try:
                # Try with semicolon separator for CSV files
                if path_obj.suffix.lower() == '.csv':
                    raw_data = file_utils.load_single_file(file_name, input_type="raw", sep=';')
                else:
                    # For non-CSV files, let FileUtils auto-detect the format
                    raw_data = file_utils.load_single_file(file_name, input_type="raw")
            except Exception:
                # Try with comma separator if semicolon fails
                raw_data = file_utils.load_single_file(file_name, input_type="raw", sep=',')
        
        logger.info(f"Loaded {len(raw_data)} rows of data")
    except (FileNotFoundError, StorageError) as e:
        logger.error(f"Could not find or load the data file at {data_file_path}: {e}")
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
                # Create a row for this qualification
                volumes_by_qual.append({
                    "tilastovuosi": year,
                    "tutkinto": qual,
                    "tutkintotyyppi": qual_data["tutkintotyyppi"].iloc[0] if not qual_data.empty else "",
                    f"{year}_järjestäjänä": provider_vol,
                    f"{year}_hankintana": subcontractor_vol,
                    f"{year}_yhteensä": provider_vol + subcontractor_vol,
                    f"{year}_market_total": market_volume,
                    f"{year}_market_share": ((provider_vol + subcontractor_vol) / market_volume * 100) if market_volume > 0 else 0
                })
    
    # Convert to DataFrame
    volumes_by_qual_df = pd.DataFrame(volumes_by_qual)
    
    # Step 8: Convert qualification volumes to long format for easier analysis
    logger.info("Converting qualification volumes to long format")
    volumes_long_df = []
    
    for _, row in volumes_by_qual_df.iterrows():
        year = row["tilastovuosi"]
        qualification = row["tutkinto"]
        
        # Create a record for the long format
        volumes_long_df.append({
            "Year": year,
            "Qualification": qualification,
            "Provider Amount": row[f"{year}_järjestäjänä"],
            "Subcontractor Amount": row[f"{year}_hankintana"],
            "Total Amount": row[f"{year}_yhteensä"],
            "Market Total": row[f"{year}_market_total"],
            "Market Share (%)": row[f"{year}_market_share"]
        })
    
    # Convert to DataFrame
    volumes_long_df = pd.DataFrame(volumes_long_df)
    
    # Step 9: Calculate market shares
    logger.info("Calculating market shares")
    market_shares = calculate_market_shares(
        df=df_filtered,
        provider_names=institution_variants
    )
    
    # Step 10: Calculate market share changes for all consecutive year pairs
    logger.info("Calculating market share changes for all consecutive year pairs")
    
    # Get all distinct years
    years = sorted(market_shares["tilastovuosi"].unique())
    
    # Calculate changes for each consecutive year pair
    market_share_changes = []
    for i in range(1, len(years)):
        current_year = years[i]
        previous_year = years[i-1]
        
        # Get data for current and previous year
        changes = calculate_market_share_changes(
            market_shares,
            current_year=current_year,
            previous_year=previous_year
        )
        
        market_share_changes.append(changes)
    
    # Combine all changes into a single DataFrame
    if market_share_changes:
        market_share_changes_df = pd.concat(market_share_changes)
    else:
        market_share_changes_df = pd.DataFrame()
    
    # Step 11: Create combined Provider's Market dataset
    logger.info("Creating combined Provider's Market dataset")
    
    # Log column names for debugging
    logger.info(f"Market shares columns: {list(market_shares.columns)}")
    logger.info(f"Market share changes columns: {list(market_share_changes_df.columns) if not market_share_changes_df.empty else []}")
    
    # Step 12: Analyze qualification growth
    logger.info("Analyzing qualification growth")
    qualification_growth = analyze_qualification_growth(volumes_by_qual_df)
    
    # Step 13: Calculate CAGR for qualifications
    logger.info("Calculating CAGR for qualifications")
    
    # Get the latest year for the CAGR calculation
    years = sorted(df_filtered["tilastovuosi"].unique())
    latest_year = years[-1] if years else None
    
    # Use volumes_long_df which has the correct structure for CAGR calculation
    # It has columns: Year, Qualification, Total Amount, etc.
    cagr_analysis = calculate_cagr_for_groups(
        df=volumes_long_df,
        groupby_columns=["Qualification"],
        value_column="Total Amount",
        qual_type_column="tutkintotyyppi" if "tutkintotyyppi" in volumes_long_df.columns else None
    )
    
    # Step 14: Ensure CAGR Analysis has at least one row
    if cagr_analysis.empty:
        logger.warning("CAGR Analysis is empty. Adding a placeholder row.")
        cagr_analysis = pd.DataFrame([{
            "tutkinto": "No qualification data available",
            "CAGR": "",
            "First Year": years[0] if years else None,
            "Last Year": years[-1] if years else None,
            "First Year Volume": 0,
            "Last Year Volume": 0,
            "Years Present": 0,
            "Qualification Type": "N/A"
        }])
    
    # Step 15: Create visualizations
    logger.info("Creating visualizations")
    
    # Plot 1: Total volumes over time
    plot_total_volumes(
        volumes_df=total_volumes,
        institution_short_name=institution_short_name,
        output_dir=plots_dir
    )
    
    # Plot 2: Top qualifications
    plot_top_qualifications(
        volumes_by_qual=volumes_by_qual_df,
        institution_short_name=institution_short_name,
        output_dir=plots_dir
    )
    
    # Plot 3: Market share heatmap
    plot_market_share_heatmap(
        market_share_df=market_shares,
        institution_short_name=institution_short_name,
        output_dir=plots_dir
    )
    
    # Plot 4: Qualification market shares
    plot_qualification_market_shares(
        volumes_df=df_filtered,
        institution_names=institution_variants,
        institution_short_name=institution_short_name,
        output_dir=plots_dir
    )
    
    # Plot 5: Qualification growth
    plot_qualification_growth(
        growth_df=qualification_growth,
        institution_short_name=institution_short_name,
        output_dir=plots_dir
    )
    
    # Plot 6: Qualification time series
    plot_qualification_time_series(
        volumes_by_qual=volumes_by_qual_df,
        institution_short_name=institution_short_name,
        output_dir=plots_dir
    )
    
    # Step 16: Export results to Excel
    logger.info("Exporting results to Excel")
    
    # Prepare data for Excel export - ensure correct structure for Provider's Market
    # Map column names from market_shares to match required output format
    providers_market = market_shares.rename(columns={
        'tilastovuosi': 'Year',
        'tutkinto': 'Qualification',
        'provider': 'Provider',
        'volume_as_provider': 'Provider Amount',
        'volume_as_subcontractor': 'Subcontractor Amount',
        'total_volume': 'Total Volume',
        'qualification_market_volume': 'Market Total',
        'market_share': 'Market Share (%)'
    }).copy()
    
    # Calculate Market Rank for each qualification and year
    providers_market["Market Rank"] = np.nan
    for (year, qual), group in providers_market.groupby(["Year", "Qualification"]):
        ranks = group["Market Share (%)"].rank(ascending=False, method="min")
        providers_market.loc[
            (providers_market["Year"] == year) & 
            (providers_market["Qualification"] == qual), 
            "Market Rank"
        ] = ranks
    
    # Add Market Share Growth and market gainer columns using market share changes data
    providers_market["Market Share Growth (%)"] = np.nan
    providers_market["market gainer"] = np.nan
    
    # For each provider and qualification, find the market share growth from the changes data
    for index, row in providers_market.iterrows():
        # Look for corresponding entry in market_share_changes
        year = row["Year"]
        qualification = row["Qualification"]
        provider = row["Provider"]
        
        # Find change data for this provider/qualification
        matching_changes = market_share_changes_df[
            (market_share_changes_df["current_year"] == year) &
            (market_share_changes_df["tutkinto"] == qualification) &
            (market_share_changes_df["provider"] == provider)
        ]
        
        if not matching_changes.empty:
            providers_market.loc[index, "Market Share Growth (%)"] = matching_changes.iloc[0]["market_share_change_percent"]
            providers_market.loc[index, "market gainer"] = matching_changes.iloc[0]["gainer_rank"]
    
    # First, identify all qualifications offered by the target institutions
    target_institution_quals = set()
    for _, row in providers_market.iterrows():
        if row["Provider"] in institution_variants:
            target_institution_quals.add((row["Year"], row["Qualification"]))
    
    # Filter to include all providers for those qualification-year combinations
    providers_market_filtered = providers_market[
        providers_market.apply(
            lambda row: (row["Year"], row["Qualification"]) in target_institution_quals, 
            axis=1
        )
    ].copy().reset_index(drop=True)
    
    # Ensure all required columns are present
    required_columns = [
        "Year", "Qualification", "Provider", "Provider Amount", "Subcontractor Amount",
        "Total Volume", "Market Total", "Market Share (%)", "Market Rank", 
        "Market Share Growth (%)", "market gainer"
    ]
    
    # Use only the required columns in the specified order
    providers_market_final = providers_market_filtered[required_columns]
    
    # Prepare Excel data
    excel_data = {
        "Total Volumes": total_volumes.drop(columns=['kouluttaja yhteensä'], errors='ignore').reset_index(drop=True) if not total_volumes.empty else pd.DataFrame(),
        "Volumes by Qualification": volumes_long_df.reset_index(drop=True) if not volumes_long_df.empty else pd.DataFrame(),
        "Provider's Market": providers_market_final.reset_index(drop=True) if not providers_market_final.empty else pd.DataFrame(),
        "CAGR Analysis": cagr_analysis.reset_index(drop=True) if not cagr_analysis.empty else pd.DataFrame()
    }
    
    # Add detailed logging for Excel export preparation
    for sheet_name, df in excel_data.items():
        if df.empty:
            logger.warning(f"Sheet '{sheet_name}' is empty and may not be included in the Excel export")
        else:
            logger.info(f"Sheet '{sheet_name}' prepared with {len(df)} rows and {len(df.columns)} columns")
            logger.debug(f"Columns for '{sheet_name}': {list(df.columns)}")
    
    # Export to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"{institution_short_name.lower()}_market_analysis{timestamp}"
    
    try:
        # Create a full path to the output file in the institution-specific directory
        excel_full_path = output_dir / f"{excel_filename}.xlsx"
        
        # Use pd.ExcelWriter directly to save to the exact path we want
        with pd.ExcelWriter(excel_full_path, engine='openpyxl') as writer:
            for sheet_name, df in excel_data.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Analysis results exported to {excel_full_path}")
        excel_path = excel_full_path
        
    except Exception as e:
        logger.error(f"Error exporting Excel file to specific directory: {e}")
        # Fall back to the FileUtils method as backup
        logger.info("Falling back to FileUtils export method")
        try:
            path_result = file_utils.save_data_to_storage(
                data=excel_data,
                file_name=excel_filename,
                output_type="reports",
                output_filetype=OutputFileType.XLSX,
                index=False,
                include_timestamp=True
            )
            
            # Extract the actual path from the result
            if isinstance(path_result, tuple) and path_result:
                if isinstance(path_result[0], dict):
                    excel_path = next(iter(path_result[0].values()))
                else:
                    excel_path = path_result[0]
            elif isinstance(path_result, dict):
                excel_path = next(iter(path_result.values()))
            else:
                excel_path = path_result
                
            excel_path = Path(excel_path)
            logger.error("Analysis complete, but saved to default reports directory instead of institution directory")
            
        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")
            excel_path = None
        
    logger.info(f"Analysis complete!")
    
    return {
        "total_volumes": total_volumes,
        "volumes_by_qualification": volumes_by_qual_df,
        "volumes_long": volumes_long_df,
        "market_shares": market_shares,
        "qualification_cagr": cagr_analysis,
        "excel_path": excel_path
    }

# Create a dummy dataset for testing or demonstration
def create_dummy_dataset():
    """Create a dummy dataset for testing or demonstration purposes."""
    # Define years
    years = range(2017, 2025)
    
    # Define qualifications
    qualifications = [
        "Liiketoiminnan AT", 
        "Johtamisen EAT", 
        "Yrittäjyyden AT", 
        "Myynnin AT", 
        "Markkinointiviestinnän AT"
    ]
    
    # Define education providers
    providers = [
        "Rastor-instituutti", 
        "Business College Helsinki", 
        "Mercuria", 
        "Markkinointi-instituutti",
        "Kauppiaitten Kauppaoppilaitos", 
        "Suomen Liikemiesten Kauppaopisto"
    ]
    
    # Generate rows for the dataframe
    rows = []
    
    # For each year and qualification
    for year in years:
        for qual in qualifications:
            # Base market size for this qualification
            qual_market_size = np.random.randint(200, 800)
            
            # For each provider
            for provider in providers:
                # Provider's share of this qualification
                provider_share = np.random.uniform(0.05, 0.25)
                provider_volume = int(qual_market_size * provider_share)
                
                # Sometimes add subcontractor relationship
                subcontractor = None
                if np.random.random() < 0.3:  # 30% chance of having a subcontractor
                    # Pick a random provider as subcontractor
                    potential_subcontractors = [p for p in providers if p != provider]
                    subcontractor = np.random.choice(potential_subcontractors)
                    
                    # Volume handled by subcontractor (30-70% of total)
                    sub_ratio = np.random.uniform(0.3, 0.7)
                    sub_volume = int(provider_volume * sub_ratio)
                    main_volume = provider_volume - sub_volume
                    
                    # Add row for main provider with subcontractor
                    rows.append({
                        "tilastovuosi": year,
                        "tutkintotyyppi": "Ammattitutkinnot" if "AT" in qual else "Erikoisammattitutkinnot",
                        "tutkinto": qual,
                        "koulutuksenJarjestaja": provider,
                        "hankintakoulutuksenJarjestaja": subcontractor,
                        "nettoopiskelijamaaraLkm": main_volume
                    })
                    
                    # Add row for the subcontractor portion
                    rows.append({
                        "tilastovuosi": year,
                        "tutkintotyyppi": "Ammattitutkinnot" if "AT" in qual else "Erikoisammattitutkinnot",
                        "tutkinto": qual,
                        "koulutuksenJarjestaja": subcontractor,
                        "hankintakoulutuksenJarjestaja": None,
                        "nettoopiskelijamaaraLkm": sub_volume
                    })
                else:
                    # No subcontractor, add row for main provider only
                    rows.append({
                        "tilastovuosi": year,
                        "tutkintotyyppi": "Ammattitutkinnot" if "AT" in qual else "Erikoisammattitutkinnot",
                        "tutkinto": qual,
                        "koulutuksenJarjestaja": provider,
                        "hankintakoulutuksenJarjestaja": None,
                        "nettoopiskelijamaaraLkm": provider_volume
                    })
    
    # Create pandas DataFrame
    df = pd.DataFrame(rows)
    
    logger.info(f"Created dummy dataset with {len(df)} rows")
    return df

if __name__ == "__main__":
    main() 