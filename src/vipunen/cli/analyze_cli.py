"""
CLI module for the Vipunen project.

This module provides the main entry point for the education market analysis
workflow, orchestrating the data loading, analysis, and export steps.
"""
import sys
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import math
import datetime # Added for date parsing
import matplotlib.pyplot as plt # Add import
import numpy as np
import plotly.graph_objects as go

from ..config.config_loader import get_config
from ..data.data_loader import load_data
from ..data.data_processor import clean_and_prepare_data
from ..export.excel_exporter import export_to_excel
from ..analysis.market_analyzer import MarketAnalyzer
from .argument_parser import parse_arguments, get_institution_variants
# Import Visualizer and constants
from ..visualization.education_visualizer import EducationVisualizer, COLOR_PALETTES, TEXT_CONSTANTS

# Configure logging
# Comment out the basicConfig here as it's handled in run_analysis.py
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger(__name__)

# --- Refactored Helper Functions ---

def prepare_analysis_data(
    config: Dict[str, Any], 
    args: Dict[str, Any]
) -> Tuple[pd.DataFrame, str, List[str], str, str, bool]:
    """
    Loads configuration, determines parameters, loads and prepares data.

    Args:
        config: Loaded configuration dictionary.
        args: Dictionary of arguments (e.g., from command-line parsing).

    Returns:
        Tuple containing:
        - df_clean (pd.DataFrame): Cleaned and prepared data.
        - institution_key (str): The key for the institution being analyzed.
        - institution_variants (List[str]): List of names/variants for the institution.
        - institution_short_name (str): Short name for the institution.
        - data_update_date_str (str): String representation of the data update date.
        - filter_qual_types (bool): Whether to filter by qualification types.
    """
    logger.info("Preparing analysis data...")
    
    # Step 1: Define parameters for the analysis from config and args
    data_file_path = args.get('data_file', config['paths']['data'])
    
    # Determine the institution key
    default_institution_name = config['institutions']['default']['name']
    arg_institution_value = args.get('institution') 
    
    if arg_institution_value is None or arg_institution_value == default_institution_name:
        institution_key = 'default'
        logger.info(f"Using default institution key: '{institution_key}'")
    else:
        institution_key = arg_institution_value
        logger.info(f"Using institution key from args: '{institution_key}'")
        if institution_key not in config['institutions']:
            msg = f"Provided institution key '{institution_key}' not found in config['institutions']. Aborting."
            logger.error(msg)
            raise KeyError(msg)
            
    # Get short_name
    institution_short_name = args.get('short_name', config['institutions'][institution_key]['short_name'])
    use_dummy = args.get('use_dummy', False)
    filter_qual_types = args.get('filter_qual_types', False)
    # filter_by_inst_quals = args.get('filter_by_inst_quals', False) # This filtering happens below now

    # Get institution variants
    if 'variants' in args and args['variants']:
        institution_variants = list(args['variants'])
        main_name = config['institutions'][institution_key]['name']
        if main_name not in institution_variants:
            institution_variants.append(main_name)
    else:
        institution_variants = config['institutions'][institution_key].get('variants', [])
        main_name = config['institutions'][institution_key]['name']
        if main_name not in institution_variants:
            institution_variants.append(main_name)

    logger.info(f"Analyzing institution key: {institution_key} (Name: {config['institutions'][institution_key]['name']})")
    logger.info(f"Institution variants used for matching: {institution_variants}")

    # Step 2: Load the raw data
    logger.info(f"Loading raw data from {data_file_path}")
    raw_data = load_data(file_path=data_file_path, use_dummy=use_dummy)
    logger.info(f"Loaded {len(raw_data)} rows of data")

    # Step 3: Extract Data Update Date
    data_update_date_str = datetime.datetime.now().strftime("%d.%m.%Y") # Default to today
    update_date_col = config.get('columns', {}).get('input', {}).get('update_date', 'tietojoukkoPaivitettyPvm')
    if not raw_data.empty and update_date_col in raw_data.columns:
        try:
            raw_date_str = str(raw_data[update_date_col].iloc[0])
            parsed_date = pd.to_datetime(raw_date_str)
            data_update_date_str = parsed_date.strftime("%d.%m.%Y")
            logger.info(f"Using data update date from column '{update_date_col}': {data_update_date_str}")
        except Exception as date_err:
            logger.warning(f"Could not parse date from column '{update_date_col}': {date_err}. Falling back to current date.")
    else:
        logger.warning(f"Update date column '{update_date_col}' not found or data is empty. Falling back to current date.")

    # Step 4: Clean and prepare the data
    logger.info("Cleaning and preparing the data")
    df_clean_initial = clean_and_prepare_data(
        raw_data, 
        institution_names=institution_variants,
        merge_qualifications=True,
        shorten_names=True
    )
    
    # Step 5: Filter data based on institution's offered qualifications
    logger.info("Filtering data based on institution and offered qualifications...")
    input_cols = config['columns']['input']
    institution_mask = (
        (df_clean_initial[input_cols['provider']].isin(institution_variants)) |
        (df_clean_initial[input_cols['subcontractor']].isin(institution_variants))
    )
    inst_qualifications = df_clean_initial[institution_mask][input_cols['qualification']].unique()
    
    if len(inst_qualifications) > 0:
        df_clean = df_clean_initial[df_clean_initial[input_cols['qualification']].isin(inst_qualifications)].copy()
        logger.info(f"Filtered data to {len(inst_qualifications)} qualifications offered by {institution_short_name}. Shape before: {df_clean_initial.shape}, after: {df_clean.shape}")
    else:
        logger.warning(f"No qualifications found for institution {institution_short_name} based on variants. Proceeding with potentially unfiltered data.")
        df_clean = df_clean_initial # Use unfiltered data if no specific qualifications found
        
    # Step 6: Conditionally filter by qualification types (applied AFTER institution qual filtering)
    # This is returned separately as it's a choice made at analysis/orchestration time
    # if filter_qual_types:
    #     qual_types = config.get('qualification_types', ['Ammattitutkinnot', 'Erikoisammattitutkinnot'])
    #     df_clean = df_clean[df_clean['tutkintotyyppi'].isin(qual_types)]
    #     logger.info(f"Filtered to qualification types: {qual_types}")

    return df_clean, institution_key, institution_variants, institution_short_name, data_update_date_str, filter_qual_types


def perform_market_analysis(
    df_clean: pd.DataFrame, 
    config: Dict[str, Any], 
    institution_variants: List[str], 
    institution_short_name: str,
    filter_qual_types: bool # Added parameter
) -> Tuple[Dict[str, pd.DataFrame], MarketAnalyzer]:
    """
    Performs the market analysis using MarketAnalyzer.

    Args:
        df_clean: Cleaned and prepared data DataFrame.
        config: Configuration dictionary.
        institution_variants: List of names/variants for the institution.
        institution_short_name: Short name for the institution.
        filter_qual_types: Whether to filter by qualification types before analysis.

    Returns:
        Tuple containing:
        - analysis_results (Dict[str, pd.DataFrame]): Dictionary of analysis result DataFrames.
        - analyzer (MarketAnalyzer): The MarketAnalyzer instance used for analysis.
    """
    logger.info("Performing market analysis...")
    
    # Apply qualification type filtering if requested
    if filter_qual_types:
        qual_types = config.get('qualification_types', ['Ammattitutkinnot', 'Erikoisammattitutkinnot'])
        df_analysis_input = df_clean[df_clean['tutkintotyyppi'].isin(qual_types)].copy()
        logger.info(f"Applied qualification type filter: {qual_types}. Shape before: {df_clean.shape}, after: {df_analysis_input.shape}")
    else:
         df_analysis_input = df_clean # Use the data as is
         
    # Initialize and run MarketAnalyzer
    try:
        logger.info("Initializing MarketAnalyzer...")
        analyzer = MarketAnalyzer(df_analysis_input, cfg=config) 
        analyzer.institution_names = institution_variants
        analyzer.institution_short_name = institution_short_name
        
        logger.info("Running analysis...")
        analysis_results = analyzer.analyze()
        
        return analysis_results, analyzer

    except Exception as e:
        logger.error(f"Error during analysis execution: {e}", exc_info=True)
        # Return empty results and a dummy analyzer? Or re-raise? For now, return empty.
        return {
            "total_volumes": pd.DataFrame(),
            "volumes_by_qualification": pd.DataFrame(),
            "detailed_providers_market": pd.DataFrame(),
            "qualification_cagr": pd.DataFrame()
        }, MarketAnalyzer(pd.DataFrame(), cfg=config) # Return a dummy analyzer instance


def export_analysis_results(
    analysis_results: Dict[str, pd.DataFrame], 
    config: Dict[str, Any], 
    institution_short_name: str, 
    base_output_path: str, # Changed from args to direct path
    metadata_df: Optional[pd.DataFrame] = None, 
    include_timestamp: bool = True
) -> Optional[str]:
    """
    Exports analysis results to an Excel file.

    Args:
        analysis_results: Dictionary of analysis result DataFrames.
        config: Configuration dictionary.
        institution_short_name: Short name for the institution.
        base_output_path: Base directory for output (e.g., 'data/reports').
        metadata_df: Optional DataFrame containing analysis metadata.
        include_timestamp: Whether to include a timestamp in the filename.

    Returns:
        Optional[str]: Path to the generated Excel file, or None if export fails.
    """
    logger.info("Exporting analysis results to Excel...")
    excel_path = None
    try:
        # Determine output directory path
        dir_name = f"education_market_{institution_short_name.lower()}"
        full_output_dir_path = Path(base_output_path) / dir_name
        
        # Determine path relative to base 'data' directory for FileUtils if possible
        try:
            base_data_dir = Path(config['paths']['data']).parts[0] # Assumes paths.data exists
            excel_output_dir_relative = str(full_output_dir_path.relative_to(base_data_dir))
        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Could not reliably determine relative path ({e}). Using full path: {full_output_dir_path}")
            excel_output_dir_relative = str(full_output_dir_path) # Fallback
            
        logger.info(f"Calculated relative output dir for export: {excel_output_dir_relative}")
        
        # Ensure the main output directory exists (FileUtils handles subdirectory creation)
        full_output_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured base output directory exists: {full_output_dir_path}")
        
        # Prepare Excel data dictionary
        excel_data = {}

        # --- Add Metadata Sheet ---
        if metadata_df is not None and not metadata_df.empty:
            # Get sheet name from config, fallback to default
            metadata_sheet_name = config.get('excel', {}).get('metadata_sheet_name', "Analysis Info")
            excel_data[metadata_sheet_name] = metadata_df
            logger.info(f"Adding metadata sheet: '{metadata_sheet_name}'")
        else:
            logger.warning("Metadata DataFrame not provided or is empty. Skipping metadata sheet.")
        # --- End Metadata Sheet ---

        # Map analysis results to sheet names from config
        sheet_configs = config.get('excel', {}).get('sheets', [])
        analysis_keys = ['total_volumes', 'volumes_by_qualification', 'detailed_providers_market', 'qualification_cagr'] 

        # Basic check: only map the first 4 sheets if config matches expected length
        if len(sheet_configs) >= len(analysis_keys):
            for i, sheet_info in enumerate(sheet_configs[:len(analysis_keys)]): # Iterate only up to available keys
                sheet_name = sheet_info.get('name', f'DataSheet{i+1}') # Fallback name
                analysis_key = analysis_keys[i] 
                df = analysis_results.get(analysis_key, pd.DataFrame())
                # Only add non-empty DataFrames to avoid empty sheets
                if isinstance(df, pd.DataFrame) and not df.empty:
                     excel_data[sheet_name] = df.reset_index(drop=True)
                     logger.info(f"Mapping analysis key '{analysis_key}' to Excel sheet '{sheet_name}'")
                elif isinstance(df, pd.Series) and not df.empty: # Handle Series if necessary
                     excel_data[sheet_name] = df.reset_index() # Convert Series to DataFrame
                     logger.info(f"Mapping analysis key '{analysis_key}' (Series) to Excel sheet '{sheet_name}'")
                else:
                     logger.warning(f"Skipping empty or invalid data for key '{analysis_key}' (Sheet: '{sheet_name}')")
        else:
            logger.warning(f"Mismatch between excel sheet configs ({len(sheet_configs)}) and expected analysis results ({len(analysis_keys)}). Exporting only metadata if available.")
            # Only metadata sheet will be present in excel_data if it was added

        # Export (only if there's data to export)
        if not excel_data:
             logger.error("No data available to export to Excel (neither analysis results nor metadata).")
             return None
             
        excel_path = export_to_excel(
            data_dict=excel_data,
            file_name=f"{institution_short_name.lower()}_market_analysis",
            output_dir=excel_output_dir_relative, # Use relative path
            include_timestamp=include_timestamp
        )
        logger.info(f"Excel export successful: {excel_path}")

    except Exception as e:
        logger.error(f"Error during Excel export: {e}", exc_info=True)
        excel_path = None # Ensure path is None on error

    return excel_path


def generate_visualizations(
    analysis_results: Dict[str, Any],
    visualizer: EducationVisualizer,
    analyzer: MarketAnalyzer,
    config: Dict[str, Any],
    data_update_date_str: str
):
    """
    Generate all standard visualizations based on analysis results.

    Args:
        analysis_results: Dictionary containing the results from MarketAnalyzer.analyze()
        visualizer: An initialized EducationVisualizer instance.
        analyzer: The MarketAnalyzer instance (to access min/max years, institution name etc.).
        config: Configuration dictionary.
        data_update_date_str: String representation of the data update date
    """
    logger.info("Starting visualization generation...")

    # Get config column names
    cols_out = config.get('columns', {}).get('output', {})
    year_col = cols_out.get('year', 'Year') # Fallback to default if missing
    qual_col = cols_out.get('qualification', 'Qualification')
    provider_col = cols_out.get('provider', 'Provider')
    provider_amount_col = cols_out.get('provider_amount', 'Provider Amount')
    subcontractor_amount_col = cols_out.get('subcontractor_amount', 'Subcontractor Amount')
    total_volume_col = cols_out.get('total_volume', 'Total Volume') # Institution's volume
    market_total_col = cols_out.get('market_total', 'Market Total')
    market_share_col = cols_out.get('market_share', 'Market Share (%)')
    market_rank_col = cols_out.get('market_rank', 'Market Rank')
    market_share_growth_col = cols_out.get('market_share_growth', 'Market Share Growth (%)')
    market_gainer_rank_col = cols_out.get('market_gainer_rank', 'Market Gainer Rank')

    inst_short_name = analyzer.institution_short_name
    inst_names = analyzer.institution_names
    min_year = analyzer.min_year
    max_year = analyzer.max_year
    # Use the provided data update date for the caption
    base_caption = TEXT_CONSTANTS["data_source"].format(date=data_update_date_str)
    
    # Determine last full year for certain plots/filters
    last_full_year = max_year
    if max_year is not None and min_year is not None and max_year > min_year:
        last_full_year = max_year - 1
        
    plot_reference_year = last_full_year if last_full_year else max_year # Use last full year for titles etc. if available

    # --- Plot 1: Stacked Area (Total Volumes) ---
    total_volumes_df = analysis_results.get('total_volumes')
    if total_volumes_df is not None and not total_volumes_df.empty \
            and all(c in total_volumes_df.columns for c in [year_col, provider_amount_col, subcontractor_amount_col]):
        try:
            logger.info("Generating Total Volumes Area Chart...")
            # Create a temporary DataFrame with Finnish role names for plotting labels/colors
            plot_df_roles = total_volumes_df.rename(columns={
                provider_amount_col: 'järjestäjänä',
                subcontractor_amount_col: 'hankintana'
            })
            fig, _ = visualizer.create_area_chart( # Capture fig
                data=plot_df_roles,
                x_col=year_col, # Use config year column
                y_cols=['järjestäjänä', 'hankintana'],
                colors=[COLOR_PALETTES["roles"]["järjestäjänä"], COLOR_PALETTES["roles"]["hankintana"]],
                labels=['järjestäjänä', 'hankintana'],
                title=f"{inst_short_name} netto-opiskelijamäärä vuosina {min_year}-{max_year}",
                caption=base_caption,
                stacked=True
            )
            visualizer.save_visualization(fig, f"{inst_short_name}_total_volumes_area") # Save separately
            plt.close(fig) # Close the figure after saving
        except Exception as e:
            logger.error(f"Failed to generate Total Volumes plot: {e}", exc_info=True)
    else:
        logger.warning(f"Skipping Total Volumes plot: Data not available or missing required columns ({year_col}, {provider_amount_col}, {subcontractor_amount_col}).")

    # --- Prepare Detailed Market Data (used by several plots) ---
    # Data is already filtered by MarketAnalyzer.analyze based on min_market_size_threshold
    detailed_df = analysis_results.get('detailed_providers_market')
    if detailed_df is None or detailed_df.empty:
        logger.warning("Skipping several plots: Detailed providers market data not available.")
        return # Exit if detailed data is missing
        
    # --- Determine Active Qualifications (for filtering institution-specific plots) ---
    active_qualifications = []
    if max_year is not None and min_year is not None:
        # Check activity based on last_full_year and the year before it
        years_to_check_activity = [last_full_year]
        prev_full_year = None
        if last_full_year > min_year:
            prev_full_year = last_full_year - 1
            years_to_check_activity.append(prev_full_year)
            
        # Filter detailed data to the institution and the relevant years
        inst_recent_df = detailed_df[
            (detailed_df[provider_col].isin(inst_names)) &
            (detailed_df[year_col].isin(years_to_check_activity))
        ].copy() # Use .copy() to avoid SettingWithCopyWarning later

        # --- Get configuration for active qualification filtering ---
        analysis_config = config.get('analysis', {})
        min_volume_sum_threshold = analysis_config.get('active_qualification_min_volume_sum', 3) # Default to 3 if not set

        # 1. Identify qualifications with sufficient volume over the last two years combined
        volume_grouped = inst_recent_df.groupby(qual_col)[total_volume_col].sum()
        quals_with_recent_volume = set(volume_grouped[volume_grouped >= min_volume_sum_threshold].index)
        logger.debug(f"Found {len(quals_with_recent_volume)} qualifications with summed volume >= {min_volume_sum_threshold} in {years_to_check_activity} for {inst_short_name}.")

        # 2. Identify qualifications where institution had 100% share in BOTH years
        quals_with_100_share_both_years = set()
        # Check only if we have two years of data to compare and share column exists
        if prev_full_year is not None and not inst_recent_df.empty and market_share_col in inst_recent_df.columns:
            # Ensure Market Share is numeric
            inst_recent_df[market_share_col] = pd.to_numeric(inst_recent_df[market_share_col], errors='coerce')
            # Filter for 100% market share (allowing for small floating point inaccuracies)
            inst_100_share_df = inst_recent_df[inst_recent_df[market_share_col].round(2) == 100.0]
            # Count how many years each qualification had 100% share
            share_counts = inst_100_share_df.groupby(qual_col)[year_col].nunique()
            # Keep only those present in both years (count == 2)
            quals_with_100_share_both_years = set(share_counts[share_counts == 2].index)
            if quals_with_100_share_both_years:
                logger.debug(f"Found {len(quals_with_100_share_both_years)} qualifications with 100% share in both {prev_full_year} and {last_full_year}: {quals_with_100_share_both_years}")
        
        # 3. Final active qualifications: Volume > threshold MINUS 100% share in both years
        active_qualifications_set = quals_with_recent_volume - quals_with_100_share_both_years
        active_qualifications = sorted(list(active_qualifications_set)) # Convert back to sorted list
        
        logger.info(f"Determined {len(active_qualifications)} active qualifications for plots (Summed Volume >= {min_volume_sum_threshold} in {years_to_check_activity} AND <100% share in both years if applicable)." )
    else:
        logger.warning("Could not determine active qualifications due to missing year data.")
        # Fallback: use all qualifications the institution ever offered in the filtered data
        active_qualifications = detailed_df[detailed_df[provider_col].isin(inst_names)][qual_col].unique().tolist()

    # --- Plot 2: Line Chart (Market Share Evolution) - Loop per Qualification ---
    logger.info(f"Generating Market Share Line Charts for {len(active_qualifications)} active qualifications...")
    for qual in active_qualifications:
        try:
            qual_df = detailed_df[detailed_df[qual_col] == qual]
            if qual_df.empty:
                continue
                
            # Find top M providers in the latest year for this qualification
            latest_qual_providers = qual_df[qual_df[year_col] == plot_reference_year]
            top_m_providers = latest_qual_providers.nlargest(6, market_share_col)[provider_col].tolist()
            
            # Pivot data for plotting
            plot_data = qual_df[qual_df[provider_col].isin(top_m_providers)].pivot(
                index=year_col, columns=provider_col, values=market_share_col
            )
            plot_data.index.name = year_col # Ensure index has a name for the function
            
            if not plot_data.empty:
                qual_filename_part = qual.replace(' ', '_').replace('/', '_').lower()[:30] # Create safe filename part
                fig, _ = visualizer.create_line_chart( # Capture fig
                    data=plot_data,
                    x_col=plot_data.index,
                    y_cols=top_m_providers,
                    colors=COLOR_PALETTES["main"],
                    labels=top_m_providers,
                    title=f"{qual}: Markkinaosuus (%)",
                    caption=base_caption,
                    markers=True
                )
                visualizer.save_visualization(fig, f"{inst_short_name}_{qual_filename_part}_market_share_lines") # Save separately
                plt.close(fig) # Close the figure after saving
        except Exception as e:
            logger.error(f"Failed to generate Market Share line plot for {qual}: {e}", exc_info=True)

    # --- Plot 3: Heatmap (Institution's Share) ---
    try:
        logger.info("Generating Institution Market Share Heatmap...")
        inst_share_df_raw = detailed_df[detailed_df[provider_col].isin(inst_names)]
        
        # --- Aggregate data across institution variants ---
        if not inst_share_df_raw.empty:
            # Define aggregation logic using config column names
            agg_logic = {
                provider_amount_col: 'sum',
                subcontractor_amount_col: 'sum',
                total_volume_col: 'sum',
                market_total_col: 'first', # Should be the same for all rows in the group
                # Keep other potentially relevant columns (take the first instance)
                provider_col: 'first', # Keep one provider name for reference
                market_share_col: 'first', # Placeholder, will recalculate
                market_rank_col: 'first', # Placeholder, maybe min() is better?
                market_share_growth_col: 'first', # Placeholder
                market_gainer_rank_col: 'first' # Placeholder
            }
            # Filter agg_logic to columns actually present in the dataframe
            agg_logic = {k: v for k, v in agg_logic.items() if k in inst_share_df_raw.columns}

            logger.debug(f"Aggregating data for {inst_short_name} across variants...")
            inst_share_df_agg = inst_share_df_raw.groupby([year_col, qual_col], as_index=False).agg(agg_logic)
            
            # Recalculate market share after aggregation
            if total_volume_col in inst_share_df_agg.columns and market_total_col in inst_share_df_agg.columns:
                # Avoid division by zero or NaN Market Total
                valid_market_total = inst_share_df_agg[market_total_col] > 0
                inst_share_df_agg[market_share_col] = 0.0 # Initialize column
                inst_share_df_agg.loc[valid_market_total, market_share_col] = \
                    (inst_share_df_agg.loc[valid_market_total, total_volume_col] / inst_share_df_agg.loc[valid_market_total, market_total_col] * 100)
            
            inst_share_df = inst_share_df_agg # Use the aggregated dataframe
            logger.debug(f"Aggregation complete. Shape before: {inst_share_df_raw.shape}, after: {inst_share_df.shape}")
            
        else:
            inst_share_df = inst_share_df_raw # Use the empty dataframe if no data found

        # --- End Aggregation ---

        # Check for duplicates before pivoting (SHOULD NOT HAPPEN after aggregation)
        # duplicates = inst_share_df.duplicated(subset=[qual_col, year_col], keep=False)
        # if duplicates.any():
        #     logger.warning(f"Found duplicate {qual_col}/{year_col} entries for {inst_short_name} AFTER AGGREGATION. This should not happen. Dropping duplicates.")
        #     logger.debug(f"Duplicate rows:\n{inst_share_df[duplicates]}")
        #     inst_share_df = inst_share_df.drop_duplicates(subset=[qual_col, year_col], keep='first')
            
        # Filter heatmap data to active qualifications
        if not inst_share_df.empty:
            inst_share_df_active = inst_share_df[inst_share_df[qual_col].isin(active_qualifications)]
            logger.info(f"Filtered heatmap data to {len(active_qualifications)} active qualifications. Shape before: {inst_share_df.shape}, after: {inst_share_df_active.shape}")
        else:
            inst_share_df_active = inst_share_df # Keep empty dataframe if no data
            
        if not inst_share_df_active.empty and market_share_col in inst_share_df_active.columns:
             try:
                 heatmap_data = inst_share_df_active.pivot_table(
                     index=qual_col, columns=year_col, values=market_share_col
                 )
                 # Sort heatmap rows alphabetically by qualification
                 heatmap_data = heatmap_data.sort_index()
             
                 if not heatmap_data.empty:
                     fig, _ = visualizer.create_heatmap( # Capture fig
                         data=heatmap_data,
                         title=f"{inst_short_name} markkinaosuus (%) aktiivisissa tutkinnoissa",
                         caption=base_caption,
                         cmap="Greens",
                         annot=True,
                         fmt=".1f"
                     )
                     visualizer.save_visualization(fig, f"{inst_short_name}_market_share_heatmap_active") # Save separately
                     plt.close(fig) # Close the figure after saving
                 else:
                      logger.warning(f"Skipping Market Share Heatmap: Pivoted data is empty for active qualifications.")
             except Exception as pivot_err:
                  logger.error(f"Error pivoting data for heatmap: {pivot_err}", exc_info=True)
        else:
            logger.warning(f"Skipping Market Share Heatmap: No data available for {inst_short_name} in active qualifications or missing column '{market_share_col}'.")
            
    except Exception as e:
        logger.error(f"Failed to generate Institution Market Share Heatmap: {e}", exc_info=True)

    # --- Plot 4: BCG Matrix (Market Share vs. Growth) ---
    logger.info("Generating BCG Growth-Share Matrix...")
    bcg_data_df = analysis_results.get('bcg_data') # Get the calculated BCG data
    
    if bcg_data_df is not None and not bcg_data_df.empty:
        try:
            # Define column names expected by create_bcg_matrix based on _calculate_bcg_data output
            bcg_growth_col = 'Market Growth (%)'
            bcg_share_col = 'Relative Market Share'
            bcg_size_col = 'Institution Volume'
            # Use the standard qualification column name from config
            bcg_qual_col = qual_col 
            
            # Check if required columns exist (already checked in analyzer, but double-check is safe)
            required_bcg_cols = [bcg_qual_col, bcg_growth_col, bcg_share_col, bcg_size_col]
            if all(c in bcg_data_df.columns for c in required_bcg_cols):
                plot_title = f"{inst_short_name}: Tutkintojen kasvu vs. markkinaosuus ({max_year})"
                # Construct caption
                bcg_caption = base_caption + f" Kuplan koko = {inst_short_name} volyymi. Suhteellinen markkinaosuus = {inst_short_name} osuus / Suurimman kilpailijan osuus."

                fig, _ = visualizer.create_bcg_matrix(
                    data=bcg_data_df,
                    growth_col=bcg_growth_col,
                    share_col=bcg_share_col,
                    size_col=bcg_size_col,
                    label_col=bcg_qual_col,
                    title=plot_title,
                    caption=bcg_caption,
                    # avg_growth=None, # Use default (mean) or provide a specific value
                    # share_threshold=1.0 # Use default
                )
                visualizer.save_visualization(fig, f"{inst_short_name}_bcg_matrix")
                plt.close(fig)
            else:
                logger.warning(f"Skipping BCG Matrix plot: Missing one or more required columns in bcg_data: {required_bcg_cols}. Found: {bcg_data_df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Failed to generate BCG Matrix plot: {e}", exc_info=True)
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                plt.close(fig)
    else:
        logger.warning("Skipping BCG Matrix plot: bcg_data not available or empty in analysis results.")

    # --- Plot 5: Combined Volume / Provider Count Plot ---
    logger.info("Generating Volume / Provider Count Plot...")
    volume_df = analysis_results.get('total_volumes') # Overall institution volume
    count_df = analysis_results.get('provider_counts_by_year') # Provider counts for institution's qualifications

    if volume_df is not None and not volume_df.empty and count_df is not None and not count_df.empty:
        logger.debug("Entering Volume/Provider Count plotting block.")
        try:
            # Get necessary column names from config (use output section)
            year_col_name = config['columns']['output']['year']
            vol_provider_col_name = config['columns']['output']['provider_amount']
            vol_subcontractor_col_name = config['columns']['output']['subcontractor_amount']
            # Use the same keys used in MarketAnalyzer when creating provider_counts_by_year
            count_provider_col_name = 'Unique_Providers_Count' # As defined in MarketAnalyzer
            count_subcontractor_col_name = 'Unique_Subcontractors_Count' # As defined in MarketAnalyzer
            
            plot_title = f"{inst_short_name}: Opiskelijamäärät ja kouluttajamarkkina ({min_year}-{max_year})"
            volume_subplot_title = "Netto-opiskelijamäärä"
            count_subplot_title = "Uniikit kouluttajat markkinassa"
            plot_caption = base_caption + f". Kouluttajamäärä perustuu tutkintoihin, joita {inst_short_name} tarjoaa."

            fig, _ = visualizer.create_volume_and_provider_count_plot(
                volume_data=volume_df,
                count_data=count_df,
                title=plot_title,
                volume_title=volume_subplot_title,
                count_title=count_subplot_title,
                year_col=year_col_name,
                vol_provider_col=vol_provider_col_name,
                vol_subcontractor_col=vol_subcontractor_col_name,
                count_provider_col=count_provider_col_name,
                count_subcontractor_col=count_subcontractor_col_name,
                caption=plot_caption
            )
            visualizer.save_visualization(fig, f"{inst_short_name}_volume_provider_counts")
            plt.close(fig)
        except KeyError as ke:
             logger.error(f"Failed to generate Volume/Provider Count plot due to missing column key: {ke}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to generate Volume/Provider Count plot: {e}", exc_info=True)
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                plt.close(fig)
    else:
        logger.warning("Skipping Volume/Provider Count plot: Required volume or count data is missing.")

    logger.info("Visualization generation finished.")

# --- Refactored Orchestrator Function ---

def run_analysis_workflow(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates the full analysis workflow: data prep, analysis, export, viz.

    Args:
        args: Dictionary of arguments from command-line parsing.

    Returns:
        Dictionary containing analysis results DataFrames and Excel path.
            Keys: 'analysis_results', 'excel_path'
    """
    logger.info("Starting analysis workflow...")
    
    # Step 1: Load configuration (can be done once here)
    logger.info("Loading configuration...")
    config = get_config()
    
    # --- Optional Logging Level Override ---
    # ... (logging override code if needed) ...
    # ---

    # Step 2: Prepare Data
    try:
        df_clean, institution_key, institution_variants, institution_short_name, data_update_date_str, filter_qual_types = prepare_analysis_data(config, args)
    except Exception as prep_err:
        logger.error(f"Data preparation failed: {prep_err}", exc_info=True)
        # Return minimal results indicating failure
        return {"analysis_results": {}, "excel_path": None} 

    # Step 3: Perform Analysis
    try:
        analysis_results, analyzer = perform_market_analysis(
            df_clean, config, institution_variants, institution_short_name, filter_qual_types
        )
    except Exception as analysis_err:
        logger.error(f"Market analysis execution failed: {analysis_err}", exc_info=True)
        # Return minimal results indicating failure, similar to data prep failure
        return {"analysis_results": {}, "excel_path": None}

    # Check if analysis produced meaningful results before proceeding
    # A simple check could be if the main DataFrame is not empty
    if analysis_results.get("detailed_providers_market", pd.DataFrame()).empty:
         logger.warning("Analysis resulted in empty 'detailed_providers_market' data. Skipping export and visualization.")
         return {"analysis_results": analysis_results, "excel_path": None}
         
    # --- Create Metadata ---
    logger.info("Creating metadata for export...")
    metadata = {
        "Analysis Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Institution Analyzed": f"{config['institutions'][institution_key]['name']} ({institution_short_name})",
        "Institution Variants Used": ", ".join(institution_variants),
        "Input Data File": args.get('data_file', config.get('paths',{}).get('data', 'N/A')),
        "Data Update Date": data_update_date_str,
        "Qualification Type Filter Applied": "Yes" if filter_qual_types else "No",
        "Min Market Size Threshold": config.get('analysis', {}).get('min_market_size_threshold', 'N/A')
    }
    metadata_df = pd.DataFrame(metadata.items(), columns=["Parameter", "Value"])
    # --- End Create Metadata ---
         
    # Step 4: Export Results to Excel
    # Explicitly handle None from args.get('output_dir')
    output_dir_arg = args.get('output_dir')
    if output_dir_arg is not None:
        base_output_path = output_dir_arg
        logger.debug(f"Using output directory from command line args: {base_output_path}")
    else:
        # Fallback to config or hardcoded default
        base_output_path = config.get('paths', {}).get('output', 'data/reports')
        logger.debug(f"Using output directory from config/default: {base_output_path}")

    # Remove the previous debug log
    # logger.debug(f"[Workflow] base_output_path before export: '{base_output_path}' (Type: {type(base_output_path)})")
    excel_path = export_analysis_results(
        analysis_results, 
        config, 
        institution_short_name, 
        base_output_path,
        metadata_df=metadata_df # Pass the created metadata DataFrame
    )

    # Step 5: Generate Visualizations
    try:
        logger.info("Initializing visualizer...")
        # Determine the full output path for visualizations (needed for Visualizer init)
        # Use the same base_output_path determined above
        dir_name = f"education_market_{institution_short_name.lower()}"
        # Remove the previous debug log
        # logger.debug(f"[Workflow] base_output_path before viz path creation: '{base_output_path}' (Type: {type(base_output_path)})")
        full_output_dir_path = Path(base_output_path) / dir_name
        full_output_dir_path.mkdir(parents=True, exist_ok=True) # Ensure dir exists
        
        visualizer = EducationVisualizer(
            output_dir=full_output_dir_path, 
            output_format=args.get('plot_format', 'pdf'), # Use arg or default
            institution_short_name=institution_short_name,
            include_timestamp=True # Match Excel timestamping logic
        )
        
        generate_visualizations(
            analysis_results=analysis_results, 
            visualizer=visualizer, 
            analyzer=analyzer, # Pass the analyzer instance
            config=config, 
            data_update_date_str=data_update_date_str 
        )
    except Exception as vis_error:
        logger.error(f"Error during visualization generation: {vis_error}", exc_info=True)
        # Continue workflow even if visualizations fail
    finally:
        # Ensure PDF is closed if visualizer was created
        if 'visualizer' in locals() and visualizer is not None:
            visualizer.close_pdf()

    logger.info("Analysis workflow completed.")
    
    # Return final results
    return {
        "analysis_results": analysis_results, 
        "excel_path": excel_path
    }

# --- Main CLI Entry Point ---

def main():
    """Main entry point for the CLI."""
    try:
        # Parse arguments first
        parsed_args = parse_arguments(sys.argv[1:]) # Pass actual args
        args_dict = vars(parsed_args)
        # --- Add Debug Log ---
        logger.debug(f"[Main] args_dict passed to workflow: {args_dict}")
        
        # Run the refactored workflow orchestrator
        results = run_analysis_workflow(args_dict)
        
        # Optionally log or use results here
        if results.get("excel_path"):
            logger.info(f"Analysis report generated: {results['excel_path']}")
        else:
            logger.warning("Analysis completed, but no Excel report was generated (possibly due to errors or empty results).")
            
        return 0 # Success
        
    except Exception as e:
        logger.error(f"Analysis workflow failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1 # Failure

if __name__ == "__main__":
    sys.exit(main()) 