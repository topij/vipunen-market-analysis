#!/usr/bin/env python
"""
Jupyter Notebook-Style Script for Vipunen Education Market Analysis.

This script replicates the analysis workflow from the Vipunen project,
displaying intermediate results and plots in a way suitable for notebooks
or interactive environments, before exporting the final results.
"""

import logging
import pandas as pd
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import sys # Keep sys for potential path manipulation if needed later

# --- Project Module Imports ---
# Assuming the script is run from the project root or the environment includes the src path
try:
    from src.vipunen.config.config_loader import get_config
    from src.vipunen.data.data_loader import load_data
    from src.vipunen.data.data_processor import clean_and_prepare_data
    from src.vipunen.export.excel_exporter import export_to_excel
    from src.vipunen.analysis.market_analyzer import MarketAnalyzer
    # Import the wrapper function from analyze_cli
    from src.vipunen.cli.analyze_cli import export_analysis_results
    # Import Visualizer and constants if needed later
    from src.vipunen.visualization.education_visualizer import EducationVisualizer, COLOR_PALETTES, TEXT_CONSTANTS
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure the script is run from the project root or 'src' is in the Python path.")
    sys.exit(1)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, # Use INFO level for notebook clarity, DEBUG can be verbose
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # Force logging to stdout for notebook-like output
    stream=sys.stdout 
)
# Silence overly verbose loggers
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("FileUtils").setLevel(logging.WARNING) 

logger = logging.getLogger("AnalysisNotebook")

# --- Analysis Configuration (Replaces Command-Line Args) ---
logger.info("--- Setting Analysis Configuration ---")

# Load main project config
try:
    config = get_config() # Assumes config.yaml is discoverable
    logger.info("Successfully loaded project configuration.")
except Exception as e:
    logger.error(f"Failed to load project configuration: {e}. Exiting.")
    sys.exit(1)

# Define analysis parameters (equivalent to args parsed in analyze_cli.py)
# These can be modified by the user for different analyses
ANALYSIS_PARAMS = {
    'data_file': None, # Set to None to use path from config, or provide a specific path string
    'institution': None, # Set to None to use default from config, or provide an institution key (e.g., 'careeria')
    'short_name': None, # Set to None to use default from config based on institution
    'use_dummy': False, # Set to True to use dummy data if available
    'filter_qual_types': False, # Set to True to filter for specific qualification types (e.g., Ammatti/Erikoisammattitutkinto)
    'output_dir': None, # Set to None to use path from config, or provide specific base dir for outputs
    'include_timestamp': True # Whether to include timestamp in output filenames
    # Add other parameters as needed, mirroring analyze_cli args
}

logger.info(f"Using Analysis Parameters: {ANALYSIS_PARAMS}")

# --- Main Analysis Workflow ---
logger.info("--- Starting Analysis Workflow ---")

# === Step 1: Prepare Data (Load, Clean, Filter) ===
logger.info("--- Step 1: Preparing Analysis Data ---")

try:
    # Resolve parameters from ANALYSIS_PARAMS and config
    data_file_path = ANALYSIS_PARAMS.get('data_file') or config['paths']['data']
    use_dummy = ANALYSIS_PARAMS.get('use_dummy', False)
    filter_qual_types_flag = ANALYSIS_PARAMS.get('filter_qual_types', False)

    # Determine institution key and variants
    default_institution_name = config['institutions']['default']['name']
    arg_institution_value = ANALYSIS_PARAMS.get('institution')

    if arg_institution_value is None or arg_institution_value == default_institution_name:
        institution_key = 'default'
    else:
        institution_key = arg_institution_value
        if institution_key not in config['institutions']:
            raise KeyError(f"Institution key '{institution_key}' from ANALYSIS_PARAMS not found in config.")

    institution_config = config['institutions'][institution_key]
    institution_name = institution_config['name']
    institution_short_name = ANALYSIS_PARAMS.get('short_name') or institution_config['short_name']
    
    # Get variants (handle potential absence in config gracefully)
    institution_variants = list(institution_config.get('variants', []))
    if institution_name not in institution_variants:
        institution_variants.append(institution_name)
        
    logger.info(f"Analyzing Institution: {institution_name} (Key: {institution_key}, Short: {institution_short_name})")
    logger.info(f"Using Institution Variants: {institution_variants}")
    logger.info(f"Data Source: {data_file_path}")

    # --- 1a: Load Raw Data ---
    logger.info("Loading raw data...")
    df_raw = load_data(file_path=data_file_path, use_dummy=use_dummy)
    logger.info(f"Loaded {len(df_raw)} rows of raw data.")
    print("\n--- Raw Data Sample (First 5 Rows) ---")
    print(df_raw.head())
    print("\n--- Raw Data Info ---")
    df_raw.info(verbose=True, show_counts=True)

    # --- 1b: Extract Data Update Date ---
    data_update_date_str = datetime.datetime.now().strftime("%d.%m.%Y") # Default
    update_date_col = config.get('columns', {}).get('input', {}).get('update_date', 'tietojoukkoPaivitettyPvm')
    if not df_raw.empty and update_date_col in df_raw.columns:
        try:
            raw_date_str = str(df_raw[update_date_col].iloc[0])
            parsed_date = pd.to_datetime(raw_date_str)
            data_update_date_str = parsed_date.strftime("%d.%m.%Y")
            logger.info(f"Extracted data update date: {data_update_date_str}")
        except Exception as date_err:
            logger.warning(f"Could not parse date from column '{update_date_col}': {date_err}. Using current date.")
    else:
        logger.warning(f"Update date column '{update_date_col}' not found or data empty. Using current date.")

    # --- 1c: Clean and Prepare Data (Initial) ---
    logger.info("Cleaning and preparing data (merging qualifications, shortening names)...")
    df_clean_initial = clean_and_prepare_data(
        df_raw,
        institution_names=institution_variants, # Pass variants here for potential use in cleaning
        merge_qualifications=True,
        shorten_names=True
    )
    logger.info(f"Initial cleaning complete. Shape: {df_clean_initial.shape}")
    print("\n--- Cleaned Data Sample (Initial) ---")
    print(df_clean_initial.head())
    print("\n--- Cleaned Data Info (Initial) ---")
    df_clean_initial.info(verbose=True, show_counts=True)

    # --- 1d: Filter by Institution's Offered Qualifications ---
    logger.info(f"Filtering data based on qualifications offered by {institution_short_name}...")
    input_cols = config['columns']['input']
    # Ensure required columns exist before filtering
    required_filter_cols = [input_cols['provider'], input_cols['subcontractor'], input_cols['qualification']]
    if not all(col in df_clean_initial.columns for col in required_filter_cols):
        raise ValueError(f"Missing one or more required columns for filtering: {required_filter_cols}")
        
    institution_mask = (
        (df_clean_initial[input_cols['provider']].isin(institution_variants)) |
        (df_clean_initial[input_cols['subcontractor']].isin(institution_variants))
    )
    inst_qualifications = df_clean_initial.loc[institution_mask, input_cols['qualification']].unique()

    if len(inst_qualifications) > 0:
        df_prepared = df_clean_initial[df_clean_initial[input_cols['qualification']].isin(inst_qualifications)].copy()
        logger.info(f"Filtered data to {len(inst_qualifications)} qualifications offered by {institution_short_name}. Final shape: {df_prepared.shape}")
    else:
        logger.warning(f"No specific qualifications found for {institution_short_name} based on variants {institution_variants}. Using data before institution qualification filtering.")
        df_prepared = df_clean_initial # Use the data before this specific filtering step

    print("\n--- Prepared Data Sample (Final for Analysis) ---")
    print(df_prepared.head())
    print("\n--- Prepared Data Info (Final for Analysis) ---")
    df_prepared.info(verbose=True, show_counts=True)
    
    logger.info("--- Step 1: Data Preparation Complete ---")

except KeyError as e:
    logger.error(f"Configuration Error during data preparation: Missing key {e}")
    sys.exit(1)
except FileNotFoundError as e:
    logger.error(f"Data File Error: {e}")
    sys.exit(1)
except ValueError as e:
    logger.error(f"Data Error during preparation: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"An unexpected error occurred during data preparation: {e}", exc_info=True)
    sys.exit(1)

# === Step 2: Perform Market Analysis ===
logger.info("--- Step 2: Performing Market Analysis ---")

try:
    # Ensure data from Step 1 is available
    if 'df_prepared' not in locals() or df_prepared is None or df_prepared.empty:
        logger.warning("Prepared data (df_prepared) is missing or empty. Skipping Market Analysis.")
        analysis_results = {} # Initialize as empty
        analyzer = None # Initialize as None
    else:
        # --- 2a: Apply Optional Qualification Type Filtering ---
        if filter_qual_types_flag:
            qual_types = config.get('qualification_types', ['Ammattitutkinnot', 'Erikoisammattitutkinnot']) # Default types if not in config
            logger.info(f"Applying qualification type filter for: {qual_types}")
            df_analysis_input = df_prepared[df_prepared['tutkintotyyppi'].isin(qual_types)].copy()
            logger.info(f"Shape before type filter: {df_prepared.shape}, after: {df_analysis_input.shape}")
        else:
            logger.info("Skipping qualification type filtering.")
            df_analysis_input = df_prepared.copy() # Use the prepared data directly

        # --- 2b: Initialize and Run Analyzer ---
        logger.info("Initializing MarketAnalyzer...")
        analyzer = MarketAnalyzer(df_analysis_input, cfg=config)
        analyzer.institution_names = institution_variants
        analyzer.institution_short_name = institution_short_name
        logger.info(f"Analyzer configured for: {institution_short_name} ({institution_variants})")

        logger.info("Running analysis...")
        analysis_results = analyzer.analyze() # This returns a dict of DataFrames
        logger.info(f"Analysis complete. Results keys: {list(analysis_results.keys())}")

        # --- 2c: Display Sample Results ---
        # Print head of key results DataFrames for notebook-like inspection
        print("\n--- Analysis Results Samples ---")
        for key, df_result in analysis_results.items():
            if isinstance(df_result, pd.DataFrame) and not df_result.empty:
                print(f"\n--- Result: {key} (Top 5 rows) ---")
                print(df_result.head())
            elif isinstance(df_result, pd.DataFrame) and df_result.empty:
                 print(f"\n--- Result: {key} (DataFrame is empty) ---")
            else:
                # Handle non-DataFrame results if any (e.g., scalars, lists)
                 print(f"\n--- Result: {key} (Type: {type(df_result)}) ---")
                 print(df_result)
                 
        logger.info("--- Step 2: Market Analysis Complete ---")

except Exception as e:
    logger.error(f"An unexpected error occurred during market analysis: {e}", exc_info=True)
    analysis_results = {} # Ensure it exists but is empty on error
    analyzer = None # Ensure it's None on error
    sys.exit(1) # Or handle more gracefully depending on desired flow

# === Step 3: Generate Visualizations ===
logger.info("--- Step 3: Generating Visualizations ---")

# Define directory for saving plots --> Will use main output dir for PDF
# plots_output_dir = None 
# figures = {} # Initialize dictionary to store figure paths --> Not needed for PDF
pdf_report_path = None # Store path to the generated PDF

try:
    # Ensure results from Step 2 are available
    if 'analysis_results' not in locals() or not analysis_results or 'analyzer' not in locals() or analyzer is None:
        logger.warning("Analysis results or analyzer instance missing. Skipping Visualizations.")
    else:
        # Determine base output directory
        base_output_path_str = ANALYSIS_PARAMS.get('output_dir') or config.get('paths', {}).get('output', 'output') # Default to 'output'
        base_output_path = Path(base_output_path_str)

        # Create the specific subdirectory for the institution's report (Excel and PDF)
        # This logic might be duplicated in export_analysis_results, but ensures dir exists for Visualizer
        report_dir_name = f"education_market_{institution_short_name.lower()}"
        report_output_dir = base_output_path / report_dir_name
        report_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PDF report will be saved in directory: {report_output_dir}")

        # Initialize Visualizer - For PDF Output
        logger.info("Initializing EducationVisualizer for PDF output...")
        visualizer = EducationVisualizer(
            style="default",
            output_dir=report_output_dir, # Use the main report dir
            output_format='pdf', # Generate PDF
            institution_short_name=institution_short_name,
            include_timestamp=ANALYSIS_PARAMS.get('include_timestamp', True) # Match Excel timestamping
        )
        visualizer.data_update_date = data_update_date_str

        logger.info("Generating plots and adding them to PDF...")

        # --- Get Config Column Names (using defaults as fallback) ---
        cols_out = config.get('columns', {}).get('output', {})
        year_col = cols_out.get('year', 'Vuosi') # Assuming Vuosi is common
        qual_col = cols_out.get('qualification', 'Tutkinto')
        provider_col = cols_out.get('provider', 'Oppilaitos') # Assuming Oppilaitos
        provider_amount_col = cols_out.get('provider_amount', 'NOM järjestäjänä')
        subcontractor_amount_col = cols_out.get('subcontractor_amount', 'NOM hankintana')
        total_volume_col = cols_out.get('total_volume', 'NOM yhteensä')
        market_total_col = cols_out.get('market_total', 'Markkina yhteensä')
        market_share_col = cols_out.get('market_share', 'Markkinaosuus (%)')
        # BCG specific columns (defined in MarketAnalyzer._calculate_bcg_data)
        bcg_growth_col = 'Market Growth (%)'
        bcg_share_col = 'Relative Market Share'
        bcg_size_col = 'Institution Volume'
        # Provider count specific columns (defined in MarketAnalyzer._calculate_provider_counts)
        count_provider_col = 'Unique_Providers_Count'
        count_subcontractor_col = 'Unique_Subcontractors_Count'

        # --- Get other needed info ---
        inst_short_name = analyzer.institution_short_name
        inst_names = analyzer.institution_names
        min_year = analyzer.min_year
        max_year = analyzer.max_year
        base_caption = TEXT_CONSTANTS["data_source"].format(date=data_update_date_str)
        last_full_year = max_year - 1 if max_year and min_year and max_year > min_year else max_year
        plot_reference_year = last_full_year if last_full_year else max_year

        # --- Plot 1: Stacked Area (Total Volumes) ---
        total_volumes_df = analysis_results.get('total_volumes')
        if total_volumes_df is not None and not total_volumes_df.empty and all(c in total_volumes_df.columns for c in [year_col, provider_amount_col, subcontractor_amount_col]):
            try:
                logger.info("Generating Total Volumes Area Chart...")
                plot_df_roles = total_volumes_df.rename(columns={provider_amount_col: 'järjestäjänä', subcontractor_amount_col: 'hankintana'})
                fig, _ = visualizer.create_area_chart(
                    data=plot_df_roles, x_col=year_col, y_cols=['järjestäjänä', 'hankintana'],
                    colors=[COLOR_PALETTES["roles"]["järjestäjänä"], COLOR_PALETTES["roles"]["hankintana"]],
                    labels=['järjestäjänä', 'hankintana'], title=f"{inst_short_name} netto-opiskelijamäärä vuosina {min_year}-{max_year}",
                    caption=base_caption, stacked=True
                )
                # Add figure to PDF
                visualizer.save_visualization(fig, f"{inst_short_name}_total_volumes_area")
            except Exception as e:
                logger.error(f"Failed to generate Total Volumes plot: {e}", exc_info=True)
                if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        else: logger.warning(f"Skipping Total Volumes plot: Data not available or missing columns.")

        # --- Determine Active Qualifications (logic from analyze_cli.py) ---
        detailed_df = analysis_results.get('detailed_providers_market')
        active_qualifications = []
        if detailed_df is not None and not detailed_df.empty and max_year is not None and min_year is not None:
            years_to_check_activity = [last_full_year]
            prev_full_year = last_full_year - 1 if last_full_year > min_year else None
            if prev_full_year: years_to_check_activity.append(prev_full_year)

            inst_recent_df = detailed_df[(detailed_df[provider_col].isin(inst_names)) & (detailed_df[year_col].isin(years_to_check_activity))].copy()
            analysis_config = config.get('analysis', {})
            min_volume_sum_threshold = analysis_config.get('active_qualification_min_volume_sum', 3)
            quals_with_recent_volume = set()
            if total_volume_col in inst_recent_df.columns:
                volume_grouped = inst_recent_df.groupby(qual_col)[total_volume_col].sum()
                quals_with_recent_volume = set(volume_grouped[volume_grouped >= min_volume_sum_threshold].index)
            
            quals_with_100_share_both_years = set()
            if prev_full_year is not None and not inst_recent_df.empty and market_share_col in inst_recent_df.columns:
                inst_recent_df[market_share_col] = pd.to_numeric(inst_recent_df[market_share_col], errors='coerce')
                inst_100_share_df = inst_recent_df[inst_recent_df[market_share_col].round(2) == 100.0]
                share_counts = inst_100_share_df.groupby(qual_col)[year_col].nunique()
                quals_with_100_share_both_years = set(share_counts[share_counts == 2].index)
                
            active_qualifications_set = quals_with_recent_volume - quals_with_100_share_both_years
            active_qualifications = sorted(list(active_qualifications_set))
            logger.info(f"Determined {len(active_qualifications)} active qualifications for plots.")
        else:
            logger.warning("Could not determine active qualifications. Using fallback or skipping related plots.")
            if detailed_df is not None and not detailed_df.empty and provider_col in detailed_df.columns and qual_col in detailed_df.columns:
                 active_qualifications = detailed_df[detailed_df[provider_col].isin(inst_names)][qual_col].unique().tolist()
                 logger.info(f"Using fallback: {len(active_qualifications)} qualifications institution ever offered.")

        # --- Plot 2: Line Chart (Market Share Evolution) - Loop per Qualification ---
        if detailed_df is not None and not detailed_df.empty:
            logger.info(f"Generating Market Share Line Charts for {len(active_qualifications)} active qualifications...")
            for qual in active_qualifications:
                try:
                    qual_df = detailed_df[detailed_df[qual_col] == qual]
                    if qual_df.empty: continue
                    latest_qual_providers = qual_df[qual_df[year_col] == plot_reference_year]
                    top_m_providers = latest_qual_providers.nlargest(6, market_share_col)[provider_col].tolist()
                    plot_data = qual_df[qual_df[provider_col].isin(top_m_providers)].pivot(index=year_col, columns=provider_col, values=market_share_col)
                    if not plot_data.empty:
                        plot_data.index.name = year_col # Ensure index name
                        qual_filename_part = qual.replace(' ', '_').replace('/', '_').replace(':', '_').replace(',', '').replace('.', '').lower()[:50] # More robust filename part
                        fig, _ = visualizer.create_line_chart(
                            data=plot_data, x_col=plot_data.index, y_cols=top_m_providers,
                            colors=COLOR_PALETTES["main"], labels=top_m_providers,
                            title=f"{qual}: Markkinaosuus (%)", caption=base_caption, markers=True
                        )
                        # Add figure to PDF
                        visualizer.save_visualization(fig, f"{inst_short_name}_{qual_filename_part}_market_share_lines")
                except Exception as e:
                    logger.error(f"Failed to generate Market Share line plot for {qual}: {e}", exc_info=True)
                    if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        else: logger.warning("Skipping Market Share Line Charts: Detailed market data missing.")

        # --- Plot 3: Heatmap (Institution's Share) ---
        if detailed_df is not None and not detailed_df.empty:
            try:
                logger.info("Generating Institution Market Share Heatmap...")
                inst_share_df_raw = detailed_df[detailed_df[provider_col].isin(inst_names)].copy() # Use .copy()
                inst_share_df = inst_share_df_raw # Start with raw, aggregate if multiple variants
                
                if len(inst_names) > 1 and not inst_share_df_raw.empty:
                     # Define aggregation logic using config column names
                    agg_logic = {
                        provider_amount_col: 'sum', subcontractor_amount_col: 'sum',
                        total_volume_col: 'sum', market_total_col: 'first',
                        market_share_col: 'first', # Placeholder, recalculate
                        # Add other columns if needed, e.g., 'first' for rank
                    }
                    agg_logic = {k: v for k, v in agg_logic.items() if k in inst_share_df_raw.columns}
                    if agg_logic:
                        inst_share_df_agg = inst_share_df_raw.groupby([year_col, qual_col], as_index=False).agg(agg_logic)
                        # Recalculate market share
                        if total_volume_col in inst_share_df_agg.columns and market_total_col in inst_share_df_agg.columns:
                            valid_market_total = inst_share_df_agg[market_total_col] > 0
                            inst_share_df_agg[market_share_col] = 0.0
                            inst_share_df_agg.loc[valid_market_total, market_share_col] = (inst_share_df_agg.loc[valid_market_total, total_volume_col] / inst_share_df_agg.loc[valid_market_total, market_total_col] * 100)
                        inst_share_df = inst_share_df_agg
                    else:
                         logger.warning("Aggregation skipped for heatmap: No relevant columns found for aggregation.")

                if not inst_share_df.empty:
                    inst_share_df_active = inst_share_df[inst_share_df[qual_col].isin(active_qualifications)]
                else: inst_share_df_active = pd.DataFrame() # Ensure it's an empty df
                
                if not inst_share_df_active.empty and market_share_col in inst_share_df_active.columns:
                    heatmap_data = inst_share_df_active.pivot_table(index=qual_col, columns=year_col, values=market_share_col)
                    heatmap_data = heatmap_data.sort_index() # Sort rows
                    if not heatmap_data.empty:
                        fig, _ = visualizer.create_heatmap(
                            data=heatmap_data, title=f"{inst_short_name} markkinaosuus (%) aktiivisissa tutkinnoissa",
                            caption=base_caption, cmap="Greens", annot=True, fmt=".1f"
                        )
                        # Add figure to PDF
                        visualizer.save_visualization(fig, f"{inst_short_name}_market_share_heatmap_active")
                    else: logger.warning(f"Skipping Heatmap: Pivoted data is empty for active qualifications.")
                else: logger.warning(f"Skipping Heatmap: No data for {inst_short_name} in active qualifications or missing market share column.")
            except Exception as e:
                logger.error(f"Failed to generate Market Share Heatmap: {e}", exc_info=True)
                if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        else: logger.warning("Skipping Heatmap: Detailed market data missing.")

        # --- Plot 4: BCG Matrix ---
        bcg_data_df = analysis_results.get('bcg_data')
        if bcg_data_df is not None and not bcg_data_df.empty:
            logger.info("Generating BCG Growth-Share Matrix...")
            try:
                required_bcg_cols = [qual_col, bcg_growth_col, bcg_share_col, bcg_size_col]
                # Check if qual_col from config is present (it might be named differently initially)
                actual_qual_col = qual_col if qual_col in bcg_data_df.columns else 'Qualification' # Check common alternative
                if actual_qual_col not in bcg_data_df.columns: 
                    logger.warning(f"BCG Matrix: Qualification column ('{qual_col}' or 'Qualification') not found.")
                elif all(c in bcg_data_df.columns for c in [bcg_growth_col, bcg_share_col, bcg_size_col]):
                    plot_title = f"{inst_short_name}: Tutkintojen kasvu vs. markkinaosuus ({plot_reference_year})"
                    bcg_caption = base_caption + f" Kuplan koko = {inst_short_name} volyymi ({plot_reference_year}). Suhteellinen markkinaosuus = {inst_short_name} osuus / Suurimman kilpailijan osuus."
                    fig, _ = visualizer.create_bcg_matrix(
                        data=bcg_data_df, growth_col=bcg_growth_col, share_col=bcg_share_col,
                        size_col=bcg_size_col, label_col=actual_qual_col, title=plot_title, caption=bcg_caption
                    )
                    # Add figure to PDF
                    visualizer.save_visualization(fig, f"{inst_short_name}_bcg_matrix")
                else: logger.warning(f"Skipping BCG Matrix: Missing required columns. Found: {bcg_data_df.columns.tolist()}")
            except Exception as e:
                logger.error(f"Failed to generate BCG Matrix plot: {e}", exc_info=True)
                if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        else: logger.warning("Skipping BCG Matrix: bcg_data not available or empty.")

        # --- Plot 5: Combined Volume / Provider Count Plot ---
        volume_df = analysis_results.get('total_volumes')
        count_df = analysis_results.get('provider_counts_by_year')
        if volume_df is not None and not volume_df.empty and count_df is not None and not count_df.empty:
            logger.info("Generating Volume / Provider Count Plot...")
            try:
                # Check if all required columns exist
                req_vol_cols = [year_col, provider_amount_col, subcontractor_amount_col]
                req_count_cols = [year_col, count_provider_col, count_subcontractor_col]
                if all(c in volume_df.columns for c in req_vol_cols) and all(c in count_df.columns for c in req_count_cols):
                    plot_title = f"{inst_short_name}: Opiskelijamäärät ja kouluttajamarkkina ({min_year}-{max_year})"
                    plot_caption = base_caption + f". Kouluttajamäärä perustuu tutkintoihin, joita {inst_short_name} tarjoaa."
                    fig, _ = visualizer.create_volume_and_provider_count_plot(
                        volume_data=volume_df, count_data=count_df, title=plot_title,
                        volume_title="Netto-opiskelijamäärä", count_title="Uniikit kouluttajat markkinassa",
                        year_col=year_col, vol_provider_col=provider_amount_col, vol_subcontractor_col=subcontractor_amount_col,
                        count_provider_col=count_provider_col, count_subcontractor_col=count_subcontractor_col,
                        caption=plot_caption
                    )
                    # Add figure to PDF
                    visualizer.save_visualization(fig, f"{inst_short_name}_volume_provider_counts")
                else:
                    logger.warning(f"Skipping Volume/Provider Count plot: Missing required columns. Vol needed: {req_vol_cols} (found {volume_df.columns.tolist()}). Count needed: {req_count_cols} (found {count_df.columns.tolist()})")
            except Exception as e:
                logger.error(f"Failed to generate Volume/Provider Count plot: {e}", exc_info=True)
                if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        else: logger.warning("Skipping Volume/Provider Count plot: Required volume or count data missing.")

        # --- Finalize and Close PDF ---
        logger.info("Finalizing PDF report...")
        visualizer.close_pdf() # Close the PDF file
        pdf_report_path = visualizer.pdf_path # Get the path where the PDF was saved
        if pdf_report_path:
             logger.info(f"PDF report saved to: {pdf_report_path}")
        else:
             logger.warning("Could not determine PDF report path from visualizer.")

        logger.info(f"--- Step 3: PDF Report Generation Complete ---")

except Exception as e:
    logger.error(f"An unexpected error occurred during PDF visualization generation: {e}", exc_info=True)
    # figures dict is already initialized -> No figures dict anymore
    pdf_report_path = None # Ensure path is None on error

# === Step 4: Export Final Results (Excel) ===
logger.info("--- Step 4: Exporting Final Results to Excel ---")

excel_path = None # Initialize path
try:
    # Ensure analysis results are available
    if 'analysis_results' not in locals() or not analysis_results:
        logger.warning("Analysis results missing. Skipping Excel Export.")
    else:
        # --- Create Metadata ---
        logger.info("Creating metadata for export...")
        metadata = {
            "Analysis Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Institution Analyzed": f"{institution_name} ({institution_short_name})", # Use variables from Step 1
            "Institution Variants Used": ", ".join(institution_variants),
            "Input Data File": data_file_path,
            "Data Update Date": data_update_date_str,
            "Qualification Type Filter Applied": "Yes" if filter_qual_types_flag else "No",
            "Min Market Size Threshold (for plots)": config.get('analysis', {}).get('min_market_size_threshold', 'N/A'),
            "Active Qualifications Filter Threshold (for plots)": config.get('analysis', {}).get('active_qualification_min_volume_sum', 'N/A')
        }
        metadata_df = pd.DataFrame(metadata.items(), columns=["Parameter", "Value"])
        logger.info("Metadata created successfully.")
        print("\n--- Analysis Metadata ---")
        print(metadata_df)
        
        # --- Determine Base Output Path ---
        base_output_path_str = ANALYSIS_PARAMS.get('output_dir') or config.get('paths', {}).get('output', 'output') # Default to 'output'
        base_output_path = Path(base_output_path_str)
        # No need to create the subdir here, export_to_excel should handle it based on its logic
        logger.info(f"Using base output directory for Excel export: {base_output_path}")

        # --- Call Exporter ---
        # Use the wrapper function from analyze_cli which handles filename and dict prep
        excel_path = export_analysis_results(
            analysis_results=analysis_results,
            config=config,
            institution_short_name=institution_short_name,
            base_output_path=str(base_output_path), # Pass as string
            metadata_df=metadata_df,
            include_timestamp=ANALYSIS_PARAMS.get('include_timestamp', True)
        )
        
        if excel_path:
            logger.info(f"Successfully exported results to Excel: {excel_path}")
        else:
            logger.warning("Excel export function did not return a valid path.")
            
        logger.info("--- Step 4: Excel Export Complete ---")

except Exception as e:
    logger.error(f"An unexpected error occurred during Excel export: {e}", exc_info=True)
    # excel_path remains None or its previous value


# === Step 5: Custom User Analysis Area ===
logger.info("--- Step 5: Custom Analysis Area ---")

# The main analysis workflow is complete.
# Key variables available for further analysis:
# - config: The loaded project configuration dictionary.
# - ANALYSIS_PARAMS: The parameters used for this specific run.
# - df_raw: The raw data loaded initially.
# - df_prepared: The cleaned and filtered data used for the main analysis.
# - analysis_results: A dictionary containing the various analysis DataFrames.
#     Keys: ['total_volumes', 'volumes_by_qualification', 'detailed_providers_market', 
#            'qualification_cagr', 'overall_total_market_volume', 
#            'qualification_market_yoy_growth', 'provider_counts_by_year', 'bcg_data']
# - analyzer: The MarketAnalyzer instance.
# - pdf_report_path: Path to the generated PDF report (if successful).
# - excel_path: The path to the generated Excel file (if export was successful).
# - institution_short_name: Short name of the analyzed institution.
# - institution_variants: List of variants used for matching.
# - data_update_date_str: The data update date string.

logger.info("You can now add your custom Python code below to explore the results further.")
print("\nExample: Accessing the detailed market data DataFrame:")
# Uncomment the lines below to print the head of the detailed market data
# if 'analysis_results' in locals() and 'detailed_providers_market' in analysis_results:
#     print("\n--- Detailed Providers Market Sample (from analysis_results) ---")
#     detailed_df_example = analysis_results['detailed_providers_market']
#     print(detailed_df_example.head())
# else:
#     print("Detailed market data not available.")

print("\nScript execution finished. Add custom analysis code below this line.")

# --- End of Script ---


# TODO: Add Section for Custom User Analysis --> This is now Step 5

if __name__ == "__main__":
    # This block allows running the script directly, mimicking notebook execution
    logger.info("Running analysis script...")
    
    # Placeholder for calling the main workflow steps
    if 'df_prepared' in locals():
        logger.info("Data preparation step appears complete.")
        if 'analysis_results' in locals() and analysis_results:
            logger.info("Market analysis step appears complete.")
            # Check for PDF path instead of figures dict
            if 'pdf_report_path' in locals() and pdf_report_path:
                logger.info("PDF Visualization step appears complete.")
            else:
                logger.warning("PDF Visualization step may have failed or was skipped.")
        elif 'analysis_results' in locals():
             logger.warning("Market analysis step completed but produced no results.")
        else:
            logger.warning("Market analysis step may have failed, 'analysis_results' not found.")
            
        if 'excel_path' in locals() and excel_path:
             logger.info("Excel export step appears complete.")
        else:
             logger.warning("Excel export step may have failed or was skipped.")
            
    else:
        logger.warning("Data preparation step may have failed, 'df_prepared' not found.")

    # Example: Accessing a config value
    try:
        data_path_from_config = config.get('paths', {}).get('data', 'Not Found')
        logger.info(f"Data path from config: {data_path_from_config}")
    except Exception as e:
        logger.error(f"Error accessing config: {e}")
        
    logger.info("Analysis script finished.") 