"""
CLI module for the Vipunen project.

This module provides the main entry point for the education market analysis
workflow, orchestrating the data loading, analysis, and export steps.
"""
import sys
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..config.config_loader import get_config
from ..data.data_loader import load_data
from ..data.data_processor import clean_and_prepare_data
from ..export.excel_exporter import export_to_excel
from ..analysis.market_analyzer import MarketAnalyzer
from .argument_parser import parse_arguments, get_institution_variants
# Import Visualizer and constants
from ..visualization.education_visualizer import EducationVisualizer, COLOR_PALETTES, TEXT_CONSTANTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_visualizations(analysis_results: Dict[str, Any], visualizer: EducationVisualizer, analyzer: MarketAnalyzer, config: Dict[str, Any]):
    """
    Generate all standard visualizations based on analysis results.

    Args:
        analysis_results: Dictionary containing the results from MarketAnalyzer.analyze()
        visualizer: An initialized EducationVisualizer instance.
        analyzer: The MarketAnalyzer instance (to access min/max years, institution name etc.).
        config: Configuration dictionary.
    """
    logger.info("Starting visualization generation...")

    inst_short_name = analyzer.institution_short_name
    inst_names = analyzer.institution_names
    min_year = analyzer.min_year
    max_year = analyzer.max_year
    current_date_str = visualizer.current_date # Get date from visualizer
    base_caption = TEXT_CONSTANTS["data_source"].format(date=current_date_str)
    
    # Determine last full year for certain plots/filters
    last_full_year = max_year
    if max_year is not None and min_year is not None and max_year > min_year:
        last_full_year = max_year - 1
        
    plot_reference_year = last_full_year if last_full_year else max_year # Use last full year for titles etc. if available

    # --- Plot 1: Stacked Area (Total Volumes) ---
    total_volumes_df = analysis_results.get('total_volumes')
    if total_volumes_df is not None and not total_volumes_df.empty:
        try:
            logger.info("Generating Total Volumes Area Chart...")
            visualizer.create_area_chart(
                data=total_volumes_df,
                x_col='tilastovuosi',
                y_cols=['järjestäjänä', 'hankintana'],
                colors=[COLOR_PALETTES["roles"]["järjestäjänä"], COLOR_PALETTES["roles"]["hankintana"]],
                labels=['järjestäjänä', 'hankintana'],
                title=f"{inst_short_name} opiskelijamäärät vuosina {min_year}-{max_year}",
                caption=base_caption,
                filename=f"{inst_short_name}_total_volumes_area"
            )
        except Exception as e:
            logger.error(f"Failed to generate Total Volumes plot: {e}", exc_info=True)
    else:
        logger.warning("Skipping Total Volumes plot: Data not available.")

    # --- Prepare Detailed Market Data (used by several plots) ---
    # Data is now pre-filtered by MarketAnalyzer.analyze based on min_market_size_threshold
    detailed_df = analysis_results.get('detailed_providers_market')
    if detailed_df is None or detailed_df.empty:
        logger.warning("Skipping several plots: Detailed providers market data not available.")
        return # Exit if detailed data is missing
        
    # --- Determine Active Qualifications (for filtering institution-specific plots) ---
    # Uses the already filtered detailed_df
    active_qualifications = []
    if max_year is not None and min_year is not None:
        # Check activity based on last_full_year and the year before it
        years_to_check_activity = [last_full_year]
        if last_full_year > min_year:
            years_to_check_activity.append(last_full_year - 1)
            
        active_df = detailed_df[
            (detailed_df['Provider'].isin(inst_names)) &
            (detailed_df['Year'].isin(years_to_check_activity))
        ]
        # Sum volume over the last two years for the institution
        active_grouped = active_df.groupby('Qualification')['Total Volume'].sum()
        # --- Use the >= 1 threshold consistent with analyzer --- 
        active_qualifications = active_grouped[active_grouped >= 1].index.tolist()
        logger.info(f"Found {len(active_qualifications)} active qualifications (volume >= 1 in {years_to_check_activity}) for {inst_short_name}: {active_qualifications}" )
    else:
        logger.warning("Could not determine active qualifications due to missing year data.")
        # Fallback: use all qualifications the institution ever offered in the filtered data
        active_qualifications = detailed_df[detailed_df['Provider'].isin(inst_names)]['Qualification'].unique().tolist()

    # --- Plot 2: Line Chart (Market Share Evolution) - Loop per Qualification ---
    logger.info(f"Generating Market Share Line Charts for {len(active_qualifications)} active qualifications...")
    # --- Iterate over ALL active qualifications --- 
    for qual in active_qualifications: 
        try:
            qual_df = detailed_df[detailed_df['Qualification'] == qual]
            if qual_df.empty:
                continue
                
            # Find top M providers in the latest year for this qualification
            latest_qual_providers = qual_df[qual_df['Year'] == plot_reference_year]
            top_m_providers = latest_qual_providers.nlargest(6, 'Market Share (%)')['Provider'].tolist()
            
            # Pivot data for plotting
            plot_data = qual_df[qual_df['Provider'].isin(top_m_providers)].pivot(
                index='Year', columns='Provider', values='Market Share (%)'
            )
            plot_data.index.name = 'Year' # Ensure index has a name for the function
            
            if not plot_data.empty:
                qual_filename_part = qual.replace(' ', '_').replace('/', '_').lower()[:30] # Create safe filename part
                visualizer.create_line_chart(
                    data=plot_data,
                    x_col=plot_data.index,
                    y_cols=top_m_providers,
                    colors=COLOR_PALETTES["main"],
                    labels=top_m_providers,
                    title=f"{qual}: Markkinaosuus (%)",
                    caption=base_caption,
                    filename=f"{inst_short_name}_{qual_filename_part}_market_share_lines"
                )
        except Exception as e:
            logger.error(f"Failed to generate Market Share line plot for {qual}: {e}", exc_info=True)

    # --- Plot 3: Heatmap (Institution's Share) ---
    try:
        logger.info("Generating Institution Market Share Heatmap...")
        inst_share_df = detailed_df[detailed_df['Provider'].isin(inst_names)]
        # Check for duplicates before pivoting
        duplicates = inst_share_df.duplicated(subset=['Qualification', 'Year'], keep=False)
        if duplicates.any():
            logger.warning(f"Found duplicate Qualification/Year entries for {inst_short_name}. Dropping duplicates before pivoting heatmap.")
            logger.debug(f"Duplicate rows:\n{inst_share_df[duplicates]}")
            inst_share_df = inst_share_df.drop_duplicates(subset=['Qualification', 'Year'], keep='first')
            
        heatmap_data = inst_share_df.pivot(index='Qualification', columns='Year', values='Market Share (%)')
        # Filter heatmap data to only active qualifications
        heatmap_data = heatmap_data[heatmap_data.index.isin(active_qualifications)]
        
        if not heatmap_data.empty:
            visualizer.create_heatmap(
                data=heatmap_data,
                title=f"{inst_short_name}: markkinaosuus tutkinnoittain",
                caption=base_caption,
                filename=f"{inst_short_name}_share_heatmap",
                cmap='Blues' # Example: Use Blues colormap
            )
    except Exception as e:
        logger.error(f"Failed to generate Institution Share heatmap: {e}", exc_info=True)

    # --- Plot 4: Heatmap with Marginals ---
    overall_volume_series = analysis_results.get('overall_total_market_volume')
    if overall_volume_series is not None and not overall_volume_series.empty and 'heatmap_data' in locals() and not heatmap_data.empty:
        try:
            logger.info("Generating Market Share Heatmap with Marginals...")
            # Prepare right data (latest year market total per qual)
            latest_detailed = detailed_df[detailed_df['Year'] == plot_reference_year]
            right_data = latest_detailed.drop_duplicates(subset='Qualification').set_index('Qualification')['Market Total']
            # Filter right_data to only active qualifications
            right_data = right_data[right_data.index.isin(active_qualifications)]

            visualizer.create_heatmap_with_marginals(
                heatmap_data=heatmap_data, # From plot 3 (already filtered)
                top_data=overall_volume_series,
                right_data=right_data,
                title=f"{inst_short_name}: markkinaosuus vs markkinakoko ({plot_reference_year})",
                top_title=f"Koko markkina (kaikki tutkinnot, {min_year}-{max_year})",
                right_title=f"Tutkinnon markkinakoko ({plot_reference_year})",
                caption=base_caption,
                filename=f"{inst_short_name}_share_heatmap_marginals",
                cmap='Blues'
            )
        except Exception as e:
            logger.error(f"Failed to generate Heatmap with Marginals: {e}", exc_info=True)
    else:
        logger.warning("Skipping Heatmap with Marginals: Data not available.")
        
    # --- Plot 5: Horizontal Bar (Qualification Growth) ---
    qual_growth_df = analysis_results.get('qualification_market_yoy_growth')
    if qual_growth_df is not None and not qual_growth_df.empty:
        try:
            logger.info("Generating Qualification Market Growth Bar Chart...")
            # Use growth data for the transition into the plot_reference_year
            growth_ref_year = qual_growth_df[qual_growth_df['Year'] == plot_reference_year].dropna(subset=['Market Total YoY Growth (%)'])
            # Filter to qualifications relevant to the institution (active ones)
            growth_ref_year = growth_ref_year[growth_ref_year['Qualification'].isin(active_qualifications)]
                
            if not growth_ref_year.empty:
                sorted_growth = growth_ref_year.sort_values('Market Total YoY Growth (%)')
                visualizer.create_horizontal_bar_chart(
                    data=sorted_growth,
                    x_col='Market Total YoY Growth (%)',
                    y_col='Qualification',
                    volume_col='Market Total',
                    title=f"Tutkinnot: nousijat ja laskijat (YoY: {plot_reference_year-1}-{plot_reference_year})",
                    caption=base_caption,
                    filename=f"{inst_short_name}_qualification_growth_bar",
                    x_label_text="Tutkinnon markkinakasvu (%)",
                    y_label_detail_format="({:.0f})" # Format volume as integer
                )
        except Exception as e:
            logger.error(f"Failed to generate Qualification Growth plot: {e}", exc_info=True)
    else:
        logger.warning("Skipping Qualification Growth plot: Data not available.")

    # --- Plot 6: Horizontal Bar (Provider Gainers/Losers) - Loop per Qualification ---
    logger.info(f"Generating Provider Gainer/Loser Bar Charts for {len(active_qualifications)} active qualifications...")
    # --- Iterate over ALL active qualifications --- 
    for qual in active_qualifications: 
        try:
            latest_qual_df = detailed_df[
                (detailed_df['Qualification'] == qual) & (detailed_df['Year'] == plot_reference_year)
            ].dropna(subset=['Market Share Growth (%)'])

            if not latest_qual_df.empty:
                gainers = latest_qual_df.nlargest(3, 'Market Share Growth (%)')
                losers = latest_qual_df.nsmallest(3, 'Market Share Growth (%)')
                # Ensure no overlap if fewer than 6 providers
                plot_data = pd.concat([losers, gainers]).drop_duplicates().sort_values('Market Share Growth (%)')

                if not plot_data.empty:
                    qual_filename_part = qual.replace(' ', '_').replace('/', '_').lower()[:30]
                    visualizer.create_horizontal_bar_chart(
                        data=plot_data,
                        x_col='Market Share Growth (%)',
                        y_col='Provider',
                        volume_col='Market Share (%)', # Show current market share in label
                        title=f"{qual}: suurimmat nousijat ja laskijat ({plot_reference_year})",
                        caption=base_caption + f" Suluissa on toimijan markkinaosuus tutkinnossa vuonna {plot_reference_year}.",
                        filename=f"{inst_short_name}_{qual_filename_part}_gainer_loser_bar",
                        x_label_text="Markkinaosuuden vuosikasvu (%)",
                        y_label_detail_format="({:.1f} %)"
                    )
        except Exception as e:
            logger.error(f"Failed to generate Gainer/Loser plot for {qual}: {e}", exc_info=True)

    # --- Plot 7: Treemap ---
    # inst_latest_df = detailed_df[(detailed_df['Provider'].isin(inst_names)) & (detailed_df['Year'] == max_year)]
    # if inst_latest_df is not None and not inst_latest_df.empty:
    #     try:
    #         logger.info("Generating Market Share Treemap...")
    #         # Use data from plot_reference_year for the treemap
    #         treemap_base_data = detailed_df[
    #             (detailed_df['Provider'].isin(inst_names)) &
    #             (detailed_df['Year'] == plot_reference_year)
    #         ].copy()

    #         # Filter for active qualifications
    #         treemap_base_data = treemap_base_data[treemap_base_data['Qualification'].isin(active_qualifications)]
            
    #         # Ensure Market Total is present for sizing
    #         if 'Market Total' not in treemap_base_data.columns:
    #              # Merge market total if missing (might happen if filtered differently)
    #              ref_year_totals = detailed_df[detailed_df['Year'] == plot_reference_year][['Qualification', 'Market Total']].drop_duplicates()
    #              treemap_base_data = pd.merge(treemap_base_data, ref_year_totals, on='Qualification', how='left')
            
    #         # Apply RI-specific adjustment for 'Liiketoiminnan PT'
    #         if analyzer.institution_short_name == "RI":
    #             pt_index = treemap_base_data[treemap_base_data['Qualification'] == 'Liiketoiminnan PT'].index
    #             if not pt_index.empty:
    #                 logger.info("Applying RI-specific adjustment: Halving Market Total for Liiketoiminnan PT in Treemap.")
    #                 treemap_base_data.loc[pt_index, 'Market Total'] = treemap_base_data.loc[pt_index, 'Market Total'] / 2
                 
    #         plot_data = treemap_base_data.sort_values('Market Total', ascending=False)
            
    #         # Check required columns exist before plotting
    #         required_cols = ['Market Total', 'Qualification', 'Market Share (%)']
    #         if not plot_data.empty and all(col in plot_data.columns for col in required_cols) and not plot_data[required_cols].isnull().any().any():
    #              visualizer.create_treemap(
    #                  data=plot_data,
    #                  value_col='Market Total', # Use adjusted value
    #                  label_col='Qualification',
    #                  detail_col='Market Share (%)', # Display institution's market share
    #                  title=f"{inst_short_name}: markkinaosuus, koko tutkintomarkkina ({plot_reference_year})",
    #                  caption=base_caption + " Laatikon koko kuvaa tutkinnon markkinakokoa.",
    #                  filename=f"{inst_short_name}_treemap"
    #              )
    #     except Exception as e:
    #         logger.error(f"Failed to generate Treemap: {e}", exc_info=True)
    # else:
    #     logger.warning("Skipping Treemap: Data not available.")

    logger.info("...visualization generation complete.")

def run_analysis(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the education market analysis workflow.
    
    Args:
        args: Dictionary of arguments, defaults to command-line arguments if None
        
    Returns:
        Dict[str, Any]: Dictionary with analysis results
    """
    # Parse arguments if not provided
    if args is None:
        # Pass empty list to prevent parsing pytest args from sys.argv
        parsed_args = parse_arguments([]) 
        args = vars(parsed_args)
    
    # Step 1: Get configuration
    config = get_config()
    
    # Step 2: Define parameters for the analysis
    data_file_path = args.get('data_file', config['paths']['data'])
    institution_name = args.get('institution', config['institutions']['default']['name'])
    institution_short_name = args.get('short_name', config['institutions']['default']['short_name'])
    use_dummy = args.get('use_dummy', False)
    filter_qual_types = args.get('filter_qual_types', False)
    filter_by_inst_quals = args.get('filter_by_inst_quals', False)
    
    # Set up institution variants
    if 'variants' in args and args['variants']:
        institution_variants = list(args['variants'])
        if institution_name not in institution_variants:
            institution_variants.append(institution_name)
    else:
        institution_variants = config['institutions']['default']['variants']
        if institution_name not in institution_variants:
            institution_variants.append(institution_name)
    
    logger.info(f"Analyzing institution: {institution_name}")
    logger.info(f"Institution variants: {institution_variants}")
    
    # Step 3: Load the raw data
    logger.info(f"Loading raw data from {data_file_path}")
    raw_data = load_data(file_path=data_file_path, use_dummy=use_dummy)
    logger.info(f"Loaded {len(raw_data)} rows of data")
    
    # Step 4: Clean and prepare the data
    logger.info("Cleaning and preparing the data")
    df_clean = clean_and_prepare_data(
        raw_data, 
        institution_names=institution_variants,
        merge_qualifications=True,
        shorten_names=True
    )
    
    # Filter data for the specific institution if needed
    if filter_by_inst_quals or filter_qual_types:
        logger.info("Filtering data based on institution and qualification types")
        # Filter for institution data
        institution_mask = (
            df_clean['koulutuksenJarjestaja'].isin(institution_variants) | 
            df_clean['hankintakoulutuksenJarjestaja'].isin(institution_variants)
        )
        
        # Get qualifications offered by the institution
        if filter_by_inst_quals:
            inst_qualifications = df_clean[institution_mask]['tutkinto'].unique()
            df_clean = df_clean[df_clean['tutkinto'].isin(inst_qualifications)]
            logger.info(f"Filtered to {len(inst_qualifications)} qualifications offered by {institution_name}")
        
        # Filter by qualification types if requested
        if filter_qual_types:
            qual_types = config.get('qualification_types', ['Ammattitutkinnot', 'Erikoisammattitutkinnot'])
            df_clean = df_clean[df_clean['tutkintotyyppi'].isin(qual_types)]
            logger.info(f"Filtered to qualification types: {qual_types}")
    
    # --- Start Analysis Block with Error Handling ---
    analysis_results = {}
    excel_path = None
    try:
        # Step 5: Perform analysis using the MarketAnalyzer
        logger.info("Initializing market analyzer")
        analyzer = MarketAnalyzer(
            data=df_clean
        )
        
        # Add institution names as an attribute to be used by the analyzer
        analyzer.institution_names = institution_variants
        analyzer.institution_short_name = institution_short_name
        
        # Run the analysis
        logger.info("Running analysis")
        analysis_results = analyzer.analyze() # This calls get_all_results
        
        # Step 6: Create directory structure for outputs (only if analysis succeeds)
        logger.info("Creating output directories")
        
        # Determine output directory
        output_dir = args.get('output_dir')
        if output_dir is None:
            output_dir = config['paths'].get('output', 'data/reports')
            
        # Create directory name based on institution
        dir_name = f"education_market_{institution_short_name.lower()}"
        output_dir = Path(output_dir) / dir_name
        
        # Create directory
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create plots directory
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Step 7: Export results to Excel (only if analysis succeeds)
        logger.info("Exporting results to Excel")
        
        # Prepare Excel data using results from analyze()
        excel_data = {
            "Total Volumes": analysis_results.get('total_volumes', pd.DataFrame()).reset_index(drop=True),
            "Volumes by Qualification": analysis_results.get('volumes_by_qualification', pd.DataFrame()).reset_index(drop=True),
            "Provider's Market": analysis_results.get("detailed_providers_market", pd.DataFrame()).reset_index(drop=True),
            "CAGR Analysis": analysis_results.get('qualification_cagr', pd.DataFrame()).reset_index(drop=True)
        }
        
        # Export to Excel
        excel_path = export_to_excel(
            data_dict=excel_data,
            file_name=f"{institution_short_name.lower()}_market_analysis",
            output_dir=output_dir,
            include_timestamp=True
        )
        
        logger.info(f"Analysis complete!")
        
        # Step 8: Generate Visualizations
        try:
            logger.info("Initializing visualizer...")
            visualizer = EducationVisualizer(output_dir=plots_dir)
            generate_visualizations(analysis_results, visualizer, analyzer, config)
        except Exception as vis_error:
            logger.error(f"Error during visualization generation: {vis_error}", exc_info=True)
            # Continue without visualizations if they fail

    except Exception as e:
        logger.error(f"Error during analysis execution: {e}", exc_info=True)
        # Ensure default empty results are available in case of error
        analysis_results = {
            "total_volumes": pd.DataFrame(),
            "volumes_by_qualification": pd.DataFrame(),
            "detailed_providers_market": pd.DataFrame(),
            "qualification_cagr": pd.DataFrame()
        }
        excel_path = None # No Excel file generated
        # Note: We allow the function to return normally, but with empty results

    # --- End Analysis Block ---

    # Return results (potentially empty if error occurred, now includes full analysis dict)
    return {
        "total_volumes": analysis_results.get('total_volumes', pd.DataFrame()),
        "volumes_by_qualification": analysis_results.get('volumes_by_qualification', pd.DataFrame()),
        "detailed_providers_market": analysis_results.get("detailed_providers_market", pd.DataFrame()),
        "qualification_cagr": analysis_results.get('qualification_cagr', pd.DataFrame()),
        "excel_path": excel_path
    }

def main():
    """Main entry point for the CLI."""
    try:
        results = run_analysis()
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 