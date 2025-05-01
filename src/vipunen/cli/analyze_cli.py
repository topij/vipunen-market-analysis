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
import math
import datetime # Added for date parsing
import matplotlib.pyplot as plt # Add import

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

def generate_visualizations(
    analysis_results: Dict[str, Any],
    visualizer: EducationVisualizer,
    analyzer: MarketAnalyzer,
    config: Dict[str, Any],
    data_update_date_str: str # Added parameter for data update date
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
            visualizer.create_area_chart(
                data=plot_df_roles,
                x_col=year_col, # Use config year column
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
                    filename=f"{inst_short_name}_{qual_filename_part}_market_share_lines"
                )
                # plt.close(fig) # Close figure after saving - Removed for PDF output
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
        # duplicates = inst_share_df.duplicated(subset=['Qualification', 'Year'], keep=False)
        # if duplicates.any():
        #     logger.warning(f"Found duplicate Qualification/Year entries for {inst_short_name} AFTER AGGREGATION. This should not happen. Dropping duplicates.")
        #     logger.debug(f"Duplicate rows:\n{inst_share_df[duplicates]}")
        #     inst_share_df = inst_share_df.drop_duplicates(subset=['Qualification', 'Year'], keep='first')
            
        heatmap_data = inst_share_df.pivot(index=qual_col, columns=year_col, values=market_share_col)
        # Filter heatmap data to only active qualifications
        heatmap_data = heatmap_data[heatmap_data.index.isin(active_qualifications)]
        
        if not heatmap_data.empty:
            # Pass the aggregated, recalculated heatmap data
            fig, _ = visualizer.create_heatmap(
                data=heatmap_data,
                title=f"{inst_short_name}: markkinaosuus tutkinnoittain",
                caption=base_caption,
                filename=f"{inst_short_name}_share_heatmap",
                cmap='Blues' # Example: Use Blues colormap
            )
            # plt.close(fig) # Close figure - Removed for PDF output
    except Exception as e:
        logger.error(f"Failed to generate Institution Share heatmap: {e}", exc_info=True)

    # --- Plot 4: Heatmap with Marginals --- COMMENTED OUT
    # overall_volume_series = analysis_results.get('overall_total_market_volume')
    # # Use the aggregated inst_share_df for the heatmap data here as well
    # if overall_volume_series is not None and not overall_volume_series.empty and 'heatmap_data' in locals() and not heatmap_data.empty:
    #     try:
    #         logger.info("Generating Market Share Heatmap with Marginals...")
    #         # Prepare right data (latest year market total per qual)
    #         # Note: Market Total comes from detailed_df, not the aggregated one
    #         latest_detailed = detailed_df[detailed_df['Year'] == plot_reference_year]
    #         right_data = latest_detailed.drop_duplicates(subset='Qualification').set_index('Qualification')['Market Total']
    #         # Filter right_data to only active qualifications
    #         right_data = right_data[right_data.index.isin(active_qualifications)]
    # 
    #         # --- Apply RI-specific adjustment for 'Liiketoiminnan PT' --- 
    #         if analyzer.institution_short_name == "RI" and 'Liiketoiminnan PT' in right_data.index:
    #             logger.info("Applying RI-specific adjustment: Halving 'Liiketoiminnan PT' market size for marginals plot.")
    #             right_data.loc['Liiketoiminnan PT'] = right_data.loc['Liiketoiminnan PT'] / 2
    #         # --- End RI Adjustment --- 
    # 
    #         # Use the heatmap_data created *after* aggregation
    #         fig = visualizer.create_heatmap_with_marginals(
    #             heatmap_data=heatmap_data, 
    #             top_data=overall_volume_series,
    #             right_data=right_data,
    #             title=f"{inst_short_name}: markkinaosuus vs markkinakoko ({plot_reference_year})",
    #             top_title=f"Koko markkina (kaikki tutkinnot, {min_year}-{max_year})",
    #             right_title=f"Tutkinnon markkinakoko ({plot_reference_year})",
    #             # caption=base_caption,
    #             caption=base_caption + (". Huom. Liiketoiminnan PT koko puolitettu (RI:n markkina)" if analyzer.institution_short_name == "RI" else ""),
    # 
    #             filename=f"{inst_short_name}_share_heatmap_marginals",
    #             cmap='Blues'
    #         )
    #         # plt.close(fig) # Close figure - Removed for PDF output
    #     except Exception as e:
    #         logger.error(f"Failed to generate Heatmap with Marginals: {e}", exc_info=True)
    # else:
    #     logger.warning("Skipping Heatmap with Marginals: Data not available.")
        
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
                # Capture fig, ax return values
                fig, ax = visualizer.create_horizontal_bar_chart(
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
                # plt.close(fig) # Close figure - Removed for PDF output
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
                # --- Filtering based on config ---
                gainer_loser_config = config.get('analysis', {}).get('gainers_losers', {})
                min_share_threshold = gainer_loser_config.get('min_market_share_threshold')
                min_rank_percentile = gainer_loser_config.get('min_market_rank_percentile')
                
                filter_notes = [] # Initialize list to store filter descriptions

                original_provider_count = len(latest_qual_df)
                filtered_df = latest_qual_df.copy() # Start with a copy

                # Apply Market Share Threshold
                if min_share_threshold is not None and 'Market Share (%)' in filtered_df.columns:
                    try:
                        min_share = float(min_share_threshold)
                        if 0 <= min_share <= 100:
                            original_len = len(filtered_df)
                            filtered_df = filtered_df[filtered_df['Market Share (%)'] >= min_share]
                            if len(filtered_df) < original_len: # Check if filter actually removed rows
                                filter_notes.append(f"Markkinaosuus < {min_share}%")
                            logger.debug(f"[{qual}] Applying min market share threshold: >= {min_share}%. Kept {len(filtered_df)}/{original_provider_count} providers.")
                        else:
                            logger.warning(f"Invalid min_market_share_threshold ({min_share_threshold}), must be between 0 and 100. Skipping.")
                    except ValueError:
                        logger.warning(f"Invalid min_market_share_threshold ({min_share_threshold}), must be a number. Skipping.")

                # Apply Market Rank Percentile Threshold
                if min_rank_percentile is not None and 'Market Rank' in filtered_df.columns:
                    try:
                        percentile = float(min_rank_percentile)
                        if 0 <= percentile <= 100:
                            if not filtered_df.empty:
                                original_len = len(filtered_df)
                                # Calculate the rank cutoff. Lower rank numbers are better (e.g., rank 1 is best).
                                # We want to keep the top 'percentile' percent.
                                # Example: 90th percentile -> keep top 10% -> keep ranks <= ceil(total_providers * 0.10)
                                cutoff_fraction = (100.0 - percentile) / 100.0
                                num_providers_to_keep = math.ceil(len(filtered_df) * cutoff_fraction)
                                # Ensure we keep at least one if possible
                                num_providers_to_keep = max(1, num_providers_to_keep) 
                                
                                # Find the rank corresponding to the cutoff
                                # Sort by rank to find the rank value at the Nth position
                                sorted_by_rank = filtered_df.sort_values('Market Rank')
                                if num_providers_to_keep <= len(sorted_by_rank):
                                    rank_threshold = sorted_by_rank.iloc[num_providers_to_keep - 1]['Market Rank']
                                    # Keep all providers with rank <= rank_threshold (handles ties)
                                    filtered_df = filtered_df[filtered_df['Market Rank'] <= rank_threshold]
                                    if len(filtered_df) < original_len: # Check if filter actually removed rows
                                        filter_notes.append(f"Sijoitus top {100-percentile:.0f}% ulkopuolella")
                                    logger.debug(f"[{qual}] Applying min market rank percentile: {percentile}th (keep top {100-percentile:.1f}% => rank <= {rank_threshold}). Kept {len(filtered_df)} providers.")
                                else:
                                    # Keep everyone if cutoff exceeds number of providers
                                    logger.debug(f"[{qual}] Rank percentile filter ({percentile}th) keeps all {len(filtered_df)} remaining providers.")
                            else:
                                logger.debug(f"[{qual}] Skipping rank percentile filter: no providers left after previous filters.")
                        else:
                            logger.warning(f"Invalid min_market_rank_percentile ({min_rank_percentile}), must be between 0 and 100. Skipping.")
                    except ValueError:
                        logger.warning(f"Invalid min_market_rank_percentile ({min_rank_percentile}), must be a number. Skipping.")
                
                # Use the filtered data frame for selecting gainers/losers
                if not filtered_df.empty:
                    gainers = filtered_df.nlargest(3, 'Market Share Growth (%)')
                    losers = filtered_df.nsmallest(3, 'Market Share Growth (%)')
                    # Ensure no overlap if fewer than 6 providers
                    plot_data = pd.concat([losers, gainers]).drop_duplicates().sort_values('Market Share Growth (%)')
                else:
                    logger.info(f"[{qual}] No providers left after filtering for gainer/loser plot.")
                    plot_data = pd.DataFrame() # Ensure plot_data is an empty DataFrame

                if not plot_data.empty:
                    # Construct caption with potential filtering notes
                    plot_caption = base_caption + f" Suluissa on toimijan markkinaosuus tutkinnossa vuonna {plot_reference_year}."
                    if filter_notes:
                        plot_caption += f" Suodatettu: {'; '.join(filter_notes)}."
                        
                    qual_filename_part = qual.replace(' ', '_').replace('/', '_').lower()[:30]
                    fig, _ = visualizer.create_horizontal_bar_chart( # Capture fig
                        data=plot_data,
                        x_col='Market Share Growth (%)',
                        y_col='Provider',
                        volume_col='Market Share (%)', # Show current market share in label
                        title=f"{qual}: suurimmat nousijat ja laskijat ({plot_reference_year})",
                        caption=plot_caption, # Use the potentially extended caption
                        filename=f"{inst_short_name}_{qual_filename_part}_gainer_loser_bar",
                        x_label_text="Markkinaosuuden vuosikasvu (%)",
                        y_label_detail_format="({:.1f} %)"
                    )
                    # plt.close(fig) # Close figure after saving - Removed for PDF output
        except Exception as e:
            logger.error(f"Failed to generate Gainer/Loser plot for {qual}: {e}", exc_info=True)

    # --- Plot 7: Treemap ---
    inst_latest_df = detailed_df[(detailed_df['Provider'].isin(inst_names)) & (detailed_df['Year'] == plot_reference_year)]
    if inst_latest_df is not None and not inst_latest_df.empty:
        try:
            logger.info("Generating Market Share Treemap...")
            # Use data from plot_reference_year for the treemap
            treemap_base_data = detailed_df[
                (detailed_df['Provider'].isin(inst_names)) &
                (detailed_df['Year'] == plot_reference_year)
            ].copy()

            # Filter for active qualifications
            treemap_base_data = treemap_base_data[treemap_base_data['Qualification'].isin(active_qualifications)]
            
            # Ensure Market Total is present for sizing
            if 'Market Total' not in treemap_base_data.columns:
                 # Merge market total if missing (might happen if filtered differently)
                 ref_year_totals = detailed_df[detailed_df['Year'] == plot_reference_year][['Qualification', 'Market Total']].drop_duplicates()
                 treemap_base_data = pd.merge(treemap_base_data, ref_year_totals, on='Qualification', how='left')
            
            # Apply RI-specific adjustment for 'Liiketoiminnan PT'
            if analyzer.institution_short_name == "RI":
                pt_index = treemap_base_data[treemap_base_data['Qualification'] == 'Liiketoiminnan PT'].index
                if not pt_index.empty:
                    logger.info("Applying RI-specific adjustment: Halving Market Total for Liiketoiminnan PT in Treemap.")
                    treemap_base_data.loc[pt_index, 'Market Total'] = treemap_base_data.loc[pt_index, 'Market Total'] / 2
                 
            # Ensure Market Total is not zero or NaN before proceeding
            treemap_data = treemap_base_data[treemap_base_data['Market Total'].fillna(0) > 0].copy()
            
            if not treemap_data.empty:
                # Sort by Market Total descending for stable treemap layout
                treemap_data = treemap_data.sort_values('Market Total', ascending=False)
                
                fig, _ = visualizer.create_treemap(
                    data=treemap_data,
                    value_col='Market Total', # Size by Market Total
                    label_col='Qualification', 
                    detail_col='Market Share (%)', # Show institution's share inside
                    title=f"{inst_short_name}: Tutkintojen markkinaosuudet ({plot_reference_year})",
                    caption=base_caption + (". Huom. Liiketoiminnan PT koko puolitettu (RI:n markkina)" if analyzer.institution_short_name == "RI" else ""),
                    filename=f"{inst_short_name}_qualification_treemap"
                )
                # plt.close(fig) # Close figure - Keep commented for PDF output
        except Exception as e:
            logger.error(f"Failed to generate Treemap plot: {e}", exc_info=True)
    else:
        logger.warning("Skipping Treemap plot: Data not available for the reference year.")

    logger.info("Visualization generation completed.")

    # --- Close the PDF file if it was opened ---
    visualizer.close_pdf()

def run_analysis(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the education market analysis workflow.
    
    Args:
        args: Dictionary of arguments, defaults to command-line arguments if None
        
    Returns:
        Dict[str, Any]: Dictionary with analysis results
    """
    # Temporarily set this module's logger to DEBUG to see duplicate rows
    # original_level = logger.getEffectiveLevel()
    # logger.setLevel(logging.DEBUG) # REMOVE
    
    # Parse arguments if not provided
    if args is None:
        # Pass empty list to prevent parsing pytest args from sys.argv
        parsed_args = parse_arguments([]) 
        args = vars(parsed_args)
    
    # Step 1: Load configuration
    logger.info("Loading configuration...")
    config = get_config()
    
    # --- Logging Level Override (Example - Remove if not needed) ---
    # ... (logging override code) ...
    # --- End Logging Level Override ---

    # Step 2: Define parameters for the analysis
    data_file_path = args.get('data_file', config['paths']['data'])
    
    # Determine the institution key to use
    # Default name from config to compare against
    default_institution_name = config['institutions']['default']['name']
    # Get the value provided by argument or default
    arg_institution_value = args.get('institution') 
    
    if arg_institution_value is None or arg_institution_value == default_institution_name:
        # If no argument provided OR if the provided value is the default name, use the 'default' key
        institution_key = 'default'
        logger.info(f"No institution argument provided or matches default name. Using default key: '{institution_key}'")
    else:
        # Otherwise, assume the provided argument is the intended key
        institution_key = arg_institution_value
        logger.info(f"Using institution key provided via argument: '{institution_key}'")
        # Optional: Validate if the key exists in config
        if institution_key not in config['institutions']:
            logger.error(f"Provided institution key '{institution_key}' not found in config['institutions']. Aborting.")
            raise KeyError(f"Institution key '{institution_key}' not found in configuration.")
            
    # print(f"DEBUG: Initial institution_key from args/default: {institution_key}") # DEBUG REMOVED
    
    # Get short_name based on the determined key
    institution_short_name = args.get('short_name', config['institutions'][institution_key]['short_name'])
    use_dummy = args.get('use_dummy', False)
    filter_qual_types = args.get('filter_qual_types', False)
    filter_by_inst_quals = args.get('filter_by_inst_quals', False)
    
    # Get institution variants based on the key
    if 'variants' in args and args['variants']:
        institution_variants = list(args['variants'])
        # Add the main name corresponding to the key if not already present
        main_name = config['institutions'][institution_key]['name']
        if main_name not in institution_variants:
            institution_variants.append(main_name)
    else:
        # Default variants from config using the key
        institution_variants = config['institutions'][institution_key].get('variants', [])
        # Add the main name corresponding to the key if not already present
        main_name = config['institutions'][institution_key]['name']
        if main_name not in institution_variants:
            institution_variants.append(main_name)

    logger.info(f"Analyzing institution key: {institution_key} (Name: {config['institutions'][institution_key]['name']})")
    logger.info(f"Institution variants used for matching: {institution_variants}")
    
    # Step 3: Load the raw data
    logger.info(f"Loading raw data from {data_file_path}")
    raw_data = load_data(file_path=data_file_path, use_dummy=use_dummy)
    logger.info(f"Loaded {len(raw_data)} rows of data")
    
    # --- Extract Data Update Date ---
    data_update_date_str = datetime.datetime.now().strftime("%d.%m.%Y") # Default to today
    update_date_col = config.get('columns', {}).get('input', {}).get('update_date', 'tietojoukkoPaivitettyPvm')
    if not raw_data.empty and update_date_col in raw_data.columns:
        try:
            # Get the date string from the first row
            raw_date_str = str(raw_data[update_date_col].iloc[0])
            # Attempt to parse the date (assuming common formats like YYYY-MM-DD HH:MM:SS or YYYY-MM-DD)
            # pd.to_datetime is flexible
            parsed_date = pd.to_datetime(raw_date_str)
            data_update_date_str = parsed_date.strftime("%d.%m.%Y")
            logger.info(f"Using data update date from column '{update_date_col}': {data_update_date_str}")
        except Exception as date_err:
            logger.warning(f"Could not parse date from column '{update_date_col}': {date_err}. Falling back to current date.")
    else:
        logger.warning(f"Update date column '{update_date_col}' not found or data is empty. Falling back to current date.")
    # --- End Extract Data Update Date ---
    
    # Step 4: Clean and prepare the data
    logger.info("Cleaning and preparing the data")
    df_clean = clean_and_prepare_data(
        raw_data, 
        institution_names=institution_variants,
        merge_qualifications=True,
        shorten_names=True
    )
    
    # Filter data for the specific institution and qualifications
    logger.info("Filtering data based on institution and offered qualifications...")
    # Filter for institution data to identify relevant qualifications
    logger.info(f"Filtering data for institution: {config['institutions'][institution_key]['name']} and variants...")
    institution_mask = (
        (df_clean[config['columns']['input']['provider']].isin(institution_variants)) |
        (df_clean[config['columns']['input']['subcontractor']].isin(institution_variants))
    )
    
    # Get qualifications offered by the institution
    inst_qualifications = df_clean[institution_mask][config['columns']['input']['qualification']].unique()
    
    # Filter the main DataFrame to only include these qualifications
    if len(inst_qualifications) > 0:
        df_clean_filtered_by_qual = df_clean[df_clean[config['columns']['input']['qualification']].isin(inst_qualifications)].copy()
        logger.info(f"Filtered data to {len(inst_qualifications)} qualifications offered by {config['institutions'][institution_key]['name']}. Shape before: {df_clean.shape}, after: {df_clean_filtered_by_qual.shape}")
        df_clean = df_clean_filtered_by_qual # Update df_clean with the filtered data
    else:
        logger.warning(f"No qualifications found for institution {config['institutions'][institution_key]['name']} based on variants. Proceeding with data potentially unfiltered by qualification.")
        # df_clean remains unfiltered by institution qualifications in this case

    # Conditionally filter by qualification types (applied AFTER institution qual filtering)
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
        # Create MarketAnalyzer instance
        logger.info("Initializing MarketAnalyzer...")
        # Pass the loaded config to the analyzer
        analyzer = MarketAnalyzer(df_clean, cfg=config)
        # Use institution_key (defaults to 'default') to access config
        # Set the list of names including variants and the main name for the analyzer
        analyzer.institution_names = institution_variants # Use the derived list
        analyzer.institution_short_name = institution_short_name # Use the derived short name
        
        # Run the analysis
        logger.info("Running analysis")
        analysis_results = analyzer.analyze() # This calls get_all_results
        
        # Step 6: Create directory structure for outputs (only if analysis succeeds)
        logger.info("Creating output directories")
        
        # Determine output directory path (e.g., project_root/data/reports/education_market_ri)
        base_output_path = config['paths'].get('output', 'data/reports')
        if args.get('output_dir'): # Allow command-line override
             base_output_path = args['output_dir']
             
        dir_name = f"education_market_{institution_short_name.lower()}"
        # full_output_dir_path includes the base 'data' dir if present in config
        full_output_dir_path = Path(base_output_path) / dir_name
        
        # --- Determine path relative to base 'data' directory for FileUtils --- 
        # FileUtils expects paths relative to its configured type directories (e.g., 'reports')
        try:
            # Find the base data dir component (e.g., 'data')
            base_data_dir = Path(config['paths']['data']).parts[0] # Assumes paths.data exists and gives base
            excel_output_dir_relative = str(full_output_dir_path.relative_to(base_data_dir))
        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Could not reliably determine path relative to base data directory ({e}). Using full path: {full_output_dir_path}")
            # Fallback: Pass the full path; export_to_excel should handle it if configured correctly
            excel_output_dir_relative = str(full_output_dir_path)
            
        logger.info(f"Calculated relative output dir for export: {excel_output_dir_relative}")
        
        # Create the full directory path locally (for plots etc.)
        # FileUtils will handle creation for the Excel file path
        plots_dir = full_output_dir_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured plots directory exists: {plots_dir}")
        
        # Step 7: Export results to Excel (only if analysis succeeds)
        logger.info("Exporting results to Excel")
        
        # Prepare Excel data using results from analyze()
        excel_data = {
            "Total Volumes": analysis_results.get('total_volumes', pd.DataFrame()).reset_index(drop=True),
            "Volumes by Qualification": analysis_results.get('volumes_by_qualification', pd.DataFrame()).reset_index(drop=True),
            "Provider's Market": analysis_results.get("detailed_providers_market", pd.DataFrame()).reset_index(drop=True),
            "CAGR Analysis": analysis_results.get('qualification_cagr', pd.DataFrame()).reset_index(drop=True)
        }
        
        # Export to Excel using the relative path
        excel_path = export_to_excel(
            data_dict=excel_data,
            file_name=f"{institution_short_name.lower()}_market_analysis",
            output_dir=excel_output_dir_relative, # Pass relative path
            include_timestamp=True
        )
        
        logger.info(f"Analysis complete!")
        
        # Step 8: Generate Visualizations
        try:
            logger.info("Initializing visualizer...")
            # Pass the full path for plots dir to visualizer
            visualizer = EducationVisualizer(
                output_dir=plots_dir, 
                output_format='pdf'
            )
            # Pass the extracted data update date
            generate_visualizations(
                analysis_results=analysis_results, 
                visualizer=visualizer, 
                analyzer=analyzer, 
                config=config, 
                data_update_date_str=data_update_date_str 
            )
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
    
    # Restore original logging level before returning
    # logger.setLevel(original_level) # REMOVE

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