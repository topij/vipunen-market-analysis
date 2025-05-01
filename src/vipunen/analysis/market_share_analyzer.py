"""
Market share analysis utilities for education market analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_market_shares(df: pd.DataFrame, provider_names: List[str], 
                        year_col: str = 'tilastovuosi', 
                        qual_col: str = 'tutkinto', 
                        provider_col: str = 'koulutuksenJarjestaja', 
                        subcontractor_col: str = 'hankintakoulutuksenJarjestaja',
                        value_col: str = 'nettoopiskelijamaaraLkm',
                        share_calculation_basis: str = 'both') -> pd.DataFrame:
    """
    Calculate market shares for qualifications.
    
    Args:
        df: DataFrame with education data
        provider_names: List of names for the institution under analysis
        year_col: Column name for the year
        qual_col: Column name for the qualification
        provider_col: Column name for the main provider
        subcontractor_col: Column name for the subcontractor
        value_col: Column name for the volume metric
        share_calculation_basis: How to calculate market share ('main_provider', 'subcontractor', 'both').
                                Defaults to 'both'. 'both' can lead to shares > 100%.
        
    Returns:
        pd.DataFrame: DataFrame with market share calculations
    """
    # Validate share_calculation_basis
    valid_bases = ['main_provider', 'subcontractor', 'both']
    if share_calculation_basis not in valid_bases:
        raise ValueError(f"Invalid share_calculation_basis: {share_calculation_basis}. Must be one of {valid_bases}")
        
    # Log warning if using 'both' method
    if share_calculation_basis == 'both':
        logger.warning(
            "Using 'both' for share_calculation_basis. "
            "Market shares are calculated based on total volume (main provider + subcontractor). "
            "This may lead to double counting and the sum of shares per qualification/year exceeding 100%."
        )
        
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Initialize lists to store results
    market_share_records = []
    
    # Process each year-qualification combination
    for year in sorted(data[year_col].unique()):
        year_data = data[data[year_col] == year]
        
        # Process each qualification
        for qual in sorted(year_data[qual_col].unique()):
            # Filter data for this qualification
            qual_data = year_data[year_data[qual_col] == qual]
            
            # Calculate total market volume for this qualification
            total_market_volume = qual_data[value_col].sum()
            
            if total_market_volume <= 0:
                logger.warning(f"Skipping {qual} in {year}: no volume data")
                continue
            
            # Determine providers involved in this qualification (both main and sub)
            main_providers = set(qual_data[provider_col].dropna().unique())
            sub_providers = set(qual_data[subcontractor_col].dropna().unique())
            all_providers = sorted(list(main_providers.union(sub_providers)))
            
            # Calculate volumes and market shares for each provider involved
            # Note: Looping through all_providers ensures we capture subcontractors who might not be main providers
            for provider in all_providers:
                # Provider's volume as main provider
                provider_volume = qual_data[qual_data[provider_col] == provider][value_col].sum()
                
                # Provider's volume as subcontractor
                subcontractor_volume = qual_data[qual_data[subcontractor_col] == provider][value_col].sum()
                
                # Total provider activity volume (main + sub)
                total_provider_activity_volume = provider_volume + subcontractor_volume
                
                # Calculate market share based on the chosen basis
                numerator_volume = 0
                if share_calculation_basis == 'main_provider':
                    numerator_volume = provider_volume
                elif share_calculation_basis == 'subcontractor':
                    numerator_volume = subcontractor_volume
                else: # 'both'
                    numerator_volume = total_provider_activity_volume
                
                market_share = (numerator_volume / total_market_volume) * 100
                
                # Total number of unique main providers and subcontractors for this qualification
                provider_count = qual_data[provider_col].nunique()
                subcontractor_count = qual_data[subcontractor_col].dropna().nunique()
                
                # Add record
                market_share_records.append({
                    year_col: year,
                    qual_col: qual,
                    'provider': provider,
                    'volume_as_provider': provider_volume,
                    'volume_as_subcontractor': subcontractor_volume,
                    'total_volume': total_provider_activity_volume, # Renamed for clarity
                    'qualification_market_volume': total_market_volume,
                    'market_share': market_share,
                    'provider_count': provider_count,
                    'subcontractor_count': subcontractor_count,
                    'is_target_provider': provider in provider_names
                })
    
    if market_share_records:
        # Convert to DataFrame
        market_share_df = pd.DataFrame(market_share_records)
        
        # Sort by year, qualification, and market share (descending)
        market_share_df = market_share_df.sort_values(
            by=[year_col, qual_col, 'market_share'], 
            ascending=[True, True, False]
        )
        
        logger.info(f"Calculated market shares for {len(market_share_df)} provider-qualification combinations")
        return market_share_df
    else:
        logger.warning("No market share data generated")
        return pd.DataFrame()

def calculate_market_share_changes(market_share_df: pd.DataFrame,
                                 year_col: str,
                                 qual_col: str,
                                 provider_col: str,
                                 market_share_col: str,
                                 # volume_col: str # Add if volume change is needed later
                                ) -> pd.DataFrame:
    """
    Calculate year-over-year market share changes for each provider-qualification group.

    Args:
        market_share_df: DataFrame with market share data (output from calculate_market_shares).
                         Must contain year, qualification, provider, and market share columns.
        year_col: Name of the year column.
        qual_col: Name of the qualification column.
        provider_col: Name of the provider column.
        market_share_col: Name of the market share column (e.g., 'market_share').
        # volume_col: Name of the volume column (if volume change needed).

    Returns:
        pd.DataFrame: DataFrame with year, qualification, provider, previous market share,
                      and market share change. Returns empty if input is empty or lacks columns.
    """
    if market_share_df.empty:
        logger.warning("Empty market share data, cannot calculate changes")
        return pd.DataFrame()

    required_cols = [year_col, qual_col, provider_col, market_share_col]
    if not all(col in market_share_df.columns for col in required_cols):
        logger.error(f"Input DataFrame missing required columns for change calculation. Need: {required_cols}")
        return pd.DataFrame()

    # Ensure data is sorted correctly for shift operation
    market_share_df = market_share_df.sort_values(by=[qual_col, provider_col, year_col])

    # Calculate previous year's market share within each group
    market_share_df['previous_market_share'] = market_share_df.groupby(
        [qual_col, provider_col]
    )[market_share_col].shift(1)

    # Calculate market share change (absolute difference)
    # Fill NaN for the first year of each group
    market_share_df['market_share_change'] = market_share_df[market_share_col] - market_share_df['previous_market_share']

    # --- Optional: Calculate percentage change (handle division by zero/NaN) ---
    # market_share_df['market_share_change_percent'] = (
    #     (market_share_df[market_share_col] / market_share_df['previous_market_share']) - 1
    # ) * 100
    # market_share_df['market_share_change_percent'] = market_share_df['market_share_change_percent'].replace([np.inf, -np.inf], np.nan)
    # market_share_df.loc[market_share_df['previous_market_share'] == 0, 'market_share_change_percent'] = np.nan # Or some indicator like 99999
    # ---------------------------------------------------------------------

    # --- Optional: Calculate gainer rank based on absolute change ---
    # market_share_df['gainer_rank'] = market_share_df.groupby([year_col, qual_col])['market_share_change']\
    #                                         .rank(method='min', ascending=False)
    # --------------------------------------------------------------

    # Select and return relevant columns
    # Keep only rows where change could be calculated (i.e., not the first year for each group)
    result_df = market_share_df.dropna(subset=['previous_market_share'])

    output_cols = [
        year_col, qual_col, provider_col,
        'previous_market_share', 'market_share_change' # Key results needed by MarketAnalyzer
        # Add 'market_share_change_percent', 'gainer_rank' if needed
    ]

    logger.info(f"Calculated market share changes for {len(result_df)} provider-qualification-year combinations.")

    return result_df[output_cols]

def calculate_total_volumes(df: pd.DataFrame, provider_names: List[str], 
                        year_col: str = 'tilastovuosi', 
                        provider_col: str = 'koulutuksenJarjestaja', 
                        subcontractor_col: str = 'hankintakoulutuksenJarjestaja',
                        value_col: str = 'nettoopiskelijamaaraLkm') -> pd.DataFrame:
    """
    Calculate total volumes for an institution, matching the original implementation.
    
    Args:
        df: DataFrame containing the education data
        provider_names: List of names for the institution being analyzed
        year_col: Column containing year information
        provider_col: Column containing main provider names
        subcontractor_col: Column containing subcontractor names
        value_col: Column containing volume values
        
    Returns:
        pd.DataFrame: Summary DataFrame with volume breakdowns by year
    """
    # Handle empty input DataFrame
    if df.empty:
        logger.warning("Input DataFrame is empty, returning empty DataFrame.")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[year_col, 'järjestäjänä', 'hankintana', 'Yhteensä', 'järjestäjä_osuus (%)'])

    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Ensure provider_names is a list
    if isinstance(provider_names, str):
        provider_names = [provider_names]
    
    # Create a dictionary to store results by year
    results = {}
    
    # Group by year to calculate volumes
    for year, year_df in df_copy.groupby(year_col):
        # For main provider role
        main_provider_data = year_df[year_df[provider_col].isin(provider_names)]
        main_volume = main_provider_data[value_col].sum()
        
        # For subcontractor role - only count rows where subcontractor is not NaN
        subcontractor_data = year_df[(year_df[subcontractor_col].isin(provider_names)) & (pd.notna(year_df[subcontractor_col]))]
        sub_volume = subcontractor_data[value_col].sum()
        
        # Total volume is the sum of both roles
        total_volume = main_volume + sub_volume
        
        # Calculate percentage as main provider
        main_percentage = (main_volume / total_volume * 100) if total_volume > 0 else 0
        
        # Store results for this year
        results[year] = {
            'järjestäjänä': main_volume,
            'hankintana': sub_volume,
            'Yhteensä': total_volume,
            'järjestäjä_osuus (%)': main_percentage
        }
    
    # Convert to DataFrame
    result_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    result_df = result_df.rename(columns={'index': year_col})
    
    # Sort by year
    result_df = result_df.sort_values(year_col)
    
    return result_df 