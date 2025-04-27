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
                                 current_year: int, 
                                 previous_year: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate market share changes between two years.
    
    Args:
        market_share_df: DataFrame with market share data
        current_year: The current year for comparison
        previous_year: The previous year for comparison (defaults to current_year - 1)
        
    Returns:
        pd.DataFrame: DataFrame with market share changes
    """
    if market_share_df.empty:
        logger.warning("Empty market share data, cannot calculate changes")
        return pd.DataFrame()
    
    # Ensure tilastovuosi is present
    if 'tilastovuosi' not in market_share_df.columns:
        logger.error("Column 'tilastovuosi' not found in market share data")
        return pd.DataFrame()
    
    # Set default previous_year if not provided
    if previous_year is None:
        previous_year = current_year - 1
    
    # Check if both years exist in the data
    if not all(year in market_share_df['tilastovuosi'].unique() for year in [current_year, previous_year]):
        logger.warning(f"Both years {previous_year} and {current_year} must exist in the data")
        return pd.DataFrame()
    
    # Get data for current and previous years
    current_year_data = market_share_df[market_share_df['tilastovuosi'] == current_year].copy()
    prev_year_data = market_share_df[market_share_df['tilastovuosi'] == previous_year].copy()
    
    # Keep only relevant columns for merging
    current_cols = ['tutkinto', 'provider', 'market_share', 'total_volume']
    prev_cols = ['tutkinto', 'provider', 'market_share', 'total_volume']
    
    current_year_data = current_year_data[['tutkinto', 'provider', 'market_share', 'total_volume']].rename(
        columns={'market_share': 'current_share', 'total_volume': 'current_volume'})
    prev_year_data = prev_year_data[['tutkinto', 'provider', 'market_share', 'total_volume']].rename(
        columns={'market_share': 'previous_share', 'total_volume': 'previous_volume'})
    
    # Merge data for both years on qualification and provider
    market_share_change = pd.merge(
        current_year_data, 
        prev_year_data, 
        on=['tutkinto', 'provider'], 
        how='inner'
    )
    
    # Calculate market share change metrics
    market_share_change['market_share_change'] = market_share_change['current_share'] - market_share_change['previous_share']
    market_share_change['market_share_change_percent'] = (
        (market_share_change['current_share'] / market_share_change['previous_share']) - 1
    ) * 100
    
    # Calculate volume change metrics
    market_share_change['volume_change'] = market_share_change['current_volume'] - market_share_change['previous_volume']
    market_share_change['volume_change_percent'] = (
        (market_share_change['current_volume'] / market_share_change['previous_volume']) - 1
    ) * 100
    
    # Add year columns for reference
    market_share_change['current_year'] = current_year
    market_share_change['previous_year'] = previous_year
    
    # Rank gainers by qualification
    market_share_change['gainer_rank'] = market_share_change.groupby('tutkinto')['market_share_change'].rank(ascending=False)
    
    # Sort by qualification and market share change (descending)
    market_share_change = market_share_change.sort_values(by=['tutkinto', 'market_share_change'], ascending=[True, False])
    
    logger.info(f"Calculated market share changes for {len(market_share_change)} provider-qualification combinations")
    return market_share_change

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