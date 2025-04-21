"""Data aggregation module for Vipunen data processing."""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def aggregate_by_provider(df: pd.DataFrame, provider: str) -> pd.DataFrame:
    """
    Aggregate data for a specific provider, considering both main provider and subcontractor roles.
    When a provider appears in both roles, their full student count is included in both roles.
    
    Args:
        df: Input DataFrame with cleaned data
        provider: Name of the provider to analyze
        
    Returns:
        DataFrame with aggregated data for the provider
    """
    try:
        # Filter for the provider in both roles
        provider_data = df[
            (df['main_provider'] == provider) |
            (df['subcontractor'] == provider)
        ].copy()
        
        # Create role indicators
        provider_data['is_main_provider'] = provider_data['main_provider'] == provider
        provider_data['is_subcontractor'] = provider_data['subcontractor'] == provider
        
        # Aggregate by year and degree
        aggregated = provider_data.groupby(['year', 'degree']).agg({
            'net_students': 'sum',
            'is_main_provider': 'any',
            'is_subcontractor': 'any'
        }).reset_index()
        
        logger.info(f"Data aggregated for provider: {provider}")
        return aggregated
        
    except Exception as e:
        logger.error(f"Error aggregating provider data: {str(e)}")
        raise

def calculate_market_shares(df: pd.DataFrame, provider: str) -> pd.DataFrame:
    """
    Calculate market shares for a specific provider.
    When a provider appears in both roles (main provider and subcontractor),
    their market share is calculated as their total student count divided by the total market size.
    
    Args:
        df: Input DataFrame with cleaned data
        provider: Name of the provider to analyze
        
    Returns:
        DataFrame with market share calculations
    """
    try:
        # Calculate total market size by year and degree
        market_totals = df.groupby(['year', 'degree'])['net_students'].sum().reset_index()
        market_totals.rename(columns={'net_students': 'total_market'}, inplace=True)
        
        # Get provider's volume
        provider_data = aggregate_by_provider(df, provider)
        
        # Merge and calculate market share
        market_shares = pd.merge(
            provider_data,
            market_totals,
            on=['year', 'degree'],
            how='left'
        )
        
        # Calculate market share by dividing provider's student count by total market size
        # When a provider appears in both roles, their total student count is divided by 2
        market_shares['market_share'] = (
            market_shares['net_students'] / 2 / market_shares['total_market'] * 100
        )
        
        logger.info(f"Market shares calculated for provider: {provider}")
        return market_shares
        
    except Exception as e:
        logger.error(f"Error calculating market shares: {str(e)}")
        raise

def calculate_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate year-over-year growth rates.
    
    Args:
        df: Input DataFrame with cleaned data
        
    Returns:
        DataFrame with growth rate calculations
    """
    try:
        # Sort by year and degree
        df_sorted = df.sort_values(['year', 'degree'])
        
        # Calculate year-over-year growth
        df_sorted['yoy_growth'] = df_sorted.groupby('degree')['net_students'].pct_change() * 100
        
        # Calculate CAGR
        def calculate_cagr(group):
            if len(group) < 2:
                return pd.Series({'cagr': np.nan})
            start_value = group['net_students'].iloc[0]
            end_value = group['net_students'].iloc[-1]
            n_years = len(group) - 1
            if start_value == 0:
                return pd.Series({'cagr': np.nan})
            cagr = ((end_value / start_value) ** (1/n_years) - 1) * 100
            return pd.Series({'cagr': cagr})
        
        # Group by degree and calculate CAGR
        cagr = df_sorted.groupby('degree').apply(calculate_cagr).reset_index()
        
        logger.info("Growth rates calculated")
        return df_sorted, cagr
        
    except Exception as e:
        logger.error(f"Error calculating growth rates: {str(e)}")
        raise

def aggregate_market_data(df: pd.DataFrame, provider: str) -> Dict[str, pd.DataFrame]:
    """
    Perform all market data aggregation operations.
    
    Args:
        df: Input DataFrame with cleaned data
        provider: Name of the provider to analyze
        
    Returns:
        Dictionary containing various aggregated dataframes
    """
    try:
        # Calculate market shares
        market_shares = calculate_market_shares(df, provider)
        
        # Calculate growth rates
        yoy_growth, cagr = calculate_growth_rates(market_shares)
        
        # Aggregate provider data
        provider_data = aggregate_by_provider(df, provider)
        
        # Create result dictionary
        results = {
            'market_shares': market_shares,
            'yoy_growth': yoy_growth,
            'cagr': cagr,
            'provider_data': provider_data
        }
        
        logger.info("Market data aggregation completed")
        return results
        
    except Exception as e:
        logger.error(f"Error in market data aggregation: {str(e)}")
        raise 