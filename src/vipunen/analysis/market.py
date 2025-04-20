"""Market analysis module for Vipunen data."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def calculate_market_shares(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    provider_col: str,
    provider_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """Calculate market shares for providers in the market.
    
    Args:
        df: DataFrame containing the market data
        group_cols: Columns to group by (e.g., ['tilastovuosi', 'tutkinto'])
        value_col: Column containing the values to sum (e.g., 'nettoopiskelijamaaraLkm')
        provider_col: Column containing provider names
        provider_list: Optional list of providers to focus on. If None, all providers are included.
        
    Returns:
        DataFrame with market shares calculated
    """
    try:
        # Filter for specific providers if list is provided
        if provider_list:
            df = df[df[provider_col].isin(provider_list)]
            
        # Calculate total market size by group
        market_totals = df.groupby(group_cols)[value_col].sum().reset_index(name='market_total')
        
        # Calculate provider totals
        provider_totals = df.groupby(group_cols + [provider_col])[value_col].sum().reset_index(name=value_col)
        
        # Merge and calculate market shares
        result = pd.merge(provider_totals, market_totals, on=group_cols)
        result['market_share'] = (result[value_col] / result['market_total'] * 100).round(2)
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating market shares: {e}")
        raise

def calculate_provider_counts(
    df: pd.DataFrame,
    group_cols: List[str],
    provider_col: str
) -> pd.DataFrame:
    """Calculate the number of providers in the market.
    
    Args:
        df: DataFrame containing the market data
        group_cols: Columns to group by (e.g., ['tilastovuosi', 'tutkinto'])
        provider_col: Column containing provider names
        
    Returns:
        DataFrame with provider counts
    """
    try:
        # Count unique providers by group
        result = df.groupby(group_cols)[provider_col].nunique().reset_index(name='provider_count')
        return result
        
    except Exception as e:
        logger.error(f"Error calculating provider counts: {e}")
        raise

def calculate_growth_trends(df: pd.DataFrame, year_col: str, value_col: str, group_cols: List[str]) -> pd.DataFrame:
    """Calculate growth trends for each group.
    
    Args:
        df: Input DataFrame
        year_col: Name of the year column
        value_col: Name of the value column to analyze
        group_cols: List of columns to group by
        
    Returns:
        DataFrame with growth trends
    """
    if df.empty:
        return pd.DataFrame(columns=group_cols + ['cagr', 'start_year', 'end_year'])
    
    # Calculate yearly totals
    yearly_totals = df.groupby([year_col] + group_cols)[value_col].sum().reset_index()
    
    # Calculate CAGR for each group
    growth_data = []
    for group, group_df in yearly_totals.groupby(group_cols):
        if len(group_df) < 2:
            continue
            
        years = group_df[year_col].sort_values()
        values = group_df[value_col].values
        
        # Calculate CAGR
        start_value = values[0]
        end_value = values[-1]
        n_years = len(years) - 1
        
        if start_value > 0:
            cagr = ((end_value / start_value) ** (1/n_years) - 1) * 100
        else:
            cagr = float('nan')
        
        growth_data.append({
            **dict(zip(group_cols, group)),
            'cagr': cagr,
            'start_year': years.iloc[0],
            'end_year': years.iloc[-1]
        })
    
    if not growth_data:
        return pd.DataFrame(columns=group_cols + ['cagr', 'start_year', 'end_year'])
    
    return pd.DataFrame(growth_data)

def analyze_market(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    provider_col: str,
    year_col: str = 'tilastovuosi',
    provider_list: Optional[List[str]] = None,
    min_years: int = 2
) -> Dict[str, pd.DataFrame]:
    """Perform comprehensive market analysis.
    
    Args:
        df: DataFrame containing the market data
        group_cols: Columns to group by (e.g., ['tutkinto'])
        value_col: Column containing the values to analyze
        provider_col: Column containing provider names
        year_col: Column containing the year
        provider_list: Optional list of providers to focus on
        min_years: Minimum number of years required for trend calculation
        
    Returns:
        Dictionary containing various market analysis DataFrames:
        - market_shares: Market shares by provider
        - provider_counts: Number of providers in the market
        - growth_trends: Growth trends and CAGR
    """
    try:
        # Calculate market shares
        market_shares = calculate_market_shares(
            df=df,
            group_cols=group_cols,
            value_col=value_col,
            provider_col=provider_col,
            provider_list=provider_list
        )
        
        # Calculate provider counts
        provider_counts = calculate_provider_counts(
            df=df,
            group_cols=group_cols,
            provider_col=provider_col
        )
        
        # Calculate growth trends
        growth_trends = calculate_growth_trends(
            df=df,
            year_col=year_col,
            value_col=value_col,
            group_cols=group_cols
        )
        
        return {
            'market_shares': market_shares,
            'provider_counts': provider_counts,
            'growth_trends': growth_trends
        }
        
    except Exception as e:
        logger.error(f"Error performing market analysis: {e}")
        raise 