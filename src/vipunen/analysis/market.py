"""
Market analysis module for Vipunen data.

This module provides functions for analyzing market shares, growth trends,
and provider statistics in the education market.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_provider_column(df: pd.DataFrame) -> str:
    """
    Get the provider column name from the DataFrame.
    
    Args:
        df: Input DataFrame containing the data
        
    Returns:
        The provider column name
    """
    if 'kouluttaja' in df.columns:
        return 'kouluttaja'
    elif 'koulutuksenJarjestaja' in df.columns:
        return 'koulutuksenJarjestaja'
    else:
        raise KeyError("Neither 'kouluttaja' nor 'koulutuksenJarjestaja' found in DataFrame columns")

def filter_qualification_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data for specific qualification types (ammattitutkinto and erikoisammattitutkinto).
    
    Args:
        df: Input DataFrame containing the data
        
    Returns:
        DataFrame filtered for specific qualification types
    """
    try:
        valid_types = ['ammattitutkinto', 'erikoisammattitutkinto']
        filtered_df = df[df['tutkintotyyppi'].isin(valid_types)].copy()
        logger.info(f"Filtered data for qualification types: {valid_types}")
        return filtered_df
    except Exception as e:
        logger.error(f"Error filtering qualification types: {str(e)}")
        raise

def calculate_total_students(
    df: pd.DataFrame,
    provider: str,
    provider_col: str = 'koulutuksenJarjestaja',
    subcontractor_col: str = 'hankintakoulutuksenJarjestaja',
    value_col: str = 'nettoopiskelijamaaraLkm'
) -> Dict[str, float]:
    """Calculate total student numbers for a provider.
    
    Args:
        df: DataFrame containing market data
        provider: Provider to analyze
        provider_col: Column containing main provider names
        subcontractor_col: Column containing subcontractor names
        value_col: Column containing volume values
        
    Returns:
        Dictionary with total student numbers and percentages
    """
    try:
        # Calculate main provider students
        main_students = df[df[provider_col] == provider][value_col].sum()
        
        # Calculate subcontractor students if the column exists
        sub_students = 0
        if subcontractor_col in df.columns:
            sub_students = df[df[subcontractor_col] == provider][value_col].sum()
        
        # Calculate totals and percentages
        total_students = main_students + sub_students
        main_percentage = (main_students / total_students * 100) if total_students > 0 else 0
        sub_percentage = (sub_students / total_students * 100) if total_students > 0 else 0
        
        return {
            'total_students': total_students,
            'main_provider_students': main_students,
            'subcontractor_students': sub_students,
            'main_provider_percentage': main_percentage,
            'subcontractor_percentage': sub_percentage
        }
        
    except Exception as e:
        logger.error(f"Error calculating total students: {str(e)}")
        raise

def calculate_market_shares(
    df: pd.DataFrame,
    group_cols: Union[list, None] = None,
    value_col: str = 'nettoopiskelijamaaraLkm',
    provider_col: str = 'koulutuksenJarjestaja',
    subcontractor_col: str = 'hankintakoulutuksenJarjestaja',
    provider_list: Union[list, None] = None,
    min_market_share: float = 0.0
) -> pd.DataFrame:
    """Calculate market shares for providers.
    
    Args:
        df: DataFrame containing market data
        group_cols: List of columns to group by (e.g. ['tilastovuosi', 'tutkinto'])
        value_col: Column containing volume values
        provider_col: Column containing provider names
        subcontractor_col: Column containing subcontractor names
        provider_list: Optional list of providers to include
        min_market_share: Minimum market share threshold
        
    Returns:
        DataFrame with market shares and rankings
    """
    try:
        # Use default group_cols if none provided
        if group_cols is None:
            group_cols = []
        elif not isinstance(group_cols, list):
            group_cols = [group_cols]
            
        # Calculate provider volumes
        def calculate_provider_volume(df_group, provider):
            main_volume = df_group[df_group[provider_col] == provider][value_col].sum()
            sub_volume = 0
            if subcontractor_col in df_group.columns:
                sub_volume = df_group[df_group[subcontractor_col] == provider][value_col].sum()
            return main_volume + sub_volume
        
        # Get unique providers
        providers = set(df[provider_col].unique())
        if subcontractor_col in df.columns:
            providers.update(df[subcontractor_col].dropna().unique())
        
        # Filter for specific providers if requested
        if provider_list is not None:
            providers = providers.intersection(provider_list)
        
        # Calculate volumes and market shares for each group
        result_data = []
        for group_values, group_df in df.groupby(group_cols):
            group_dict = dict(zip(group_cols, group_values)) if group_cols else {}
            total_volume = group_df[value_col].sum()
            
            for provider in providers:
                provider_volume = calculate_provider_volume(group_df, provider)
                market_share = (provider_volume / total_volume * 100) if total_volume > 0 else 0
                
                if market_share >= min_market_share:
                    result_data.append({
                        **group_dict,
                        'provider': provider,
                        'volume': provider_volume,
                        'market_share': market_share
                    })
        
        result = pd.DataFrame(result_data)
        if not result.empty:
            # Add rankings within each group
            result['rank'] = result.groupby(group_cols)['volume'].rank(ascending=False)
            
            # Sort by provider and group columns
            sort_cols = ['provider'] + (group_cols if group_cols else [])
            result = result.sort_values(sort_cols)
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating market shares: {str(e)}")
        raise

def calculate_growth_trends(df: pd.DataFrame, year_col: str = 'tilastovuosi', value_col: str = 'nettoopiskelijamaaraLkm',
                          group_cols: List[str] = None, provider_col: str = 'koulutuksenJarjestaja') -> pd.DataFrame:
    """Calculate growth trends for providers.
    
    Args:
        df: DataFrame containing market data
        year_col: Column containing year information (default: 'tilastovuosi')
        value_col: Column containing values to analyze (default: 'nettoopiskelijamaaraLkm')
        group_cols: Columns to group by (default: None)
        provider_col: Column containing provider information (default: 'koulutuksenJarjestaja')
        
    Returns:
        DataFrame containing growth trend analysis
    """
    try:
        # Calculate provider volumes (both roles)
        main_volumes = df.groupby([year_col, provider_col])[value_col].sum().reset_index()
        main_volumes = main_volumes.rename(columns={provider_col: 'provider'})
        
        # Add subcontractor volumes if the column exists
        if 'hankintakoulutuksenJarjestaja' in df.columns:
            sub_volumes = df.groupby([year_col, 'hankintakoulutuksenJarjestaja'])[value_col].sum().reset_index()
            sub_volumes = sub_volumes.rename(columns={'hankintakoulutuksenJarjestaja': 'provider'})
            volumes = pd.concat([main_volumes, sub_volumes])
            volumes = volumes.groupby([year_col, 'provider'])[value_col].sum().reset_index()
        else:
            volumes = main_volumes
            
        # Calculate year-over-year changes
        volumes = volumes.sort_values([year_col, 'provider'])
        volumes['volume_change'] = volumes.groupby('provider')[value_col].pct_change() * 100
        
        # Calculate compound annual growth rate (CAGR)
        min_year = volumes[year_col].min()
        max_year = volumes[year_col].max()
        n_years = max_year - min_year
        
        if n_years > 0:
            cagr = volumes.groupby('provider').apply(
                lambda x: ((x[value_col].iloc[-1] / x[value_col].iloc[0]) ** (1/n_years) - 1) * 100
            ).reset_index()
            cagr.columns = ['provider', 'cagr']
            volumes = pd.merge(volumes, cagr, on='provider', how='left')
        else:
            volumes['cagr'] = 0
            
        return volumes
        
    except Exception as e:
        logger.error(f"Error calculating growth trends: {str(e)}")
        raise

def calculate_provider_counts(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Calculate the number of unique providers for each group.
    
    Args:
        df: DataFrame containing market data
        group_cols: Columns to group by
        
    Returns:
        DataFrame with provider counts
    """
    try:
        # Count unique main providers
        main_providers = df.groupby(group_cols)['koulutuksenJarjestaja'].nunique()
        
        # Count unique subcontractors if the column exists
        if 'hankintakoulutuksenJarjestaja' in df.columns:
            sub_providers = df.groupby(group_cols)['hankintakoulutuksenJarjestaja'].nunique()
            provider_count = main_providers.add(sub_providers, fill_value=0)
        else:
            provider_count = main_providers
        
        # Create result DataFrame
        result = pd.DataFrame({
            'provider_count': provider_count
        }).reset_index()
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating provider counts: {str(e)}")
        raise

def calculate_provider_rankings(
    df: pd.DataFrame,
    group_cols: Union[list, None] = None,
    value_col: str = 'nettoopiskelijamaaraLkm',
    provider_col: str = 'koulutuksenJarjestaja'
) -> pd.DataFrame:
    """Calculate provider rankings based on volumes.
    
    Args:
        df: DataFrame containing market data
        group_cols: List of columns to group by
        value_col: Column containing volume values
        provider_col: Column containing provider names
        
    Returns:
        DataFrame with provider rankings
    """
    try:
        # Calculate market shares to get volumes and rankings
        result = calculate_market_shares(
            df=df,
            group_cols=group_cols,
            value_col=value_col,
            provider_col=provider_col
        )
        
        # Keep only necessary columns
        result = result[['provider', 'volume', 'rank'] + (group_cols if group_cols else [])]
        
        return result.sort_values('rank')
        
    except Exception as e:
        logger.error(f"Error in calculate_provider_rankings: {str(e)}")
        raise

def analyze_provider_roles(
    df: pd.DataFrame,
    provider: str,
    provider_col: str = 'koulutuksenJarjestaja',
    subcontractor_col: str = 'hankintakoulutuksenJarjestaja',
    value_col: str = 'nettoopiskelijamaaraLkm'
) -> pd.DataFrame:
    """Analyze provider roles (main provider vs subcontractor).
    
    Args:
        df: DataFrame containing market data
        provider: Provider to analyze
        provider_col: Column containing main provider names
        subcontractor_col: Column containing subcontractor names
        value_col: Column containing volume values
        
    Returns:
        DataFrame with provider role analysis
    """
    try:
        # Calculate main provider volumes
        main_volumes = df[df[provider_col] == provider].groupby('tutkinto')[value_col].sum()
        
        # Calculate subcontractor volumes if the column exists
        if subcontractor_col in df.columns:
            sub_volumes = df[df[subcontractor_col] == provider].groupby('tutkinto')[value_col].sum()
        else:
            sub_volumes = pd.Series(0, index=main_volumes.index)
        
        # Combine results
        result = pd.DataFrame({
            'tutkinto': main_volumes.index,
            'main_provider_volume': main_volumes.values,
            'subcontractor_volume': sub_volumes.reindex(main_volumes.index).fillna(0).values
        })
        
        # Calculate totals and percentages
        result['total_volume'] = result['main_provider_volume'] + result['subcontractor_volume']
        result['main_provider_percentage'] = (result['main_provider_volume'] / result['total_volume'] * 100).fillna(0)
        result['subcontractor_percentage'] = (result['subcontractor_volume'] / result['total_volume'] * 100).fillna(0)
        
        return result.sort_values('total_volume', ascending=False)
        
    except Exception as e:
        logger.error(f"Error analyzing provider roles: {str(e)}")
        raise

def track_market_shares(
    df: pd.DataFrame,
    provider: str,
    provider_col: str = 'koulutuksenJarjestaja',
    subcontractor_col: str = 'hankintakoulutuksenJarjestaja',
    year_col: str = 'tilastovuosi',
    value_col: str = 'nettoopiskelijamaaraLkm'
) -> pd.DataFrame:
    """Track market shares over time.
    
    Args:
        df: DataFrame containing market data
        provider: Provider to analyze
        provider_col: Column containing main provider names
        subcontractor_col: Column containing subcontractor names
        year_col: Column containing year values
        value_col: Column containing volume values
        
    Returns:
        DataFrame with market share tracking
    """
    try:
        # Calculate provider volumes
        def calculate_provider_volume(df_group):
            main_volume = df_group[df_group[provider_col] == provider][value_col].sum()
            sub_volume = 0
            if subcontractor_col in df_group.columns:
                sub_volume = df_group[df_group[subcontractor_col] == provider][value_col].sum()
            return main_volume + sub_volume
        
        # Calculate total market volumes
        def calculate_total_volume(df_group):
            return df_group[value_col].sum()
        
        # Group by year and qualification
        groups = df.groupby([year_col, 'tutkinto'])
        
        # Calculate volumes and market shares
        result = []
        for (year, qual), group in groups:
            provider_volume = calculate_provider_volume(group)
            total_volume = calculate_total_volume(group)
            market_share = (provider_volume / total_volume * 100) if total_volume > 0 else 0
            
            result.append({
                'tilastovuosi': year,
                'tutkinto': qual,
                'provider_volume': provider_volume,
                'total_market_volume': total_volume,
                'market_share': market_share
            })
        
        return pd.DataFrame(result).sort_values(['tilastovuosi', 'tutkinto'])
        
    except Exception as e:
        logger.error(f"Error tracking market shares: {str(e)}")
        raise

def analyze_market(df: pd.DataFrame, year_col: str = 'tilastovuosi', value_col: str = 'nettoopiskelijamaaraLkm', 
                  group_cols: List[str] = None, provider_col: str = 'koulutuksenJarjestaja') -> Dict[str, pd.DataFrame]:
    """Perform comprehensive market analysis.
    
    Args:
        df: DataFrame containing market data
        year_col: Column containing year information (default: 'tilastovuosi')
        value_col: Column containing values to analyze (default: 'nettoopiskelijamaaraLkm')
        group_cols: Columns to group by (default: None)
        provider_col: Column containing provider information (default: 'koulutuksenJarjestaja')
        
    Returns:
        Dictionary containing various market analysis results
    """
    try:
        # Calculate market shares
        market_shares = calculate_market_shares(df, group_cols=group_cols, value_col=value_col, provider_col=provider_col)
        
        # Calculate provider rankings
        rankings = calculate_provider_rankings(df, group_cols=group_cols, value_col=value_col, provider_col=provider_col)
        
        # Calculate growth trends
        growth_trends = calculate_growth_trends(df, year_col=year_col, value_col=value_col, 
                                             group_cols=group_cols, provider_col=provider_col)
        
        # Calculate market concentration
        concentration = calculate_market_concentration(df, value_col=value_col, group_cols=group_cols)
        
        return {
            'market_shares': market_shares,
            'rankings': rankings,
            'growth_trends': growth_trends,
            'concentration': concentration
        }
        
    except Exception as e:
        logger.error(f"Error analyzing market: {str(e)}")
        raise

def get_provider_qualifications(
    df: pd.DataFrame,
    provider: str,
    provider_col: Optional[str] = None
) -> Set[str]:
    """
    Get unique qualifications for a specific provider.
    
    Args:
        df: Input DataFrame containing the data
        provider: Provider name to analyze
        provider_col: Optional column name for provider filtering
        
    Returns:
        Set of unique qualifications
    """
    try:
        # Get the provider column name
        provider_col = provider_col or _get_provider_column(df)
        
        # Get provider's qualifications
        provider_data = df[
            (df[provider_col] == provider) |
            (df['hankintakoulutuksenJarjestaja'] == provider)
        ].copy()
        
        return set(provider_data['tutkinto'].unique())
        
    except Exception as e:
        logger.error(f"Error getting provider qualifications: {str(e)}")
        raise

def analyze_qualification_volumes(
    df: pd.DataFrame,
    provider: str,
    year: int,
    provider_col: str = 'koulutuksenJarjestaja',
    subcontractor_col: str = 'hankintakoulutuksenJarjestaja',
    year_col: str = 'tilastovuosi',
    value_col: str = 'nettoopiskelijamaaraLkm'
) -> pd.DataFrame:
    """Analyze qualification volumes for a provider.
    
    Args:
        df: DataFrame containing market data
        provider: Provider to analyze
        year: Year to analyze
        provider_col: Column containing main provider names
        subcontractor_col: Column containing subcontractor names
        year_col: Column containing year values
        value_col: Column containing volume values
        
    Returns:
        DataFrame with qualification volumes and percentages
    """
    try:
        # Filter data for the specified year
        df = df[df[year_col] == year].copy()
        
        # Calculate main provider volumes
        main_volumes = df[df[provider_col] == provider].groupby('tutkinto')[value_col].sum()
        
        # Calculate subcontractor volumes if the column exists
        if subcontractor_col in df.columns:
            sub_volumes = df[df[subcontractor_col] == provider].groupby('tutkinto')[value_col].sum()
        else:
            sub_volumes = pd.Series(0, index=main_volumes.index)
        
        # Combine results
        result = pd.DataFrame({
            'tutkinto': main_volumes.index,
            'main_provider_volume': main_volumes.values,
            'subcontractor_volume': sub_volumes.reindex(main_volumes.index).fillna(0).values
        })
        
        # Calculate totals and percentages
        result['total_volume'] = result['main_provider_volume'] + result['subcontractor_volume']
        result['main_provider_percentage'] = (result['main_provider_volume'] / result['total_volume'] * 100).fillna(0)
        result['subcontractor_percentage'] = (result['subcontractor_volume'] / result['total_volume'] * 100).fillna(0)
        
        return result.sort_values('total_volume', ascending=False)
        
    except Exception as e:
        logger.error(f"Error analyzing qualification volumes: {str(e)}")
        raise

def calculate_year_over_year_changes(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    provider_col: str = 'koulutuksenJarjestaja',
    subcontractor_col: str = 'hankintakoulutuksenJarjestaja',
    year_col: str = 'tilastovuosi',
    value_col: str = 'nettoopiskelijamaaraLkm'
) -> pd.DataFrame:
    """Calculate year-over-year changes in volumes and market shares.
    
    Args:
        df: DataFrame containing market data
        start_year: Starting year for comparison
        end_year: Ending year for comparison
        provider_col: Column containing main provider names
        subcontractor_col: Column containing subcontractor names
        year_col: Column containing year values
        value_col: Column containing volume values
        
    Returns:
        DataFrame with year-over-year changes
    """
    try:
        # Filter data for the specified years
        df_start = df[df[year_col] == start_year].copy()
        df_end = df[df[year_col] == end_year].copy()
        
        # Calculate volumes for each year
        def calculate_provider_volumes(df):
            # Calculate main provider volumes
            main_volumes = df.groupby(provider_col)[value_col].sum()
            
            # Add subcontractor volumes if available
            if subcontractor_col in df.columns:
                sub_volumes = df.groupby(subcontractor_col)[value_col].sum()
                # Combine volumes, handling missing values
                total_volumes = main_volumes.add(sub_volumes, fill_value=0)
            else:
                total_volumes = main_volumes
                
            return total_volumes
        
        # Get volumes for both years
        start_volumes = calculate_provider_volumes(df_start)
        end_volumes = calculate_provider_volumes(df_end)
        
        # Get all providers
        all_providers = pd.Index(set(start_volumes.index) | set(end_volumes.index))
        
        # Create result DataFrame
        result = pd.DataFrame(index=all_providers)
        result['volume_start'] = start_volumes.reindex(all_providers).fillna(0)
        result['volume_end'] = end_volumes.reindex(all_providers).fillna(0)
        
        # Calculate changes
        result['volume_change'] = result['volume_end'] - result['volume_start']
        result['volume_change_pct'] = (result['volume_change'] / result['volume_start'] * 100).fillna(0)
        
        # Calculate market shares
        total_start = result['volume_start'].sum()
        total_end = result['volume_end'].sum()
        result['market_share_start'] = (result['volume_start'] / total_start * 100).fillna(0)
        result['market_share_end'] = (result['volume_end'] / total_end * 100).fillna(0)
        result['market_share_change'] = result['market_share_end'] - result['market_share_start']
        
        # Reset index and rename
        result = result.reset_index().rename(columns={'index': 'provider'})
        
        return result.sort_values('volume_change_pct', ascending=False)
        
    except Exception as e:
        logger.error(f"Error calculating year-over-year changes: {str(e)}")
        raise

def analyze_market_share_changes(df: pd.DataFrame, start_year: int, end_year: int, min_market_share: float = 0.0) -> pd.DataFrame:
    """Analyze changes in market shares between two years.
    
    Args:
        df: DataFrame containing market data
        start_year: Starting year for analysis
        end_year: Ending year for analysis
        min_market_share: Minimum market share threshold (default: 0.0)
        
    Returns:
        DataFrame with market share changes
    """
    try:
        # Filter data for start and end years
        start_data = df[df['tilastovuosi'] == start_year].copy()
        end_data = df[df['tilastovuosi'] == end_year].copy()
        
        # Handle missing subcontractor data
        if 'hankintakoulutuksenJarjestaja' in start_data.columns:
            start_data['hankintakoulutuksenJarjestaja'] = start_data['hankintakoulutuksenJarjestaja'].replace('Tieto puuttuu', None)
        if 'hankintakoulutuksenJarjestaja' in end_data.columns:
            end_data['hankintakoulutuksenJarjestaja'] = end_data['hankintakoulutuksenJarjestaja'].replace('Tieto puuttuu', None)
        
        # Calculate market shares for both years
        start_shares = calculate_market_shares(start_data, min_market_share)
        end_shares = calculate_market_shares(end_data, min_market_share)
        
        # If either period has no data, return empty DataFrame
        if start_shares.empty or end_shares.empty:
            return pd.DataFrame(columns=[
                'provider', 'market_share_start', 'market_share_end',
                'volume_start', 'volume_end', 'volume_change', 'volume_change_pct',
                'market_share_change', 'market_share_change_pct'
            ])
        
        # Merge start and end data
        result = pd.merge(
            start_shares[['provider', 'market_share', 'volume']],
            end_shares[['provider', 'market_share', 'volume']],
            on='provider',
            how='outer',
            suffixes=('_start', '_end')
        ).fillna(0)
        
        # Calculate changes
        result['volume_change'] = result['volume_end'] - result['volume_start']
        result['market_share_change'] = result['market_share_end'] - result['market_share_start']
        
        # Calculate percentage changes
        result['volume_change_pct'] = result.apply(
            lambda row: ((row['volume_end'] - row['volume_start']) / row['volume_start'] * 100) 
            if row['volume_start'] > 0 else (float('inf') if row['volume_end'] > 0 else 0),
            axis=1
        )
        
        result['market_share_change_pct'] = result.apply(
            lambda row: ((row['market_share_end'] - row['market_share_start']) / row['market_share_start'] * 100)
            if row['market_share_start'] > 0 else (float('inf') if row['market_share_end'] > 0 else 0),
            axis=1
        )
        
        # Sort by market share change
        return result.sort_values('market_share_change', ascending=False)
        
    except Exception as e:
        logger.error(f"Error analyzing market share changes: {str(e)}")
        raise

def calculate_market_concentration(
    df: pd.DataFrame,
    group_cols: Union[list, None] = None,
    value_col: str = 'nettoopiskelijamaaraLkm',
    provider_col: str = 'koulutuksenJarjestaja',
    subcontractor_col: str = 'hankintakoulutuksenJarjestaja'
) -> pd.DataFrame:
    """Calculate market concentration metrics.
    
    Args:
        df: DataFrame containing market data
        group_cols: List of columns to group by (e.g. ['tilastovuosi', 'tutkinto'])
        value_col: Column containing volume values
        provider_col: Column containing provider names
        subcontractor_col: Column containing subcontractor names
        
    Returns:
        DataFrame with market concentration metrics
    """
    try:
        # Calculate market shares first
        market_shares = calculate_market_shares(
            df=df,
            group_cols=group_cols,
            value_col=value_col,
            provider_col=provider_col,
            subcontractor_col=subcontractor_col
        )
        
        # Calculate concentration metrics for each group
        result_data = []
        for group_values, group_df in market_shares.groupby(group_cols if group_cols else []):
            group_dict = dict(zip(group_cols, group_values)) if group_cols else {}
            
            # Sort market shares in descending order
            sorted_shares = group_df['market_share'].sort_values(ascending=False)
            
            # Calculate concentration metrics
            metrics = {
                'provider_count': len(sorted_shares),
                'cr1': sorted_shares.iloc[0] if len(sorted_shares) >= 1 else 0,  # Top provider share
                'cr3': sorted_shares.iloc[:3].sum() if len(sorted_shares) >= 3 else sorted_shares.sum(),  # Top 3 share
                'cr5': sorted_shares.iloc[:5].sum() if len(sorted_shares) >= 5 else sorted_shares.sum(),  # Top 5 share
                'hhi': (sorted_shares ** 2).sum() / 100  # Herfindahl-Hirschman Index
            }
            
            result_data.append({**group_dict, **metrics})
        
        result = pd.DataFrame(result_data)
        
        # Sort by group columns if any
        if group_cols:
            result = result.sort_values(group_cols)
            
        return result
        
    except Exception as e:
        logger.error(f"Error calculating market concentration: {str(e)}")
        raise 