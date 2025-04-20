"""Growth analysis functions for the Vipunen project."""
import numpy as np
import pandas as pd
from typing import List, Optional, Union

def calculate_cagr(
    df: pd.DataFrame,
    groupby_columns: List[str],
    value_column: str,
    last_year: Optional[int] = None
) -> pd.DataFrame:
    """Calculates the Compound Annual Growth Rate (CAGR) for each group in a DataFrame.
    
    Args:
        df: The DataFrame containing the data
        groupby_columns: Columns to group by
        value_column: Column containing the values to calculate CAGR for
        last_year: Optional last year to include in the calculation
        
    Returns:
        DataFrame with CAGR calculations for each group
    """
    # Filter by last year if specified
    if last_year is not None:
        df = df[df['tilastovuosi'] <= last_year]
    
    # Initialize list to store results
    cagr_list = []
    
    # Group data
    grouped = df.groupby(groupby_columns)
    
    for name, group in grouped:
        # Create result row base
        cagr_dict = {}
        for col, val in zip(groupby_columns, name if isinstance(name, tuple) else [name]):
            cagr_dict[col] = [val]
        
        if len(group) <= 1:
            # For single value or empty group, CAGR is 0
            cagr_dict['CAGR (%)'] = [0]
            cagr_list.append(pd.DataFrame(cagr_dict))
            continue
        
        # Sort by year
        group = group.sort_values('tilastovuosi')
        
        # Get values for CAGR calculation
        end_value = group[value_column].iloc[-1]
        start_value = group[value_column].iloc[0]
        periods = group['tilastovuosi'].iloc[-1] - group['tilastovuosi'].iloc[0]
        
        # Handle special cases
        if start_value == 0 and end_value == 0:
            cagr = 0
        elif start_value == 0:
            cagr = np.inf
        elif end_value == 0:
            cagr = -np.inf
        elif end_value / start_value > 1e5:
            cagr = np.inf
        else:
            try:
                cagr = ((end_value / start_value) ** (1 / periods) - 1) * 100
            except ZeroDivisionError:
                cagr = np.nan
        
        cagr_dict['CAGR (%)'] = [cagr]
        cagr_list.append(pd.DataFrame(cagr_dict))
    
    # Handle empty list case
    if not cagr_list:
        empty_dict = {col: [] for col in groupby_columns}
        empty_dict['CAGR (%)'] = []
        return pd.DataFrame(empty_dict)
    
    # Combine results
    return pd.concat(cagr_list, ignore_index=True)

def calculate_yoy_growth(
    df: pd.DataFrame,
    groupby_col: str,
    target_col: str,
    output_col: str,
    end_year: int,
    time_window: int,
    most_recent_complete_year: Optional[int] = None
) -> pd.DataFrame:
    """Calculate Year-over-Year (YoY) growth for specified columns.
    
    Args:
        df: Input DataFrame
        groupby_col: Column to group by
        target_col: Column to calculate growth for
        output_col: Name for the output growth column
        end_year: End year for the calculation
        time_window: Number of years to consider
        most_recent_complete_year: Optional most recent complete year
        
    Returns:
        DataFrame with YoY growth calculations
    """
    # Validate end year
    if most_recent_complete_year and end_year > most_recent_complete_year:
        raise ValueError(
            f"End year must be <= most recent complete year ({most_recent_complete_year})"
        )
    
    # Calculate start year
    start_year = end_year - time_window + 1
    
    # Filter data
    df_filtered = df[df['tilastovuosi'].isin(range(start_year, end_year + 1))]
    
    # If no data for the time window, return empty DataFrame with correct structure
    if len(df_filtered) == 0:
        return pd.DataFrame({
            groupby_col: [],
            output_col: [],
            f'{output_col}_trendi': [],
            'tilastovuosi': [],
            'tilastovuodet': []
        })
    
    # Group and aggregate
    df_grouped = df_filtered.groupby([groupby_col, 'tilastovuosi']).agg({
        target_col: 'sum'
    }).reset_index()
    
    # Calculate growth
    if time_window == 1:
        df_grouped[output_col] = df_grouped.groupby(groupby_col)[target_col].pct_change() * 100
    else:
        df_grouped[output_col] = df_grouped.groupby(groupby_col)[target_col].pct_change(
            periods=time_window - 1
        ) * 100
    
    # Filter to end year
    df_yoy = df_grouped[df_grouped['tilastovuosi'] == end_year].copy()
    
    # Add metadata
    df_yoy['tilastovuodet'] = f"{start_year}-{end_year}"
    df_yoy[f'{output_col}_trendi'] = np.where(
        df_yoy[output_col] > 0, 'Kasvussa', 'Laskussa'
    )
    
    # Handle NaN values
    df_yoy[output_col] = df_yoy[output_col].fillna(0)
    df_yoy[f'{output_col}_trendi'] = df_yoy[f'{output_col}_trendi'].fillna('Ei muutosta')
    
    return df_yoy[[
        groupby_col,
        output_col,
        f'{output_col}_trendi',
        'tilastovuosi',
        'tilastovuodet'
    ]]

def calculate_multiple_yoy_growth(
    df: pd.DataFrame,
    variables_dict: dict,
    time_window: int,
    groupby_col: str,
    most_recent_complete_year: Optional[int] = None
) -> pd.DataFrame:
    """Calculate YoY growth for multiple variables.
    
    Args:
        df: Input DataFrame
        variables_dict: Dictionary mapping target columns to output column names
        time_window: Number of years to consider
        groupby_col: Column to group by
        most_recent_complete_year: Optional most recent complete year
        
    Returns:
        DataFrame with multiple YoY growth calculations
    """
    dfs_dict = {}
    
    for target_col, output_col in variables_dict.items():
        dfs = []
        unique_years = sorted(df['tilastovuosi'].unique())
        unique_groupby_values = df[groupby_col].unique()
        
        # Skip if not enough years for the time window
        if len(unique_years) < time_window:
            continue
        
        for group_value in unique_groupby_values:
            df_filtered = df[df[groupby_col] == group_value]
            
            for year in unique_years[time_window-1:]:
                years_list = [year - x for x in reversed(range(time_window))]
                df_years = df_filtered[df_filtered['tilastovuosi'].isin(years_list)]
                
                # Only calculate if we have data for all years in the window
                if len(df_years) == time_window:
                    df_yoy = calculate_yoy_growth(
                        df=df_years,
                        groupby_col=groupby_col,
                        target_col=target_col,
                        output_col=output_col,
                        time_window=time_window,
                        end_year=years_list[-1],
                        most_recent_complete_year=most_recent_complete_year
                    )
                    dfs.append(df_yoy)
        
        if dfs:  # Only add to dict if we have results
            dfs_dict[target_col] = pd.concat(dfs, ignore_index=True)
    
    # Handle case where no results were calculated
    if not dfs_dict:
        empty_df = pd.DataFrame({
            groupby_col: [],
            'tilastovuosi': [],
            'tilastovuodet': []
        })
        for _, output_col in variables_dict.items():
            empty_df[output_col] = []
            empty_df[f'{output_col}_trendi'] = []
        return empty_df
    
    # Merge results
    result_df = dfs_dict.popitem()[1]
    for _, df in dfs_dict.items():
        result_df = pd.merge(
            result_df,
            df,
            on=[groupby_col, 'tilastovuosi', 'tilastovuodet'],
            how='outer'
        )
    
    return result_df 