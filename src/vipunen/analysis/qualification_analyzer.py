"""
Qualification analysis utilities for education market analysis.
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

def calculate_cagr_for_groups(df: pd.DataFrame, 
                         groupby_columns: List[str], 
                         value_column: str, 
                         last_year: Optional[int] = None,
                         qual_type_column: Optional[str] = None,
                         year_column: str = None) -> pd.DataFrame:
    """
    Calculate the Compound Annual Growth Rate (CAGR) for each group in a DataFrame.
    
    Args:
        df: DataFrame containing the data with a year column
        groupby_columns: List of column names to group the DataFrame by
        value_column: Name of column containing values for CAGR calculation
            If this is a year-specific column like "2022_yhteensä", the function will
            look for corresponding first year column.
        last_year: Optional last year to include in the CAGR calculation
        qual_type_column: Optional column name for qualification type information
        year_column: Column name containing year information. If None, will try
                    'tilastovuosi' and 'Year' in that order.
    
    Returns:
        DataFrame containing calculated CAGR for each group, with columns:
        - tutkinto (or first groupby column)
        - CAGR
        - First Year
        - Last Year
        - First Year Volume
        - Last Year Volume
        - Years Present
        - Qualification Type (if qual_type_column is provided)
    
    Notes:
        - Handles cases where start or end value is zero
        - Infinite growth represented by np.inf
        - Decline to zero represented by -np.inf
    """
    # Initialize an empty list for CAGR DataFrames
    cagr_list = []
    
    # Determine the year column name
    if year_column is None:
        if 'tilastovuosi' in df.columns:
            year_column = 'tilastovuosi'
        elif 'Year' in df.columns:
            year_column = 'Year'
        else:
            logger.error("No year column found in the DataFrame")
            return pd.DataFrame()
    
    # Filter the DataFrame if a last_year parameter is provided
    if last_year is not None:
        df = df[df[year_column] <= last_year]
    
    # Group the DataFrame by the columns specified in groupby_columns
    grouped = df.groupby(groupby_columns)
    
    # Check if value_column is a year-specific column (like "2022_yhteensä")
    year_specific = False
    if "_" in value_column:
        try:
            end_year = int(value_column.split("_")[0])
            column_suffix = "_".join(value_column.split("_")[1:])
            year_specific = True
            logger.info(f"Detected year-specific column: {value_column} (year: {end_year}, suffix: {column_suffix})")
        except (ValueError, IndexError):
            year_specific = False
    
    # Iterate over each group to calculate CAGR
    for name, group in grouped:
        # For year-specific columns, we need different handling
        if year_specific:
            # Find the first year with data
            years = sorted([int(col.split("_")[0]) for col in group.columns 
                           if "_" in col and col.split("_")[1:] == column_suffix.split("_")])
            
            if len(years) < 2:
                logger.warning(f"Not enough years for CAGR calculation for {name}")
                continue
                
            start_year = min(years)
            end_year = max(years)
            
            # Construct column names
            start_col = f"{start_year}_{column_suffix}"
            end_col = f"{end_year}_{column_suffix}"
            
            # Skip if columns don't exist
            if start_col not in group.columns or end_col not in group.columns:
                logger.warning(f"Missing columns for CAGR calculation: {start_col} or {end_col}")
                continue
                
            # Get start and end values
            start_value = group[start_col].iloc[0] if not group.empty else 0
            end_value = group[end_col].iloc[0] if not group.empty else 0
            
            periods = end_year - start_year
            first_year_offered = start_year
            last_year_offered = end_year
            
        # Handle standard case for non-year-specific columns
        else:
            # Ensure there is more than one year of data for the group
            if len(group) > 1:
                # Sort the group by year to ensure start and end values are correct
                group = group.sort_values(year_column)
                
                # Get the first and last value and the number of periods (years)
                end_value = group[value_column].iloc[-1]
                start_value = group[value_column].iloc[0]
                
                # Track the actual first and last year offered
                first_year_offered = group[year_column].iloc[0]
                last_year_offered = group[year_column].iloc[-1]
                periods = last_year_offered - first_year_offered
            else:
                # Not enough data for CAGR
                continue
                
        # Skip if periods is zero (same year)
        if periods == 0:
            continue
            
        # Handle special or extreme cases for CAGR calculation
        if start_value == 0 and end_value == 0:
            cagr = 0  # No growth, value remains zero
        elif start_value == 0:
            cagr = np.inf  # Infinite growth from zero
        elif end_value == 0:
            cagr = -np.inf  # Decline to zero
        elif end_value / start_value > 1e5:  # Arbitrarily chosen threshold for extreme growth
            cagr = np.inf  # Represent extreme growth as infinite
        else:
            try:
                cagr = ((end_value / start_value) ** (1 / periods) - 1) * 100  # CAGR formula
            except ZeroDivisionError:
                cagr = np.nan  # Undefined, occurs if start and end years are the same
        
        # Create a DataFrame for this group's CAGR
        if isinstance(name, tuple):
            # For multiple groupby columns, create a dict with column names as keys
            cagr_dict = {groupby_columns[i]: name[i] for i in range(len(groupby_columns))}
        else:
            # For a single groupby column
            cagr_dict = {groupby_columns[0]: name}
        
        # Assign the numeric CAGR value (handle inf/nan)
        cagr_dict['CAGR'] = cagr 
        cagr_dict['First Year'] = first_year_offered
        cagr_dict['Last Year'] = last_year_offered
        cagr_dict['First Year Volume'] = start_value
        cagr_dict['Last Year Volume'] = end_value
        
        # Count the actual number of distinct years with data
        if year_specific:
            # Count years from column names
            years_present = len([col for col in group.columns if col.startswith(str(start_year)) or 
                                col.startswith(str(end_year))])
        else:
            years_present = len(group[year_column].unique())
            
        cagr_dict['Years Present'] = years_present
        
        # Add qualification type if the column was provided
        if qual_type_column is not None and qual_type_column in group.columns and not group.empty:
            # Get the qualification type, splitting at whitespace and taking the last word if needed
            qual_type = group[qual_type_column].iloc[0]
            if isinstance(qual_type, str) and " " in qual_type:
                qual_type = qual_type.split()[-1]
            cagr_dict['Qualification Type'] = qual_type
        
        cagr_list.append(pd.DataFrame([cagr_dict]))
    
    # Combine all individual CAGR DataFrames
    if cagr_list:
        result_df = pd.concat(cagr_list, ignore_index=True)
        
        logger.info(f"Calculated CAGR for {len(result_df)} groups")
        return result_df
    else:
        # Return an empty DataFrame with appropriate columns
        columns = groupby_columns + [
            'CAGR', 'First Year', 'Last Year', 
            'First Year Volume', 'Last Year Volume', 'Years Present'
        ]
        
        if qual_type_column is not None:
            columns.append('Qualification Type')
            
        logger.warning("No CAGR data generated")
        return pd.DataFrame(columns=columns)

def calculate_yoy_growth(df: pd.DataFrame, 
                    groupby_col: str, 
                    target_col: str, 
                    output_col: str, 
                    end_year: int, 
                    time_window: int, 
                    most_recent_complete_year: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate Year-over-Year (YoY) growth for a specified target column.
    
    Args:
        df: DataFrame containing the data
        groupby_col: Column to group by (e.g., 'tutkinto')
        target_col: Column containing values to calculate growth for
        output_col: Name for the new column containing growth rates
        end_year: The year for which to calculate growth
        time_window: Number of years to consider for growth calculation
        most_recent_complete_year: Optional most recent year with complete data
    
    Returns:
        DataFrame containing YoY growth calculations
    """
    # Replace infinities and NaNs with 0
    working_df = df.copy()
    working_df[target_col] = working_df[target_col].replace([np.inf, -np.inf, np.nan], 0)
    
    # Calculate years range to include
    years_list = [end_year - x for x in reversed(range(time_window))]
    years_str = ", ".join(map(str, years_list))
    
    # Filter the DataFrame to only include the relevant years
    filtered_df = working_df[working_df['tilastovuosi'].isin(years_list)]
    
    # Group by the specified column and calculate growth
    if len(filtered_df) > 0:
        # Group by the specified column and tilastovuosi, then sum the target column
        grouped_df = filtered_df.groupby([groupby_col, 'tilastovuosi'])[target_col].sum().reset_index()
        
        # Pivot to have years as columns
        pivot_df = grouped_df.pivot(index=groupby_col, columns='tilastovuosi', values=target_col)
        
        # Calculate growth percentage
        if time_window > 1 and len(pivot_df.columns) >= 2:
            start_year = years_list[0]
            end_year = years_list[-1]
            
            # Calculate percentage change from start to end
            if start_year in pivot_df.columns and end_year in pivot_df.columns:
                # Handle zeros in the start column to avoid division by zero
                pivot_df[output_col] = np.where(
                    pivot_df[start_year] == 0,
                    np.where(
                        pivot_df[end_year] == 0,
                        0,  # Both years are zero: 0% growth
                        np.inf  # Start is zero, end is not: infinite growth
                    ),
                    (pivot_df[end_year] - pivot_df[start_year]) / pivot_df[start_year] * 100
                )
            else:
                pivot_df[output_col] = np.nan
                logger.warning(f"Missing year data for growth calculation: start_year={start_year}, end_year={end_year}")
        else:
            pivot_df[output_col] = np.nan
            logger.warning(f"Insufficient time window or missing data for growth calculation")
        
        # Add trend column (growing or declining)
        pivot_df[f'{output_col}_trendi'] = np.where(pivot_df[output_col] > 0, 'Kasvussa', 'Laskussa')
        
        # Add year information
        pivot_df['tilastovuosi'] = end_year
        pivot_df['tilastovuodet'] = years_str
        
        # Reset index and select only needed columns
        result_df = pivot_df.reset_index()
        logger.info(f"Calculated {time_window}-year growth for {len(result_df)} groups ending in {end_year}")
        return result_df[[groupby_col, output_col, f'{output_col}_trendi', 'tilastovuosi', 'tilastovuodet']]
    else:
        # Return empty DataFrame with appropriate columns
        logger.warning(f"No data available for years {years_list}")
        return pd.DataFrame(columns=[groupby_col, output_col, f'{output_col}_trendi', 'tilastovuosi', 'tilastovuodet'])

def analyze_qualification_growth(volumes_by_qual: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Year-over-Year (YoY) change in qualification volumes.
    
    Args:
        volumes_by_qual: DataFrame with volume data by qualification and year
        
    Returns:
        pd.DataFrame: DataFrame with growth analysis for qualifications
    """
    # Check if we have the required columns
    if 'tutkinto' not in volumes_by_qual.columns:
        logger.error("Required column 'tutkinto' not found in input data")
        return pd.DataFrame()
    
    # Initialize list to store growth data
    growth_data = []
    
    # Process each qualification
    for qualification in volumes_by_qual['tutkinto'].unique():
        # Get data for this qualification
        qual_data = volumes_by_qual[
            volumes_by_qual['tutkinto'] == qualification
        ]
        
        # Extract years from column names
        years = sorted([int(col.split('_')[0]) for col in qual_data.columns if '_järjestäjänä' in col])
        
        # Need at least two years of data to calculate growth
        if len(years) < 2:
            logger.warning(f"Skipping growth calculation for {qualification} - not enough data")
            continue
        
        # Calculate year-over-year growth for each pair of consecutive years
        for i in range(1, len(years)):
            current_year = years[i]
            previous_year = years[i-1]
            
            # Skip if years are not consecutive
            if current_year != previous_year + 1:
                logger.warning(f"Skipping {previous_year}-{current_year} for {qualification} - years not consecutive")
                continue
            
            # Get columns for current and previous years
            current_provider_col = f"{current_year}_järjestäjänä"
            current_subcontractor_col = f"{current_year}_hankintana"
            prev_provider_col = f"{previous_year}_järjestäjänä"
            prev_subcontractor_col = f"{previous_year}_hankintana"
            
            # Check if all columns exist
            if not all(col in qual_data.columns for col in [current_provider_col, current_subcontractor_col, prev_provider_col, prev_subcontractor_col]):
                logger.warning(f"Missing volume data for {qualification} in years {previous_year}-{current_year}")
                continue
            
            # Extract volumes
            row = qual_data.iloc[0]
            current_provider_vol = row[current_provider_col]
            current_subcontractor_vol = row[current_subcontractor_col]
            current_total = current_provider_vol + current_subcontractor_vol
            
            prev_provider_vol = row[prev_provider_col]
            prev_subcontractor_vol = row[prev_subcontractor_col]
            prev_total = prev_provider_vol + prev_subcontractor_vol
            
            # Calculate growth percentages
            if prev_total > 0:
                total_growth = ((current_total / prev_total) - 1) * 100
            else:
                total_growth = np.inf if current_total > 0 else 0
            
            if prev_provider_vol > 0:
                provider_growth = ((current_provider_vol / prev_provider_vol) - 1) * 100
            else:
                provider_growth = np.inf if current_provider_vol > 0 else 0
            
            if prev_subcontractor_vol > 0:
                subcontractor_growth = ((current_subcontractor_vol / prev_subcontractor_vol) - 1) * 100
            else:
                subcontractor_growth = np.inf if current_subcontractor_vol > 0 else 0
            
            # Calculate role shift (change in the proportion of provider vs. subcontractor)
            prev_provider_share = prev_provider_vol / prev_total * 100 if prev_total > 0 else 0
            current_provider_share = current_provider_vol / current_total * 100 if current_total > 0 else 0
            provider_share_change = current_provider_share - prev_provider_share
            
            # Add record
            growth_data.append({
                'tilastovuosi': current_year,
                'tutkinto': qualification,
                'Current Total': current_total,
                'Previous Total': prev_total,
                'Total Growth (%)': round(total_growth, 2) if not np.isinf(total_growth) else None,
                'Provider Growth (%)': round(provider_growth, 2) if not np.isinf(provider_growth) else None,
                'Subcontractor Growth (%)': round(subcontractor_growth, 2) if not np.isinf(subcontractor_growth) else None,
                'Provider Share Change (pp)': round(provider_share_change, 2)
            })
    
    # Convert to DataFrame
    growth_df = pd.DataFrame(growth_data)
    
    if not growth_df.empty:
        # Sort by year (newest first) and then by qualification
        growth_df = growth_df.sort_values(
            by=['tilastovuosi', 'tutkinto'], 
            ascending=[False, True]
        )
        
        logger.info(f"Generated qualification growth analysis with {len(growth_df)} rows")
    else:
        logger.warning("No growth data generated - check if there are multiple years of data")
    
    return growth_df

def get_top_providers(df: pd.DataFrame, year: int, top_n: int = 5, 
                   provider_col: str = 'koulutuksenJarjestaja',
                   qual_col: str = 'tutkinto',
                   value_col: str = 'nettoopiskelijamaaraLkm') -> Dict[str, List[str]]:
    """
    Get the top institutions for each qualification for a given year.
    
    Args:
        df: DataFrame containing the educational data
        year: Year to analyze
        top_n: Number of top institutions to include
        provider_col: Column name for the provider
        qual_col: Column name for the qualification
        value_col: Column name for the volume metric
        
    Returns:
        Dictionary with qualification as key and list of top institutions as value
    """
    # Filter the DataFrame to only include the given year
    year_data = df[df['tilastovuosi'] == year].copy()
    
    if year_data.empty:
        logger.warning(f"No data found for year {year}")
        return {}
    
    # Aggregate volumes by qualification and provider
    agg_data = year_data.groupby([qual_col, provider_col])[value_col].sum().reset_index()
    
    # Rank providers within each qualification
    agg_data['rank'] = agg_data.groupby(qual_col)[value_col].rank(ascending=False, method='min')
    
    # Filter to include only top N providers for each qualification
    top_providers_df = agg_data[agg_data['rank'] <= top_n]
    
    # Convert to dictionary
    results = {}
    for qual in top_providers_df[qual_col].unique():
        providers = top_providers_df[top_providers_df[qual_col] == qual].sort_values('rank')[provider_col].tolist()
        results[qual] = providers
    
    logger.info(f"Identified top {top_n} providers for {len(results)} qualifications in {year}")
    return results 