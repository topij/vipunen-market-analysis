"""
Education Market Analysis module that focuses on analyzing vocational qualification data 
from a specific provider's perspective.
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

# Utility functions for education market analysis
def sum_new_columns(df: pd.DataFrame, suffixes: List[str] = ['_as_jarjestaja', '_as_hankinta']) -> pd.Series:
    """
    Sum the corresponding columns with the given suffixes.
    
    Args:
        df: DataFrame containing the columns to sum
        suffixes: List of suffixes used to identify the columns to sum
        
    Returns:
        Series containing the sum of the columns for each row
    """
    # Find columns ending with the suffixes
    for suffix in suffixes:
        for col in df.columns:
            if col.endswith(suffix):
                original_col = col.replace(suffix, '')
                col1 = original_col + suffixes[0]
                col2 = original_col + suffixes[1]
                
    return (df[col1] + df[col2])

def calculate_cagr_for_groups(
    df: pd.DataFrame, 
    groupby_columns: List[str], 
    value_column: str, 
    last_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate the Compound Annual Growth Rate (CAGR) for each group in a DataFrame.
    
    Args:
        df: DataFrame containing the data with a 'tilastovuosi' column for years
        groupby_columns: List of column names to group the DataFrame by
        value_column: Name of column containing values for CAGR calculation
        last_year: Optional last year to include in the CAGR calculation
    
    Returns:
        DataFrame containing calculated CAGR for each group, with CAGR in percentage format
    
    Notes:
        - Handles cases where start or end value is zero
        - Infinite growth represented by np.inf
        - Decline to zero represented by -np.inf
    """
    # Initialize an empty list for CAGR DataFrames
    cagr_list = []
    
    # Filter the DataFrame if a last_year parameter is provided
    if last_year is not None:
        df = df[df['tilastovuosi'] <= last_year]
    
    # Group the DataFrame by the columns specified in groupby_columns
    grouped = df.groupby(groupby_columns)
    
    # Iterate over each group to calculate CAGR
    for name, group in grouped:
        # Ensure there is more than one year of data for the group
        if len(group) > 1:
            # Sort the group by year to ensure start and end values are correct
            group = group.sort_values('tilastovuosi')
            
            # Get the first and last value and the number of periods (years)
            end_value = group[value_column].iloc[-1]
            start_value = group[value_column].iloc[0]
            
            # Track the actual first and last year offered
            first_year_offered = group['tilastovuosi'].iloc[0]
            last_year_offered = group['tilastovuosi'].iloc[-1]
            periods = last_year_offered - first_year_offered
            
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
            
            cagr_dict['CAGR'] = cagr
            cagr_dict['First Year Offered'] = first_year_offered
            cagr_dict['Last Year Offered'] = last_year_offered
            cagr_list.append(pd.DataFrame([cagr_dict]))
    
    # Combine all individual CAGR DataFrames
    if cagr_list:
        return pd.concat(cagr_list, ignore_index=True)
    else:
        # Return an empty DataFrame with appropriate columns if no CAGR could be calculated
        columns = groupby_columns + ['CAGR', 'First Year Offered', 'Last Year Offered']
        return pd.DataFrame(columns=columns)

def calculate_yoy_growth(
    df: pd.DataFrame, 
    groupby_col: str, 
    target_col: str, 
    output_col: str, 
    end_year: int, 
    time_window: int, 
    most_recent_complete_year: Optional[int] = None
) -> pd.DataFrame:
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
        return result_df[[groupby_col, output_col, f'{output_col}_trendi', 'tilastovuosi', 'tilastovuodet']]
    else:
        # Return empty DataFrame with appropriate columns
        return pd.DataFrame(columns=[groupby_col, output_col, f'{output_col}_trendi', 'tilastovuosi', 'tilastovuodet'])

def hae_tutkinto_nom(df: pd.DataFrame, role: str, qualifications: List[str]) -> pd.DataFrame:
    """
    Get the yearly metrics for specific qualifications, filtered by a particular role.
    
    Args:
        df: DataFrame containing the educational data
        role: Role of the institution (either 'koulutuksenJarjestaja' or 'hankintakoulutuksenJarjestaja')
        qualifications: List of qualifications to focus on
        
    Returns:
        DataFrame with aggregated yearly metrics for the specified qualifications and role
    """
    # Define the columns to be used in grouping
    columns = ['tilastovuosi', 'tutkinto']
    
    # Filter the DataFrame based on the list of qualifications
    filtered_df = df[df['tutkinto'].isin(qualifications)]
    
    # Append the role to the list of columns and group by those columns
    columns.append(role)
    filtered_df = filtered_df.groupby(by=columns)['nettoopiskelijamaaraLkm'].sum().reset_index()
    
    # Remove rows where the role information is missing
    filtered_df = filtered_df[filtered_df[role] != 'Tieto puuttuu']
    
    logger.info(f"Generated metrics for {role} with {filtered_df.shape[0]} rows")
    return filtered_df

def hae_tutkinnon_suurimmat_toimijat(df: pd.DataFrame, year: int, top_n: int = 5) -> Dict[str, List[str]]:
    """
    Get the top institutions for each qualification for a given year.
    
    Args:
        df: DataFrame containing the educational data
        year: Year to analyze
        top_n: Number of top institutions to include
        
    Returns:
        Dictionary with qualification as key and list of top institutions as value
    """
    # Filter the DataFrame to only include the given year and top N institutions
    # Use basic filtering instead of query to avoid variable context issues
    filtered_df = df.copy()
    if 'tilastovuosi' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['tilastovuosi'] == year]
    
    if 'Sijoitus markkinassa' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Sijoitus markkinassa'] <= top_n]
    
    # Group the data by qualification and get the top institutions for each qualification
    if not filtered_df.empty and 'tutkinto' in filtered_df.columns and 'kouluttaja' in filtered_df.columns:
        top_n_by_qualification = filtered_df.groupby('tutkinto')['kouluttaja'].apply(list).to_dict()
        return top_n_by_qualification
    else:
        return {}

def hae_kouluttaja_nom(
    df: pd.DataFrame, 
    kouluttaja: List[str],
    nom_as_jarjestaja: str = 'nettoopiskelijamaaraLkm', 
    nom_as_hankinta: str = 'nettoopiskelijamaaraLkm'
) -> pd.DataFrame:
    """
    Get yearly metrics for a specific educational institution, separated by roles as 
    main organizer and contracted training arranger.
    
    Args:
        df: DataFrame containing the educational data
        kouluttaja: List of institution names to focus on
        nom_as_jarjestaja: Column name for the metric as main organizer
        nom_as_hankinta: Column name for the metric as contracted arranger
        
    Returns:
        DataFrame with aggregated yearly metrics for the specified institution
    """
    # Filter for rows where the institution is the main organizer and aggregate by year
    df_koulutuksenjarjestaja = df[df['koulutuksenJarjestaja'].isin(kouluttaja)].groupby(
        by=['tilastovuosi', 'koulutuksenJarjestaja']
    )[nom_as_jarjestaja].sum().reset_index()
    
    # Filter for rows where the institution is the contracted arranger and aggregate by year
    df_hankintakoulutuksenjarjestaja = df[df['hankintakoulutuksenJarjestaja'].isin(kouluttaja)].groupby(
        by=['tilastovuosi', 'hankintakoulutuksenJarjestaja']
    )[nom_as_hankinta].sum().reset_index()
    
    suffixes = ('_as_jarjestaja', '_as_hankinta')
    
    # Rename the organization columns to a common name for merging
    df_koulutuksenjarjestaja.rename(columns={'koulutuksenJarjestaja': 'kouluttaja'}, inplace=True)
    df_hankintakoulutuksenjarjestaja.rename(columns={'hankintakoulutuksenJarjestaja': 'kouluttaja'}, inplace=True)
    
    # Merge the two DataFrames on year and institution
    df_molemmat = pd.merge(
        df_koulutuksenjarjestaja, 
        df_hankintakoulutuksenjarjestaja, 
        on=['tilastovuosi', 'kouluttaja'], 
        how='outer',
        suffixes=suffixes
    )
    
    # Calculate the sum of metrics across both roles
    df_molemmat['Yhteensä'] = sum_new_columns(df_molemmat, suffixes)
    
    # Calculate proportion of students served by the institution as main organizer
    df_molemmat['järjestäjä_osuus (%)'] = (
        df_molemmat[nom_as_jarjestaja + suffixes[0]] / df_molemmat['Yhteensä'] * 100
    ).fillna(0)
    
    return df_molemmat

class EducationMarketAnalyzer:
    """
    Analyzer for education market data focusing on a specific institution.
    
    This class implements the analysis steps outlined in the project description:
    1. Calculate the total volume of students for the institution under analysis
    2. Break down volume by provider role (main provider vs subcontractor)
    """
    
    def __init__(
        self, 
        data: pd.DataFrame,
        institution_names: List[str],
        year_col: str = 'tilastovuosi',
        degree_type_col: str = 'tutkintotyyppi',
        qualification_col: str = 'tutkinto',
        provider_col: str = 'koulutuksenJarjestaja',
        subcontractor_col: str = 'hankintakoulutuksenJarjestaja',
        volume_col: str = 'nettoopiskelijamaaraLkm',
        filter_degree_types: bool = False,
        filter_qualifications: Optional[List[str]] = None,
        reference_year: Optional[int] = None
    ):
        """
        Initialize the EducationMarketAnalyzer.
        
        Args:
            data: DataFrame containing the raw data
            institution_names: List of names that represent the institution under analysis
            year_col: Column name for the year
            degree_type_col: Column name for the degree type
            qualification_col: Column name for the qualification/degree name
            provider_col: Column name for the main provider
            subcontractor_col: Column name for the subcontractor
            volume_col: Column name for the student volume
            filter_degree_types: Whether to filter by degree types in total volume analysis
            filter_qualifications: Optional list of qualification names to filter by
            reference_year: Optional year to determine qualifications to include
                (will include all qualifications offered by the institution in this year)
        """
        self.data = data.copy()
        self.institution_names = institution_names
        self.year_col = year_col
        self.degree_type_col = degree_type_col
        self.qualification_col = qualification_col
        self.provider_col = provider_col
        self.subcontractor_col = subcontractor_col
        self.volume_col = volume_col
        self.filter_degree_types = filter_degree_types
        self.filter_qualifications = filter_qualifications
        
        # Store degree types for later filtering when needed
        self.degree_types = ['Ammattitutkinnot', 'Erikoisammattitutkinnot']
        
        # If reference year is provided, find qualifications offered by the institution in that year
        if reference_year is not None:
            self.filter_qualifications = self._get_qualifications_in_year(reference_year)
            logger.info(f"Using {len(self.filter_qualifications)} qualifications from reference year {reference_year}")
        
        # Log data dimensions
        logger.info(f"Loaded data with {len(self.data)} rows")
    
    def _get_qualifications_in_year(self, year: int) -> List[str]:
        """
        Get a list of qualifications offered by the institution in a specific year.
        
        Args:
            year: The reference year
            
        Returns:
            List[str]: List of qualification names
        """
        # First filter by degree types to ensure we only include the relevant types
        filtered_data = self._filter_data_by_degree_types()
        
        # Then filter to the specific year
        year_data = filtered_data[filtered_data[self.year_col] == year]
        
        if len(year_data) == 0:
            logger.warning(f"No data found for reference year {year}")
            return []
        
        # Find qualifications where the institution is either provider or subcontractor
        institution_quals = year_data[
            (year_data[self.provider_col].isin(self.institution_names)) |
            (year_data[self.subcontractor_col].isin(self.institution_names))
        ][self.qualification_col].unique().tolist()
        
        logger.info(f"Found {len(institution_quals)} qualifications offered by the institution in year {year}")
        return institution_quals
    
    def _filter_data_by_degree_types(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Filter data for ammattitutkinto and erikoisammattitutkinto degrees.
        
        Args:
            data: DataFrame to filter. If None, uses self.data
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if data is None:
            data = self.data
            
        filtered_data = data[data[self.degree_type_col].isin(self.degree_types)].copy()
        logger.info(f"Filtered data to {len(filtered_data)} rows for degree types: {self.degree_types}")
        return filtered_data
    
    def _filter_data_by_qualifications(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to include only specified qualifications.
        
        Args:
            data: DataFrame to filter
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if self.filter_qualifications is None or len(self.filter_qualifications) == 0:
            return data
            
        filtered_data = data[data[self.qualification_col].isin(self.filter_qualifications)].copy()
        logger.info(f"Filtered data to {len(filtered_data)} rows for {len(self.filter_qualifications)} specified qualifications")
        return filtered_data
    
    def analyze_total_volume(self) -> pd.DataFrame:
        """
        Calculate the total volume of students and break it down by provider role.
        
        Returns:
            pd.DataFrame: Summary DataFrame with volume breakdowns by year
        """
        # Apply degree type filtering if configured
        working_data = self._filter_data_by_degree_types() if self.filter_degree_types else self.data
        
        # Apply qualification filtering if specified
        if self.filter_qualifications:
            working_data = self._filter_data_by_qualifications(working_data)
        
        # For main provider role (koulutuksenJarjestaja)
        main_provider_data = working_data[working_data[self.provider_col].isin(self.institution_names)]
        main_provider_volumes = main_provider_data.groupby(self.year_col)[self.volume_col].sum()
        
        # For subcontractor role (hankintakoulutuksenJarjestaja)
        subcontractor_data = working_data[working_data[self.subcontractor_col].isin(self.institution_names)]
        subcontractor_volumes = subcontractor_data.groupby(self.year_col)[self.volume_col].sum()
        
        # Log the volume for 2018 for debugging
        debug_2018_main = main_provider_data[main_provider_data[self.year_col] == 2018][self.volume_col].sum()
        logger.info(f"DEBUG: Main provider volume for 2018: {debug_2018_main}")
        
        # Create summary dataframe
        summary_df = pd.DataFrame(index=sorted(working_data[self.year_col].unique()))
        summary_df['kouluttaja'] = 'RI'  # Shortened name for the institution
        summary_df['järjestäjänä'] = main_provider_volumes
        summary_df['hankintana'] = subcontractor_volumes
        
        # Fill NaN values with 0
        summary_df = summary_df.fillna(0)
        
        # Calculate total volumes and percentage as main provider
        summary_df['Yhteensä'] = summary_df['järjestäjänä'] + summary_df['hankintana']
        summary_df['järjestäjä_osuus (%)'] = (
            summary_df['järjestäjänä'] / summary_df['Yhteensä'] * 100
        ).round(2)
        
        # Reset index to have year as a column
        summary_df = summary_df.reset_index().rename(columns={'index': self.year_col})
        
        logger.info(f"Generated volume summary for {len(summary_df)} years")
        return summary_df
        
    def analyze_volumes_by_qualification(self) -> pd.DataFrame:
        """
        Calculate volumes of students separately for each qualification offered by the institution.
        This implements step 3 from the project description.
        
        Returns:
            pd.DataFrame: Summary DataFrame with volumes by qualification and year
        """
        # Always filter data by degree types for qualification analysis
        filtered_data = self._filter_data_by_degree_types()
        
        # Apply qualification filtering if specified
        if self.filter_qualifications:
            filtered_data = self._filter_data_by_qualifications(filtered_data)
            
        # For main provider role
        main_provider_data = filtered_data[filtered_data[self.provider_col].isin(self.institution_names)]
        provider_volumes = main_provider_data.groupby([self.year_col, self.qualification_col])[self.volume_col].sum().reset_index()
        
        # For subcontractor role
        subcontractor_data = filtered_data[filtered_data[self.subcontractor_col].isin(self.institution_names)]
        subcontractor_volumes = subcontractor_data.groupby([self.year_col, self.qualification_col])[self.volume_col].sum().reset_index()
        
        # Add role identifier
        provider_volumes['role'] = 'järjestäjänä'
        subcontractor_volumes['role'] = 'hankintana'
        
        # Combine data
        combined_volumes = pd.concat([provider_volumes, subcontractor_volumes], ignore_index=True)
        
        # Create a pivot table with years as columns, qualifications as rows, and roles as values
        pivot_volumes = combined_volumes.pivot_table(
            index=self.qualification_col,
            columns=[self.year_col, 'role'],
            values=self.volume_col,
            fill_value=0
        )
        
        # Flatten column multi-index for easier handling
        pivot_volumes.columns = [f"{year}_{role}" for year, role in pivot_volumes.columns]
        
        # Reset index to have qualification as a regular column
        result_df = pivot_volumes.reset_index()
        
        logger.info(f"Generated volume breakdown for {len(result_df)} qualifications")
        return result_df
    
    def calculate_qualification_cagr(self, start_year: int = None, end_year: int = None) -> pd.DataFrame:
        """
        Calculate Compound Annual Growth Rate (CAGR) for qualifications offered by the institution.
        
        Args:
            start_year: Optional start year for CAGR calculation (defaults to earliest available)
            end_year: Optional end year for CAGR calculation (defaults to latest available)
            
        Returns:
            pd.DataFrame: DataFrame with CAGR values for each qualification
        """
        # Filter data by degree types and qualifications
        filtered_data = self._filter_data_by_degree_types()
        if self.filter_qualifications:
            filtered_data = self._filter_data_by_qualifications(filtered_data)
            
        # Prepare data for CAGR calculation
        institution_data = filtered_data[
            (filtered_data[self.provider_col].isin(self.institution_names)) |
            (filtered_data[self.subcontractor_col].isin(self.institution_names))
        ]
        
        # Set default years if not provided
        if start_year is None:
            start_year = institution_data[self.year_col].min()
        if end_year is None:
            end_year = institution_data[self.year_col].max()
            
        # Filter to the selected year range
        year_range_data = institution_data[
            (institution_data[self.year_col] >= start_year) &
            (institution_data[self.year_col] <= end_year)
        ]
        
        # Group by qualification and year, then sum volumes
        grouped_data = year_range_data.groupby([self.qualification_col, self.year_col])[self.volume_col].sum().reset_index()
        
        # Calculate CAGR for each qualification
        cagr_results = calculate_cagr_for_groups(
            df=grouped_data, 
            groupby_columns=[self.qualification_col], 
            value_column=self.volume_col
        )
        
        # The start_year and end_year are now replaced by First Year Offered and Last Year Offered
        # which are calculated within calculate_cagr_for_groups based on actual years present
        
        # Add year range information for consistency with previous versions
        cagr_results['year_range'] = cagr_results.apply(
            lambda row: f"{row['First Year Offered']}-{row['Last Year Offered']}",
            axis=1
        )
        
        logger.info(f"Calculated CAGR for {len(cagr_results)} qualifications based on actual years offered")
        return cagr_results
    
    def calculate_yearly_growth(self, target_column: str, year: int, time_window: int = 3) -> pd.DataFrame:
        """
        Calculate Year-over-Year (YoY) growth rates for a specific metric.
        
        Args:
            target_column: Column to calculate growth for
            year: End year for the calculation
            time_window: Number of years to include in the calculation
            
        Returns:
            pd.DataFrame: DataFrame with YoY growth rates
        """
        # Filter data by degree types and qualifications
        filtered_data = self._filter_data_by_degree_types()
        if self.filter_qualifications:
            filtered_data = self._filter_data_by_qualifications(filtered_data)
            
        # Prepare data for growth calculation
        institution_data = filtered_data[
            (filtered_data[self.provider_col].isin(self.institution_names)) |
            (filtered_data[self.subcontractor_col].isin(self.institution_names))
        ]
        
        # Calculate growth
        growth_results = calculate_yoy_growth(
            df=institution_data,
            groupby_col=self.qualification_col,
            target_col=target_column,
            output_col=f"{target_column}_growth",
            end_year=year,
            time_window=time_window
        )
        
        logger.info(f"Calculated {time_window}-year growth for {len(growth_results)} qualifications ending in {year}")
        return growth_results
    
    def get_top_providers_by_qualification(self, year: int, top_n: int = 5) -> Dict[str, List[str]]:
        """
        Get the top institutions for each qualification for a given year.
        
        Args:
            year: Year to analyze
            top_n: Number of top institutions to include
            
        Returns:
            Dict: Dictionary mapping qualifications to lists of top providers
        """
        try:
            # First filter and aggregate the data
            filtered_data = self._filter_data_by_degree_types()
            if self.filter_qualifications:
                filtered_data = self._filter_data_by_qualifications(filtered_data)
                
            # Filter to the specific year
            year_data = filtered_data[filtered_data[self.year_col] == year].copy()
            
            if year_data.empty:
                logger.warning(f"No data found for year {year}")
                return {}
                
            # Calculate market position (ranking) for each provider
            # Group by qualification and provider
            provider_volumes = year_data.groupby([self.qualification_col, self.provider_col])[self.volume_col].sum().reset_index()
            
            # Rank providers within each qualification
            provider_volumes['Sijoitus markkinassa'] = provider_volumes.groupby(self.qualification_col)[self.volume_col].rank(ascending=False, method='min')
            provider_volumes['kouluttaja'] = provider_volumes[self.provider_col]  # Rename for compatibility
            
            # Get top providers
            top_providers = hae_tutkinnon_suurimmat_toimijat(provider_volumes, year, top_n)
            
            logger.info(f"Identified top {top_n} providers for {len(top_providers)} qualifications in {year}")
            return top_providers
        except Exception as e:
            logger.error(f"Error in get_top_providers_by_qualification: {e}")
            return {}
    
    def analyze_institution_roles(self) -> pd.DataFrame:
        """
        Analyze the institution's roles as main provider and subcontractor over time.
        
        Returns:
            pd.DataFrame: DataFrame with metrics for each role by year
        """
        # Filter data by degree types and qualifications
        filtered_data = self._filter_data_by_degree_types()
        if self.filter_qualifications:
            filtered_data = self._filter_data_by_qualifications(filtered_data)
            
        # Use the hae_kouluttaja_nom function to get the analysis
        role_analysis = hae_kouluttaja_nom(
            df=filtered_data,
            kouluttaja=self.institution_names,
            nom_as_jarjestaja=self.volume_col,
            nom_as_hankinta=self.volume_col
        )
        
        # Filter out rows where the total is zero
        role_analysis = role_analysis[role_analysis['Yhteensä'] > 0]
        
        logger.info(f"Analyzed institution roles across {len(role_analysis)} years")
        return role_analysis
        
    def analyze_qualification_growth(self) -> pd.DataFrame:
        """
        Calculate Year-over-Year change (%) in volumes of students for each qualification
        offered by the institution under analysis. This implements step 4 from the project description.
        
        Returns:
            pd.DataFrame: DataFrame with YoY growth analysis for qualifications
        """
        # First get the volumes by qualification
        volumes_df = self.analyze_volumes_by_qualification()
        
        # Calculate YoY changes
        growth_data = []
        
        # For each qualification
        for qualification in volumes_df[self.qualification_col].unique():
            # Get data for this qualification, sorted by year
            qual_data = volumes_df[
                volumes_df[self.qualification_col] == qualification
            ].sort_values(by=self.year_col)
            
            # Need at least two years of data to calculate growth
            if len(qual_data) < 2:
                logger.warning(f"Skipping growth calculation for {qualification} - not enough data")
                continue
                
            # Calculate YoY changes for each consecutive year pair
            for i in range(1, len(qual_data)):
                current_year = qual_data.iloc[i][self.year_col]
                previous_year = qual_data.iloc[i-1][self.year_col]
                
                # Skip if years are not consecutive
                if current_year != previous_year + 1:
                    logger.warning(f"Skipping {previous_year}-{current_year} for {qualification} - years not consecutive")
                    continue
                
                # Total market change
                current_market = qual_data.iloc[i]['tutkinto yhteensä']
                previous_market = qual_data.iloc[i-1]['tutkinto yhteensä']
                market_growth = ((current_market / previous_market) - 1) * 100 if previous_market > 0 else np.nan
                
                # Institution's volume change
                current_inst_vol = qual_data.iloc[i]['kouluttaja yhteensä']
                previous_inst_vol = qual_data.iloc[i-1]['kouluttaja yhteensä']
                inst_growth = ((current_inst_vol / previous_inst_vol) - 1) * 100 if previous_inst_vol > 0 else np.nan
                
                # Market share change (percentage points)
                current_share = qual_data.iloc[i]['markkinaosuus (%)']
                previous_share = qual_data.iloc[i-1]['markkinaosuus (%)']
                share_change = current_share - previous_share
                
                # Relative market share change (%)
                relative_share_change = ((current_share / previous_share) - 1) * 100 if previous_share > 0 else np.nan
                
                # Provider count changes
                current_providers = qual_data.iloc[i]['koulutuksenJarjestaja_n']
                previous_providers = qual_data.iloc[i-1]['koulutuksenJarjestaja_n']
                providers_growth = ((current_providers / previous_providers) - 1) * 100 if previous_providers > 0 else np.nan
                
                current_subcontractors = qual_data.iloc[i]['hankintakoulutuksenJarjestaja_n']
                previous_subcontractors = qual_data.iloc[i-1]['hankintakoulutuksenJarjestaja_n']
                subcontractors_growth = ((current_subcontractors / previous_subcontractors) - 1) * 100 if previous_subcontractors > 0 else np.nan
                
                growth_data.append({
                    self.year_col: current_year,
                    self.qualification_col: qualification,
                    'tutkinto yhteensä': current_market,
                    'tutkinto kasvu (%)': round(market_growth, 2) if not np.isnan(market_growth) else None,
                    'kouluttaja yhteensä': current_inst_vol,
                    'kouluttaja kasvu (%)': round(inst_growth, 2) if not np.isnan(inst_growth) else None,
                    'markkinaosuus (%)': current_share,
                    'markkinaosuus muutos (%-yks)': round(share_change, 2),
                    'markkinaosuus kasvu (%)': round(relative_share_change, 2) if not np.isnan(relative_share_change) else None,
                    'koulutuksenJarjestaja_n': current_providers,
                    'koulutuksenJarjestaja_n kasvu (%)': round(providers_growth, 2) if not np.isnan(providers_growth) else None,
                    'hankintakoulutuksenJarjestaja_n': current_subcontractors,
                    'hankintakoulutuksenJarjestaja_n kasvu (%)': round(subcontractors_growth, 2) if not np.isnan(subcontractors_growth) else None
                })
        
        # Convert to DataFrame
        growth_df = pd.DataFrame(growth_data)
        
        if not growth_df.empty:
            # Sort by year (newest first) and then by qualification
            growth_df = growth_df.sort_values(
                by=[self.year_col, self.qualification_col], 
                ascending=[False, True]
            )
            
            logger.info(f"Generated qualification growth analysis with {len(growth_df)} rows")
        else:
            logger.warning("No growth data generated - check if there are multiple years of data")
            
        return growth_df
        
    def get_qualification_list(self) -> pd.DataFrame:
        """
        Get a list of all qualifications in the filtered data with their latest volumes.
        
        Returns:
            pd.DataFrame: DataFrame with qualification list and their volumes in the latest year
        """
        # Filter data by degree types
        filtered_data = self._filter_data_by_degree_types()
        
        # Get the latest year in the data
        latest_year = filtered_data[self.year_col].max()
        
        # Filter for the latest year
        latest_data = filtered_data[filtered_data[self.year_col] == latest_year]
        
        # Group by qualification and calculate total volume
        qualification_volumes = latest_data.groupby(self.qualification_col)[self.volume_col].sum().reset_index()
        
        # Get institution volumes for each qualification
        qualification_inst_volumes = []
        
        for qualification in qualification_volumes[self.qualification_col]:
            qual_data = latest_data[latest_data[self.qualification_col] == qualification]
            
            # Check if institution is involved in this qualification
            is_provider = qual_data[self.provider_col].isin(self.institution_names).any()
            is_subcontractor = qual_data[self.subcontractor_col].isin(self.institution_names).any()
            
            qualification_inst_volumes.append({
                self.qualification_col: qualification,
                'tutkinto yhteensä': qual_data[self.volume_col].sum(),
                'järjestäjänä': qual_data[qual_data[self.provider_col].isin(self.institution_names)][self.volume_col].sum(),
                'hankintana': qual_data[qual_data[self.subcontractor_col].isin(self.institution_names)][self.volume_col].sum(),
                'institution_involved': is_provider or is_subcontractor,
                self.year_col: latest_year
            })
        
        # Create DataFrame
        qual_list_df = pd.DataFrame(qualification_inst_volumes)
        
        # Calculate institution's volume in each qualification (as both provider and subcontractor)
        qual_list_df['kouluttaja yhteensä'] = qual_list_df['järjestäjänä'] + qual_list_df['hankintana']
        
        # Calculate market share
        qual_list_df['markkinaosuus (%)'] = (
            qual_list_df['kouluttaja yhteensä'] / qual_list_df['tutkinto yhteensä'] * 100
        ).round(2)
        
        # Sort by institution volume (descending)
        qual_list_df = qual_list_df.sort_values('kouluttaja yhteensä', ascending=False)
        
        logger.info(f"Generated qualification list with {len(qual_list_df)} qualifications for year {latest_year}")
        return qual_list_df 