"""
Market analyzer module for the Vipunen project.

This module provides a wrapper around the EducationMarketAnalyzer class
to simplify integration with the CLI.
"""
import pandas as pd
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from .qualification_analyzer import calculate_cagr_for_groups

from ..config.config_loader import get_config
from .market_share_analyzer import calculate_market_shares as calculate_detailed_market_shares
from .market_share_analyzer import calculate_market_share_changes
from .education_market import EducationMarketAnalyzer, hae_kouluttaja_nom

logger = logging.getLogger(__name__)

# Load config once at module level, or pass it in. Passing it in is cleaner.
# config = get_config() # Avoid global config if possible

class MarketAnalyzer:
    """
    Main analyzer for market data, focusing on qualifications offered by educational institutions.
    
    This class is a simplified wrapper around EducationMarketAnalyzer, focusing on the most important
    market analysis metrics like total volumes, qualification breakdown, market shares, growth and CAGR.
    """
    
    def __init__(self, data: pd.DataFrame, cfg: dict):
        """
        Initialize the MarketAnalyzer with market data and configuration.
        
        Args:
            data: DataFrame containing the market data with columns for year, degree type, 
                 qualification, provider, subcontractor and volume.
            cfg: Configuration dictionary containing column mappings and settings.
        """
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg # Store config
        self.cols_in = self.cfg['columns']['input']
        self.cols_out = self.cfg['columns']['output']
        
        self.data = data
        
        # Store the institution names (to be set by caller)
        self.institution_names = []
        self.institution_short_name = ""
        
        # Calculate min and max years using config input mapping
        if not data.empty and self.cols_in['year'] in data.columns:
            self.min_year = data[self.cols_in['year']].min()
            self.max_year = data[self.cols_in['year']].max()
        else:
            self.min_year = None
            self.max_year = None
            
    def _get_institution_qualifications(self) -> List[str]:
        """
        Get all unique qualifications offered across all institutions.
        
        Returns:
            List of qualification names
        """
        if self.data.empty:
            return []
            
        all_qualifications = self.data[self.cols_in['qualification']].unique().tolist()
        return all_qualifications
        
    def calculate_total_volumes(self) -> pd.DataFrame:
        """
        Calculate the total student volumes for each year.
        
        Returns:
            DataFrame with years as index and total_volume as column
        """
        if self.data.empty or self.min_year is None or self.max_year is None:
            return pd.DataFrame()
        
        # Initialize the analyzer with institution names and use input mappings
        analyzer = EducationMarketAnalyzer(
            data=self.data,
            institution_names=self.institution_names,
            year_col=self.cols_in['year'],
            degree_type_col=self.cols_in['degree_type'],
            qualification_col=self.cols_in['qualification'],
            provider_col=self.cols_in['provider'],
            subcontractor_col=self.cols_in['subcontractor'],
            volume_col=self.cols_in['volume']
        )
        
        # Calculate total volumes
        total_volumes = analyzer.analyze_total_volume()

        # Ensure the kouluttaja column has the right short name
        # The analyzer likely uses the input provider column name, adjust if needed
        # Assuming 'analyze_total_volume' returns columns based on input names?
        # Let's rename to output name standard if it exists
        # --- UPDATE: analyze_total_volume seems to return hardcoded Finnish names ---
        # if self.cols_in['provider'] in total_volumes.columns:
        #      total_volumes[self.cols_out['provider']] = self.institution_short_name
        if 'kouluttaja' in total_volumes.columns:
             total_volumes[self.cols_out['provider']] = self.institution_short_name
             total_volumes = total_volumes.drop(columns=['kouluttaja']) # Drop original

        # Rename year and volume columns to output standard
        rename_map = {
            # self.cols_in['year']: self.cols_out['year'], # Assuming year is returned as input name?
            # self.cols_in['volume']: self.cols_out['total_volume']
            # --- UPDATE: Rename from actual returned columns --- 
            'tilastovuosi': self.cols_out['year'],
            'järjestäjänä': self.cols_out['provider_amount'],
            'hankintana': self.cols_out['subcontractor_amount'],
            'Yhteensä': self.cols_out['total_volume'], # This is institution's total
            # Keep/rename 'järjestäjä_osuus (%)' if needed?
        }
        # Only rename columns that exist
        rename_map = {k: v for k, v in rename_map.items() if k in total_volumes.columns}
        total_volumes = total_volumes.rename(columns=rename_map)

        return total_volumes
    
    def calculate_volumes_by_qualification(self) -> pd.DataFrame:
        """
        Calculate student volumes broken down by qualification type and year for the primary institution.
        
        Returns:
            DataFrame with qualification volumes by year, using output column names.
        """
        if self.data.empty or self.min_year is None or self.max_year is None:
            # Return DataFrame with expected output columns
            return pd.DataFrame(columns=[
                self.cols_out['year'], self.cols_out['qualification'],
                self.cols_out['provider_amount'], self.cols_out['subcontractor_amount'],
                self.cols_out['total_volume'], self.cols_out['market_total'],
                self.cols_out['market_share']
            ])

        results = []
        year_col = self.cols_in['year']
        qual_col = self.cols_in['qualification']
        provider_col = self.cols_in['provider']
        subcontractor_col = self.cols_in['subcontractor']
        volume_col = self.cols_in['volume']

        for year in range(self.min_year, self.max_year + 1):
            year_data = self.data[self.data[year_col] == year]

            if year_data.empty:
                continue

            qualifications = year_data[qual_col].unique()

            for qual in qualifications:
                qual_data = year_data[year_data[qual_col] == qual]

                # Calculate total market volume for this qualification
                market_total = qual_data[volume_col].sum()

                # Check if institution is involved with this qualification
                institution_data = qual_data[
                    (qual_data[provider_col].isin(self.institution_names)) |
                    (qual_data[subcontractor_col].isin(self.institution_names))
                ]

                if institution_data.empty:
                    continue

                # Calculate provider amount
                provider_amount = institution_data[
                    institution_data[provider_col].isin(self.institution_names)
                ][volume_col].sum()

                # Calculate subcontractor amount
                subcontractor_amount = institution_data[
                    institution_data[subcontractor_col].isin(self.institution_names)
                ][volume_col].sum()

                # Calculate total amount for the institution
                total_amount = provider_amount + subcontractor_amount

                # Calculate market share
                market_share = (total_amount / market_total * 100) if market_total > 0 else 0

                results.append({
                    self.cols_out['year']: year,
                    self.cols_out['qualification']: qual,
                    self.cols_out['provider_amount']: provider_amount,
                    self.cols_out['subcontractor_amount']: subcontractor_amount,
                    self.cols_out['total_volume']: total_amount, # Institution's total volume for this qual
                    self.cols_out['market_total']: market_total, # Total market volume for this qual
                    self.cols_out['market_share']: market_share
                })

        if not results:
            # Return DataFrame with expected output columns
            return pd.DataFrame(columns=[
                self.cols_out['year'], self.cols_out['qualification'],
                self.cols_out['provider_amount'], self.cols_out['subcontractor_amount'],
                self.cols_out['total_volume'], self.cols_out['market_total'],
                self.cols_out['market_share']
            ])

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(by=[self.cols_out['year'], self.cols_out['qualification']])

        return result_df
    
    def calculate_market_shares(self) -> pd.DataFrame:
        """
        DEPRECATED? This seems like an old calculation.
        calculate_providers_market is likely the intended function.
        Leaving stub here for now, but should probably be removed or refactored.

        Calculate market shares for each institution.

        Returns:
            DataFrame containing market shares, growth rates, and market gainer ranks
        """
        self.logger.warning("MarketAnalyzer.calculate_market_shares() might be deprecated. Use calculate_providers_market() instead.")
        # Returning empty DataFrame matching expected structure of calculate_providers_market
        return pd.DataFrame(columns=[
            self.cols_out['year'], self.cols_out['qualification'], self.cols_out['provider'],
            self.cols_out['provider_amount'], self.cols_out['subcontractor_amount'],
            self.cols_out['total_volume'], self.cols_out['market_total'],
            self.cols_out['market_share'], self.cols_out['market_rank'],
            self.cols_out['market_share_growth'], self.cols_out['market_gainer_rank'] # Using new config key
        ])

        # Old logic below (commented out):
        # if self.data.empty or self.min_year is None or self.max_year is None or self.min_year >= self.max_year:
        #     return pd.DataFrame()
        # ... (rest of old implementation) ...

    def calculate_providers_market(self) -> pd.DataFrame:
        """
        Calculate detailed market shares and ranks for all providers across all qualifications and years.

        Uses the calculate_detailed_market_shares function and adds ranking and growth.

        Returns:
            pd.DataFrame: Detailed market analysis with columns specified in config output mapping.
        """
        if self.data.empty:
            return pd.DataFrame()

        self.logger.info("Calculating detailed market shares for all providers...")

        # Use the dedicated market share calculation function
        # Pass input column names from config
        detailed_shares = calculate_detailed_market_shares(
            df=self.data,
            provider_names=self.institution_names, # Pass institution names for potential filtering/tagging if needed by function
            provider_col=self.cols_in['provider'],
            subcontractor_col=self.cols_in['subcontractor'],
            year_col=self.cols_in['year'],
            qual_col=self.cols_in['qualification'],
            value_col=self.cols_in['volume'], # Correct argument name for calculate_market_shares
            share_calculation_basis='both' # Example: Calculate share based on combined vol
        )

        if detailed_shares.empty:
            return pd.DataFrame()

        self.logger.info("Calculating market share changes and ranks...")

        # Calculate year-over-year market share changes
        # This function likely expects specific column names, adjust if needed
        # Assuming it works with the output of calculate_detailed_market_shares
        market_share_change = calculate_market_share_changes(
            market_share_df=detailed_shares,
            year_col=self.cols_in['year'], # Input year col used by detailed_shares
            qual_col=self.cols_in['qualification'], # Input qual col used by detailed_shares
            provider_col='provider', # detailed_shares uses 'provider' column name
            market_share_col='market_share' # detailed_shares uses 'market_share' column name
        )

        # Merge changes back into the detailed shares
        if not market_share_change.empty:
            # Ensure the merge keys match the columns returned by calculate_market_share_changes
            merge_keys = [self.cols_in['year'], self.cols_in['qualification'], 'provider']
            # Ensure keys exist in both dataframes before merging
            if not all(key in detailed_shares.columns for key in merge_keys):
                 raise ValueError(f"Merge keys missing in detailed_shares: Need {merge_keys}, have {detailed_shares.columns}")
            if not all(key in market_share_change.columns for key in merge_keys):
                 raise ValueError(f"Merge keys missing in market_share_change: Need {merge_keys}, have {market_share_change.columns}")
                 
            detailed_shares = pd.merge(
                detailed_shares,
                market_share_change[merge_keys + ['market_share_change', 'previous_market_share']], # Select cols
                on=merge_keys,
                how='left'
            )
            # Rename market_share_change to the config output name
            detailed_shares = detailed_shares.rename(columns={'market_share_change': self.cols_out['market_share_growth']})
            # Fill NaN growth for the first year or where previous share was zero/NaN
            detailed_shares[self.cols_out['market_share_growth']] = detailed_shares[self.cols_out['market_share_growth']].fillna(0)
        else:
            # Add the column with default value if no changes were calculated
             detailed_shares[self.cols_out['market_share_growth']] = 0.0

        # Step 3: Calculate Market Rank within each year-qualification group
        # Rank based on the 'market_share' column output by calculate_detailed_market_shares
        detailed_shares[self.cols_out['market_rank']] = detailed_shares.groupby(
            [self.cols_in['year'], self.cols_in['qualification']]
        )['market_share'].rank(ascending=False, method='min').astype(int)

        # Step 4: Calculate Market Gainer Rank (based on Market Share Growth)
        # Rank gainers within each year-qualification group based on the growth calculated
        # Handle potential NaNs or infinities in growth if necessary (assuming market_share_change handled this)
        # Sort by growth rate to determine market gainer rank
        detailed_shares = detailed_shares.sort_values(
             by=[self.cols_in['year'], self.cols_in['qualification'], self.cols_out['market_share_growth']],
             ascending=[True, True, False] # Higher growth = better rank
        )
        # Assign rank within year-qualification groups
        detailed_shares[self.cols_out['market_gainer_rank']] = detailed_shares.groupby(
             [self.cols_in['year'], self.cols_in['qualification']]
        )[self.cols_out['market_share_growth']].rank(ascending=False, method='min')

        # Step 5: Rename columns to match the output config specification
        # Identify columns to keep and rename. Source names are from 
        # calculate_detailed_market_shares output and the calculated columns.
        rename_map = {
            self.cols_in['year']: self.cols_out['year'],                   # Input year -> Output year
            self.cols_in['qualification']: self.cols_out['qualification'], # Input qual -> Output qual
            'provider': self.cols_out['provider'],                   # 'provider' -> Output provider
            'volume_as_provider': self.cols_out['provider_amount'],
            'volume_as_subcontractor': self.cols_out['subcontractor_amount'],
            'total_volume': self.cols_out['total_volume'],
            'qualification_market_volume': self.cols_out['market_total'],
            'market_share': self.cols_out['market_share'],
            # market_rank, market_share_growth, market_gainer_rank are already named correctly
        }

        # Apply renaming only for columns that exist in detailed_shares
        rename_map = {k: v for k, v in rename_map.items() if k in detailed_shares.columns}
        final_df = detailed_shares.rename(columns=rename_map)

        # --- DEBUG: Check dtypes before conversion ---
        # print("\nDEBUG: Dtypes before numeric conversion:")
        # print(final_df.dtypes)
        # --- END DEBUG ---
        
        # Ensure correct data types and handle NaNs
        # Example: Convert percentage columns
        if self.cols_out['market_share'] in final_df.columns:
             final_df[self.cols_out['market_share']] = final_df[self.cols_out['market_share']].round(2)
        if self.cols_out['market_share_growth'] in final_df.columns:
             final_df[self.cols_out['market_share_growth']] = final_df[self.cols_out['market_share_growth']].round(2)

        # --- DEBUG: Check for non-numeric values ---
        numeric_cols = [
            self.cols_out['provider_amount'], self.cols_out['subcontractor_amount'],
            self.cols_out['total_volume'], self.cols_out['market_total']
        ]
        # for col in numeric_cols:
        #     if col in final_df.columns:
        #         try:
        #             pd.to_numeric(final_df[col], errors='raise')
        #         except (ValueError, TypeError) as e:
        #             print(f"\nDEBUG: Non-numeric value found in column '{col}': {e}")
        #             print("Sample problematic values:")
        #             print(final_df[pd.to_numeric(final_df[col], errors='coerce').isna()][col].unique()[:10])
        # --- END DEBUG ---
        
        # Convert numeric columns to appropriate types (e.g., Int64 for nullable integers)
        for col in numeric_cols:
            if col in final_df.columns:
                # Existing values might be float, NaN, or potentially other objects
                # 1. Use errors='coerce' to force non-numerics to NaN
                numeric_series = pd.to_numeric(final_df[col], errors='coerce')
                # 2. Round floats to nearest integer (optional, but might help with casting)
                numeric_series = numeric_series.round(0) 
                # 3. Convert to nullable Int64
                try:
                    final_df[col] = numeric_series.astype('Int64')
                except Exception as e:
                    logger.error(f"Failed to convert column '{col}' to Int64 after pd.to_numeric: {e}")
                    # Print unique non-NA values that might be causing issues
                    # problem_values = numeric_series.dropna().astype(str).unique()
                    # logger.error(f"Unique non-NA values in '{col}' before final conversion attempt: {problem_values[:20]}") # Log first 20 unique values
                    raise # Re-raise the exception

        # Keep Market Rank as integer
        if self.cols_out['market_rank'] in final_df.columns:
            # Convert Market Rank to Int64 first to handle potential NAs from grouping/ranking
            final_df[self.cols_out['market_rank']] = final_df[self.cols_out['market_rank']].astype('Int64') 
            # Fill any remaining NA with -1 if needed AFTER conversion to nullable int
            final_df[self.cols_out['market_rank']] = final_df[self.cols_out['market_rank']].fillna(-1) 

        # Keep Market Gainer Rank as nullable integer (might have NaNs for first year)
        if self.cols_out['market_gainer_rank'] in final_df.columns:
            # Gainer rank might be float due to NaNs, convert to Int64 after potential fillna if needed
             final_df[self.cols_out['market_gainer_rank']] = final_df[self.cols_out['market_gainer_rank']].round().astype('Int64') # Use round() before converting
             # Explicitly set gainer rank to NA for the minimum year
             if self.min_year is not None:
                 min_year_mask = final_df[self.cols_out['year']] == self.min_year
                 final_df.loc[min_year_mask, self.cols_out['market_gainer_rank']] = pd.NA

        # Select and order final columns based on config output order (optional but nice)
        final_columns_in_config_order = [
            self.cols_out['year'], self.cols_out['qualification'], self.cols_out['provider'],
            self.cols_out['provider_amount'], self.cols_out['subcontractor_amount'],
            self.cols_out['total_volume'], self.cols_out['market_total'],
            self.cols_out['market_share'], self.cols_out['market_rank'],
            self.cols_out['market_share_growth'], self.cols_out['market_gainer_rank']
        ]
        # Filter this list to only include columns that *actually exist* in the final_df AFTER renaming
        final_columns_present = [col for col in final_columns_in_config_order if col in final_df.columns]
        
        # Select the present columns in the desired order
        final_df = final_df[final_columns_present]

        # Sort final output
        final_df = final_df.sort_values(
            by=[self.cols_out['year'], self.cols_out['qualification'], self.cols_out['market_rank']],
            ascending=[True, True, True]
        )

        self.logger.info("Finished calculating detailed market shares.")
        return final_df.reset_index(drop=True)

    def calculate_qualification_growth(self) -> pd.DataFrame:
        """
        Calculate growth rate between the first and last year for each qualification type.
        
        Returns:
            DataFrame with qualification types as index and growth rate as column
        """
        if self.data.empty or self.min_year is None or self.max_year is None or self.min_year == self.max_year:
            return pd.DataFrame()
        
        # For qualification growth we need to manually create a simpler DataFrame
        # rather than using the more complex EducationMarketAnalyzer implementation
        # Group by qualification type and year to calculate volumes
        grouped = self.data.groupby(['tutkintotyyppi', 'tilastovuosi'])['nettoopiskelijamaaraLkm'].sum().reset_index()
        
        # Pivot to have years as columns
        pivot_df = grouped.pivot(index='tutkintotyyppi', columns='tilastovuosi', values='nettoopiskelijamaaraLkm').fillna(0)
        
        # Calculate growth for each qualification type from first to last year
        result = pd.DataFrame(index=pivot_df.index)
        result['First Year'] = self.min_year
        result['Last Year'] = self.max_year
        result['First Year Volume'] = pivot_df[self.min_year]
        result['Last Year Volume'] = pivot_df[self.max_year]
        
        # Calculate growth rate
        result['growth_rate'] = ((result['Last Year Volume'] / result['First Year Volume']) - 1) * 100
        result['growth_rate'] = result['growth_rate'].replace([np.inf, -np.inf], np.nan)
        
        return result
    
    def calculate_qualification_cagr(self) -> pd.DataFrame:
        """
        Calculate Compound Annual Growth Rate (CAGR) for qualifications offered by the target institution.
        
        Returns:
            DataFrame with qualification CAGRs, sorted by qualification name.
            Columns match the 'CAGR Analysis' section in docs/export_file_structure.txt, 
            with the grouping column named 'Qualification'.
            Columns: Qualification | CAGR | First Year | Last Year | First Year Volume | Last Year Volume | Years Present
        """
        # Return empty DataFrame if no data or institution
        if self.data.empty or not self.institution_names or self.min_year is None or self.max_year is None:
            return pd.DataFrame()
        
        # Filter for the institution's data
        institution_data = self.data[
            (self.data['koulutuksenJarjestaja'].isin(self.institution_names)) |
            (self.data['hankintakoulutuksenJarjestaja'].isin(self.institution_names))
        ].copy()  # Create a copy to avoid SettingWithCopyWarning

        # Determine start and end years for CAGR
        start_year = institution_data['tilastovuosi'].min() if not institution_data.empty else None
        end_year = institution_data['tilastovuosi'].max() if not institution_data.empty else None

        if start_year is None or end_year is None or start_year == end_year:
            logger.warning("Not enough data points for CAGR calculation.")
            return pd.DataFrame()

        # Group by qualification and year, sum volumes
        grouped_data = institution_data.groupby(['tutkinto', 'tilastovuosi'])['nettoopiskelijamaaraLkm'].sum().reset_index()

        # Calculate CAGR using the utility function
        cagr_results = calculate_cagr_for_groups(
            df=grouped_data,
            groupby_columns=['tutkinto'],
            value_column='nettoopiskelijamaaraLkm',
            year_column='tilastovuosi'
        )

        # Ensure required columns exist in the result
        required_cols = ['Qualification', 'CAGR', 'First Year', 'Last Year', 'First Year Volume', 'Last Year Volume'] # Keep English names
        
        # Initial check
        missing_cols = [col for col in required_cols if col not in cagr_results.columns]
        has_tutkinto = 'tutkinto' in cagr_results.columns

        if missing_cols:
            # Check if the only missing column is 'Qualification' and 'tutkinto' is present
            if missing_cols == ['Qualification'] and has_tutkinto:
                logger.info("'Qualification' column missing, but 'tutkinto' found. Renaming 'tutkinto' to 'Qualification'.")
                cagr_results = cagr_results.rename(columns={'tutkinto': 'Qualification'}) # Rename
                # No further action needed here, the check passed after rename
            else:
                # If other columns are missing, or 'Qualification' is missing without 'tutkinto' present
                logger.warning(f"CAGR calculation result missing required columns. Missing: {missing_cols}. Found: {cagr_results.columns}. Expected: {required_cols}")
                # Check if potentially optional columns like 'Years Present' exist, just for logging info
                optional_cols = ['Years Present']
                found_optional = [col for col in optional_cols if col in cagr_results.columns]
                if found_optional:
                    logger.info(f"Found optional columns: {found_optional}")
                
                # Return empty DataFrame with the expected columns
                return pd.DataFrame(columns=required_cols)
        
        # If we reach here, either the columns were initially correct, 
        # or 'tutkinto' was successfully renamed to 'Qualification'.

        # --- Rename columns based on output config --- 
        cagr_rename_map = {
            'Qualification': self.cols_out.get('qualification', 'Qualification'), # Reuse existing key
            'CAGR': self.cols_out.get('cagr_rate', 'CAGR'),
            'First Year': self.cols_out.get('cagr_first_year', 'First Year'),
            'Last Year': self.cols_out.get('cagr_last_year', 'Last Year'),
            'First Year Volume': self.cols_out.get('cagr_first_year_volume', 'First Year Volume'),
            'Last Year Volume': self.cols_out.get('cagr_last_year_volume', 'Last Year Volume'),
            'Years Present': self.cols_out.get('cagr_years_present', 'Years Present')
            # Add 'Qualification Type' if it's handled and needs mapping
        }
        # Apply renaming only for columns that exist
        cagr_rename_map = {k: v for k, v in cagr_rename_map.items() if k in cagr_results.columns}
        cagr_results_renamed = cagr_results.rename(columns=cagr_rename_map)

        # Select and order columns based on the renamed keys (optional)
        final_cagr_cols = [
            self.cols_out.get('qualification', 'Qualification'),
            self.cols_out.get('cagr_rate', 'CAGR'),
            self.cols_out.get('cagr_first_year', 'First Year'),
            self.cols_out.get('cagr_last_year', 'Last Year'),
            self.cols_out.get('cagr_first_year_volume', 'First Year Volume'),
            self.cols_out.get('cagr_last_year_volume', 'Last Year Volume'),
            self.cols_out.get('cagr_years_present', 'Years Present')
        ]
        final_cagr_cols_present = [col for col in final_cagr_cols if col in cagr_results_renamed.columns]
        final_cagr_df = cagr_results_renamed[final_cagr_cols_present]

        logger.info(f"Calculated CAGR for {len(cagr_results)} qualifications")
        return final_cagr_df
    
    def calculate_overall_total_market_volume(self) -> pd.Series:
        """
        Calculate the total market volume across all qualifications and providers for each year.

        Returns:
            pd.Series: Index=Year (output name), Values=Total Market Volume (output name)
        """
        if self.data.empty or self.cols_in['year'] not in self.data.columns or self.cols_in['volume'] not in self.data.columns:
            return pd.Series(dtype=float)

        # Group by year and sum the volume column using input names
        overall_volume = self.data.groupby(self.cols_in['year'])[self.cols_in['volume']].sum()

        # Rename the index and series name to output names
        overall_volume.index.name = self.cols_out['year']
        overall_volume.name = self.cols_out['market_total'] # Represents overall market total volume

        return overall_volume

    def calculate_qualification_market_yoy_growth(self, detailed_providers_market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Year-over-Year (YoY) growth of the *total market size* for each qualification.

        Args:
            detailed_providers_market_df: DataFrame output from calculate_providers_market.
                                          Must contain 'Year', 'Qualification', and 'Market Total'.

        Returns:
            pd.DataFrame: Columns: Qualification, Year, Market Total, Market Total YoY Growth (%)
                          using output column names.
        """
        year_col = self.cols_out['year']
        qual_col = self.cols_out['qualification']
        market_total_col = self.cols_out['market_total']
        market_growth_col = f"{market_total_col} YoY Growth (%)" # Define output growth column name

        if detailed_providers_market_df.empty or not all(col in detailed_providers_market_df.columns for col in [year_col, qual_col, market_total_col]):
            logger.warning(f"Cannot calculate market total YoY growth: Input DataFrame is empty or missing required columns ({year_col}, {qual_col}, {market_total_col}).")
            return pd.DataFrame(columns=[qual_col, year_col, market_total_col, market_growth_col])

        # Get unique market totals per qualification per year
        market_totals = detailed_providers_market_df[[year_col, qual_col, market_total_col]].drop_duplicates()

        # Sort by qualification and year to ensure correct shift calculation
        market_totals = market_totals.sort_values(by=[qual_col, year_col])

        # Calculate previous year's market total
        market_totals['Previous Market Total'] = market_totals.groupby(qual_col)[market_total_col].shift(1)

        # Calculate YoY growth percentage
        market_totals[market_growth_col] = (
            (market_totals[market_total_col] - market_totals['Previous Market Total']) /
             market_totals['Previous Market Total'] * 100
        )

        # Handle division by zero and infinite growth cases
        # Replace inf/-inf with NaN first (safer for comparisons)
        market_totals[market_growth_col] = market_totals[market_growth_col].replace([np.inf, -np.inf], np.nan)
        # If Previous Market Total was 0 or NaN and Current is > 0, growth is undefined (NaN)
        market_totals.loc[(market_totals['Previous Market Total'].isna() | (market_totals['Previous Market Total'] == 0)) & (market_totals[market_total_col] > 0), market_growth_col] = np.nan
        # If both Previous and Current are 0 or NaN, growth is 0%
        market_totals.loc[(market_totals['Previous Market Total'].isna() | (market_totals['Previous Market Total'] == 0)) & (market_totals[market_total_col].isna() | (market_totals[market_total_col] == 0)), market_growth_col] = 0.0

        # Round growth percentage
        market_totals[market_growth_col] = market_totals[market_growth_col].round(2)

        return market_totals[[qual_col, year_col, market_total_col, market_growth_col]]

    def _calculate_provider_counts_by_year(self) -> pd.DataFrame:
        """
        Calculates the count of unique providers and subcontractors operating 
        within the markets relevant to the target institution's qualifications for each year.

        Returns:
            DataFrame with columns for year, unique provider count, and unique subcontractor count.
            Uses output column names defined in config.
        """
        if self.data.empty or self.min_year is None or self.max_year is None:
            return pd.DataFrame(columns=[self.cols_out['year'], 'Unique_Providers_Count', 'Unique_Subcontractors_Count'])

        year_in_col = self.cols_in['year']
        qual_in_col = self.cols_in['qualification']
        provider_in_col = self.cols_in['provider']
        subcontractor_in_col = self.cols_in['subcontractor']
        year_out_col = self.cols_out['year']

        # 1. Identify qualifications offered by the target institution
        inst_mask = (
            (self.data[provider_in_col].isin(self.institution_names)) |
            (self.data[subcontractor_in_col].isin(self.institution_names))
        )
        inst_quals = self.data.loc[inst_mask, qual_in_col].unique()
        
        if len(inst_quals) == 0:
            self.logger.warning("No qualifications found for the target institution. Cannot calculate provider counts.")
            return pd.DataFrame(columns=[year_out_col, 'Unique_Providers_Count', 'Unique_Subcontractors_Count'])
            
        self.logger.debug(f"Identified {len(inst_quals)} qualifications offered by {self.institution_short_name} for provider counting.")
        # Add INFO log for qualification count
        self.logger.info(f"Provider Count Calc: Found {len(inst_quals)} qualifications for {self.institution_short_name}.")

        # 2. Filter the original data to include only these qualifications
        market_data_for_inst_quals = self.data[self.data[qual_in_col].isin(inst_quals)].copy()
        
        # Add INFO log for filtered data shape
        self.logger.info(f"Provider Count Calc: Shape of data filtered by institution's qualifications: {market_data_for_inst_quals.shape}")
        
        if market_data_for_inst_quals.empty:
             self.logger.warning("Filtered market data for institution's qualifications is empty.")
             return pd.DataFrame(columns=[year_out_col, 'Unique_Providers_Count', 'Unique_Subcontractors_Count'])

        # 3. Group by year and count unique non-null providers/subcontractors
        # Use output config names for the new count columns
        provider_count_col_name = self.cols_out.get('unique_providers_count', 'Unique_Providers_Count')
        subcontractor_count_col_name = self.cols_out.get('unique_subcontractors_count', 'Unique_Subcontractors_Count')
        
        provider_counts = market_data_for_inst_quals.groupby(year_in_col).agg(
            # Use the dynamically determined column names
            **{provider_count_col_name: (provider_in_col, lambda x: x.nunique())},
            **{subcontractor_count_col_name: (subcontractor_in_col, lambda x: x.nunique())}
        ).reset_index()
        
        # Rename year column to output standard
        provider_counts = provider_counts.rename(columns={year_in_col: year_out_col})
        
        # Add INFO log for final counts shape
        self.logger.info(f"Provider Count Calc: Final provider counts shape: {provider_counts.shape}")
        self.logger.info("Finished calculating provider counts by year.")
        
        return provider_counts

    def _calculate_bcg_data(self, 
                            detailed_providers_market_df: pd.DataFrame, 
                            qualification_market_yoy_growth_df: pd.DataFrame,
                            volumes_by_qualification_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the data required for a BCG Growth-Share Matrix.

        Combines latest market growth, relative market share, and institution volume 
        for each qualification.

        Args:
            detailed_providers_market_df: DataFrame from calculate_providers_market().
            qualification_market_yoy_growth_df: DataFrame from calculate_qualification_market_yoy_growth().
            volumes_by_qualification_df: DataFrame from calculate_volumes_by_qualification().

        Returns:
            DataFrame indexed by qualification with columns for 'Market Growth (%)', 
            'Relative Market Share', and 'Institution Volume'. Returns empty if input is insufficient.
        """
        self.logger.info("Calculating data for BCG Matrix...")
        if detailed_providers_market_df.empty or qualification_market_yoy_growth_df.empty or volumes_by_qualification_df.empty or self.max_year is None:
            self.logger.warning("Cannot calculate BCG data: Missing required input DataFrames or max_year.")
            # Return DataFrame with expected columns, including Qualification before setting index
            return pd.DataFrame(columns=[self.cols_out['qualification'], 'Market Growth (%)', 'Relative Market Share', 'Institution Volume'])#.set_index(self.cols_out['qualification'])

        year_col = self.cols_out['year']
        qual_col = self.cols_out['qualification']
        provider_col = self.cols_out['provider']
        market_share_col = self.cols_out['market_share']
        inst_volume_col = self.cols_out['total_volume'] # Institution's volume in volumes_by_qualification_df
        # Dynamically determine the growth column name based on config
        market_total_col_name = self.cols_out['market_total']
        growth_col = f"{market_total_col_name} YoY Growth (%)" # Use the dynamic name

        latest_year = self.max_year

        # 1. Get latest Market Growth Rate per qualification
        latest_growth = qualification_market_yoy_growth_df[qualification_market_yoy_growth_df[year_col] == latest_year]
        # Check if growth_col exists before selecting
        if growth_col not in latest_growth.columns:
             self.logger.warning(f"Growth column '{growth_col}' not found in qualification_market_yoy_growth_df. Cannot calculate BCG growth.")
             # Create an empty DataFrame with the expected columns for joining later
             latest_growth = pd.DataFrame(columns=[qual_col, growth_col]).set_index(qual_col)
        else:
             latest_growth = latest_growth[[qual_col, growth_col]].set_index(qual_col)
        
        # Fill NaN growth with 0? Or drop? Let's fill with 0 for now.
        latest_growth = latest_growth.fillna(0) 

        # 2. Get latest Institution Volume per qualification
        latest_volumes = volumes_by_qualification_df[volumes_by_qualification_df[year_col] == latest_year]
        latest_volumes = latest_volumes[[qual_col, inst_volume_col]].set_index(qual_col)
        # Ensure volume is numeric and fill NaNs with 0
        latest_volumes[inst_volume_col] = pd.to_numeric(latest_volumes[inst_volume_col], errors='coerce').fillna(0)

        # 3. Calculate latest Relative Market Share per qualification
        latest_market_data = detailed_providers_market_df[detailed_providers_market_df[year_col] == latest_year].copy()
        
        # Ensure market share is numeric
        latest_market_data[market_share_col] = pd.to_numeric(latest_market_data[market_share_col], errors='coerce').fillna(0)


        relative_shares = {}
        for qual in latest_market_data[qual_col].unique():
            qual_data = latest_market_data[latest_market_data[qual_col] == qual]
            
            # Find institution's share (handle multiple variants by summing)
            inst_data = qual_data[qual_data[provider_col].isin(self.institution_names)]
            inst_share = inst_data[market_share_col].sum() # Sum if multiple variants appear as separate providers

            # Find competitors' shares
            competitor_data = qual_data[~qual_data[provider_col].isin(self.institution_names)]
            
            if competitor_data.empty:
                # Institution has 100% market share or is the only one
                largest_competitor_share = 0
            else:
                largest_competitor_share = competitor_data[market_share_col].max()

            # Calculate relative share
            if largest_competitor_share > 0:
                relative_share = inst_share / largest_competitor_share
            elif inst_share > 0: 
                # Institution has share, but no competitors (or competitors have 0 share)
                relative_share = np.inf # Assign inf, handle in plotting.
            else:
                # Institution has 0 share
                relative_share = 0
                
            relative_shares[qual] = relative_share

        relative_share_df = pd.DataFrame.from_dict(relative_shares, orient='index', columns=['Relative Market Share'])
        relative_share_df.index.name = qual_col

        # 4. Combine the data
        bcg_df = latest_growth.join(relative_share_df, how='inner') # Inner join ensures we only have quals with both growth and share data
        bcg_df = bcg_df.join(latest_volumes, how='inner') # Inner join ensures we only have quals with volume data too

        # Rename columns for clarity
        bcg_df = bcg_df.rename(columns={
            growth_col: 'Market Growth (%)',
            inst_volume_col: 'Institution Volume'
        })
        
        # Reset index to have qualification as a column
        bcg_df = bcg_df.reset_index() # Keep Qualification as a column

        self.logger.info(f"Finished calculating BCG data. Found data for {len(bcg_df)} qualifications.")
        return bcg_df

    def get_all_results(self) -> Dict[str, pd.DataFrame]:
        """
        Calculates the data required for a BCG Growth-Share Matrix.

        Combines latest market growth, relative market share, and institution volume 
        for each qualification.

        Args:
            detailed_providers_market_df: DataFrame from calculate_providers_market().
            qualification_market_yoy_growth_df: DataFrame from calculate_qualification_market_yoy_growth().
            volumes_by_qualification_df: DataFrame from calculate_volumes_by_qualification().

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing results for:
                - 'total_volumes'
                - 'volumes_by_qualification'
                - 'market_shares' (Note: This is the simplified YoY view for all providers)
                - 'detailed_providers_market' (Detailed view for relevant qualifications/providers/years)
                - 'qualification_cagr'
                - 'overall_total_market_volume' (Series: Index=Year, Values=Total Volume)
                - 'qualification_market_yoy_growth'
                - 'provider_counts_by_year'
                - 'bcg_data'
        """
        self.logger.info(f"Starting market analysis for: {self.institution_names}")

        # Perform core calculations
        results = {}
        results['total_volumes'] = self.calculate_total_volumes()
        results['volumes_by_qualification'] = self.calculate_volumes_by_qualification()
        results['detailed_providers_market'] = self.calculate_providers_market()
        results['qualification_cagr'] = self.calculate_qualification_cagr()
        results['overall_total_market_volume'] = self.calculate_overall_total_market_volume()

        # Calculate qualification market growth based on detailed provider market data
        # Do this *before* filtering detailed_providers_market, so growth calculation has full context
        qualification_market_yoy_growth_df = pd.DataFrame()
        if 'detailed_providers_market' in results and not results['detailed_providers_market'].empty:
            qualification_market_yoy_growth_df = self.calculate_qualification_market_yoy_growth(
                results['detailed_providers_market']
            )
        results['qualification_market_yoy_growth'] = qualification_market_yoy_growth_df # Store potentially unfiltered growth

        # --- Calculate Provider Counts --- 
        try:
            provider_counts_df = self._calculate_provider_counts_by_year()
            results['provider_counts_by_year'] = provider_counts_df
        except Exception as e:
            self.logger.error(f"Failed to calculate provider counts by year: {e}", exc_info=True)
            # Ensure the key exists even if calculation fails
            results['provider_counts_by_year'] = pd.DataFrame(columns=[
                self.cols_out['year'], 'Unique_Providers_Count', 'Unique_Subcontractors_Count'
            ])
        # --- End Provider Counts --- 

        # --- Calculate BCG Data ---
        try:
            bcg_data_df = self._calculate_bcg_data(
                detailed_providers_market_df=results['detailed_providers_market'],
                qualification_market_yoy_growth_df=results['qualification_market_yoy_growth'],
                volumes_by_qualification_df=results['volumes_by_qualification']
            )
            results['bcg_data'] = bcg_data_df
            self.logger.info(f"Calculated bcg_data. Shape: {results['bcg_data'].shape}")
        except Exception as e:
            self.logger.error(f"Failed to calculate BCG data: {e}", exc_info=True)
            # Ensure the key exists even if calculation fails
            results['bcg_data'] = pd.DataFrame(columns=[
                 self.cols_out['qualification'], 'Market Growth (%)', 'Relative Market Share', 'Institution Volume'
             ]) # Removed set_index for consistency
        # --- End BCG Data ---

        # --- Filter results based on minimum market size threshold and institution inactivity ---
        quals_to_exclude_low_volume = set()
        quals_to_exclude_inactive = set()
        
        # Determine the last full year (usually max_year - 1, unless only one year exists)
        last_full_year = self.max_year
        if self.max_year is not None and self.min_year is not None and self.max_year > self.min_year:
            last_full_year = self.max_year - 1 
        
        # Check if we have detailed data and at least one reference year (last_full_year or max_year)
        if 'detailed_providers_market' in results and not results['detailed_providers_market'].empty and (last_full_year is not None or self.max_year is not None):
            detailed_df = results['detailed_providers_market'] # Get the original, unfiltered detailed data
            # Use config output column names
            year_col = self.cols_out['year']
            qual_col = self.cols_out['qualification']
            market_total_col = self.cols_out['market_total']
            total_volume_col = self.cols_out['total_volume']
            provider_col = self.cols_out['provider'] # Get provider column name from output config
            inst_names_list = self.institution_names
            inst_short_name_log = self.institution_short_name

            # --- Calculate Qualifications to Exclude (for specific filtering later) ---
            # 1. Check for Low Total Market Volume
            # ... (existing low volume check logic - calculates quals_to_exclude_low_volume) ...
            last_year_totals = pd.DataFrame()
            if last_full_year is not None and last_full_year in detailed_df[year_col].unique():
                 last_year_totals = detailed_df[detailed_df[year_col] == last_full_year][[qual_col, market_total_col]].drop_duplicates()
            current_year_totals = pd.DataFrame()
            if self.max_year is not None and self.max_year in detailed_df[year_col].unique():
                current_year_totals = detailed_df[detailed_df[year_col] == self.max_year][[qual_col, market_total_col]].drop_duplicates()
            quals_below_last = set()
            if not last_year_totals.empty:
                 # Access threshold from config
                 threshold = self.cfg.get('analysis', {}).get('min_market_size_threshold', 5)
                 quals_below_last = set(last_year_totals[last_year_totals[market_total_col] < threshold][qual_col].unique())
            quals_below_current = set()
            if not current_year_totals.empty:
                 # Access threshold from config
                 threshold = self.cfg.get('analysis', {}).get('min_market_size_threshold', 5)
                 quals_below_current = set(current_year_totals[current_year_totals[market_total_col] < threshold][qual_col].unique())
            log_year_text_low_vol = ""
            if last_full_year is not None and self.max_year is not None and last_full_year != self.max_year:
                quals_to_exclude_low_volume = quals_below_last.intersection(quals_below_current)
                log_year_text_low_vol = f"BOTH {last_full_year} and {self.max_year}"
            elif quals_below_current:
                 quals_to_exclude_low_volume = quals_below_current
                 log_year_text_low_vol = f"{self.max_year}"
            elif quals_below_last:
                 quals_to_exclude_low_volume = quals_below_last
                 log_year_text_low_vol = f"{last_full_year}"
            else:
                largest_competitor_share = competitor_data[market_share_col].max()

            # Calculate relative share
            if largest_competitor_share > 0:
                relative_share = inst_share / largest_competitor_share
            elif inst_share > 0: 
                # Institution has share, but no competitors (or competitors have 0 share)
                relative_share = np.inf # Assign inf, handle in plotting.
            else:
                # Institution has 0 share
                relative_share = 0
                
            relative_shares[qual] = relative_share

        relative_share_df = pd.DataFrame.from_dict(relative_shares, orient='index', columns=['Relative Market Share'])
        relative_share_df.index.name = qual_col

        # 4. Combine the data
        bcg_df = latest_growth.join(relative_share_df, how='inner') # Inner join ensures we only have quals with both growth and share data
        bcg_df = bcg_df.join(latest_volumes, how='inner') # Inner join ensures we only have quals with volume data too

        # Rename columns for clarity
        bcg_df = bcg_df.rename(columns={
            growth_col: 'Market Growth (%)',
            inst_volume_col: 'Institution Volume'
        })
        
        # Reset index to have qualification as a column
        bcg_df = bcg_df.reset_index()

        self.logger.info(f"Finished calculating BCG data. Found data for {len(bcg_df)} qualifications.")
        return bcg_df

    def get_all_results(self) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Calculate all standard analysis results and return them in a dictionary.
        
        This is the main calculation hub before applying filters in analyze().

        Returns:
            Dictionary where keys are result names (e.g., 'total_volumes', 'detailed_providers_market') 
            and values are the corresponding DataFrames or Series.
        """
        results = {}
        
        self.logger.info("Calculating all analysis results...")
        
        results['total_volumes'] = self.calculate_total_volumes()
        self.logger.info(f"Calculated total_volumes. Shape: {results['total_volumes'].shape}")
        
        results['volumes_by_qualification'] = self.calculate_volumes_by_qualification()
        self.logger.info(f"Calculated volumes_by_qualification. Shape: {results['volumes_by_qualification'].shape}")
        
        results['detailed_providers_market'] = self.calculate_providers_market()
        self.logger.info(f"Calculated detailed_providers_market. Shape: {results['detailed_providers_market'].shape}")
        
        results['qualification_cagr'] = self.calculate_qualification_cagr()
        self.logger.info(f"Calculated qualification_cagr. Shape: {results['qualification_cagr'].shape}")
        
        results['overall_total_market_volume'] = self.calculate_overall_total_market_volume()
        self.logger.info(f"Calculated overall_total_market_volume. Length: {len(results['overall_total_market_volume'])}")
        
        # YoY growth depends on detailed market data
        results['qualification_market_yoy_growth'] = self.calculate_qualification_market_yoy_growth(results['detailed_providers_market'])
        self.logger.info(f"Calculated qualification_market_yoy_growth. Shape: {results['qualification_market_yoy_growth'].shape}")

        # Provider counts calculation
        results['provider_counts_by_year'] = self._calculate_provider_counts_by_year()
        self.logger.info(f"Calculated provider_counts_by_year. Shape: {results['provider_counts_by_year'].shape}")

        # BCG data calculation depends on detailed market, YoY growth, and volumes by qual
        results['bcg_data'] = self._calculate_bcg_data(
            detailed_providers_market_df=results['detailed_providers_market'],
            qualification_market_yoy_growth_df=results['qualification_market_yoy_growth'],
            volumes_by_qualification_df=results['volumes_by_qualification']
        )
        self.logger.info(f"Calculated bcg_data. Shape: {results['bcg_data'].shape}")

        self.logger.info("Finished calculating all analysis results.")
        return results

    def analyze(self, min_market_size_threshold: int = 5) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Perform the full market analysis, including calculations and filtering.

        Args:
            min_market_size_threshold (int): Minimum total market size for a qualification
                                             in the latest year to be included in the results.
                                             Defaults to 5.
                                             (Note: Threshold currently applied within get_all_results)

        Returns:
            Dict[str, Union[pd.DataFrame, pd.Series]]: Dictionary containing results for:
                - 'total_volumes'
                - 'volumes_by_qualification'
                - 'detailed_providers_market' (Detailed view for relevant qualifications/providers/years)
                - 'qualification_cagr'
                - 'overall_total_market_volume' (Series: Index=Year, Values=Total Volume)
                - 'qualification_market_yoy_growth'
                - 'provider_counts_by_year'
                - 'bcg_data'
        """
        self.logger.info(f"Calling get_all_results to perform market analysis for: {self.institution_names}")
        # The filtering based on threshold is now handled within get_all_results
        return self.get_all_results()
        
        # REMOVED DUPLICATED LOGIC FROM HERE

    def export_to_csv(self, base_path: str, min_market_size_threshold: int = 5) -> Dict[str, str]:
        """
        Run the analysis and export all resulting DataFrames/Series to CSV files.

        Args:
            base_path: Base path for output files (without extension).
            min_market_size_threshold (int): Minimum total market size threshold for filtering.
            
        Returns:
            Dictionary mapping analysis type (key from analyze results) to the saved file path.
        """
        self.logger.info(f"Starting analysis and export to CSV with base path: {base_path}")
        
        # Run the analysis first to get the potentially filtered results
        analysis_results = self.analyze(min_market_size_threshold=min_market_size_threshold)
        
        exported_files = {}
        
        for key, df_or_series in analysis_results.items():
            if df_or_series is not None and not df_or_series.empty:
                file_path = f"{base_path}_{key}.csv"
                try:
                    # Handle both DataFrames and Series
                    if isinstance(df_or_series, pd.Series):
                        # Give Series a default name if needed or use its name attribute
                        series_name = df_or_series.name if df_or_series.name else 'Value'
                        df_or_series.to_csv(file_path, index=True, header=[series_name]) # Include index as it's likely the year
                    else: # It's a DataFrame
                        df_or_series.to_csv(file_path, index=False)
                        
                    exported_files[key] = file_path
                    self.logger.info(f"Successfully exported '{key}' to {file_path}")
                except Exception as e:
                    self.logger.error(f"Failed to export '{key}' to {file_path}: {e}")
            else:
                self.logger.warning(f"Skipping export for '{key}': Data is empty or None.")
                
        self.logger.info(f"CSV export process finished. Exported files: {list(exported_files.keys())}")
        return exported_files 