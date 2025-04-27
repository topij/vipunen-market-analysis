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

class MarketAnalyzer:
    """
    Main analyzer for market data, focusing on qualifications offered by educational institutions.
    
    This class is a simplified wrapper around EducationMarketAnalyzer, focusing on the most important
    market analysis metrics like total volumes, qualification breakdown, market shares, growth and CAGR.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the MarketAnalyzer with market data.
        
        Args:
            data: DataFrame containing the market data with columns for year, degree type, 
                 qualification, provider, subcontractor and volume.
        """
        self.logger = logging.getLogger(__name__)
        
        self.data = data
        
        # Store the institution names (to be set by caller)
        self.institution_names = []
        self.institution_short_name = ""
        
        # Calculate min and max years
        if not data.empty and 'tilastovuosi' in data.columns:
            self.min_year = data['tilastovuosi'].min()
            self.max_year = data['tilastovuosi'].max()
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
            
        all_qualifications = self.data['tutkinto'].unique().tolist()
        return all_qualifications
        
    def calculate_total_volumes(self) -> pd.DataFrame:
        """
        Calculate the total student volumes for each year.
        
        Returns:
            DataFrame with years as index and total_volume as column
        """
        if self.data.empty or self.min_year is None or self.max_year is None:
            return pd.DataFrame()
        
        # Initialize the analyzer with institution names
        analyzer = EducationMarketAnalyzer(
            data=self.data,
            institution_names=self.institution_names
        )
        
        # Calculate total volumes
        total_volumes = analyzer.analyze_total_volume()
        
        # Ensure the kouluttaja column has the right short name
        if 'kouluttaja' in total_volumes.columns:
            total_volumes['kouluttaja'] = self.institution_short_name
            
        return total_volumes
    
    def calculate_volumes_by_qualification(self) -> pd.DataFrame:
        """
        Calculate student volumes broken down by qualification type and year.
        
        Returns:
            DataFrame with qualification volumes by year
        """
        if self.data.empty or self.min_year is None or self.max_year is None:
            return pd.DataFrame()
        
        # Group by year, qualification, and role (provider/subcontractor)
        results = []
        
        for year in range(self.min_year, self.max_year + 1):
            year_data = self.data[self.data['tilastovuosi'] == year]
            
            # Skip if no data for this year
            if year_data.empty:
                continue
                
            # Get unique qualifications
            qualifications = year_data['tutkinto'].unique()
            
            for qual in qualifications:
                # Filter data for this qualification
                qual_data = year_data[year_data['tutkinto'] == qual]
                
                # Calculate total market volume for this qualification
                market_total = qual_data['nettoopiskelijamaaraLkm'].sum()
                
                # Check if institution is involved with this qualification
                institution_data = qual_data[
                    (qual_data['koulutuksenJarjestaja'].isin(self.institution_names)) | 
                    (qual_data['hankintakoulutuksenJarjestaja'].isin(self.institution_names))
                ]
                
                if institution_data.empty:
                    continue
                
                # Calculate provider amount
                provider_amount = institution_data[
                    institution_data['koulutuksenJarjestaja'].isin(self.institution_names)
                ]['nettoopiskelijamaaraLkm'].sum()
                
                # Calculate subcontractor amount
                subcontractor_amount = institution_data[
                    institution_data['hankintakoulutuksenJarjestaja'].isin(self.institution_names)
                ]['nettoopiskelijamaaraLkm'].sum()
                
                # Calculate total amount
                total_amount = provider_amount + subcontractor_amount
                
                # Calculate market share
                market_share = (total_amount / market_total * 100) if market_total > 0 else 0
                
                results.append({
                    'Year': year,
                    'Qualification': qual,
                    'Provider Amount': provider_amount,
                    'Subcontractor Amount': subcontractor_amount,
                    'Total Amount': total_amount,
                    'Market Total': market_total,
                    'Market Share (%)': market_share
                })
        
        if not results:
            return pd.DataFrame()
            
        # Convert to DataFrame and sort
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(by=['Year', 'Qualification'])
        
        return result_df
    
    def calculate_market_shares(self) -> pd.DataFrame:
        """
        Calculate market shares for each institution.
        
        Returns:
            DataFrame containing market shares, growth rates, and market gainer ranks
        """
        if self.data.empty or self.min_year is None or self.max_year is None or self.min_year >= self.max_year:
            return pd.DataFrame()
        
        # Get the most recent year with full data and the previous year
        current_year = self.max_year
        previous_year = current_year - 1
        
        if previous_year < self.min_year:
            return pd.DataFrame()
        
        # Filter to current and previous year
        current_year_data = self.data[self.data['tilastovuosi'] == current_year].copy()
        previous_year_data = self.data[self.data['tilastovuosi'] == previous_year].copy()
        
        # Check if we have data for both years
        if current_year_data.empty or previous_year_data.empty:
            return pd.DataFrame()
        
        # Calculate volumes for each institution
        def calculate_volumes(df):
            institution_volumes = {}
            
            # Process providers
            provider_df = df.groupby('koulutuksenJarjestaja').agg({
                'nettoopiskelijamaaraLkm': 'sum'
            }).reset_index()
            
            for _, row in provider_df.iterrows():
                institution = row['koulutuksenJarjestaja']
                volume = row['nettoopiskelijamaaraLkm']
                institution_volumes[institution] = institution_volumes.get(institution, 0) + volume
            
            # Process subcontractors
            subcontractor_df = df.groupby('hankintakoulutuksenJarjestaja').agg({
                'nettoopiskelijamaaraLkm': 'sum'
            }).reset_index()
            
            for _, row in subcontractor_df.iterrows():
                institution = row['hankintakoulutuksenJarjestaja']
                if pd.notna(institution) and institution != '':
                    volume = row['nettoopiskelijamaaraLkm']
                    institution_volumes[institution] = institution_volumes.get(institution, 0) + volume
            
            return institution_volumes
        
        current_volumes = calculate_volumes(current_year_data)
        previous_volumes = calculate_volumes(previous_year_data)
        
        # Calculate total volume
        total_current_volume = sum(current_volumes.values())
        total_previous_volume = sum(previous_volumes.values())
        
        # Prepare results DataFrame
        results = []
        
        # Create a set of all institutions
        all_institutions = set(list(current_volumes.keys()) + list(previous_volumes.keys()))
        all_institutions = {inst for inst in all_institutions if pd.notna(inst) and inst != ''}
        
        for institution in all_institutions:
            current_vol = current_volumes.get(institution, 0)
            previous_vol = previous_volumes.get(institution, 0)
            
            # Calculate market shares
            current_share = (current_vol / total_current_volume) if total_current_volume > 0 else 0
            previous_share = (previous_vol / total_previous_volume) if total_previous_volume > 0 else 0
            
            # Calculate growth rate
            if previous_vol > 0:
                growth_rate = (current_vol - previous_vol) / previous_vol
            else:
                growth_rate = float('inf') if current_vol > 0 else 0
            
            # Calculate market share change
            share_change = current_share - previous_share
            
            # Add to results
            results.append({
                'Institution': institution,
                'Current Volume': current_vol,
                'Previous Volume': previous_vol,
                'Current Market Share': current_share,
                'Previous Market Share': previous_share, 
                'Volume Growth': growth_rate,
                'Market Share Change': share_change,
                'Is Target': institution in self.institution_names
            })
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        
        # Sort by growth rate to determine market gainer rank
        growth_sorted_df = result_df.sort_values('Volume Growth', ascending=False).reset_index(drop=True)
        
        # Assign rank numbers to each institution (starting from 1 for the highest growth)
        for i, idx in enumerate(growth_sorted_df.index):
            rank = i + 1
            growth_sorted_df.loc[idx, 'market_gainer'] = f"#{rank}"
        
        # Restore original data
        result_df = growth_sorted_df.copy()
        
        # Sort by current volume for final display
        result_df = result_df.sort_values('Current Volume', ascending=False).reset_index(drop=True)
        
        # Rename columns to match expected output structure where possible
        result_df = result_df.rename(columns={
            'Institution': 'Provider',
            'Current Volume': 'Total Volume', # Represents total volume in the Current Year
            'Current Market Share': 'Market Share (%)' # Represents share in the Current Year
        })
        
        # Select columns for final result
        return result_df[[
            'Provider', 'Total Volume', 'Previous Volume', 
            'Market Share (%)', 'Previous Market Share',
            'Volume Growth', 'Market Share Change', 'market_gainer' # 'Is Target' is less relevant here
        ]]
    
    def calculate_providers_market(self) -> pd.DataFrame:
        """
        Calculate the detailed provider market data as expected for the 'Provider's Market' sheet.
        This includes market shares and ranks for all providers within qualifications
        relevant to the target institution, across all years.

        Returns:
            DataFrame matching the structure defined in docs/export_file_structure.txt:
            Year | Qualification | Provider | Provider Amount | Subcontractor Amount | Total Volume | 
            Market Total | Market Share (%) | Market Rank | Market Share Growth (%) | Market Gainer Rank
        """
        if self.data.empty or not self.institution_names or self.min_year is None or self.max_year is None:
            logger.warning("Cannot calculate provider's market: missing data or institution names.")
            return pd.DataFrame()

        # Step 1: Filter data to relevant qualifications (those offered by the target institution)
        # Use the internal method to get the list of qualifications offered by the institution across all years
        # This requires instantiating EducationMarketAnalyzer temporarily or duplicating the logic
        # Let's duplicate the core logic for simplicity here, focusing on all years
        inst_quals = self.data[
            (self.data['koulutuksenJarjestaja'].isin(self.institution_names)) |
            (self.data['hankintakoulutuksenJarjestaja'].isin(self.institution_names))
        ]['tutkinto'].unique().tolist()

        if not inst_quals:
            logger.warning(f"Target institution '{self.institution_names}' does not offer any qualifications in the dataset.")
            return pd.DataFrame()

        logger.info(f"Analyzing market for {len(inst_quals)} qualifications offered by {self.institution_short_name}")
        filtered_data = self.data[self.data['tutkinto'].isin(inst_quals)].copy()

        # Step 2: Calculate detailed market shares for all providers in these qualifications across all years
        detailed_shares = calculate_detailed_market_shares(
            df=filtered_data,
            provider_names=self.institution_names, # Used to flag target provider, not for filtering here
            year_col='tilastovuosi', 
            qual_col='tutkinto',
            provider_col='koulutuksenJarjestaja',
            subcontractor_col='hankintakoulutuksenJarjestaja',
            value_col='nettoopiskelijamaaraLkm'
        )

        if detailed_shares.empty:
            logger.warning("Calculation of detailed market shares returned no data.")
            return pd.DataFrame()

        # Step 3: Calculate Market Rank within each year-qualification group
        detailed_shares['Market Rank'] = detailed_shares.groupby(['tilastovuosi', 'tutkinto'])['market_share'].rank(ascending=False, method='min').astype(int)

        # Step 4: Calculate YoY Market Share Growth and Gainers
        all_years = sorted(detailed_shares['tilastovuosi'].unique())
        market_changes_all_years = []
        for i in range(1, len(all_years)):
            current_year = all_years[i]
            previous_year = all_years[i-1]
            
            # Calculate changes between these two years
            changes_df = calculate_market_share_changes(
                market_share_df=detailed_shares, # Use the full detailed shares dataframe
                current_year=current_year,
                previous_year=previous_year
            )
            if not changes_df.empty:
                market_changes_all_years.append(changes_df)

        # Combine all yearly changes
        if market_changes_all_years:
            combined_changes = pd.concat(market_changes_all_years, ignore_index=True)
            # Select and rename relevant columns
            change_cols = combined_changes[[
                'tutkinto', 'provider', 'current_year',
                'market_share_change_percent', # Use the percentage growth
                'gainer_rank'
            ]].rename(columns={
                'current_year': 'tilastovuosi',
                'market_share_change_percent': 'Market Share Growth (%)', # Rename correctly
                'gainer_rank': 'Market Gainer Rank'
            })
            
            # Merge changes back into the main dataframe
            detailed_shares = pd.merge(
                detailed_shares, 
                change_cols, 
                on=['tilastovuosi', 'tutkinto', 'provider'], 
                how='left'
            )
        else:
            # Add empty columns if no changes calculated (e.g., only one year of data)
            detailed_shares['Market Share Growth (%)'] = np.nan
            detailed_shares['Market Gainer Rank'] = np.nan

        # Step 5: Format and select final columns according to docs/export_file_structure.txt
        final_df = detailed_shares.rename(columns={
            'tilastovuosi': 'Year',
            'tutkinto': 'Qualification',
            'provider': 'Provider',
            'volume_as_provider': 'Provider Amount',
            'volume_as_subcontractor': 'Subcontractor Amount',
            'total_volume': 'Total Volume', # This is provider's total volume
            'qualification_market_volume': 'Market Total', # This is the total market size for the qual
            'market_share': 'Market Share (%)'
        })

        # Ensure columns are numeric, replacing inf with NaN
        numeric_cols = ['Provider Amount', 'Subcontractor Amount', 'Total Volume', 
                        'Market Total', 'Market Share (%)', 'Market Share Growth (%)']
        for col in numeric_cols:
            if col in final_df.columns:
                # Convert to numeric, coercing errors to NaN
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
                # Replace any infinities that might result from division by zero
                final_df[col] = final_df[col].replace([np.inf, -np.inf], np.nan)
        
        # Keep Market Rank as integer
        if 'Market Rank' in final_df.columns:
            final_df['Market Rank'] = final_df['Market Rank'].fillna(-1).astype(int) # Fill NaN ranks with -1 or similar
        if 'Market Gainer Rank' in final_df.columns:
            # Convert to nullable integer type to keep NaNs for the first year
            # Round first in case of float ranks, then convert to nullable integer
            final_df['Market Gainer Rank'] = final_df['Market Gainer Rank'].round().astype('Int64')

        # Define final column order based on the spec
        final_columns = [
            'Year', 'Qualification', 'Provider', 'Provider Amount', 'Subcontractor Amount',
            'Total Volume', 'Market Total', 'Market Share (%)', 'Market Rank',
            'Market Share Growth (%)', 'Market Gainer Rank' # Use the correct column name
        ]

        # Select and reorder columns, dropping extras
        final_df = final_df[final_columns]

        # Sort for readability
        final_df = final_df.sort_values(by=['Year', 'Qualification', 'Market Rank'], ascending=[True, True, True])

        logger.info(f"Calculated provider's market data with {len(final_df)} rows.")
        return final_df
    
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

        logger.info(f"Calculated CAGR for {len(cagr_results)} qualifications")
        return cagr_results
    
    def calculate_overall_total_market_volume(self) -> pd.Series:
        """
        Calculate the total market volume across all qualifications and providers for each year.

        Returns:
            pd.Series: Index=Year, Values=Total Volume
        """
        if self.data.empty or 'tilastovuosi' not in self.data.columns or 'nettoopiskelijamaaraLkm' not in self.data.columns:
            logger.warning("Cannot calculate overall total market volume: missing required columns or data.")
            return pd.Series(dtype=float)

        overall_volume = self.data.groupby('tilastovuosi')['nettoopiskelijamaaraLkm'].sum()
        logger.info(f"Calculated overall total market volume per year.")
        return overall_volume

    def calculate_qualification_market_yoy_growth(self, detailed_providers_market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Year-over-Year growth of the total market size for each qualification.

        Args:
            detailed_providers_market_df: The DataFrame returned by calculate_providers_market().
                                          Must contain 'Year', 'Qualification', and 'Market Total'.

        Returns:
            pd.DataFrame: Columns: Qualification, Year, Market Total, Market Total YoY Growth (%)
                          Returns empty DataFrame if only one year of data is present.
        """
        if detailed_providers_market_df.empty or not all(col in detailed_providers_market_df.columns for col in ['Year', 'Qualification', 'Market Total']):
            logger.warning("Cannot calculate qualification market YoY growth: Missing required columns or data.")
            return pd.DataFrame()
            
        # Return empty if only one year of data exists
        if detailed_providers_market_df['Year'].nunique() < 2:
            logger.warning("Skipping YoY growth calculation: Only one year of data available.")
            return pd.DataFrame(columns=['Qualification', 'Year', 'Market Total', 'Market Total YoY Growth (%)'])

        # Get unique market totals per qualification per year
        market_totals = detailed_providers_market_df[['Year', 'Qualification', 'Market Total']].drop_duplicates()

        # Sort by Qualification and Year to ensure correct YoY calculation
        market_totals = market_totals.sort_values(by=['Qualification', 'Year'])

        # Calculate YoY growth
        market_totals['Previous Market Total'] = market_totals.groupby('Qualification')['Market Total'].shift(1)
        
        # Calculate growth percentage
        market_totals['Market Total YoY Growth (%)'] = (
            (market_totals['Market Total'] - market_totals['Previous Market Total']) / 
            market_totals['Previous Market Total'] * 100
        )
        
        # Handle division by zero or growth from zero (replace inf with NaN, fill NaN where previous was 0)
        market_totals['Market Total YoY Growth (%)'] = market_totals['Market Total YoY Growth (%)'].replace([np.inf, -np.inf], np.nan)
        # If Previous Market Total was 0 and Current is > 0, growth is effectively infinite, represent as NaN or a large number?
        # Let's stick to NaN for now, consistent with market share growth.
        market_totals.loc[(market_totals['Previous Market Total'] == 0) & (market_totals['Market Total'] > 0), 'Market Total YoY Growth (%)'] = np.nan
        # If both are 0, growth is 0
        market_totals.loc[(market_totals['Previous Market Total'] == 0) & (market_totals['Market Total'] == 0), 'Market Total YoY Growth (%)'] = 0

        logger.info(f"Calculated qualification market YoY growth data.")
        return market_totals[['Qualification', 'Year', 'Market Total', 'Market Total YoY Growth (%)']]

    def get_all_results(self) -> Dict[str, pd.DataFrame]:
        """
        Run all analyses and return a dictionary of results.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing results for:
                - 'total_volumes'
                - 'volumes_by_qualification'
                - 'market_shares' (Note: This is the simplified YoY view for all providers)
                - 'detailed_providers_market' (Detailed view for relevant qualifications/providers/years)
                - 'qualification_cagr'
                - 'overall_total_market_volume' (Series: Index=Year, Values=Total Volume)
                - 'qualification_market_yoy_growth'
        """
        self.logger.info(f"Starting market analysis for: {self.institution_names}")

        # Perform core calculations
        results = {}
        results['total_volumes'] = self.calculate_total_volumes()
        results['volumes_by_qualification'] = self.calculate_volumes_by_qualification()
        results['market_shares'] = self.calculate_market_shares()
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
            provider_col = 'Provider' 
            inst_names_list = self.institution_names 
            inst_short_name_log = self.institution_short_name

            # --- Calculate Qualifications to Exclude (for specific filtering later) --- 
            # 1. Check for Low Total Market Volume
            # ... (existing low volume check logic - calculates quals_to_exclude_low_volume) ...
            last_year_totals = pd.DataFrame()
            if last_full_year is not None and last_full_year in detailed_df['Year'].unique():
                 last_year_totals = detailed_df[detailed_df['Year'] == last_full_year][['Qualification', 'Market Total']].drop_duplicates()
            current_year_totals = pd.DataFrame()
            if self.max_year is not None and self.max_year in detailed_df['Year'].unique():
                current_year_totals = detailed_df[detailed_df['Year'] == self.max_year][['Qualification', 'Market Total']].drop_duplicates()
            quals_below_last = set()
            if not last_year_totals.empty:
                quals_below_last = set(last_year_totals[last_year_totals['Market Total'] < min_market_size_threshold]['Qualification'].unique())
            quals_below_current = set()
            if not current_year_totals.empty:
                 quals_below_current = set(current_year_totals[current_year_totals['Market Total'] < min_market_size_threshold]['Qualification'].unique())
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
                 quals_to_exclude_low_volume = set()
                 log_year_text_low_vol = "relevant years"
            if quals_to_exclude_low_volume:
                self.logger.info(f"Identified {len(quals_to_exclude_low_volume)} qualifications with total market size < {min_market_size_threshold} in {log_year_text_low_vol}: {list(quals_to_exclude_low_volume)}")
            else:
                self.logger.info(f"No qualifications met the low total market size threshold ({min_market_size_threshold}) for exclusion.")

            # 2. Check for Institution Inactivity (Using variants `inst_names_list`, threshold >= 1)
            # ... (existing inactivity check logic - calculates quals_to_exclude_inactive) ...
            inst_quals_last = set()
            if last_full_year is not None and last_full_year in detailed_df['Year'].unique():
                inst_data_last = detailed_df[(detailed_df['Year'] == last_full_year) & (detailed_df[provider_col].isin(inst_names_list)) & (detailed_df['Total Volume'] >= 1)]
                inst_quals_last = set(inst_data_last['Qualification'].unique())
            inst_quals_current = set()
            if self.max_year is not None and self.max_year in detailed_df['Year'].unique():
                inst_data_current = detailed_df[(detailed_df['Year'] == self.max_year) & (detailed_df[provider_col].isin(inst_names_list)) & (detailed_df['Total Volume'] >= 1)]
                inst_quals_current = set(inst_data_current['Qualification'].unique())
            all_inst_quals = set(detailed_df[detailed_df[provider_col].isin(inst_names_list)]['Qualification'].unique())
            log_year_text_inactive = ""
            if last_full_year is not None and self.max_year is not None and last_full_year != self.max_year:
                inactive_in_both = all_inst_quals - inst_quals_last - inst_quals_current
                quals_to_exclude_inactive = inactive_in_both
                log_year_text_inactive = f"BOTH {last_full_year} and {self.max_year}"
            elif self.max_year is not None:
                 inactive_in_current = all_inst_quals - inst_quals_current
                 quals_to_exclude_inactive = inactive_in_current
                 log_year_text_inactive = f"{self.max_year}"
            elif last_full_year is not None:
                 inactive_in_last = all_inst_quals - inst_quals_last
                 quals_to_exclude_inactive = inactive_in_last
                 log_year_text_inactive = f"{last_full_year}"
            else:
                 quals_to_exclude_inactive = set()
                 log_year_text_inactive = "relevant years"
            if quals_to_exclude_inactive:
                self.logger.info(f"Identified {len(quals_to_exclude_inactive)} qualifications as inactive (volume < 1) for {inst_short_name_log} in {log_year_text_inactive}: {list(quals_to_exclude_inactive)}")
            else:
                self.logger.info(f"No qualifications identified as inactive for {inst_short_name_log} (volume < 1 in relevant years)." )

            # --- Apply Filtering Selectively --- 
            quals_to_exclude_final = list(quals_to_exclude_low_volume.union(quals_to_exclude_inactive))
            if quals_to_exclude_final:
                 self.logger.info(f"Combined list of {len(quals_to_exclude_final)} qualifications identified for potential exclusion (low volume or inactive): {quals_to_exclude_final}")
            
            # 1. Filter Qualification Market YoY Growth (Apply combined filter)
            if 'qualification_market_yoy_growth' in results and not results['qualification_market_yoy_growth'].empty:
                growth_df = results['qualification_market_yoy_growth']
                initial_rows = len(growth_df)
                filtered_growth_df = growth_df[~growth_df['Qualification'].isin(quals_to_exclude_final)]
                results['qualification_market_yoy_growth'] = filtered_growth_df
                rows_removed = initial_rows - len(filtered_growth_df)
                if rows_removed > 0:
                    self.logger.info(f"Filtered {rows_removed} rows from 'qualification_market_yoy_growth' based on combined exclusion list.")
            
            # 2. Filter Detailed Provider Market (Remove Zero Volume Rows ONLY)
            #    Do NOT filter by quals_to_exclude_final here. Keep all quals institution participated in.
            if 'detailed_providers_market' in results and not results['detailed_providers_market'].empty:
                detailed_df_to_filter = results['detailed_providers_market'] # Use the DF from results dict
                initial_rows = len(detailed_df_to_filter)
                # Filter rows where Total Volume is exactly 0
                results['detailed_providers_market'] = detailed_df_to_filter[detailed_df_to_filter['Total Volume'] > 0]
                final_rows = len(results['detailed_providers_market'])
                rows_removed = initial_rows - final_rows
                if rows_removed > 0:
                    self.logger.info(f"Removed {rows_removed} zero-volume rows from 'detailed_providers_market'. Qualification filtering NOT applied.")
                else:
                     self.logger.info("No zero-volume rows found in 'detailed_providers_market'. Qualification filtering NOT applied.")

            # 3. Volumes by Qualification (No additional filtering by low volume/inactivity)
            if 'volumes_by_qualification' in results and not results['volumes_by_qualification'].empty:
                 self.logger.info("'volumes_by_qualification' remains unfiltered by low volume/inactivity criteria.")
                 # Optional: could add filtering here to only show quals where inst_volume >=1 if needed

            # 4. Qualification CAGR (No filtering)
            if 'qualification_cagr' in results and not results['qualification_cagr'].empty:
                 self.logger.info("'qualification_cagr' remains unfiltered.")
                 
        else: # Case where no detailed_providers_market data exists or only one year
            self.logger.warning("Skipping low market size and inactivity filtering due to missing/insufficient detailed market data.")

        self.logger.info(f"Market analysis complete.")
        return results

    def analyze(self, min_market_size_threshold: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Perform the complete market analysis and return all results.

        Args:
            min_market_size_threshold (int): Minimum total market size for a qualification
                                             in the latest year to be included in the results.
                                             Defaults to 5.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing results for:
                - 'total_volumes'
                - 'volumes_by_qualification'
                - 'market_shares' (Note: This is the simplified YoY view for all providers)
                - 'detailed_providers_market' (Detailed view for relevant qualifications/providers/years)
                - 'qualification_cagr'
                - 'overall_total_market_volume' (Series: Index=Year, Values=Total Volume)
                - 'qualification_market_yoy_growth'
        """
        self.logger.info(f"Starting market analysis for: {self.institution_names}")

        # Perform core calculations
        results = {}
        results['total_volumes'] = self.calculate_total_volumes()
        results['volumes_by_qualification'] = self.calculate_volumes_by_qualification()
        results['market_shares'] = self.calculate_market_shares()
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
            provider_col = 'Provider' 
            inst_names_list = self.institution_names 
            inst_short_name_log = self.institution_short_name

            # --- Calculate Qualifications to Exclude (for specific filtering later) --- 
            # 1. Check for Low Total Market Volume
            # ... (existing low volume check logic - calculates quals_to_exclude_low_volume) ...
            last_year_totals = pd.DataFrame()
            if last_full_year is not None and last_full_year in detailed_df['Year'].unique():
                 last_year_totals = detailed_df[detailed_df['Year'] == last_full_year][['Qualification', 'Market Total']].drop_duplicates()
            current_year_totals = pd.DataFrame()
            if self.max_year is not None and self.max_year in detailed_df['Year'].unique():
                current_year_totals = detailed_df[detailed_df['Year'] == self.max_year][['Qualification', 'Market Total']].drop_duplicates()
            quals_below_last = set()
            if not last_year_totals.empty:
                quals_below_last = set(last_year_totals[last_year_totals['Market Total'] < min_market_size_threshold]['Qualification'].unique())
            quals_below_current = set()
            if not current_year_totals.empty:
                 quals_below_current = set(current_year_totals[current_year_totals['Market Total'] < min_market_size_threshold]['Qualification'].unique())
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
                 quals_to_exclude_low_volume = set()
                 log_year_text_low_vol = "relevant years"
            if quals_to_exclude_low_volume:
                self.logger.info(f"Identified {len(quals_to_exclude_low_volume)} qualifications with total market size < {min_market_size_threshold} in {log_year_text_low_vol}: {list(quals_to_exclude_low_volume)}")
            else:
                self.logger.info(f"No qualifications met the low total market size threshold ({min_market_size_threshold}) for exclusion.")

            # 2. Check for Institution Inactivity (Using variants `inst_names_list`, threshold >= 1)
            # ... (existing inactivity check logic - calculates quals_to_exclude_inactive) ...
            inst_quals_last = set()
            if last_full_year is not None and last_full_year in detailed_df['Year'].unique():
                inst_data_last = detailed_df[(detailed_df['Year'] == last_full_year) & (detailed_df[provider_col].isin(inst_names_list)) & (detailed_df['Total Volume'] >= 1)]
                inst_quals_last = set(inst_data_last['Qualification'].unique())
            inst_quals_current = set()
            if self.max_year is not None and self.max_year in detailed_df['Year'].unique():
                inst_data_current = detailed_df[(detailed_df['Year'] == self.max_year) & (detailed_df[provider_col].isin(inst_names_list)) & (detailed_df['Total Volume'] >= 1)]
                inst_quals_current = set(inst_data_current['Qualification'].unique())
            all_inst_quals = set(detailed_df[detailed_df[provider_col].isin(inst_names_list)]['Qualification'].unique())
            log_year_text_inactive = ""
            if last_full_year is not None and self.max_year is not None and last_full_year != self.max_year:
                inactive_in_both = all_inst_quals - inst_quals_last - inst_quals_current
                quals_to_exclude_inactive = inactive_in_both
                log_year_text_inactive = f"BOTH {last_full_year} and {self.max_year}"
            elif self.max_year is not None:
                 inactive_in_current = all_inst_quals - inst_quals_current
                 quals_to_exclude_inactive = inactive_in_current
                 log_year_text_inactive = f"{self.max_year}"
            elif last_full_year is not None:
                 inactive_in_last = all_inst_quals - inst_quals_last
                 quals_to_exclude_inactive = inactive_in_last
                 log_year_text_inactive = f"{last_full_year}"
            else:
                 quals_to_exclude_inactive = set()
                 log_year_text_inactive = "relevant years"
            if quals_to_exclude_inactive:
                self.logger.info(f"Identified {len(quals_to_exclude_inactive)} qualifications as inactive (volume < 1) for {inst_short_name_log} in {log_year_text_inactive}: {list(quals_to_exclude_inactive)}")
            else:
                self.logger.info(f"No qualifications identified as inactive for {inst_short_name_log} (volume < 1 in relevant years)." )

            # --- Apply Filtering Selectively --- 
            quals_to_exclude_final = list(quals_to_exclude_low_volume.union(quals_to_exclude_inactive))
            if quals_to_exclude_final:
                 self.logger.info(f"Combined list of {len(quals_to_exclude_final)} qualifications identified for potential exclusion (low volume or inactive): {quals_to_exclude_final}")
            
            # 1. Filter Qualification Market YoY Growth (Apply combined filter)
            if 'qualification_market_yoy_growth' in results and not results['qualification_market_yoy_growth'].empty:
                growth_df = results['qualification_market_yoy_growth']
                initial_rows = len(growth_df)
                filtered_growth_df = growth_df[~growth_df['Qualification'].isin(quals_to_exclude_final)]
                results['qualification_market_yoy_growth'] = filtered_growth_df
                rows_removed = initial_rows - len(filtered_growth_df)
                if rows_removed > 0:
                    self.logger.info(f"Filtered {rows_removed} rows from 'qualification_market_yoy_growth' based on combined exclusion list.")
            
            # 2. Filter Detailed Provider Market (Remove Zero Volume Rows ONLY)
            #    Do NOT filter by quals_to_exclude_final here. Keep all quals institution participated in.
            if 'detailed_providers_market' in results and not results['detailed_providers_market'].empty:
                detailed_df_to_filter = results['detailed_providers_market'] # Use the DF from results dict
                initial_rows = len(detailed_df_to_filter)
                # Filter rows where Total Volume is exactly 0
                results['detailed_providers_market'] = detailed_df_to_filter[detailed_df_to_filter['Total Volume'] > 0]
                final_rows = len(results['detailed_providers_market'])
                rows_removed = initial_rows - final_rows
                if rows_removed > 0:
                    self.logger.info(f"Removed {rows_removed} zero-volume rows from 'detailed_providers_market'. Qualification filtering NOT applied.")
                else:
                     self.logger.info("No zero-volume rows found in 'detailed_providers_market'. Qualification filtering NOT applied.")

            # 3. Volumes by Qualification (No additional filtering by low volume/inactivity)
            if 'volumes_by_qualification' in results and not results['volumes_by_qualification'].empty:
                 self.logger.info("'volumes_by_qualification' remains unfiltered by low volume/inactivity criteria.")
                 # Optional: could add filtering here to only show quals where inst_volume >=1 if needed

            # 4. Qualification CAGR (No filtering)
            if 'qualification_cagr' in results and not results['qualification_cagr'].empty:
                 self.logger.info("'qualification_cagr' remains unfiltered.")
                 
        else: # Case where no detailed_providers_market data exists or only one year
            self.logger.warning("Skipping low market size and inactivity filtering due to missing/insufficient detailed market data.")

        self.logger.info(f"Market analysis complete.")
        return results

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