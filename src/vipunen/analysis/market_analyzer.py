"""
Market analyzer module for the Vipunen project.

This module provides a wrapper around the EducationMarketAnalyzer class
to simplify integration with the CLI.
"""
import pandas as pd
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union

from ..config.config_loader import get_config
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
        
        # Format percentage columns
        result_df['Current Market Share'] = result_df['Current Market Share'].apply(lambda x: f"{x*100:.2f}%")
        result_df['Previous Market Share'] = result_df['Previous Market Share'].apply(lambda x: f"{x*100:.2f}%")
        result_df['Volume Growth'] = result_df['Volume Growth'].apply(lambda x: f"{x*100:.2f}%" if not np.isinf(x) else "New")
        result_df['Market Share Change'] = result_df['Market Share Change'].apply(lambda x: f"{x*100:.2f}%")
        
        # Sort by current volume for final display
        result_df = result_df.sort_values('Current Volume', ascending=False).reset_index(drop=True)
        
        # Select columns for final result
        return result_df[[
            'Institution', 'Current Volume', 'Previous Volume', 
            'Current Market Share', 'Previous Market Share',
            'Volume Growth', 'Market Share Change', 'market_gainer', 'Is Target'
        ]]
    
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
            DataFrame with qualification CAGRs, sorted by qualification name
        """
        # Return empty DataFrame if no data or institution
        if self.data.empty or not self.institution_names or self.min_year is None or self.max_year is None:
            return pd.DataFrame()
        
        # We'll track qualification data for the target institution
        qualification_data = {}
        
        # Filter data to include only rows from the target institution
        # (either as provider or subcontractor)
        institution_data = self.data[
            (self.data['koulutuksenJarjestaja'].isin(self.institution_names)) | 
            (self.data['hankintakoulutuksenJarjestaja'].isin(self.institution_names))
        ].copy()
        
        if institution_data.empty:
            return pd.DataFrame()
        
        # Identify all unique qualifications offered by the target institution
        unique_qualifications = institution_data['tutkintokoodi'].unique()
        
        # For each qualification, calculate volumes by year for both the institution and the total market
        for qualification in unique_qualifications:
            # Get data for this qualification
            qual_data = self.data[self.data['tutkintokoodi'] == qualification].copy()
            
            # Get the qualification name from any row (should be consistent)
            qual_name = qual_data['tutkinto'].iloc[0] if not qual_data.empty else f"Unknown ({qualification})"
            
            # Initialize tracking for this qualification
            if qualification not in qualification_data:
                qualification_data[qualification] = {
                    'name': qual_name,
                    'institution_volumes_by_year': {},
                    'market_volumes_by_year': {},
                    'first_year': None,
                    'last_year': None
                }
            
            # Calculate volumes by year for the institution
            for year in range(self.min_year, self.max_year + 1):
                year_data = qual_data[qual_data['tilastovuosi'] == year]
                
                # Calculate market volume for this qualification in this year
                market_volume = year_data['nettoopiskelijamaaraLkm'].sum()
                qualification_data[qualification]['market_volumes_by_year'][year] = market_volume
                
                # Calculate institution volume for this qualification in this year
                institution_year_data = year_data[
                    (year_data['koulutuksenJarjestaja'].isin(self.institution_names)) | 
                    (year_data['hankintakoulutuksenJarjestaja'].isin(self.institution_names))
                ]
                institution_volume = institution_year_data['nettoopiskelijamaaraLkm'].sum()
                
                # Only track years where the institution offered this qualification
                if institution_volume > 0:
                    qualification_data[qualification]['institution_volumes_by_year'][year] = institution_volume
                    
                    # Update first and last years
                    if qualification_data[qualification]['first_year'] is None or year < qualification_data[qualification]['first_year']:
                        qualification_data[qualification]['first_year'] = year
                    
                    if qualification_data[qualification]['last_year'] is None or year > qualification_data[qualification]['last_year']:
                        qualification_data[qualification]['last_year'] = year
        
        # Calculate CAGR and create result DataFrame
        results = []
        
        for qualification, data in qualification_data.items():
            # Skip if we don't have first/last year data
            if data['first_year'] is None or data['last_year'] is None:
                continue
            
            # Skip if only offered in one year (can't calculate growth)
            if data['first_year'] == data['last_year']:
                continue
            
            first_year = data['first_year']
            last_year = data['last_year']
            years_present = len(data['institution_volumes_by_year'])
            
            first_year_inst_volume = data['institution_volumes_by_year'].get(first_year, 0)
            last_year_inst_volume = data['institution_volumes_by_year'].get(last_year, 0)
            
            # Calculate market volumes
            first_year_market_volume = data['market_volumes_by_year'].get(first_year, 0)
            last_year_market_volume = data['market_volumes_by_year'].get(last_year, 0)
            
            # Calculate market shares
            first_year_share = (first_year_inst_volume / first_year_market_volume) if first_year_market_volume > 0 else 0
            last_year_share = (last_year_inst_volume / last_year_market_volume) if last_year_market_volume > 0 else 0
            
            # Calculate CAGR
            years_diff = last_year - first_year
            if years_diff > 0 and first_year_inst_volume > 0:
                cagr = (last_year_inst_volume / first_year_inst_volume) ** (1 / years_diff) - 1
            else:
                cagr = 0
            
            results.append({
                'Qualification': data['name'],
                'First Year': first_year,
                'Last Year': last_year,
                'Years Present': years_present,
                'First Year Volume': first_year_inst_volume,
                'Last Year Volume': last_year_inst_volume,
                'First Year Market Volume': first_year_market_volume,
                'Last Year Market Volume': last_year_market_volume,
                'First Year Market Share': first_year_share,
                'Last Year Market Share': last_year_share,
                'CAGR': cagr
            })
        
        if not results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        
        # Format percentages
        result_df['First Year Market Share'] = result_df['First Year Market Share'].apply(lambda x: f"{x*100:.2f}%")
        result_df['Last Year Market Share'] = result_df['Last Year Market Share'].apply(lambda x: f"{x*100:.2f}%")
        result_df['CAGR'] = result_df['CAGR'].apply(lambda x: f"{x*100:.2f}%")
        
        # Sort by qualification name
        result_df = result_df.sort_values('Qualification').reset_index(drop=True)
        
        return result_df
    
    def get_all_results(self) -> Dict[str, pd.DataFrame]:
        """
        Get all analysis results in a single dictionary.
        
        Returns:
            Dictionary containing all analysis results
        """
        return {
            "total_volumes": self.calculate_total_volumes(),
            "volumes_by_qualification": self.calculate_volumes_by_qualification(),
            "provider's_market": self.calculate_market_shares(),
            "cagr_analysis": self.calculate_qualification_cagr()
        }
    
    def analyze(self) -> Dict[str, pd.DataFrame]:
        """
        Run the full analysis suite.
        
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Starting full analysis")
        try:
            results = self.get_all_results()
            self.logger.info("Full analysis completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise 

    def export_to_csv(self, base_path: str = "market_analysis") -> Dict[str, str]:
        """
        Export analysis results to CSV files
        
        Args:
            base_path: Base path for output files (without extension)
            
        Returns:
            Dictionary mapping analysis type to file path
        """
        results = {}
        
        # Market Share Analysis
        market_share_df = self.calculate_market_shares()
        if not market_share_df.empty:
            market_share_path = f"{base_path}_market_share.csv"
            market_share_df.to_csv(market_share_path, index=False)
            results['market_share'] = market_share_path
        
        # Qualification Analysis
        qual_df = self.calculate_volumes_by_qualification()
        if not qual_df.empty:
            qual_path = f"{base_path}_qualification_volume.csv"
            qual_df.to_csv(qual_path, index=False)
            results['qualification_volume'] = qual_path
        
        # Field Analysis
        field_df = self.calculate_field_volumes()
        if not field_df.empty:
            field_path = f"{base_path}_field_volume.csv"
            field_df.to_csv(field_path, index=False)
            results['field_volume'] = field_path
            
        # CAGR Analysis
        cagr_df = self.calculate_qualification_cagr()
        if not cagr_df.empty:
            cagr_path = f"{base_path}_qualification_cagr.csv"
            cagr_df.to_csv(cagr_path, index=False)
            results['qualification_cagr'] = cagr_path
        
        return results 