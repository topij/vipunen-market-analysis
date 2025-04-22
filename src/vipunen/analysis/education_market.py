"""
Education Market Analysis module that focuses on analyzing vocational qualification data 
from a specific provider's perspective.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        filter_degree_types: bool = False
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
        
        # Store degree types for later filtering when needed
        self.degree_types = ['Ammattitutkinnot', 'Erikoisammattitutkinnot']
        
        # Log data dimensions
        logger.info(f"Loaded data with {len(self.data)} rows")
    
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
    
    def analyze_total_volume(self) -> pd.DataFrame:
        """
        Calculate the total volume of students and break it down by provider role.
        
        Returns:
            pd.DataFrame: Summary DataFrame with volume breakdowns by year
        """
        # Apply degree type filtering if configured
        working_data = self._filter_data_by_degree_types() if self.filter_degree_types else self.data
        
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
        
        # Find all qualifications offered by the institution
        institution_qualifications = (
            filtered_data[
                (filtered_data[self.provider_col].isin(self.institution_names)) |
                (filtered_data[self.subcontractor_col].isin(self.institution_names))
            ][self.qualification_col].unique()
        )
        
        logger.info(f"Found {len(institution_qualifications)} qualifications offered by the institution")
        
        # Filter data to include only these qualifications
        qualification_data = filtered_data[
            filtered_data[self.qualification_col].isin(institution_qualifications)
        ]
        
        # For each qualification, calculate:
        # 1. Total market volume
        # 2. Institution's volume as main provider
        # 3. Institution's volume as subcontractor
        
        results = []
        
        for qualification in institution_qualifications:
            qual_data = qualification_data[qualification_data[self.qualification_col] == qualification]
            
            for year in qual_data[self.year_col].unique():
                year_qual_data = qual_data[qual_data[self.year_col] == year]
                
                # Total market for this qualification
                total_market = year_qual_data[self.volume_col].sum()
                
                # Institution as main provider
                inst_as_provider = year_qual_data[
                    year_qual_data[self.provider_col].isin(self.institution_names)
                ][self.volume_col].sum()
                
                # Institution as subcontractor
                inst_as_subcontractor = year_qual_data[
                    year_qual_data[self.subcontractor_col].isin(self.institution_names)
                ][self.volume_col].sum()
                
                # Total for institution (both roles)
                inst_total = inst_as_provider + inst_as_subcontractor
                
                # Market share
                market_share = (inst_total / total_market * 100) if total_market > 0 else 0
                
                results.append({
                    self.year_col: year,
                    self.qualification_col: qualification,
                    'tutkinto yhteensä': total_market,
                    'järjestäjänä': inst_as_provider,
                    'hankintana': inst_as_subcontractor,
                    'kouluttaja yhteensä': inst_total,
                    'markkinaosuus (%)': round(market_share, 2)
                })
        
        # Convert to DataFrame and sort
        volumes_by_qual_df = pd.DataFrame(results)
        volumes_by_qual_df = volumes_by_qual_df.sort_values(
            by=[self.qualification_col, self.year_col]
        )
        
        logger.info(f"Generated qualification volume summary with {len(volumes_by_qual_df)} rows")
        return volumes_by_qual_df 