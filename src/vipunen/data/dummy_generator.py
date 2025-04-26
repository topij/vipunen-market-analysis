"""
Dummy data generator for the Vipunen project.

This module provides functions to generate dummy education market data
for testing and demonstration purposes.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional

from ..config.config_loader import get_config

logger = logging.getLogger(__name__)

def create_dummy_dataset(
    start_year: int = 2017,
    end_year: int = 2024,
    qualification_count: int = 5,
    provider_count: int = 6
) -> pd.DataFrame:
    """
    Create a dummy dataset for testing or demonstration purposes.
    
    Args:
        start_year: First year to include in the dataset
        end_year: Last year to include in the dataset
        qualification_count: Number of qualifications to generate
        provider_count: Number of providers to generate
        
    Returns:
        pd.DataFrame: Generated dummy dataset
    """
    # Get column names from config
    config = get_config()
    columns = config.get('columns', {}).get('input', {})
    
    year_col = columns.get('year', 'tilastovuosi')
    degree_type_col = columns.get('degree_type', 'tutkintotyyppi')
    qualification_col = columns.get('qualification', 'tutkinto')
    provider_col = columns.get('provider', 'koulutuksenJarjestaja')
    subcontractor_col = columns.get('subcontractor', 'hankintakoulutuksenJarjestaja')
    volume_col = columns.get('volume', 'nettoopiskelijamaaraLkm')
    
    # Define years
    years = range(start_year, end_year + 1)
    
    # Define qualifications
    qualifications = [
        "Liiketoiminnan AT", 
        "Johtamisen EAT", 
        "Yrittäjyyden AT", 
        "Myynnin AT", 
        "Markkinointiviestinnän AT"
    ][:qualification_count]
    
    # Define education providers
    providers = [
        "Rastor-instituutti ry", 
        "Business College Helsinki", 
        "Mercuria", 
        "Markkinointi-instituutti",
        "Kauppiaitten Kauppaoppilaitos", 
        "Suomen Liikemiesten Kauppaopisto"
    ][:provider_count]
    
    # Generate rows for the dataframe
    rows = []
    
    # For each year and qualification
    for year in years:
        for qual in qualifications:
            # Base market size for this qualification
            qual_market_size = np.random.randint(200, 800)
            
            # For each provider
            for provider in providers:
                # Provider's share of this qualification
                provider_share = np.random.uniform(0.05, 0.25)
                provider_volume = int(qual_market_size * provider_share)
                
                # Sometimes add subcontractor relationship
                subcontractor = None
                if np.random.random() < 0.3:  # 30% chance of having a subcontractor
                    # Pick a random provider as subcontractor
                    potential_subcontractors = [p for p in providers if p != provider]
                    subcontractor = np.random.choice(potential_subcontractors)
                    
                    # Volume handled by subcontractor (30-70% of total)
                    sub_ratio = np.random.uniform(0.3, 0.7)
                    sub_volume = int(provider_volume * sub_ratio)
                    main_volume = provider_volume - sub_volume
                    
                    # Add row for main provider with subcontractor
                    rows.append({
                        year_col: year,
                        degree_type_col: "Ammattitutkinnot" if "AT" in qual else "Erikoisammattitutkinnot",
                        qualification_col: qual,
                        provider_col: provider,
                        subcontractor_col: subcontractor,
                        volume_col: main_volume
                    })
                    
                    # Add row for the subcontractor portion
                    rows.append({
                        year_col: year,
                        degree_type_col: "Ammattitutkinnot" if "AT" in qual else "Erikoisammattitutkinnot",
                        qualification_col: qual,
                        provider_col: subcontractor,
                        subcontractor_col: None,
                        volume_col: sub_volume
                    })
                else:
                    # No subcontractor, add row for main provider only
                    rows.append({
                        year_col: year,
                        degree_type_col: "Ammattitutkinnot" if "AT" in qual else "Erikoisammattitutkinnot",
                        qualification_col: qual,
                        provider_col: provider,
                        subcontractor_col: None,
                        volume_col: provider_volume
                    })
    
    # Create pandas DataFrame
    df = pd.DataFrame(rows)
    
    logger.info(f"Created dummy dataset with {len(df)} rows")
    return df 