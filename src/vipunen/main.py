"""Main script for Vipunen data analysis."""
import logging
from pathlib import Path
import pandas as pd
from vipunen.api.client import VipunenAPIClient, APIConfig
from vipunen.analysis.market import analyze_market
from vipunen.utils.data_utils import clean_column_names, convert_to_numeric, handle_missing_values
from vipunen.utils.file_handler import VipunenFileHandler
from vipunen.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    API_CONFIG,
    ANALYSIS_CONFIG
)
from FileUtils import OutputFileType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize file handler
file_handler = VipunenFileHandler()

def main():
    """Main function to fetch and analyze Vipunen data."""
    try:
        # Initialize API client with configuration
        api_config = APIConfig(
            dataset=API_CONFIG['dataset'],
            limit=API_CONFIG['limit'],
            max_retries=API_CONFIG['max_retries']
        )
        client = VipunenAPIClient(config=api_config)
        
        # Fetch data from API
        logger.info("Fetching data from Vipunen API...")
        client.fetch_and_save_data(RAW_DATA_DIR)
        raw_data_path = RAW_DATA_DIR / f"{api_config.dataset}.csv"
        
        # Load and preprocess data using FileUtils
        logger.info("Loading and preprocessing data...")
        df = file_handler.load_data(
            raw_data_path, 
            input_type="raw",
            sep=';'  # Try with semicolon separator first
        )
        
        # Print column names for debugging
        logger.info("\nAvailable columns:")
        for col in df.columns:
            logger.info(f"- {col}")
        
        # Clean and prepare data
        df = clean_column_names(df)
        
        # Print cleaned column names
        logger.info("\nCleaned columns:")
        for col in df.columns:
            logger.info(f"- {col}")
        
        df = convert_to_numeric(df)
        df = handle_missing_values(df, method='fill')
        
        # Filter data by year range
        df = df[
            (df['tilastovuosi'] >= ANALYSIS_CONFIG['min_year']) &
            (df['tilastovuosi'] <= ANALYSIS_CONFIG['max_year'])
        ]
        
        # Perform market analysis
        logger.info("Performing market analysis...")
        market_analysis = analyze_market(
            df=df,
            group_cols=['tutkinto'],
            value_col='opiskelijatlkm',
            provider_col='koulutuksenjarjestaja',
            year_col='tilastovuosi',
            min_years=2
        )
        
        # Save processed data using FileUtils
        logger.info("Saving processed data...")
        file_path = file_handler.save_data(
            market_analysis['market_shares'],
            file_name='processed_market_data',
            output_type='processed',
            output_filetype=OutputFileType.CSV,
            sep=';'
        )
        logger.info(f"Processed data saved to {file_path}")
        
        # Export detailed analysis to Excel
        excel_data = {
            'Market Shares': market_analysis['market_shares'],
            'Market Growth': market_analysis.get('market_growth', pd.DataFrame()),
            'Provider Ranking': market_analysis.get('provider_ranking', pd.DataFrame())
        }
        
        excel_path = file_handler.export_to_excel(
            excel_data,
            file_name='market_analysis_report',
            output_type='reports'
        )
        logger.info(f"Excel report saved to {excel_path}")
        
        # Print summary of results
        logger.info("\nMarket Analysis Summary:")
        logger.info(f"Total years analyzed: {len(market_analysis['market_shares']['tilastovuosi'].unique())}")
        logger.info(f"Total degrees analyzed: {len(market_analysis['market_shares']['tutkinto'].unique())}")
        
        # Print top 5 degrees by market share in the latest year
        latest_year = market_analysis['market_shares']['tilastovuosi'].max()
        latest_shares = market_analysis['market_shares'][market_analysis['market_shares']['tilastovuosi'] == latest_year]
        top_degrees = latest_shares.nlargest(5, 'market_share')
        
        logger.info("\nTop 5 degrees by market share in latest year:")
        for _, row in top_degrees.iterrows():
            logger.info(f"{row['tutkinto']}: {row['market_share']:.1%}")
            
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 