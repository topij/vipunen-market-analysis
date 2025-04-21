"""Test the implementation with real data from Vipunen API."""
import logging
from pathlib import Path
import pandas as pd
from vipunen.api.client import VipunenAPIClient, APIConfig
from vipunen.analysis.market import analyze_market
from vipunen.visualization.plotter import create_market_analysis_plots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test the complete workflow with real data from Vipunen API."""
    try:
        # Set up directories
        data_dir = Path('data/raw')
        output_dir = Path('output/real_data_test')
        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the API client with configuration
        config = APIConfig(
            dataset='amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto',
            limit=10000,
            caller_id="0201689-0.rastor-instituutti"
        )
        client = VipunenAPIClient(config)
        
        # Fetch data from the API and save to CSV
        logger.info("Fetching data from Vipunen API...")
        client.fetch_and_save_data(data_dir)
        
        # Load the saved data
        data_file = data_dir / f"{config.dataset}.csv"
        if not data_file.exists():
            logger.error("Data file not found after API fetch")
            return
            
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file, sep=';')
        
        if df.empty:
            logger.error("No data loaded from the CSV file")
            return
            
        logger.info(f"Successfully loaded {len(df)} rows of data")
        
        # Filter data for analysis
        df = df[
            (df['tutkintotyyppi'] == 'Ammatilliset perustutkinnot') &
            (df['tilastovuosi'].between(2020, 2022))
        ]
        
        if df.empty:
            logger.error("No data left after filtering")
            return
            
        logger.info(f"Filtered data contains {len(df)} rows")
        
        # Analyze the market data
        logger.info("Analyzing market data...")
        analysis_results = analyze_market(
            df=df,
            year_col='tilastovuosi',
            provider_col='koulutuksenJarjestaja',
            value_col='nettoopiskelijamaaraLkm',
            group_cols=['tutkinto']  # Group by qualification/degree
        )
        
        # Save analysis results
        logger.info("Saving analysis results...")
        for name, result_df in analysis_results.items():
            result_df.to_csv(output_dir / f"{name}.csv", index=False)
        
        # Create and save visualizations
        logger.info("Creating visualizations...")
        plots = create_market_analysis_plots(analysis_results)
        for name, fig in plots.items():
            fig.write_html(output_dir / f"{name}.html")
            fig.write_image(output_dir / f"{name}.png")
        
        logger.info(f"Analysis complete! Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 