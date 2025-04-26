"""
CLI module for the Vipunen project.

This module provides the main entry point for the education market analysis
workflow, orchestrating the data loading, analysis, and export steps.
"""
import sys
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..config.config_loader import get_config
from ..data.data_loader import load_data, ensure_data_directory
from ..data.data_processor import clean_and_prepare_data
from ..export.excel_exporter import export_to_excel
from ..analysis.market_analyzer import MarketAnalyzer
from .argument_parser import parse_arguments, get_institution_variants

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_analysis(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the education market analysis workflow.
    
    Args:
        args: Dictionary of arguments, defaults to command-line arguments if None
        
    Returns:
        Dict[str, Any]: Dictionary with analysis results
    """
    # Parse arguments if not provided
    if args is None:
        parsed_args = parse_arguments()
        args = vars(parsed_args)
    
    # Step 1: Get configuration
    config = get_config()
    
    # Step 2: Define parameters for the analysis
    data_file_path = ensure_data_directory(args.get('data_file', config['paths']['data']))
    institution_name = args.get('institution', config['institutions']['default']['name'])
    institution_short_name = args.get('short_name', config['institutions']['default']['short_name'])
    use_dummy = args.get('use_dummy', False)
    filter_qual_types = args.get('filter_qual_types', False)
    filter_by_inst_quals = args.get('filter_by_inst_quals', False)
    
    # Set up institution variants
    if 'variants' in args and args['variants']:
        institution_variants = list(args['variants'])
        if institution_name not in institution_variants:
            institution_variants.append(institution_name)
    else:
        institution_variants = config['institutions']['default']['variants']
        if institution_name not in institution_variants:
            institution_variants.append(institution_name)
    
    logger.info(f"Analyzing institution: {institution_name}")
    logger.info(f"Institution variants: {institution_variants}")
    
    # Step 3: Load the raw data
    logger.info(f"Loading raw data from {data_file_path}")
    raw_data = load_data(file_path=data_file_path, use_dummy=use_dummy)
    logger.info(f"Loaded {len(raw_data)} rows of data")
    
    # Step 4: Clean and prepare the data
    logger.info("Cleaning and preparing the data")
    df_clean = clean_and_prepare_data(
        raw_data, 
        institution_names=institution_variants,
        merge_qualifications=True,
        shorten_names=True
    )
    
    # Filter data for the specific institution if needed
    if filter_by_inst_quals or filter_qual_types:
        logger.info("Filtering data based on institution and qualification types")
        # Filter for institution data
        institution_mask = (
            df_clean['koulutuksenJarjestaja'].isin(institution_variants) | 
            df_clean['hankintakoulutuksenJarjestaja'].isin(institution_variants)
        )
        
        # Get qualifications offered by the institution
        if filter_by_inst_quals:
            inst_qualifications = df_clean[institution_mask]['tutkinto'].unique()
            df_clean = df_clean[df_clean['tutkinto'].isin(inst_qualifications)]
            logger.info(f"Filtered to {len(inst_qualifications)} qualifications offered by {institution_name}")
        
        # Filter by qualification types if requested
        if filter_qual_types:
            qual_types = config.get('qualification_types', ['Ammattitutkinnot', 'Erikoisammattitutkinnot'])
            df_clean = df_clean[df_clean['tutkintotyyppi'].isin(qual_types)]
            logger.info(f"Filtered to qualification types: {qual_types}")
    
    # Step 5: Perform analysis using the MarketAnalyzer
    logger.info("Initializing market analyzer")
    analyzer = MarketAnalyzer(
        data=df_clean
    )
    
    # Add institution names as an attribute to be used by the analyzer
    analyzer.institution_names = institution_variants
    analyzer.institution_short_name = institution_short_name
    
    # Run the analysis
    logger.info("Running analysis")
    analysis_results = analyzer.analyze()
    
    # Extract results
    total_volumes = analysis_results.get('total_volumes', pd.DataFrame())
    volumes_by_qual = analysis_results.get('volumes_by_qualification', pd.DataFrame())
    market_shares = analysis_results.get("provider's_market", pd.DataFrame())
    qualification_cagr = analysis_results.get('cagr_analysis', pd.DataFrame())
    
    # Set the kouluttaja column for compatibility with the original implementation
    if not total_volumes.empty and 'kouluttaja' not in total_volumes.columns:
        total_volumes['kouluttaja'] = institution_short_name
    
    # Step 6: Create directory structure for outputs
    logger.info("Creating output directories")
    
    # Determine output directory
    output_dir = args.get('output_dir')
    if output_dir is None:
        output_dir = config['paths'].get('output', 'data/reports')
        
    # Create directory name based on institution
    dir_name = f"education_market_{institution_short_name.lower()}"
    output_dir = Path(output_dir) / dir_name
    
    # Create directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Step 7: Export results to Excel
    logger.info("Exporting results to Excel")
    
    # Prepare Excel data
    excel_data = {
        "Total Volumes": total_volumes.reset_index(drop=True) if not total_volumes.empty else pd.DataFrame(),
        "Volumes by Qualification": volumes_by_qual.reset_index(drop=True) if not volumes_by_qual.empty else pd.DataFrame(),
        "Provider's Market": market_shares.reset_index(drop=True) if not market_shares.empty else pd.DataFrame(),
        "CAGR Analysis": qualification_cagr.reset_index(drop=True) if not qualification_cagr.empty else pd.DataFrame()
    }
    
    # Export to Excel
    excel_path = export_to_excel(
        data_dict=excel_data,
        file_name=f"{institution_short_name.lower()}_market_analysis",
        output_dir=output_dir,
        include_timestamp=True
    )
    
    logger.info(f"Analysis complete!")
    
    return {
        "total_volumes": total_volumes,
        "volumes_by_qualification": volumes_by_qual,
        "provider's_market": market_shares,
        "cagr_analysis": qualification_cagr,
        "excel_path": excel_path
    }

def main():
    """Main entry point for the CLI."""
    try:
        results = run_analysis()
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 