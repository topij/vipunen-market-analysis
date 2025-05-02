"""
Argument parser for the Vipunen project CLI.

This module provides functions to parse command-line arguments for the
education market analysis workflow.
"""
import argparse
import logging
from typing import Dict, Any, Optional, List

from ..config.config_loader import get_config

logger = logging.getLogger(__name__)

def parse_arguments(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the education market analysis.
    
    Args:
        args_list: Optional list of strings to parse. If None, defaults to sys.argv[1:].
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    config = get_config()
    
    # Get default values from config
    default_data_file = config['paths'].get('data', 'data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv')
    default_institution = config['institutions']['default'].get('name', 'Rastor-instituutti')
    default_short_name = config['institutions']['default'].get('short_name', 'RI')
    
    parser = argparse.ArgumentParser(description="Education market analysis workflow")
    
    parser.add_argument(
        "--data-file", "-d",
        dest="data_file",
        help=f"Path to the data file (CSV), default: {default_data_file}",
        default=default_data_file
    )
    
    parser.add_argument(
        "--institution", "-i",
        dest="institution",
        help=f"Main name of the institution to analyze, default: {default_institution}",
        default=default_institution
    )
    
    parser.add_argument(
        "--short-name", "-s",
        dest="short_name",
        help=f"Short name for the institution (used in titles and file names), default: {default_short_name}",
        default=default_short_name
    )
    
    parser.add_argument(
        '--variant', '-V',
        action='append',
        dest='variants',
        default=[],
        help='Name variant for the institution (can be specified multiple times)'
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        dest="output_dir",
        help=f"Base directory for output files, default: {config['paths'].get('output', 'data/reports')}"
    )
    
    parser.add_argument(
        "--use-dummy", "-u",
        dest="use_dummy",
        action="store_true",
        help="Use dummy data instead of loading from file",
        default=False
    )
    
    parser.add_argument(
        "--filter-qual-types",
        dest="filter_qual_types",
        action="store_true",
        help="Filter data to include only ammattitutkinto and erikoisammattitutkinto",
        default=False
    )
    
    parser.add_argument(
        "--filter-by-institution-quals",
        dest="filter_by_inst_quals",
        action="store_true",
        help="Filter data to include only qualifications offered by the institution under analysis during the current and previous year",
        default=False
    )
    
    # Parse arguments (from args_list if provided, else from sys.argv)
    return parser.parse_args(args_list)

def get_institution_variants(args: argparse.Namespace, config: Dict[str, Any]) -> List[str]:
    """
    Get the institution name variants, combining command-line arguments with config defaults.
    
    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary
        
    Returns:
        list: List of institution name variants
    """
    # Use variants from command line if provided
    if args.variants:
        variants = list(args.variants)
        # Add the main institution name if not already included
        if args.institution not in variants:
            variants.append(args.institution)
        return variants
    
    # Otherwise use defaults from config
    variants = config['institutions']['default'].get('variants', [])
    if args.institution not in variants:
        variants.append(args.institution)
    
    return variants 