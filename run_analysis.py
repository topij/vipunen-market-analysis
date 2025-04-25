#!/usr/bin/env python
"""
Run the education market analysis with command-line arguments.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run education market analysis")
    
    parser.add_argument(
        "--data-file", "-d",
        dest="data_file",
        help="Path to the data file (CSV)",
        default="data/raw/ammatillinen_koulutus_2018_2022.csv"
    )
    
    parser.add_argument(
        "--institution", "-i",
        dest="institution",
        help="Main name of the institution to analyze",
        default="Rastor-instituutti"
    )
    
    parser.add_argument(
        "--short-name", "-s",
        dest="short_name",
        help="Short name for the institution (used in titles and file names)",
        default="RI"
    )
    
    parser.add_argument(
        "--variants", "-v",
        dest="variants",
        nargs="+",
        help="Additional name variants for the institution",
        default=["Rastor-instituutti ry", "RASTOR OY", "Rastor Oy"]
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        dest="output_dir",
        help="Base directory for output files (default: data/reports)",
        default=None
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
    
    return parser.parse_args()

def ensure_data_directory(file_path):
    """
    Ensure the file path includes the data directory.
    If the path starts with 'raw/', prepend 'data/' to it.
    """
    if file_path.startswith("raw/"):
        return f"data/{file_path}"
    return file_path

def main():
    """Main function to run the analysis."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Ensure data file path includes the data directory
    args.data_file = ensure_data_directory(args.data_file)
    
    try:
        # Import and run the workflow
        try:
            from analyze_education_market import main as run_workflow, create_dummy_dataset
            
            # Override sys.argv to pass our arguments to the workflow
            orig_argv = sys.argv
            
            # Build new args based on parsed arguments
            new_argv = [
                sys.argv[0],
                "--data-file", args.data_file,
                "--institution", args.institution,
                "--short-name", args.short_name,
            ]
            
            # Add institution variants
            for variant in args.variants:
                new_argv.extend(["--variant", variant])
            
            # Add output directory if specified
            if args.output_dir:
                new_argv.extend(["--output-dir", args.output_dir])
            
            # Add dummy data flag if specified
            if args.use_dummy:
                new_argv.append("--use-dummy")
            
            # Add filtering options if specified
            if args.filter_qual_types:
                new_argv.append("--filter-qual-types")
            
            if args.filter_by_inst_quals:
                new_argv.append("--filter-by-institution-quals")
            
            # Replace sys.argv
            sys.argv = new_argv
            
            # Run the workflow
            logger.info("Starting education market analysis")
            logger.info(f"Institution: {args.institution}")
            logger.info(f"Data file: {args.data_file}")
            
            if args.use_dummy:
                logger.info("Using dummy data for demonstration")
            
            run_workflow()
            
            # Restore original sys.argv
            sys.argv = orig_argv
            
            logger.info("Analysis completed successfully")
            return 0
            
        except ImportError as e:
            logger.error(f"Failed to import analyze_education_market module: {e}")
            logger.error("Make sure analyze_education_market.py is in the current directory")
            return 1
            
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 