#!/usr/bin/env python
"""Script to fetch data from the Vipunen API.

Allows specifying the dataset via command line argument.
Checks data update date before downloading.
"""
import argparse
import logging
from pathlib import Path

from vipunen.api.client import VipunenAPIClient
from vipunen.config import get_config # Use the config loader

# Configure logging (basic setup, consider moving to a shared utility)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Fetch data from the Vipunen API and save it to a CSV file."""
    parser = argparse.ArgumentParser(description="Fetch data from Vipunen API.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Name of the dataset to fetch (overrides config default)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the output CSV (overrides config default derived from paths.data)."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download data even if the update date hasn't changed."
    )
    args = parser.parse_args()

    # Load configuration
    config = get_config()
    api_config = config.get('api', {})
    paths_config = config.get('paths', {})

    # Determine dataset name
    dataset_name = args.dataset or api_config.get('default_dataset')
    if not dataset_name:
        logger.error("Dataset name not specified in arguments or config file ('api.default_dataset').")
        return 1 # Indicate error

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Infer output directory from the configured data path (removing filename)
        raw_data_path_str = paths_config.get('data')
        if raw_data_path_str:
            output_dir = Path(raw_data_path_str).parent
        else:
            logger.warning("Output directory not specified and 'paths.data' not found in config. Defaulting to 'data/raw'.")
            output_dir = Path("data/raw")

    # Create output directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return 1

    # Initialize the API client
    logger.info(f"Initializing API client for dataset: {dataset_name}")
    try:
        client = VipunenAPIClient(dataset_name=dataset_name, config=api_config)
    except Exception as e:
        logger.error(f"Failed to initialize API client: {e}")
        return 1

    # Fetch and save the data
    check_update = not args.force_download
    logger.info(f"Starting data fetch (check update date: {check_update})... -> {output_dir}")
    try:
        client.fetch_and_save_data(output_dir, check_update_date=check_update)
        logger.info("Data fetching process finished.")
    except Exception as e:
        logger.error(f"Data fetching failed: {e}", exc_info=True) # Include traceback
        return 1

    return 0 # Indicate success

if __name__ == "__main__":
    import sys
    sys.exit(main()) 