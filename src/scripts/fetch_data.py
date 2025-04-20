#!/usr/bin/env python
"""Script to fetch data from the Vipunen API."""
from pathlib import Path
from vipunen.api.client import VipunenAPIClient
from vipunen.config import RAW_DATA_DIR

def main():
    """Fetch data from the Vipunen API and save it to a CSV file."""
    # Create output directory if it doesn't exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize the API client
    client = VipunenAPIClient()
    
    # Fetch and save the data
    client.fetch_and_save_data(RAW_DATA_DIR)

if __name__ == "__main__":
    main() 