"""API client for fetching data from the Vipunen API."""
import requests
from pathlib import Path
import shutil
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import logging
import json
import time
from tqdm import tqdm
from ..config import get_config # Import config loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for update date checking
UPDATE_DATE_COLUMN = "tietojoukkoPaivitettyPvm" # API field name for update timestamp

class VipunenAPIClient:
    """Client for interacting with the Vipunen API."""

    def __init__(self, dataset_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the Vipunen API client.

        Args:
            dataset_name: The name of the dataset to fetch (e.g., 'amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto').
            config: Optional configuration dictionary. If not provided, loads from config file.
        """
        self.config = config or get_config().get('api', {}) # Load API section or default to empty dict
        if not self.config:
             logger.warning("API configuration not found in config file. Using default values.")

        self.dataset_name = dataset_name
        self.base_api_url = self.config.get('base_url', "https://api.vipunen.fi/api/resources")
        self.limit = self.config.get('limit', 5000)
        self.max_retries = self.config.get('max_retries', 3)
        self.caller_id = self.config.get('caller_id', "YOUR_DEFAULT_CALLER_ID") # Provide a default or raise error
        self.retry_delay = self.config.get('retry_delay', 5)
        self.timeout = self.config.get('timeout', 60)
        self.csv_separator = self.config.get('csv_separator', ';')
        self.csv_encoding = self.config.get('csv_encoding', 'utf-8')
        self.backup_dir_name = self.config.get('backup_dir_name', 'old_api_calls_output')
        self.metadata_dir_name = self.config.get('metadata_dir_name', '.metadata')


        if self.caller_id == "YOUR_DEFAULT_CALLER_ID":
             logger.warning("Using default Caller-Id. Please configure 'api.caller_id' in your config.yaml.")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Caller-Id": self.caller_id
        }
        self.session = requests.Session()
        self.session.headers.update(headers)

        # Construct dataset-specific URLs
        self.data_url_base = f"{self.base_api_url}/{self.dataset_name}/data"
        self.count_url = f"{self.data_url_base}/count"

    def _get_metadata_path(self, output_dir: Path) -> Path:
        """Gets the path to the metadata file for the current dataset."""
        metadata_dir = output_dir / self.metadata_dir_name
        metadata_dir.mkdir(parents=True, exist_ok=True)
        return metadata_dir / f"{self.dataset_name}_metadata.json"

    def _get_last_update_date(self, output_dir: Path) -> Optional[str]:
        """Reads the last known update date from the metadata file."""
        metadata_path = self._get_metadata_path(output_dir)
        if not metadata_path.exists():
            return None
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            last_update = metadata.get(UPDATE_DATE_COLUMN)
            logger.info(f"Found previous update date for {self.dataset_name}: {last_update}")
            return last_update
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning(f"Could not read or parse metadata file {metadata_path}: {e}")
            return None

    def _save_last_update_date(self, output_dir: Path, update_date: str) -> None:
        """Saves the latest update date to the metadata file."""
        metadata_path = self._get_metadata_path(output_dir)
        metadata = {UPDATE_DATE_COLUMN: update_date}
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            logger.info(f"Saved current update date for {self.dataset_name}: {update_date}")
        except OSError as e:
            logger.error(f"Failed to save metadata file {metadata_path}: {e}")

    def _backup_existing_file(self, file_path: Path) -> None:
        """Backs up an existing file by moving it to a backup directory."""
        if not file_path.exists():
            return

        try:
            modified_date = datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d')
            backup_dir = file_path.parent / self.backup_dir_name # Use config value
            backup_dir.mkdir(exist_ok=True)
            backup_file_path = backup_dir / f"{file_path.stem}_{modified_date}{file_path.suffix}" # Preserve suffix
            shutil.move(str(file_path), str(backup_file_path))
            logger.info(f"Backed up existing file to {backup_file_path}")
        except OSError as e:
            logger.error(f"Failed to backup file {file_path}: {e}")
            # Decide if this should raise or just warn

    def _validate_response(self, response: Any) -> bool:
        """Validate the API response data. (Basic type check)"""
        logger.debug(f"Response type: {type(response)}")
        # Basic check - could be enhanced with schema validation if needed
        if isinstance(response, (int, list, dict)):
            return True
        logger.error(f"Invalid response type: {type(response)}")
        return False

    def _fetch_data(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Fetches data from the specified URL using GET request with retries."""
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Fetching data from {url} with params {params} (attempt {attempt + 1})")
                response = self.session.get( # Use session
                    url,
                    params=params, # Pass params separately
                    timeout=self.timeout # Use config value
                )
                response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
                data = response.json()

                if not self._validate_response(data):
                    # Consider specific exception type
                    raise ValueError(f"Invalid response data structure from {url}")

                return data

            except requests.exceptions.Timeout as e:
                logger.warning(f"Attempt {attempt + 1} timed out: {e}")
            except requests.exceptions.HTTPError as e:
                logger.warning(f"Attempt {attempt + 1} failed with HTTP status {e.response.status_code}: {e}")
                # Stop retrying on client errors (4xx) unless it's 429 (Too Many Requests)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                     logger.error(f"Client error {e.response.status_code}, aborting retries.")
                     raise
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed with network error: {e}")

            if attempt < self.max_retries - 1:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay) # Use config value
            else:
                logger.error(f"Max retries ({self.max_retries}) reached for {url}")
                raise # Re-raise the last exception

    def _save_data(self, data: List[Dict[str, Any]], file_path: Path, mode: str = 'a') -> None:
        """Save data (list of dicts) to a CSV file."""
        if not isinstance(data, list) or not data:
             logger.warning(f"No data provided or invalid format ({type(data)}) for saving to {file_path}, skipping.")
             return

        try:
            header = (mode == 'w') # Write header only if mode is 'w'
            df = pd.DataFrame(data)

            # Validate DataFrame - already checked if list is empty
            if df.empty:
                logger.warning(f"Empty DataFrame created from data, skipping save to {file_path}")
                return

            df.to_csv(
                file_path,
                mode=mode,
                sep=self.csv_separator, # Use config value
                na_rep='',
                header=header,
                index=False,
                encoding=self.csv_encoding, # Use config value
                quoting=0, # Assuming this was intentional, could be config param
                quotechar='"', # Assuming this was intentional, could be config param
                lineterminator="\n", # Assuming this was intentional, could be config param
                escapechar="$" # Assuming this was intentional, could be config param
            )
            logger.debug(f"Saved {len(df)} rows to {file_path} (mode: {mode})")

        except OSError as e:
            logger.error(f"Failed to save data to {file_path}: {e}")
            raise
        except Exception as e: # Catch other potential pandas errors
             logger.error(f"An unexpected error occurred during data saving to {file_path}: {e}")
             raise

    def fetch_and_save_data(self, output_dir: Union[str, Path], check_update_date: bool = True) -> None:
        """Fetches data from the API, checks update date, and saves it to a CSV file.

        Args:
            output_dir: The directory where the CSV file and metadata will be saved.
            check_update_date: If True, checks the API data's update date against
                               the last known date and skips download if unchanged.

        Raises:
            OSError: If file operations fail.
            requests.RequestException: If API requests fail after retries.
            ValueError: If response data or row count is invalid.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / f"{self.dataset_name}.csv"

        # --- Update Date Check ---
        previous_update_date = self._get_last_update_date(output_dir) if check_update_date else None
        current_update_date = None

        # Fetch first chunk to check date (limit=1 is enough)
        try:
             first_chunk_params = {'limit': 1, 'offset': 0}
             first_chunk = self._fetch_data(self.data_url_base, params=first_chunk_params)
             if isinstance(first_chunk, list) and first_chunk:
                 first_record = first_chunk[0]
                 if UPDATE_DATE_COLUMN in first_record:
                     current_update_date = first_record[UPDATE_DATE_COLUMN]
                     logger.info(f"Current API data update date for {self.dataset_name}: {current_update_date}")
                 else:
                     logger.warning(f"Update date column '{UPDATE_DATE_COLUMN}' not found in first record.")
                     check_update_date = False # Cannot perform check if column is missing
             else:
                 logger.warning(f"Could not retrieve first record to check update date.")
                 check_update_date = False # Cannot perform check if first record is unavailable

        except (requests.RequestException, ValueError) as e:
             logger.error(f"Failed to fetch initial data chunk for update date check: {e}")
             # Decide if we should proceed without the check or raise error
             check_update_date = False # Disable check and proceed cautiously
             # raise # Uncomment to make date check failure critical

        if check_update_date and current_update_date and previous_update_date == current_update_date:
             logger.info(f"Data update date ({current_update_date}) has not changed since last download. Skipping download.")
             return # Exit function, no download needed

        # --- Proceed with Full Download ---
        logger.info(f"Proceeding with full data download for {self.dataset_name}...")
        self._backup_existing_file(file_path)

        # Get total number of rows
        try:
            max_rows = self._fetch_data(self.count_url)
            if not isinstance(max_rows, int) or max_rows < 0: # Allow 0 rows
                raise ValueError(f"Invalid row count received: {max_rows}")
        except (requests.RequestException, ValueError) as e:
             logger.error(f"Failed to get total row count: {e}")
             raise # Row count is critical

        logger.info(f"Total rows to fetch: {max_rows}")
        if max_rows == 0:
             logger.info("No rows to fetch. Creating empty file.")
             # Create empty file with header? Or just do nothing?
             # Let's create an empty file for consistency.
             try:
                  # Fetch one record to get headers if available, otherwise create empty file
                  header_check_params = {'limit': 1, 'offset': 0}
                  header_data = self._fetch_data(self.data_url_base, params=header_check_params)
                  if isinstance(header_data, list) and header_data:
                       self._save_data([], file_path, mode='w') # Write header using empty list
                       pd.DataFrame(columns=header_data[0].keys()).to_csv(file_path, sep=self.csv_separator, index=False, encoding=self.csv_encoding)
                  else:
                       file_path.touch() # Create empty file
                  logger.info(f"Empty data file created at {file_path}")
             except Exception as e:
                  logger.error(f"Failed to create empty file or write header: {e}")
             # Optionally save update date even if empty?
             if current_update_date:
                 self._save_last_update_date(output_dir, current_update_date)
             return # Exit after handling zero rows

        # Fetch data in chunks with progress bar
        mode = 'w' # Start with write mode
        download_successful = False
        try:
             with tqdm(total=max_rows, desc=f"Fetching {self.dataset_name}", unit="rows") as pbar:
                 for offset in range(0, max_rows, self.limit):
                     fetch_limit = min(self.limit, max_rows - offset)
                     params = {'limit': fetch_limit, 'offset': offset}
                     data = self._fetch_data(self.data_url_base, params=params)

                     # Additional validation on fetched data chunk
                     if not isinstance(data, list):
                          logger.error(f"Received non-list data chunk for offset {offset}. Type: {type(data)}")
                          # Decide how to handle: skip chunk, abort, etc.
                          # For now, let's try to continue but log error
                          continue # Skip this chunk

                     self._save_data(data, file_path, mode=mode)
                     mode = 'a'  # Switch to append mode after first successful write
                     pbar.update(len(data) if isinstance(data, list) else 0) # Update progress based on actual data length

             download_successful = True # Mark as successful if loop completes

        except (requests.RequestException, ValueError, OSError) as e:
            logger.error(f"Data download failed during chunk processing: {e}")
            # Keep the potentially partial file or delete it? Current logic keeps it.
            raise # Re-raise the exception

        finally:
            if download_successful and current_update_date:
                 self._save_last_update_date(output_dir, current_update_date)
            elif download_successful:
                 logger.warning("Download completed, but could not determine update date to save.")

        if download_successful:
             logger.info(f"Data fetching complete. Data saved to {file_path}")
        else:
             logger.error(f"Data fetching failed. Check logs for details. Partial data might be present in {file_path}") 