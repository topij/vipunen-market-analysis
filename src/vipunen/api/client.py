"""API client for fetching data from the Vipunen API."""
import requests
from pathlib import Path
import shutil
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import logging
from dataclasses import dataclass
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for the Vipunen API client."""
    dataset: str = 'amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto'
    limit: int = 5000
    max_retries: int = 3
    caller_id: str = "0201689-0.rastor-instituutti"
    base_url: str = "https://api.vipunen.fi/api/resources"  # Updated to correct API endpoint
    retry_delay: int = 5  # seconds to wait between retries
    timeout: int = 30  # seconds to wait for response

class VipunenAPIClient:
    """Client for interacting with the Vipunen API."""
    
    def __init__(self, config: Optional[APIConfig] = None):
        """Initialize the Vipunen API client.
        
        Args:
            config: Optional configuration for the API client. If not provided,
                   default values will be used.
        """
        self.config = config or APIConfig()
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Caller-Id": self.config.caller_id
        }
        self.base_url = f"{self.config.base_url}/{self.config.dataset}/data"
        self.count_url = f"{self.base_url}/count"
        
    def _backup_existing_file(self, file_path: Path) -> None:
        """Backs up an existing file by moving it to a backup directory.
        
        Args:
            file_path: The path to the file to be backed up
            
        Raises:
            OSError: If the backup operation fails
        """
        if not file_path.exists():
            return
            
        try:
            modified_date = datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d')
            backup_dir = file_path.parent / "old_api_calls_output"
            backup_dir.mkdir(exist_ok=True)
            backup_file_path = backup_dir / f"{file_path.stem}_{modified_date}.csv"
            shutil.move(str(file_path), str(backup_file_path))
            logger.info(f"Backed up existing file to {backup_file_path}")
        except OSError as e:
            logger.error(f"Failed to backup file {file_path}: {e}")
            raise
            
    def _validate_response(self, response: Any) -> bool:
        """Validate the API response data.
        
        Args:
            response: The API response to validate
            
        Returns:
            bool: True if the response is valid, False otherwise
        """
        # Log the response type and content for debugging
        logger.debug(f"Response type: {type(response)}")
        logger.debug(f"Response content: {response}")
        
        # Handle different response formats
        if isinstance(response, int):  # For count endpoint
            return True
        elif isinstance(response, list):  # For data endpoint
            return True
        elif isinstance(response, dict):  # For other endpoints
            return True
            
        logger.error(f"Invalid response type: {type(response)}")
        return False
            
    def _fetch_data(self, url: str) -> Any:
        """Fetches data from the specified URL using GET request.
        
        Args:
            url: The URL to fetch data from
            
        Returns:
            The JSON response from the URL
            
        Raises:
            requests.RequestException: If an error occurs during the request
            ValueError: If the response data is invalid
        """
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Fetching data from {url} (attempt {attempt + 1})")
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                if not self._validate_response(data):
                    raise ValueError(f"Invalid response data: {data}")
                    
                return data
                
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    logger.info(f"Retrying in {self.config.retry_delay} seconds...")
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error("Max retries reached")
                    raise
                    
    def _save_data(self, data: Union[List[Dict[str, Any]], Dict[str, Any]], file_path: Path, mode: str = 'a') -> None:
        """Save data to a CSV file.
        
        Args:
            data: The data to be saved
            file_path: The file path where the data will be saved
            mode: The file mode ('a' for append, 'w' for write)
            
        Raises:
            OSError: If the file operation fails
        """
        try:
            header = (mode == 'w')
            df = pd.DataFrame(data)
            
            # Validate DataFrame
            if df.empty:
                logger.warning("Empty DataFrame received, skipping save")
                return
                
            df.to_csv(
                file_path,
                mode=mode,
                sep=';',
                na_rep='',
                header=header,
                index=False,
                encoding='utf-8',
                quoting=0,
                quotechar='"',
                lineterminator="\n",
                escapechar="$"
            )
            logger.debug(f"Saved data to {file_path} (mode: {mode})")
            
        except OSError as e:
            logger.error(f"Failed to save data to {file_path}: {e}")
            raise
        
    def fetch_and_save_data(self, output_dir: Union[str, Path]) -> None:
        """Fetches data from the API and saves it to a CSV file.
        
        Args:
            output_dir: The directory where the CSV file will be saved
            
        Raises:
            OSError: If file operations fail
            requests.RequestException: If API requests fail
            ValueError: If response data is invalid
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / f"{self.config.dataset}.csv"
        self._backup_existing_file(file_path)
        
        # Get total number of rows
        max_rows = self._fetch_data(self.count_url)
        if not isinstance(max_rows, int) or max_rows <= 0:
            raise ValueError(f"Invalid row count received: {max_rows}")
            
        logger.info(f"Total rows to fetch: {max_rows}")
        
        # Fetch data in chunks with progress bar
        mode = 'w'
        with tqdm(total=max_rows, desc="Fetching data") as pbar:
            for offset in range(0, max_rows, self.config.limit):
                url = f"{self.base_url}?limit={self.config.limit}&offset={offset}"
                data = self._fetch_data(url)
                self._save_data(data, file_path, mode=mode)
                mode = 'a'  # Switch to append mode after first write
                pbar.update(min(self.config.limit, max_rows - offset))
            
        logger.info(f"Data fetching complete. Data saved to {file_path}") 