import pytest
from unittest.mock import patch, MagicMock, mock_open, call, ANY
from pathlib import Path
import json
import pandas as pd
import requests
import os

# Assume the client is located here relative to the project root
from src.vipunen.api.client import VipunenAPIClient, UPDATE_DATE_COLUMN

MOCK_DATASET_NAME = "test_dataset"
MOCK_OUTPUT_DIR_STR = "/fake/output/dir" # Use string for paths passed to methods
MOCK_API_CONFIG = {
    'base_url': "https://fake.api.vipunen.fi/api/resources",
    'limit': 100,
    'max_retries': 2,
    'caller_id': "test-caller-id",
    'retry_delay': 1,
    'timeout': 10,
    'csv_separator': ';',
    'csv_encoding': 'utf-8',
    'backup_dir_name': 'backup',
    'metadata_dir_name': '.meta'
}

@pytest.fixture
def mock_config():
    # Provides a mock of the loaded API configuration
    with patch('src.vipunen.api.client.get_config') as mock_get:
        mock_get.return_value = {'api': MOCK_API_CONFIG}
        yield mock_get

@pytest.fixture
def mock_requests_session():
    # Provides a mock requests.Session
    with patch('src.vipunen.api.client.requests.Session') as mock_session_cls:
        mock_session_instance = MagicMock()
        # Mock get method - can be customized per test
        mock_session_instance.get.return_value = MagicMock(spec=requests.Response)
        mock_session_instance.get.return_value.raise_for_status.return_value = None
        mock_session_instance.get.return_value.json.return_value = {} # Default empty json
        mock_session_cls.return_value = mock_session_instance
        yield mock_session_instance

@pytest.fixture
def client_for_internal_methods(mock_config, mock_requests_session):
    """Provides a client instance suitable for testing internal methods."""
    client = VipunenAPIClient(dataset_name=MOCK_DATASET_NAME, config=MOCK_API_CONFIG)
    client.metadata_dir_name = '.meta'
    client.dataset_name = MOCK_DATASET_NAME
    return client

# Use tmp_path fixture for this test to allow real directory creation
def test_get_metadata_path_internal(tmp_path, client_for_internal_methods):
    """Test the _get_metadata_path internal logic using tmp_path."""
    output_dir = tmp_path
    client = client_for_internal_methods
    client.metadata_dir_name = ".test_meta"
    client.dataset_name = "my_dataset"

    expected_meta_dir = output_dir / client.metadata_dir_name
    expected_meta_file = expected_meta_dir / f"{client.dataset_name}_metadata.json"

    assert not expected_meta_dir.exists()
    assert not expected_meta_file.exists()

    returned_path = client._get_metadata_path(output_dir) # Pass real Path

    assert expected_meta_dir.exists()
    assert expected_meta_dir.is_dir()
    assert returned_path == expected_meta_file
    assert not expected_meta_file.exists()

@patch('builtins.open', new_callable=mock_open, read_data=json.dumps({UPDATE_DATE_COLUMN: "2024-01-01T12:00:00Z"}))
@patch.object(VipunenAPIClient, '_get_metadata_path') # Mock the helper method
def test_get_last_update_date_exists(mock_get_meta_path, mock_open_func, client_for_internal_methods):
    """Test reading update date when metadata file exists and is valid."""
    mock_meta_file_instance = MagicMock(spec=Path)
    mock_meta_file_instance.exists.return_value = True
    mock_get_meta_path.return_value = mock_meta_file_instance

    # Pass a REAL Path object as expected by the function signature
    date = client_for_internal_methods._get_last_update_date(Path(MOCK_OUTPUT_DIR_STR))

    mock_get_meta_path.assert_called_once()
    # Check the argument passed was the correct Path object
    assert isinstance(mock_get_meta_path.call_args[0][0], Path)
    assert str(mock_get_meta_path.call_args[0][0]) == MOCK_OUTPUT_DIR_STR

    mock_meta_file_instance.exists.assert_called_once()
    mock_open_func.assert_called_once_with(mock_meta_file_instance, 'r')
    assert date == "2024-01-01T12:00:00Z"

@patch.object(VipunenAPIClient, '_get_metadata_path')
def test_get_last_update_date_not_exists(mock_get_meta_path, client_for_internal_methods):
    """Test reading update date when metadata file does not exist."""
    mock_meta_file_instance = MagicMock(spec=Path)
    mock_meta_file_instance.exists.return_value = False
    mock_get_meta_path.return_value = mock_meta_file_instance

    # Pass a REAL Path object
    date = client_for_internal_methods._get_last_update_date(Path(MOCK_OUTPUT_DIR_STR))

    mock_get_meta_path.assert_called_once()
    assert isinstance(mock_get_meta_path.call_args[0][0], Path)
    assert str(mock_get_meta_path.call_args[0][0]) == MOCK_OUTPUT_DIR_STR
    mock_meta_file_instance.exists.assert_called_once()
    assert date is None

@patch('builtins.open', new_callable=mock_open, read_data="invalid json")
@patch.object(VipunenAPIClient, '_get_metadata_path')
def test_get_last_update_date_invalid_json(mock_get_meta_path, mock_open_func, client_for_internal_methods):
    """Test reading update date when metadata file has invalid JSON."""
    mock_meta_file_instance = MagicMock(spec=Path)
    mock_meta_file_instance.exists.return_value = True
    mock_get_meta_path.return_value = mock_meta_file_instance

    # Pass a REAL Path object
    date = client_for_internal_methods._get_last_update_date(Path(MOCK_OUTPUT_DIR_STR))

    mock_get_meta_path.assert_called_once()
    assert isinstance(mock_get_meta_path.call_args[0][0], Path)
    assert str(mock_get_meta_path.call_args[0][0]) == MOCK_OUTPUT_DIR_STR

    assert date is None # Should return None on error

@patch('builtins.open', new_callable=mock_open)
@patch('src.vipunen.api.client.json.dump')
@patch.object(VipunenAPIClient, '_get_metadata_path')
def test_save_last_update_date(mock_get_meta_path, mock_json_dump, mock_open_func, client_for_internal_methods):
    """Test saving the update date to the metadata file."""
    mock_meta_file_instance = MagicMock(spec=Path)
    mock_get_meta_path.return_value = mock_meta_file_instance
    test_date = "2024-02-01T10:00:00Z"

    # Pass a REAL Path object
    client_for_internal_methods._save_last_update_date(Path(MOCK_OUTPUT_DIR_STR), test_date)

    mock_get_meta_path.assert_called_once()
    assert isinstance(mock_get_meta_path.call_args[0][0], Path)
    assert str(mock_get_meta_path.call_args[0][0]) == MOCK_OUTPUT_DIR_STR

    mock_open_func.assert_called_once_with(mock_meta_file_instance, 'w')
    mock_json_dump.assert_called_once_with({UPDATE_DATE_COLUMN: test_date}, mock_open_func())


# --- fetch_and_save_data High-Level Tests ---

@pytest.fixture
def client_for_fetch(mock_config, mock_requests_session):
    """ Fixture providing a client instance suitable for fetch_and_save tests """
    client = VipunenAPIClient(dataset_name=MOCK_DATASET_NAME, config=MOCK_API_CONFIG)
    return client

# Patch Path.mkdir specifically for this test to prevent OSError
@patch('pathlib.Path.mkdir')
@patch.object(VipunenAPIClient, '_get_last_update_date', return_value="2024-03-01T00:00:00Z")
@patch.object(VipunenAPIClient, '_fetch_data')
@patch.object(VipunenAPIClient, '_backup_existing_file')
@patch.object(VipunenAPIClient, '_save_data')
@patch.object(VipunenAPIClient, '_save_last_update_date')
def test_fetch_and_save_skips_on_same_date(
    mock_save_date, mock_save_data, mock_backup, mock_fetch, mock_get_date, mock_mkdir, # Add mock_mkdir
    client_for_fetch
):
    """Test that download is skipped if update date hasn't changed."""
    test_date = "2024-03-01T00:00:00Z"
    mock_fetch.return_value = [{UPDATE_DATE_COLUMN: test_date}]

    # Pass string path
    client_for_fetch.fetch_and_save_data(MOCK_OUTPUT_DIR_STR, check_update_date=True)

    # Check mkdir was called (on the real Path object created internally)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    # Check _get_last_update_date was called with a Path object
    mock_get_date.assert_called_once()
    assert isinstance(mock_get_date.call_args[0][0], Path)
    assert str(mock_get_date.call_args[0][0]) == MOCK_OUTPUT_DIR_STR

    mock_fetch.assert_called_once_with(client_for_fetch.data_url_base, params={'limit': 1, 'offset': 0})
    mock_backup.assert_not_called()
    mock_save_data.assert_not_called()
    mock_save_date.assert_not_called()

# Patch Path.mkdir specifically for this test
@patch('pathlib.Path.mkdir')
@patch.object(VipunenAPIClient, '_get_last_update_date', return_value="2024-03-01T00:00:00Z")
@patch.object(VipunenAPIClient, '_fetch_data')
@patch.object(VipunenAPIClient, '_backup_existing_file')
@patch.object(VipunenAPIClient, '_save_data')
@patch.object(VipunenAPIClient, '_save_last_update_date')
@patch('src.vipunen.api.client.tqdm') # Mock tqdm progress bar
def test_fetch_and_save_proceeds_on_new_date(
    mock_tqdm, mock_save_date, mock_save_data, mock_backup, mock_fetch, mock_get_date, mock_mkdir, # Add mock_mkdir
    client_for_fetch
):
    """Test that download proceeds if update date is different."""
    previous_date = "2024-03-01T00:00:00Z"
    new_date = "2024-03-02T00:00:00Z"
    total_rows = 150
    limit = client_for_fetch.limit

    mock_fetch.side_effect = [
        [{UPDATE_DATE_COLUMN: new_date}],
        total_rows,
        [{'col1': 'a', UPDATE_DATE_COLUMN: new_date}] * limit,
        [{'col1': 'b', UPDATE_DATE_COLUMN: new_date}] * (total_rows - limit)
    ]

    # Pass string path
    client_for_fetch.fetch_and_save_data(MOCK_OUTPUT_DIR_STR, check_update_date=True)

    # Check mkdir was called
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    # Assert _get_last_update_date called with Path object
    mock_get_date.assert_called_once()
    assert isinstance(mock_get_date.call_args[0][0], Path)
    assert str(mock_get_date.call_args[0][0]) == MOCK_OUTPUT_DIR_STR

    # Check fetch calls
    expected_fetch_calls = [
        call(client_for_fetch.data_url_base, params={'limit': 1, 'offset': 0}),
        call(client_for_fetch.count_url),
        call(client_for_fetch.data_url_base, params={'limit': limit, 'offset': 0}),
        call(client_for_fetch.data_url_base, params={'limit': total_rows - limit, 'offset': limit})
    ]
    mock_fetch.assert_has_calls(expected_fetch_calls)
    assert mock_fetch.call_count == 4

    # Check other actions were performed, verifying path arguments are Path objects
    mock_backup.assert_called_once()
    assert isinstance(mock_backup.call_args[0][0], Path)
    assert str(mock_backup.call_args[0][0]) == str(Path(MOCK_OUTPUT_DIR_STR) / f"{MOCK_DATASET_NAME}.csv")

    assert mock_save_data.call_count == 2
    # Check first save call args
    assert isinstance(mock_save_data.call_args_list[0][0][1], Path)
    assert str(mock_save_data.call_args_list[0][0][1]) == str(Path(MOCK_OUTPUT_DIR_STR) / f"{MOCK_DATASET_NAME}.csv")
    # Check second save call args
    assert isinstance(mock_save_data.call_args_list[1][0][1], Path)
    assert str(mock_save_data.call_args_list[1][0][1]) == str(Path(MOCK_OUTPUT_DIR_STR) / f"{MOCK_DATASET_NAME}.csv")

    mock_save_date.assert_called_once()
    assert isinstance(mock_save_date.call_args[0][0], Path)
    assert str(mock_save_date.call_args[0][0]) == MOCK_OUTPUT_DIR_STR
    assert mock_save_date.call_args[0][1] == new_date

    mock_tqdm.assert_called_once()

# TODO: Add more tests for:
# - _fetch_data retry logic, error handling (timeouts, HTTP errors)
# - _save_data CSV writing details (if needed beyond mocking pd.to_csv)
# - _backup_existing_file logic (file exists/doesn't exist)
# - fetch_and_save_data:
#   - check_update_date=False behavior
#   - Handling missing UPDATE_DATE_COLUMN in API response
#   - Handling zero rows from count
#   - Handling errors during count fetch or chunk fetch
#   - Handling empty list returned from fetch chunks
#   - Saving date correctly in finally block
#   - Error handling and exceptions 