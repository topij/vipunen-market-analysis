import pytest
from unittest.mock import patch, MagicMock, ANY
import sys
from pathlib import Path

# Assume the script is runnable via this import path
import src.scripts.fetch_data as fetch_data_script

# Mock config values that the script might use
MOCK_CONFIG_DATA = {
    'api': {
        'default_dataset': 'config_default_dataset',
        # Add other api config keys if fetch_data.py uses them directly
    },
    'paths': {
        'data': 'config/path/data/config_data.csv'
    }
}

@pytest.fixture
def mock_argparse():
    """Mocks argparse.ArgumentParser and parse_args."""
    with patch('argparse.ArgumentParser') as mock_parser_cls:
        mock_parser_instance = MagicMock()
        mock_parser_cls.return_value = mock_parser_instance
        # Default args - can be overridden in tests
        mock_parser_instance.parse_args.return_value = MagicMock(
            dataset=None,
            output_dir=None,
            force_download=False
        )
        yield mock_parser_instance

@pytest.fixture
def mock_get_config():
    """Mocks the get_config function used by the script."""
    with patch('src.scripts.fetch_data.get_config') as mock_get:
        mock_get.return_value = MOCK_CONFIG_DATA
        yield mock_get

@pytest.fixture
def mock_api_client():
    """Mocks the VipunenAPIClient class."""
    with patch('src.scripts.fetch_data.VipunenAPIClient') as mock_client_cls:
        mock_client_instance = MagicMock()
        # Mock the method called by the script
        mock_client_instance.fetch_and_save_data.return_value = None
        mock_client_cls.return_value = mock_client_instance
        yield mock_client_cls, mock_client_instance

@pytest.fixture
def mock_path():
    """Mocks pathlib.Path used by the script."""
    # This mock needs to handle Path(string) and return an instance
    # where methods like mkdir and parent are also mocked.
    with patch('src.scripts.fetch_data.Path') as mock_path_cls:
        mock_instances = {}
        def path_side_effect(*args):
            path_key = tuple(str(arg) for arg in args) # Use string representation as key
            if path_key not in mock_instances:
                instance = MagicMock(spec=Path)
                # Mock attributes and methods used by the script
                instance.parent = MagicMock(spec=Path)
                instance.mkdir = MagicMock()
                instance.__str__ = MagicMock(return_value=str(Path(*args))) # For logging
                # Simulate parent relationship slightly more realistically
                if len(args) == 1 and isinstance(args[0], str) and Path(args[0]).parent != Path(args[0]):
                     parent_path_str = str(Path(args[0]).parent)
                     instance.parent = path_side_effect(parent_path_str)
                else:
                     instance.parent = instance # Simple fallback
                mock_instances[path_key] = instance
            return mock_instances[path_key]

        mock_path_cls.side_effect = path_side_effect
        yield mock_path_cls, mock_instances

@pytest.fixture(autouse=True)
def mock_sys_exit():
    """Mocks sys.exit to prevent tests from exiting."""
    with patch('sys.exit') as mock_exit:
        yield mock_exit

# --- Test Cases ---

def test_main_defaults(mock_argparse, mock_get_config, mock_api_client, mock_path):
    """Test script runs with default arguments, using config values."""
    mock_path_cls, mock_path_instances = mock_path
    mock_client_cls, mock_client_instance = mock_api_client

    result = fetch_data_script.main()

    # Assert config was loaded
    mock_get_config.assert_called_once()

    # Assert output dir derived from config paths.data
    expected_output_dir_path = 'config/path/data' # Parent of paths.data
    # Check Path was called with the config path string
    mock_path_cls.assert_any_call(MOCK_CONFIG_DATA['paths']['data'])
    # Define the expected key for the mock instance dictionary
    expected_dir_key = ('config/path/data',) # Key based on parent path string
    # Check mkdir was called on the derived path instance using the correct key
    assert expected_dir_key in mock_path_instances, f"Key {expected_dir_key} not found in mock_path_instances"
    mock_path_instances[expected_dir_key].mkdir.assert_called_once_with(parents=True, exist_ok=True)

    # Assert API client initialized with default dataset and derived output dir
    expected_dataset = MOCK_CONFIG_DATA['api']['default_dataset']
    mock_client_cls.assert_called_once_with(
        dataset_name=expected_dataset,
        config=MOCK_CONFIG_DATA['api'] # Passes the whole API config dict
    )

    # Assert fetch_and_save_data called correctly
    mock_client_instance.fetch_and_save_data.assert_called_once_with(
        mock_path_instances[expected_dir_key], # The Path instance representing the output dir
        check_update_date=True # Default is True
    )
    assert result == 0 # Success exit code

def test_main_args_override_config(mock_argparse, mock_get_config, mock_api_client, mock_path):
    """Test script uses command line args over config values."""
    mock_path_cls, mock_path_instances = mock_path
    mock_client_cls, mock_client_instance = mock_api_client

    # Simulate command line arguments
    args_dataset = "cmdline_dataset"
    args_output_dir_str = "/cmdline/output/dir" # <<< Use string path for argument
    # The script's argparse will call Path(args_output_dir_str)
    # Get the expected mock Path instance that Path(args_output_dir_str) should return
    expected_mock_path_instance = mock_path_cls(args_output_dir_str)

    mock_argparse.parse_args.return_value = MagicMock(
        dataset=args_dataset,
        output_dir=expected_mock_path_instance, # <<< Pass the MOCK Path object
        force_download=True
    )

    result = fetch_data_script.main()

    mock_get_config.assert_called_once()

    # Get the mock instance that should have been created for the output dir
    # (Already done above using expected_mock_path_instance)
    # Check mkdir called on the specific mock instance
    expected_mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    # Assert API client initialized with command line args
    mock_client_cls.assert_called_once_with(
        dataset_name=args_dataset,
        config=MOCK_CONFIG_DATA['api']
    )

    # Assert fetch_and_save_data called with command line args
    # Ensure the mock Path instance created from the arg string is passed
    mock_client_instance.fetch_and_save_data.assert_called_once_with(
        expected_mock_path_instance, # <<< Pass the mock instance
        check_update_date=False # Due to force_download=True
    )
    assert result == 0

def test_main_missing_dataset(mock_argparse, mock_get_config, mock_api_client, mock_path, mock_sys_exit):
    """Test script exits with error if dataset is not specified anywhere."""
    # Simulate no dataset in args and no default_dataset in config
    mock_argparse.parse_args.return_value = MagicMock(
        dataset=None, output_dir=None, force_download=False
    )
    mock_get_config.return_value = {'api': {}, 'paths': {}} # No default_dataset

    result = fetch_data_script.main()

    mock_api_client[0].assert_not_called() # Client should not be initialized
    assert result == 1 # Error exit code

def test_main_default_output_dir(mock_argparse, mock_get_config, mock_api_client, mock_path):
    """Test script defaults to 'data/raw' if output_dir and paths.data are missing."""
    mock_path_cls, mock_path_instances = mock_path
    mock_client_cls, mock_client_instance = mock_api_client

    # Simulate no output dir in args and no paths.data in config
    mock_argparse.parse_args.return_value = MagicMock(
        dataset=None, output_dir=None, force_download=False
    )
    mock_get_config.return_value = {'api': MOCK_CONFIG_DATA['api'], 'paths': {}}

    result = fetch_data_script.main()

    # Check Path was called with the default string
    mock_path_cls.assert_any_call("data/raw")
    # Get the specific mock instance for the default path
    mock_default_output_instance = mock_path_cls("data/raw")
    # Check mkdir called on the default path mock instance
    mock_default_output_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    # Check fetch called with the default path mock instance
    mock_client_instance.fetch_and_save_data.assert_called_once_with(
        mock_default_output_instance, # Should be the instance representing data/raw
        check_update_date=True
    )
    assert result == 0

def test_main_client_init_fails(mock_argparse, mock_get_config, mock_api_client, mock_path, mock_sys_exit):
    """Test script exits if API client initialization fails."""
    mock_client_cls, mock_client_instance = mock_api_client
    mock_client_cls.side_effect = Exception("Client init error") # Simulate failure

    result = fetch_data_script.main()

    mock_client_cls.assert_called_once() # Attempted to init
    mock_client_instance.fetch_and_save_data.assert_not_called() # Fetch not called
    assert result == 1 # Error exit code

def test_main_fetch_fails(mock_argparse, mock_get_config, mock_api_client, mock_path, mock_sys_exit):
    """Test script exits if client.fetch_and_save_data fails."""
    mock_client_cls, mock_client_instance = mock_api_client
    mock_client_instance.fetch_and_save_data.side_effect = Exception("Fetch error")

    result = fetch_data_script.main()

    mock_client_instance.fetch_and_save_data.assert_called_once() # Attempted to fetch
    assert result == 1 # Error exit code 