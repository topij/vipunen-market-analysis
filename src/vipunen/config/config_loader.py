"""
Configuration loader for the Vipunen project.

This module provides functions to load configuration from a YAML file.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def find_project_root(marker='.git') -> Optional[Path]:
    """Find the project root directory by looking upwards for a marker file/directory."""
    current_path = Path(__file__).resolve()
    while current_path != current_path.parent: # Stop at the filesystem root
        if (current_path / marker).exists():
            return current_path
        current_path = current_path.parent
    # If marker not found, maybe return None or raise error
    logger.error(f"Project root marker '{marker}' not found starting from {Path(__file__).resolve()}.")
    return None # Or raise an exception

def get_config_path() -> Path:
    """
    Get the path to the configuration file (config.yaml) expected at the project root.

    Returns:
        Path: Path to the configuration file.
    Raises:
        FileNotFoundError: If the project root or config.yaml cannot be found.
    """
    project_root = find_project_root()
    if project_root:
        config_path = project_root / "config.yaml"
        if config_path.exists():
            logger.info(f"Found config at project root: {config_path}")
            return config_path
        else:
            msg = f"config.yaml not found at project root: {project_root}"
            logger.error(msg)
            raise FileNotFoundError(msg)
    else:
        # Handle case where project root wasn't found
        msg = "Could not determine project root to find config.yaml."
        logger.error(msg)
        # As a fallback, maybe try the original relative path? Or just error out.
        # For simplicity, let's raise an error here.
        raise FileNotFoundError(msg)

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Optional path to the configuration file.
                    If not provided, will use get_config_path()
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if config_path is None:
        try:
            config_path = get_config_path() # Try to find it using the new logic
        except FileNotFoundError as e:
             # If get_config_path fails (e.g., no root found), fall back to default
            logger.warning(f"Could not find config.yaml via project root: {e}")
            logger.warning("Using default configuration")
            return _get_default_config()

    # Proceed with loading if path was found
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        # Fall back to default configuration if loading fails for other reasons
        logger.warning(f"Failed to load config from {config_path}: {e}")
        logger.warning("Using default configuration")
        return _get_default_config()

def _get_default_config() -> Dict[str, Any]:
    """
    Get default configuration as a fallback.
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        "columns": {
            "input": {
                "year": "tilastovuosi",
                "degree_type": "tutkintotyyppi",
                "qualification": "tutkinto",
                "provider": "koulutuksenJarjestaja",
                "subcontractor": "hankintakoulutuksenJarjestaja",
                "volume": "nettoopiskelijamaaraLkm"
            },
            "output": {
                "year": "Year",
                "qualification": "Qualification",
                "provider": "Provider",
                "provider_amount": "Provider Amount",
                "subcontractor_amount": "Subcontractor Amount",
                "total_volume": "Total Volume",
                "market_total": "Market Total",
                "market_share": "Market Share (%)"
            }
        },
        "institutions": {
            "default": {
                "name": "Rastor-instituutti",
                "short_name": "RI",
                "variants": [
                    "Rastor-instituutti ry",
                    "Rastor-instituutti",
                    "RASTOR OY",
                    "Rastor Oy"
                ]
            }
        },
        "qualification_types": [
            "Ammattitutkinnot",
            "Erikoisammattitutkinnot"
        ],
        "paths": {
            "data": "data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv",
            "output": "data/reports"
        }
    }

# Singleton-like access to configuration
_config = None

def get_config() -> Dict[str, Any]:
    """
    Get the configuration dictionary, loading it if needed.
    
    This function implements a singleton pattern to avoid reloading
    the configuration file multiple times.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    global _config
    if _config is None:
        _config = load_config() # load_config now handles finding the path or defaulting
    return _config

def reload_config() -> Dict[str, Any]:
    """
    Force reload of the configuration.
    
    Returns:
        Dict[str, Any]: Freshly loaded configuration dictionary
    """
    global _config
    _config = load_config() # load_config now handles finding the path or defaulting
    return _config 