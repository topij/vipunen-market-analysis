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

def get_config_path() -> Path:
    """
    Get the path to the configuration file.
    
    Checks multiple possible locations for config.yaml file.
    
    Returns:
        Path: Path to the configuration file
    """
    # Look in several possible locations
    possible_locations = [
        Path("config.yaml"),
        Path("config/config.yaml"),
        Path(__file__).parent / "config.yaml",
        Path.home() / ".vipunen" / "config.yaml",
    ]
    
    for path in possible_locations:
        if path.exists():
            return path
    
    # Default to the one in the package
    return Path(__file__).parent / "config.yaml"

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
        config_path = get_config_path()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        # Fall back to default configuration
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
        _config = load_config()
    return _config

def reload_config() -> Dict[str, Any]:
    """
    Force reload of the configuration.
    
    Returns:
        Dict[str, Any]: Freshly loaded configuration dictionary
    """
    global _config
    _config = load_config()
    return _config 