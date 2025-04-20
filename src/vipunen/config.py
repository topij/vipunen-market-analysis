"""Configuration settings for the Vipunen project."""
from pathlib import Path
from typing import Dict, List

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PLOTS_DIR = PROJECT_ROOT / 'plots'

# API settings
API_CONFIG = {
    'base_url': 'https://vipunen.fi/api',
    'dataset': 'amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto',
    'limit': 1000,
    'max_retries': 3,
    'retry_delay': 5,
    'timeout': 30
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'style': 'whitegrid',
    'figure_size': (12, 8),
    'dpi': 300,
    'font_family': 'Arial',
    'brand_colors': {
        'primary': '#007AC9',
        'secondary': '#00A97A',
        'accent1': '#FFB800',
        'accent2': '#E31E24',
        'neutral': '#666666',
        'background': '#FFFFFF',
        'text': '#333333'
    }
}

# Analysis settings
ANALYSIS_CONFIG = {
    'min_year': 2010,
    'max_year': 2023,
    'market_share_threshold': 0.01,  # 1% minimum for market share
    'growth_threshold': 0.05,        # 5% minimum for significant growth
    'provider_threshold': 3          # Minimum number of providers for analysis
}

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# PowerPoint export settings
SLIDE_HEIGHT_RATIO = 9
SLIDE_WIDTH_RATIO = 16
SLIDE_HEIGHT_INCHES = 5.625
SLIDE_WIDTH_INCHES = 10
PLOT_IMAGE_TOP_PCT = 0.8 