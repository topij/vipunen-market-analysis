# Vipunen - Finnish Education Market Analysis

This project provides tools for analyzing the Finnish vocational education market from a provider's perspective. It focuses on analyzing ammattitutkinto (further vocational qualifications) and erikoisammattitutkinto (specialist vocational qualifications).

## Project Overview

The Vipunen project is a specialized data analysis toolkit designed for educational institutions to gain insights into the Finnish vocational education market. It allows institutions to:

- Analyze their market position and trends
- Identify growing and declining qualifications
- Compare performance against competitors
- Track student volumes across years
- Calculate compound annual growth rates (CAGR) for qualifications
- Generate comprehensive reports and visualizations

## Installation and Setup

### Prerequisites

- Python 3.8+
- FileUtils package (v0.6.1+)
- Conda (recommended for environment management)

### Using Conda

The recommended way to set up the environment is using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate vipunen-analytics
```

### Manual Installation

If you prefer not to use Conda, you can install the dependencies manually:

```bash
pip install pandas numpy matplotlib seaborn pathlib PyYAML FileUtils
```

### Development Installation

For development work, install the package in development mode:

```bash
pip install -e .
```

## Code Structure

The code is organized into a package structure with clear separation of concerns:

```
src/vipunen/
├── analysis/              # Market and qualification analysis
│   ├── education_market.py       # Core analysis functionality
│   ├── market_analyzer.py        # Market share analysis
│   ├── market_share_analyzer.py  # Specialized market share calculations
│   └── qualification_analyzer.py # Qualification-specific metrics
├── cli/                  # Command-line interface
│   ├── analyze_cli.py     # Main CLI workflow
│   └── argument_parser.py # CLI argument parsing
├── config/               # Configuration management
├── data/                 # Data loading and processing
│   ├── data_loader.py    # Functions for loading the raw data
│   ├── data_processor.py # Functions for cleaning and preparing data
│   └── dummy_generator.py # Generates dummy data for testing
├── export/               # Data export functionality
│   └── excel_exporter.py # Excel export utilities
├── utils/                # Utility functions and helpers
│   └── file_utils_config.py # FileUtils integration
└── visualization/        # Plotting and visualization
    ├── volume_plots.py   # Plots for volume metrics
    ├── market_plots.py   # Plots for market share metrics
    └── growth_plots.py   # Plots for growth metrics
```

## FileUtils Integration

This project uses the [FileUtils](https://github.com/topi-python/FileUtils) package for standardized file operations. The integration provides:

- Consistent data loading and saving across the application
- Standardized directory structure management
- Automatic timestamp-based file naming
- Metadata tracking for data lineage

The FileUtils configuration is managed in `src/vipunen/utils/file_utils_config.py`. 