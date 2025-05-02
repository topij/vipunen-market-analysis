# Vipunen Project Documentation

Welcome to the Vipunen Project documentation! This documentation provides comprehensive information about the Vipunen educational market analysis toolset.

## Table of Contents

### Core Documentation

- [Project Overview](../README.md) - Introduction to the Vipunen project and its capabilities
- [Command-Line Interface Guide](CLI_GUIDE.md) - How to use the CLI for quick analysis
- [Programmatic Usage Guide](PROGRAMMATIC_USAGE.md) - How to use the library programmatically
- [Tutorial](TUTORIAL.md) - Step-by-step guide to get started

### Data and Analysis

- [Data Requirements](DATA_REQUIREMENTS.md) - Input data format and specifications
- [Market Analysis Features](MARKET_ANALYSIS.md) - Available market analysis functionality
- [Excel Export Functionality](EXCEL_EXPORT.md) - How to export results to Excel
- [Visualization Features](VISUALIZATION.md) - Basic visualization capabilities

## Quick Start

To get started with Vipunen quickly:

1. Install dependencies with `conda env create -f environment.yaml`
2. Activate the environment with `conda activate vipunen-analytics`
3. **(Optional) Fetch latest data:** Run `python src/scripts/fetch_data.py` to download the default dataset from the Vipunen API. See [Data Requirements](DATA_REQUIREMENTS.md#obtaining-data-from-vipunen-api).
4. Run an analysis with `python run_analysis.py --use-dummy --institution "Example Institute"` (uses dummy data) or `python run_analysis.py --institution "Your Institution"` (uses fetched/existing data).

See the [Tutorial](TUTORIAL.md) for more detailed instructions.

## Architecture Overview

The Vipunen project is organized into a modular structure:

```
src/vipunen/
├── analysis/              # Market and qualification analysis
├── cli/                   # Command-line interface
├── config/                # Configuration management
├── data/                  # Data loading and processing
├── export/                # Data export functionality
├── utils/                 # Utility functions
└── visualization/         # Plotting and visualization
```

## Key Features

- Comprehensive market analysis for vocational education providers
- Analysis of student volumes and market shares
- Growth trend identification and CAGR calculations
- Provider ranking and competitive analysis
- Excel reporting with multiple worksheets
- Multi-page PDF report with standard visualizations
- Command-line interface for easy usage
- Programmatic access for custom analysis workflows
