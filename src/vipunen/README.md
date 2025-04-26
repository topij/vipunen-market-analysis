# Vipunen Analysis Package

This package provides tools for analyzing education market data in Finland with a focus on vocational qualifications.

## Structure

The package is organized into the following modules:

- **config**: Configuration management using YAML
- **data**: Data loading and processing utilities
- **analysis**: Core analysis functionality
- **export**: Export results to various formats
- **cli**: Command-line interface
- **utils**: General utility functions
- **visualization**: (Not yet refactored) Visualization utilities

## Usage

The package can be used either through the command-line interface or directly from Python.

### Command-line

```bash
# Run with default parameters (analyzes Rastor-instituutti)
python run_analysis.py

# Run with custom parameters
python run_analysis.py --institution "Another Institution" --short-name "AI" --filter-qual-types
```

### Python API

```python
from src.vipunen.cli.analyze_cli import run_analysis

# Run with default parameters
results = run_analysis()

# Run with custom parameters
custom_args = {
    'institution': 'Another Institution',
    'short_name': 'AI',
    'filter_qual_types': True
}
results = run_analysis(custom_args)

# Access the results
total_volumes = results['total_volumes']
market_shares = results['market_shares']
```

### Using the Market Analyzer Directly

```python
import pandas as pd
from src.vipunen.data.data_loader import load_data
from src.vipunen.data.data_processor import clean_and_prepare_data
from src.vipunen.analysis.market_analyzer import MarketAnalyzer

# Load and prepare data
raw_data = load_data()
clean_data = clean_and_prepare_data(raw_data)

# Create analyzer
analyzer = MarketAnalyzer(
    data=clean_data,
    institution_names=["Rastor-instituutti", "Rastor-instituutti ry"],
    filter_degree_types=True,
    filter_by_institution_quals=True
)

# Run analysis
results = analyzer.analyze()

# Access results
total_volumes = results['total_volumes']
market_shares = results['market_shares']
qualification_growth = results['qualification_growth']
```

## Configuration

The package uses YAML configuration files. The default configuration is in `src/vipunen/config/config.yaml`. You can override this by:

1. Creating a `config.yaml` file in the current directory
2. Creating a `config.yaml` file in a `config/` subdirectory
3. Creating a `.vipunen/config.yaml` file in your home directory

Configuration includes:
- Column name mappings for input and output
- Institution information (names, variants)
- Qualification types to filter
- File paths for data and outputs
- Excel export settings

## Refactoring Plan

This package is being refactored in phases:

### Phase 1 (Completed)
- ✓ Set up YAML configuration structure
- ✓ Create modular code structure
- ✓ Implement basic CLI wrapper
- ✓ Extract data loading/processing

### Phase 2 (Current)
- ✓ Create MarketAnalyzer wrapper for EducationMarketAnalyzer
- ✓ Update CLI to use MarketAnalyzer
- ◻ Complete implementation of market analysis functions
- ◻ Add unit tests

### Phase 3 (Future)
- Update FileUtils integration after API changes
- Refactor visualization modules
- Add more documentation

## Dependencies

- pandas
- numpy
- FileUtils (custom package)
- PyYAML
- matplotlib (for visualization)
- openpyxl (for Excel export) 