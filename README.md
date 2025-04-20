# Vipunen Project

A tool for fetching and analyzing data on educational institutions and their results from the Vipunen API.

## Project Structure

```
vipunen_project/
├── src/                    # Source code
│   ├── vipunen/           # Main package
│   │   ├── api/           # API client
│   │   ├── analysis/      # Analysis modules
│   │   └── utils/         # Utility functions
│   └── scripts/           # Command-line scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── data/                  # Data files
│   ├── raw/              # Raw data from API
│   └── processed/        # Processed data
└── docs/                 # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vipunen-project.git
cd vipunen-project
```

2. Create and activate a conda environment:
```bash
conda env create -f environment.yaml
conda activate vipunen-analytics
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Fetching Data

To fetch data from the Vipunen API:

```bash
python src/scripts/fetch_data.py
```

This will save the data to `data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv`.

### Analysis

The analysis can be performed using the Jupyter notebooks in the `notebooks/` directory:

1. `exploration.ipynb`: For data exploration
2. `analysis.ipynb`: For analysis and visualization

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

To format the code:

```bash
black .
isort .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 