# Finnish Education Market Analysis

This project provides tools for analyzing the Finnish vocational education market using data fetched from the open [Vipunen API](https://vipunen.fi/fi-fi/Sivut/Vipunen-API.aspx). The API is maintained by the [Finnish education administration](https://vipunen.fi/en-gb/). This project focuses on one particular data resource available the API which gives data on students in vocational education and training, as well as completed qualifications, categorized by year, qualification level, and education provider.

The data analysis toolkit allows institutions to:

- Analyze their market position and trends
- Identify growing and declining qualifications
- Compare performance against competitors
- Track student volumes across years
- Calculate compound annual growth rates (CAGR) for qualifications
- Generate comprehensive reports and visualizations

The focus in this iteration of the project has been on analyzing the market based on the annual net student count ("netto-opiskelijamäärä" in Finnish). It means the number of students an educational institution has for each day of the year divided by the number of days in the year. 

In this way, the net student count accurately reflects the average student volume for the year and complements the calendar year student count data, which includes everyone who studied during that year, regardless of the duration of their studies ([source](https://www.oph.fi/fi/uutiset/2020/vipunen-tilastopalveluun-nopeasti-paivittyvaa-tietoa-ammatillisesta-koulutuksesta)). In the analysis this is refered to as _student volume_.

See the full documentation in the [docs](docs/INDEX.md) directory.

### Project Background
I used to work at an [organization](https://rastorinst.fi) offering, among other things, vocational qualifications. As the person in charge of growth and development, I did a lot of data analysis. I did the early versions of this project during that time for my own data wrangling needs. After leaving that organization I wanted to re-write and tidy up the project to accomodate wider set of use cases. I don't maintain the project actively, but anyone is free to use, modify and improve it for their own needs. Have fun :-)

## Installation and Setup

### Prerequisites

- Python 3.8+
- Conda (recommended for environment management)

### Using Conda

The recommended way to set up the environment is using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate vipunen-analytics
```

### Manual Installation (Not Recommended)

If you prefer not to use Conda, you can install the dependencies manually (ensure versions are compatible):

```bash
pip install pandas numpy matplotlib seaborn pathlib PyYAML FileUtils squarify
```
*(See `environment.yaml` for specific versions).*

### Development Installation

For development work, install the package in development mode after setting up the environment:

```bash
pip install -e .
```

## Usage

### Command-line Interface

The easiest way to run the analysis is using the command-line interface provided by `run_analysis.py`.

```bash
python run_analysis.py --data-file <path_to_data> --institution <institution_name> --short-name <short_name>
```

For example:

```bash
python run_analysis.py --data-file amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv --institution "Rastor-instituutti ry" --variant "Rastor Oy" --short-name "RI"
python run_analysis.py --data-file amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv --institution "Rastor-instituutti ry" --variant "Rastor Oy" --short-name "RI"
```

See the full [CLI Guide](docs/CLI_GUIDE.md) for all available arguments and options like filtering and specifying output directories.

### Programmatic Usage

For more advanced customization, you can use the `MarketAnalyzer` class directly in your Python scripts. The primary workflow involves:

1.  Loading and preparing data (see `src/vipunen/data/`).
2.  Initializing `MarketAnalyzer` with the data and institution details.
3.  Calling the `analyze()` method.
4.  Using the resulting dictionary of DataFrames for export or further analysis.

```python
from src.vipunen.data.data_loader import load_data
from src.vipunen.data.data_processor import clean_and_prepare_data
from src.vipunen.analysis.market_analyzer import MarketAnalyzer
from src.vipunen.export.excel_exporter import export_to_excel # Example exporter

# Configuration (example)
data_path = "data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv"
institution_name = "Rastor-instituutti ry"
institution_variants = ["Rastor-instituutti ry", "Rastor Oy"]
institution_short_name = "RI"
min_market_size_threshold = 5
output_dir = "data/reports/education_market_ri"

# Load and prepare data
raw_data = load_data(data_path)
processed_data = clean_and_prepare_data(raw_data) # Apply cleaning as needed

# Initialize analyzer
analyzer = MarketAnalyzer(processed_data)
analyzer.institution_names = institution_variants
analyzer.institution_short_name = institution_short_name

# Run analysis
analysis_results = analyzer.analyze(min_market_size_threshold=min_market_size_threshold)

# Export (example)
excel_file = export_to_excel(
    analysis_results,
    f"{institution_short_name}_analysis",
    output_dir=output_dir
)
print(f"Exported results to {excel_file}")



```

See the [Programmatic Usage Guide](docs/PROGRAMMATIC_USAGE.md) and the [Market Analysis Features](docs/MARKET_ANALYSIS.md) documentation for more details.

## Input Data Format

The analysis expects a CSV file with specific columns representing Finnish Vipunen education data. Key columns include:

- `tilastovuosi`: The year of the data.
- `tutkintotyyppi`: Type of qualification (e.g., Ammattitutkinnot, Erikoisammattitutkinnot).
- `tutkinto`: Name of the qualification.
- `koulutuksenJarjestaja`: Main education provider.
- `hankintakoulutuksenJarjestaja`: Subcontractor provider (if it exists).
- `nettoopiskelijamaaraLkm`: Net student count (average yearly volume).

See [Data Requirements](docs/DATA_REQUIREMENTS.md) for full details.

## Output

The analysis typically produces:

1.  An **Excel file** containing multiple sheets with detailed analysis results (total volumes, volumes by qualification, detailed provider market data, CAGR, etc.). See [Excel Export Documentation](docs/EXCEL_EXPORT.md).
2.  A set of **visualization plots** (PNG images) saved in a `plots` subdirectory, showing trends in volumes, market shares, and growth. See [Visualization Documentation](docs/VISUALIZATION.md).
3.  Console logs detailing the analysis progress.

Outputs are saved by default under `data/reports/[institution_short_name]_market_analysis_[timestamp]/`.
Outputs are saved by default under `data/reports/[institution_short_name]_market_analysis_[timestamp]/`.

## Code Structure

The code is organized into a package structure:

```
src/vipunen/
├── analysis/              # Market and qualification analysis
│   ├── education_market.py       # (Potentially older analysis functions)
│   ├── market_analyzer.py        # **Core analysis orchestration class**
│   ├── market_share_analyzer.py  # Specialized market share calculations
│   └── qualification_analyzer.py # Qualification-specific metrics (CAGR)
├── cli/                  # Command-line interface
│   ├── analyze_cli.py     # Main CLI workflow logic
│   └── argument_parser.py # CLI argument parsing
├── config/               # Configuration management (config.yaml)
├── data/                 # Data loading and processing
│   ├── data_loader.py    # Functions for loading the raw data
│   ├── data_processor.py # Functions for cleaning and preparing data
│   └── dummy_generator.py # Generates dummy data for testing
├── export/               # Data export functionality
│   └── excel_exporter.py # Excel export utilities
├── utils/                # Utility functions and helpers
│   └── file_utils_config.py # FileUtils integration configuration
└── visualization/        # Plotting and visualization
    └── education_visualizer.py # Class for creating standard plots
```

## Dependencies

Key dependencies include:

- pandas
- numpy
- matplotlib
- seaborn
- pathlib
- PyYAML
- FileUtils (v0.6.1+)
- squarify

See `environment.yaml` for a full list and versions.

## FileUtils Integration

This project utilizes the [FileUtils](https://github.com/topi-python/FileUtils) package for standardized file operations like loading data and saving reports/plots. This aims to provide consistent path handling and directory structures. The configuration is managed in `src/vipunen/utils/file_utils_config.py`.

## Troubleshooting

- **FileUtils Issues**: If you encounter path or export errors related to FileUtils, ensure the package is installed and check the configuration in `src/vipunen/utils/file_utils_config.py`.
- **Missing Data Files**: Verify the expected data file exists at the specified location.
- **Dependency Conflicts**: Use the provided `environment.yaml` with Conda to avoid potential conflicts. 