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

## Documentation

**For the full documentation, see the [Documentation Index](docs/INDEX.md).**

Key guides include:
*   [Tutorial](docs/TUTORIAL.md)
*   [Command-Line Interface (CLI) Guide](docs/CLI_GUIDE.md)
*   [Data Requirements (including fetching data)](docs/DATA_REQUIREMENTS.md)
*   [Market Analysis Features](docs/MARKET_ANALYSIS.md)

---

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
```

See the full [CLI Guide](docs/CLI_GUIDE.md) for all available arguments and options like filtering and specifying output directories.

### Programmatic Usage

For more advanced customization, you can use the analysis functions directly. The recommended approach is to import and use the workflow functions from `src/vipunen/cli/analyze_cli.py` which mirror the CLI execution flow:

```python
import pandas as pd
from pathlib import Path
from src.vipunen.config.config_loader import get_config
from src.vipunen.cli.analyze_cli import (
    prepare_analysis_data,
    perform_market_analysis,
    export_analysis_results,
    generate_visualizations
)
from src.vipunen.visualization.education_visualizer import EducationVisualizer

# --- 1. Configuration & Arguments ---
# Load main config
config = get_config()

# Simulate command-line arguments or define them programmatically
# These override defaults in config where applicable
args = {
    'data_file': config['paths']['data'],  # Use path from config
    'institution': config['institutions']['default']['name'],
    'short_name': config['institutions']['default']['short_name'],
    'variants': [],  # Or specify variants: ['Var1', 'Var2']
    'output_dir': config.get('paths', {}).get('output', 'data/reports'), # Use output path from config
    'use_dummy': False,
    'filter_qual_types': False,
    'plot_format': 'pdf'
}

# --- 2. Prepare Data ---
# This loads data, cleans it, and determines institution details
df_clean, inst_key, inst_variants, inst_short_name, data_update_date, filter_quals = prepare_analysis_data(config, args)

# --- 3. Perform Analysis ---
# This runs the MarketAnalyzer
analysis_results, analyzer_instance = perform_market_analysis(
    df_clean, config, inst_variants, inst_short_name, filter_quals
)

# --- 4. Export Results (Optional) ---
if not analysis_results.get("detailed_providers_market", pd.DataFrame()).empty:
    excel_file = export_analysis_results(
        analysis_results, config, inst_short_name, args['output_dir']
    )
    print(f"Exported results to {excel_file}")

# --- 5. Generate Visualizations (Optional) ---
if not analysis_results.get("detailed_providers_market", pd.DataFrame()).empty:
    viz_dir = Path(args['output_dir']) / f"education_market_{inst_short_name.lower()}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    visualizer = EducationVisualizer(
        output_dir=viz_dir,
        output_format=args.get('plot_format', 'pdf'),
        institution_short_name=inst_short_name
    )
    try:
        generate_visualizations(
            analysis_results, visualizer, analyzer_instance, config, data_update_date
        )
    finally:
        visualizer.close_pdf() # Ensure PDF is closed
    print(f"Visualizations saved in {viz_dir}")

```

### Notebook Analysis

For a more interactive, step-by-step analysis workflow, see the Jupyter Notebooks in the `notebooks/` directory. These notebooks replicate the core analysis and allow for easier exploration of intermediate results and visualizations.

Refer to the **[Notebooks README](notebooks/README.md)** for detailed instructions on configuration and execution.

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

## Configuration (`config.yaml`)

The analysis behavior, input/output paths, column names, Excel sheet names, and visualization filters can be customized via the `config.yaml` file located in the project root.

Key sections include:

*   `columns`: Defines mappings between input data columns and the desired output column names. Modifying `columns.output` allows you to change the headers used in the analysis results and the exported Excel file.
*   `excel.sheets`: Defines the names and descriptions for the sheets generated in the Excel export. Modify the `name` fields here to control the output sheet names.
*   `institutions`: Specify the target institution, its short name, and known name variants for data matching.
*   `paths`: Define input data location and the base directory for output reports.
*   `analysis`: Configure analysis thresholds, such as `min_market_size_threshold` for filtering qualifications and settings for the gainers/losers plots.

### Language Customization

By modifying the `columns.output` and `excel.sheets` sections in `config.yaml`, you can translate the output column headers and Excel sheet names into different languages. The default configuration includes examples for Finnish:

```yaml
# config.yaml (snippet)
columns:
  output:
    year: 'Vuosi'
    qualification: 'Tutkinto'
    provider: 'Oppilaitos'
    # ... other column names ...
excel:
  sheets:
    - name: "NOM Yhteensä"
    - name: "Oppilaitoksen NOM tutkinnoittain"
    - name: "Oppilaitoksen koko markkina"
    - name: "CAGR analyysi"
```

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

1.  An **Excel file** containing multiple sheets with detailed analysis results (including an "Analysis Info" sheet with run parameters, plus sheets for total volumes, volumes by qualification, detailed provider market data, CAGR, etc.). The names of the sheets and the column headers within them are configurable via `config.yaml` (see Configuration section). See [Excel Export Documentation](docs/EXCEL_EXPORT.md).
2.  A **PDF report** saved in the main output folder (e.g., `ri_visualizations_[timestamp].pdf`). This PDF contains multiple pages, each displaying a plot with a 16:9 aspect ratio. The plots include:
    *   Institution's total student volumes over time (Stacked Area Chart)
    *   Institution's student volumes vs. total market providers count (Combined Stacked Bar + Grouped Bar)
    *   Institution's qualifications positioned by Market Growth vs. Relative Market Share (BCG Matrix / Bubble Chart)
    *   Market share evolution for top competitors within active qualifications (Line Charts - one per qualification)
    *   Institution's market share across active qualifications over time (Heatmap)
    *   Market share gainers/losers for active qualifications (Horizontal Bar Charts - one per qualification)
    *   Treemap showing institution's market share vs. qualification size for the reference year (Static plot using Matplotlib/Squarify).
    *   BCG Growth-Share Matrix (Bubble Chart) showing qualification growth vs. relative market share. (**Note:** Interpret with care, see [BCG article](https://www.bcg.com/publications/2014/growth-share-matrix-bcg-classics-revisited)).
3.  Console logs detailing the analysis progress.

The main output folder is named based on the institution's short name (e.g., `education_market_ri`).
- The Excel file (e.g., `ri_market_analysis_[timestamp].xlsx`) is saved in this folder.
- The PDF report (e.g., `ri_visualizations_[timestamp].pdf`) containing all generated plots is also saved in this folder.
Outputs are saved by default under `data/reports/`. For example:
`data/reports/education_market_ri/ri_market_analysis_[timestamp].xlsx` and 
`data/reports/education_market_ri/ri_visualizations_[timestamp].pdf`

## Code Structure

The code is organized into a package structure:

```
src/vipunen/
├── analysis/              # Market and qualification analysis
│   ├── education_market.py       # (Potentially older analysis functions)
│   ├── market_analyzer.py        # **Core analysis orchestration class**
│   ├── market_share_analyzer.py  # Specialized market share calculations
│   └── qualification_analyzer.py # Qualification-specific metrics (CAGR)
├── cli/                  # Command-line interface logic
│   ├── analyze_cli.py          # Orchestrates CLI workflow
│   └── argument_parser.py      # Argument parsing logic
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
- FileUtils (v0.6.3+)
- squarify

See `environment.yaml` for a full list and versions.

## FileUtils Integration

This project utilizes the [FileUtils](https://github.com/topij/FileUtils) package for standardized file operations like loading data and saving reports/plots. This aims to provide consistent path handling and directory structures. The configuration is managed in `src/vipunen/utils/file_utils_config.py`.

## Troubleshooting

- **FileUtils Issues**: If you encounter path or export errors related to FileUtils, ensure the package is installed and check the configuration in `src/vipunen/utils/file_utils_config.py`.
- **Missing Data Files**: Verify the expected data file exists at the specified location.
- **Dependency Conflicts**: Use the provided `environment.yaml` with Conda to avoid potential conflicts. 