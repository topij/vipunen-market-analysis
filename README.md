# Finnish Education Market Analysis

This project provides tools for analyzing the Finnish vocational education market from a provider's perspective. It focuses on analyzing ammattitutkinto (further vocational qualifications) and erikoisammattitutkinto (specialist vocational qualifications).

## Project Structure

The code is organized into a package structure with clear separation of concerns:

```
src/vipunen/
├── data/               # Data loading and processing
│   ├── data_loader.py  # Functions for loading the raw data
│   └── data_processor.py  # Functions for cleaning and preparing data
├── analysis/           # Market and qualification analysis
│   ├── market_share_analyzer.py  # Market share calculations
│   └── qualification_analyzer.py  # Qualification-specific metrics
├── visualization/      # Plotting and visualization
│   ├── volume_plots.py  # Plots for volume metrics
│   ├── market_plots.py  # Plots for market share metrics
│   └── growth_plots.py  # Plots for growth metrics
├── utils/             # Utility functions and helpers
│   └── file_handler.py  # FileUtils integration
└── export/             # Data export functionality
    └── excel_exporter.py  # Excel export utilities
```

## FileUtils Integration

This project uses the [FileUtils](https://github.com/topi-python/FileUtils) package for standardized file operations. The integration provides:

- Consistent data loading and saving across the application
- Standardized directory structure management
- Automatic timestamp-based file naming
- Metadata tracking for data lineage

For details on how FileUtils is integrated, see [FileUtils Integration Guide](docs/FILEUTILS_INTEGRATION.md).

## Usage

### Command-line Interface

The easiest way to run the analysis is using the command-line interface provided by `run_analysis.py`:

```bash
python run_analysis.py --data-file data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv --institution "Rastor-instituutti" --short-name "RI"
```

Optional arguments:
- `--filter-qual-types`: Filter data to include only ammattitutkinto and erikoisammattitutkinto (default: False)
- `--filter-by-institution-quals`: Filter data to include only qualifications offered by the institution under analysis (default: False)
- `--output-dir`: Base directory for output files (default: data/reports)
- `--variants`: Additional name variants for the institution (can be specified multiple times)
- `--use-dummy`: Use dummy data instead of loading from file (for testing purposes)

### Programmatic Usage

For more advanced customization, the main workflow is demonstrated in `analyze_education_market.py`:

```python
python analyze_education_market.py
```

This script shows the complete workflow:
1. Loading and preprocessing data
2. Calculating volumes and market shares
3. Analyzing qualification growth trends
4. Visualizing the results
5. Exporting to Excel

## Input Data Format

The analysis expects a CSV file with the following columns:
- `tilastovuosi`: The year of the data
- `suorituksenTyyppi`: Type of completion 
- `tutkintotyyppi`: Type of qualification (Ammattitutkinnot/Erikoisammattitutkinnot)
- `tutkinto`: Name of the qualification
- `koulutuksenJarjestaja`: Main education provider
- `hankintakoulutuksenJarjestaja`: Subcontractor provider (if exists)
- `nettoopiskelijamaaraLkm`: Net student count, average yearly volume

## Output

The analysis produces:
1. Visual plots saved in the specified output directory
2. Excel file with the following worksheets:
   - **Total Volumes**: Shows the institution's total volumes by year
   - **Volumes by Qualification**: Long-format table with year, qualification, provider amount, and subcontractor amount
   - **Provider's Market**: Comprehensive market data including market share and year-over-year growth for each qualification and provider
   - **CAGR Analysis**: Detailed qualification history with first/last years offered, start/end volumes, and growth rates
3. Console logs showing the progress of the analysis

## Example Workflow

```python
# Load and prepare data
raw_data = load_data("amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv")
df_clean = clean_and_prepare_data(raw_data, institution_names=institution_variants)

# Apply optional filters
if filter_qual_types:
    df_filtered = df_clean[df_clean["tutkintotyyppi"].isin(["Ammattitutkinnot", "Erikoisammattitutkinnot"])]

# Calculate market shares
market_shares = calculate_market_shares(df_filtered, institution_variants)

# Calculate market share changes for all consecutive year pairs
market_share_changes = calculate_market_share_changes_for_all_years(market_shares, all_years)

# Create visualizations
plot_total_volumes(total_volumes, institution_short_name="Rastor")

# Export to Excel
exporter = ExcelExporter(output_dir, prefix="rastor")
excel_path = exporter.export_to_excel(excel_data)
```
See more in [WORKFLOW.md](docs/WORKFLOW.md)

## Key Metrics

The analysis calculates various metrics for vocational education providers:
- Total student volumes by year
- Volumes by qualification (as both main provider and subcontractor)
- Market shares in different qualifications
- Year-over-Year growth for all years in the dataset
- Market rank and market gainer rank
- Compound Annual Growth Rate (CAGR) for qualifications

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- pathlib
- logging
- FileUtils (v0.6.1+)

You can set up the environment using conda:

```bash
conda env create -f environment.yaml
conda activate vipunen-analytics
```

## Troubleshooting

If you encounter issues with FileUtils integration, check:

1. **Path Resolution Issues**: Ensure paths don't have duplicated directory prefixes (e.g., `data/raw/data/raw/`)
2. **Excel Export Errors**: If you see `'str' object has no attribute 'value'` errors, the issue might be with how FileUtils handles enums in your version
3. **Missing Data Files**: Verify the expected data file exists at the specified location

For more troubleshooting tips, refer to the [FileUtils Integration Guide](docs/FILEUTILS_INTEGRATION.md). 