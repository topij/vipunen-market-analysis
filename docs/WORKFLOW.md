# Education Market Analysis Workflow Guide

This guide explains how to use the refactored vipunen package to analyze the Finnish vocational education market from a provider's perspective.

## Quick Start

```bash
# Set up the conda environment
conda env create -f environment.yaml
conda activate vipunen-analytics

# Run the analysis with dummy data (for demonstration)
python run_analysis.py --use-dummy

# Run the analysis with real data and custom parameters
python run_analysis.py --data-file data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv --institution "Provider Name" --short-name "Provider"
```

## Complete Workflow

### 1. Setup

```bash
# Clone the repository (if needed)
git clone https://github.com/your-username/vipunen-project.git
cd vipunen-project

# Set up the conda environment
conda env create -f environment.yaml
conda activate vipunen-analytics
```

### 2. Prepare Your Data

The analysis requires Finnish vocational education data in CSV format with these columns:
- `tilastovuosi`: Year
- `tutkintotyyppi`: Qualification type
- `tutkinto`: Qualification name
- `koulutuksenJarjestaja`: Main provider
- `hankintakoulutuksenJarjestaja`: Subcontractor (if exists)
- `nettoopiskelijamaaraLkm`: Student volume

Place your data file in `data/raw/` directory. The default expected file is `amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv`.

### 3. Run the Analysis

For basic analysis with default parameters:

```bash
python run_analysis.py --data-file data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv
```

With custom parameters and filtering options:

```bash
python run_analysis.py \
  --data-file data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv \
  --institution "Provider Full Name" \
  --short-name "Provider" \
  --variant "Provider Name Variant 1" \
  --variant "Provider Name Variant 2" \
  --output-dir "custom/output/path" \
  --filter-qual-types \
  --filter-by-institution-quals
```

#### Optional Arguments:

- `--filter-qual-types`: Filter data to include only ammattitutkinto and erikoisammattitutkinto
- `--filter-by-institution-quals`: Filter data to include only qualifications offered by the institution under analysis during the current and previous year
- `--output-dir`: Specify a custom output directory
- `--variant`: Add institution name variants (can be specified multiple times)
- `--use-dummy`: Use dummy data instead of loading from file (for testing purposes)

### 4. Analysis Steps

The analysis follows these steps:

1. **Data Loading**: Load data from CSV file using FileUtils
   - The `load_data` function in `src/vipunen/data/data_loader.py` uses the `VipunenFileHandler` to load the data
   - Only the filename is needed; the path structure is managed by FileUtils

2. **Data Cleaning**: Clean and prepare the data
   - Replace missing values
   - Standardize qualification names

3. **Data Filtering** (Optional):
   - Filter by qualification types (ammattitutkinto/erikoisammattitutkinto)
   - Filter by qualifications offered by the institution

4. **Volume Calculation**: 
   - Calculate volumes where the institution is the main provider
   - Calculate volumes where the institution is a subcontractor
   - Calculate total volumes by qualification
   - Convert volumes to long format for easier analysis

5. **Market Share Analysis**:
   - Calculate market shares for each qualification
   - Calculate year-over-year changes in market shares for all consecutive year pairs
   - Calculate market ranks and market gainer ranks

6. **Growth Analysis**:
   - Calculate qualification growth metrics
   - Calculate CAGR (Compound Annual Growth Rate) with detailed history

7. **Visualization**:
   - Generate volume plots
   - Generate market share plots
   - Generate growth trend plots

8. **Export Results**:
   - Export analysis results to Excel with focused worksheets using `ExcelExporter`
   - Files are saved with timestamps for versioning

### 5. Output Files

The analysis generates output in `data/reports/education_market_[institution_name]/`:

- **Excel File**: Comprehensive data tables with all metrics
  - `[institution]_market_analysis_[timestamp].xlsx` with the following worksheets:
    - **Total Volumes**: Institution's total volumes by year
    - **Volumes by Qualification**: Long-format table with volumes by year and qualification
    - **Provider's Market**: Combined market data with market shares and growth for all years
    - **CAGR Analysis**: Detailed qualification history and growth rates

- **Plot Files** (in `plots/` subdirectory):
  - `[institution]_total_volumes.png`: Total volumes by year and role
  - `[institution]_top_qualifications.png`: Top qualifications by volume
  - `[institution]_market_share_heatmap.png`: Market share heatmap
  - `[institution]_qualification_market_shares.png`: Market shares in top qualifications
  - `[institution]_qualification_time_series.png`: Volume trends over time

### 6. Troubleshooting FileUtils Integration

The project uses FileUtils v0.6.1 for standardized file operations. Here are solutions to common issues:

#### Path Resolution Issues

If you encounter errors like:
```
/path/to/data/raw/data/raw/filename.csv: No such file or directory
```

This indicates a path resolution issue where directory prefixes are being duplicated. The solution:

1. Make sure to pass only the filename to `load_data`, not the full path:
   ```python
   # Correct
   df = load_data("amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv")
   
   # Incorrect - might cause path duplication
   df = load_data("data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv")
   ```

2. If you need to use the full path, ensure the `ensure_data_directory` function correctly handles paths with existing "data/" prefix.

#### Excel Export Errors

If you see errors like:
```
'str' object has no attribute 'value'
```

This is typically caused by incompatibilities with how FileUtils handles enums. The solution implemented in this project:

1. We've customized the `export_to_excel` method in `VipunenFileHandler` to use pandas directly, bypassing some of the FileUtils functionality that causes issues.

2. For your own custom exports, use this pattern:
   ```python
   file_handler = VipunenFileHandler()
   excel_path = file_handler.export_to_excel(
       data_dict=your_data_dict,
       file_name="your_file_name",
       output_type="reports",
       include_timestamp=True
   )
   ```

#### File Not Found Issues

If the expected data file isn't found:

1. Check that the file exists in the correct location (FileUtils looks in the `/data/raw/` directory by default)
2. Verify the filename matches exactly what's expected (case-sensitive)
3. If using the CLI, double-check your `--data-file` parameter value

### 7. General Troubleshooting

- **Missing Data**: If your data file is missing, the script will generate dummy data for demonstration
- **Column Names**: Ensure your data has the required column names (see above)
- **Name Variants**: Provide all known name variants of the institution to ensure accurate analysis
- **Filtering Issues**: If filtering results in empty data, try running without filtering options

## Advanced Usage

### Using the Modules Directly

You can also use the modules directly in your own scripts:

```python
from src.vipunen.data.data_loader import load_data
from src.vipunen.data.data_processor import clean_and_prepare_data
from src.vipunen.analysis.market_share_analyzer import calculate_market_shares, calculate_market_share_changes
from src.vipunen.visualization.volume_plots import plot_total_volumes

# Load and prepare data
raw_data = load_data("amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv")
df_clean = clean_and_prepare_data(raw_data, institution_names=["Provider Name"])

# Optional filtering
if filter_by_qualification_type:
    df_filtered = df_clean[df_clean["tutkintotyyppi"].isin(["Ammattitutkinnot", "Erikoisammattitutkinnot"])]
else:
    df_filtered = df_clean.copy()

# Calculate market metrics
market_shares = calculate_market_shares(df_filtered, ["Provider Name"])

# Calculate market share changes for all consecutive year pairs
all_years = sorted(df_filtered["tilastovuosi"].unique())
market_share_changes_all = []

for i in range(1, len(all_years)):
    current_year = all_years[i]
    previous_year = all_years[i-1]
    year_changes = calculate_market_share_changes(market_shares, current_year, previous_year)
    market_share_changes_all.append(year_changes)

# Combine all year changes
market_share_changes = pd.concat(market_share_changes_all, ignore_index=True)

# Create visualizations
plot_total_volumes(volumes_df, institution_short_name="Provider")
```

### Using the FileUtils Integration Directly

For direct access to file operations:

```python
from src.vipunen.utils.file_handler import VipunenFileHandler

# Get the singleton instance
file_handler = VipunenFileHandler()

# Load data
df = file_handler.load_data(
    "your_data_file.csv", 
    input_type="raw",
    file_type=InputFileType.CSV
)

# Export processed data
output_path = file_handler.save_data(
    data=processed_df,
    file_name="processed_data", 
    output_type="processed"
)
```

### Extending the Analysis

To add new metrics or visualizations:

1. Add new analysis functions to the appropriate module
2. Update the main workflow script to include your new analysis
3. Add new visualization functions if needed
4. Update the Excel export to include your new metrics

## Conclusion

This workflow allows for comprehensive analysis of the Finnish vocational education market from a provider's perspective, focusing on market shares, volumes, and growth trends across different qualifications. The optional filtering features allow you to focus your analysis on specific qualification types or only those qualifications offered by the institution under analysis. 