# Data Requirements

This document outlines the data format requirements for the Vipunen education market analysis project. Following these specifications ensures proper functioning of the analysis tools.

## Input Data Format

The analysis expects data in CSV format with the following columns:

| Column Name | Description | Data Type | Required |
|-------------|-------------|-----------|----------|
| `tilastovuosi` | Statistical year | Integer | Yes |
| `suorituksenTyyppi` | Type of completion | String | Yes |
| `tutkintotyyppi` | Type of qualification | String | Yes |
| `tutkinto` | Name of the qualification | String | Yes |
| `koulutuksenJarjestaja` | Main education provider | String | Yes |
| `hankintakoulutuksenJarjestaja` | Subcontractor provider (if exists) | String | No |
| `nettoopiskelijamaaraLkm` | Net student count, average yearly volume | Float | Yes |

### Expected Values

- `tutkintotyyppi`: Expected values include "Ammattitutkinnot" (further vocational qualifications) and "Erikoisammattitutkinnot" (specialist vocational qualifications). Other types may be filtered out using the `--filter-qual-types` option.
- `koulutuksenJarjestaja` and `hankintakoulutuksenJarjestaja`: Institution names. Missing values for `hankintakoulutuksenJarjestaja` are represented as "Tieto puuttuu" (information missing).

## Sample Data

Here's an example of properly formatted data:

```csv
tilastovuosi,suorituksenTyyppi,tutkintotyyppi,tutkinto,koulutuksenJarjestaja,hankintakoulutuksenJarjestaja,nettoopiskelijamaaraLkm
2018,Tutkinto,Ammattitutkinnot,Liiketoiminnan ammattitutkinto,Rastor-instituutti ry,Tieto puuttuu,125.5
2018,Tutkinto,Ammattitutkinnot,Liiketoiminnan ammattitutkinto,Helsinki Business College Oy,Tieto puuttuu,89.2
2018,Tutkinto,Erikoisammattitutkinnot,Johtamisen erikoisammattitutkinto,Rastor-instituutti ry,Tieto puuttuu,78.3
2019,Tutkinto,Ammattitutkinnot,Liiketoiminnan ammattitutkinto,Rastor-instituutti ry,Tieto puuttuu,142.1
2019,Tutkinto,Ammattitutkinnot,Liiketoiminnan ammattitutkinto,Helsinki Business College Oy,Tieto puuttuu,95.6
2019,Tutkinto,Erikoisammattitutkinnot,Johtamisen erikoisammattitutkinto,Rastor-instituutti ry,Tieto puuttuu,82.7
```

## Data Preparation

The Vipunen package includes tools for data preparation:

1. The `data_loader.py` module handles loading raw data from CSV files
2. The `data_processor.py` module provides functions for cleaning and standardizing the data

Example of loading and preparing data:

```python
from src.vipunen.data.data_loader import load_data
from src.vipunen.data.data_processor import clean_and_prepare_data

# Load raw data
raw_data = load_data("data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv")

# Clean and prepare data
institution_variants = ["Rastor-instituutti", "Rastor-instituutti ry", "Rastor"]
clean_data = clean_and_prepare_data(raw_data, institution_names=institution_variants)
```

## Data Processing Steps

The data processing pipeline performs the following operations:

1. **Data Loading**: Reads CSV files using pandas
2. **Data Cleaning**:
   - Removes rows with missing critical values
   - Standardizes institution names
   - Converts data types (e.g., ensures `tilastovuosi` is integer)
   - Handles missing values for `hankintakoulutuksenJarjestaja`
3. **Data Transformation**:
   - Creates additional columns for analysis (e.g., `kouluttaja` combining both provider columns)
   - Shortens qualification names for better visualization
   - Calculates derived metrics (e.g., total student volumes)

## Filtering Options

Several filtering options are available to focus the analysis:

1. **Filter by Qualification Types**: Include only ammattitutkinto and erikoisammattitutkinto
2. **Filter by Institution's Qualifications**: Include only qualifications offered by the institution under analysis
3. **Filter by Years**: Focus on specific year ranges for trend analysis

These can be applied both via the CLI and programmatically.

## Using Dummy Data

For testing or demonstration purposes, you can generate dummy data using:

```python
from src.vipunen.data.dummy_generator import create_dummy_dataset

dummy_data = create_dummy_dataset(
    num_years=5,              # Number of years to generate
    num_qualifications=20,    # Number of unique qualifications
    num_providers=15,         # Number of education providers
    start_year=2018,          # First year in the dataset
    provider_name="Example Institute"  # Name of the main provider for analysis
)
```

The dummy data maintains the same structure as the expected input format.

## Obtaining Data from Vipunen API

While the raw CSV data can sometimes be found pre-downloaded, the recommended way to obtain the latest data is using the provided fetching script which interacts directly with the Vipunen API.

**Script:** `src/scripts/fetch_data.py`

**Functionality:**
- Connects to the Vipunen API.
- Downloads data for a specified dataset (defaults to `amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto`).
- Saves the data into a CSV file in the configured raw data directory (default: `data/raw/`).
- **Update Check:** Before downloading, the script checks a metadata file (`.metadata/{dataset_name}_metadata.json`) for the last known update timestamp of the data on the API. If the timestamp hasn't changed, the download is skipped to save time and resources.
- **Backup:** If a new download occurs, the previous version of the CSV file is backed up into a subdirectory (default: `old_api_calls_output`).

**Configuration:**
API connection details (base URL, caller ID, retry settings, timeouts) and output formatting (CSV separator, encoding, backup directory name) are configured in the `api` section of `config/config.yaml`.

**Usage:**
```bash
# Fetch default dataset specified in config
python src/scripts/fetch_data.py

# Fetch a specific dataset
python src/scripts/fetch_data.py --dataset other_dataset_name

# Specify output directory
python src/scripts/fetch_data.py --output-dir path/to/save

# Force download even if update date hasn't changed
python src/scripts/fetch_data.py --force-download
```

Refer to the [CLI Guide](CLI_GUIDE.md) for more script details.

## Common Data Issues and Solutions

1. **Missing Provider Information**: Rows with missing `koulutuksenJarjestaja` are excluded from analysis
2. **Institution Name Variants**: Specify known variants using the `--variant` option or `institution_names` parameter
3. **Qualification Name Length**: Long qualification names are automatically shortened for better visualization
4. **Year Coverage**: Ensure data covers multiple years for trend analysis to work properly (minimum 2 years)
5. **Zero Volumes**: Rows with zero student volumes are retained but may affect growth calculations

## Data Source

The expected data format is based on the publicly available education statistics from Vipunen (Finnish Education Statistics Database). The raw data can be downloaded from [Vipunen](https://vipunen.fi/fi-fi). 