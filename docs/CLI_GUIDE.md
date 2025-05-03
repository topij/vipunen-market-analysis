# Command-Line Interface Guide

The Vipunen project provides a command-line interface (`run_analysis.py` wrapping `src/vipunen/cli/analyze_cli.py`) for easy analysis of education market data without writing code. This guide explains how to use this interface effectively.

See also: [Market Analysis Features](MARKET_ANALYSIS.md), [Data Requirements](DATA_REQUIREMENTS.md)

## Basic Usage

The simplest way to run an analysis is using the `run_analysis.py` script:

```bash
python run_analysis.py --data-file <path_to_data> --institution <institution_name> --short-name <short_name>
```

For example:

```bash
python run_analysis.py --data-file data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv --institution "Rastor-instituutti ry" --short-name "RI"
```

## Available Arguments

### `run_analysis.py` Arguments

| Argument | Short Flag | Description | Default Value |
|----------|------------|-------------|---------------|
| `--data-file` | `-d` | Path to the input CSV data file | `data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv` |
| `--institution` | `-i` | Main name of the institution to analyze | `Rastor-instituutti ry` |
| `--short-name` | `-s` | Short name for the institution (used in titles and file names) | `RI` |
| `--variant` | `-v` | Name variant for the institution (can be specified multiple times) | `[]` |
| `--output-dir` | `-o` | Base directory for output files | `data/reports/` (subfolder `education_market_[short_name]` is created) |
| `--use-dummy` | `-u` | Use dummy data instead of loading from file | `False` |
| `--filter-qual-types` | N/A | Filter data to include only ammattitutkinto and erikoisammattitutkinto | `False` |

### `fetch_data.py` Arguments

This script fetches the latest data directly from the Vipunen API.

| Argument | Short Flag | Description | Default Value |
|----------|------------|-------------|---------------|
| `--dataset` | N/A | Name of the dataset to fetch from the API | Value of `api.default_dataset` in `config.yaml` |
| `--output-dir`| N/A | Directory to save the output CSV and `.metadata` file | Parent directory of `paths.data` in `config.yaml` (or `data/raw/` if not set) |
| `--force-download` | N/A | Download data even if the update date hasn't changed | `False` |

See [Data Requirements](DATA_REQUIREMENTS.md#obtaining-data-from-vipunen-api) for more details on data fetching and configuration.

## Configuration File (`config.yaml`)

While many options are available via command-line arguments, further customization, especially related to analysis thresholds and input/output mappings, is controlled via the `config.yaml` file located in the project root.

Key configuration options relevant to the analysis workflow include:

*   `columns`: Defines the mapping between expected column names (like `year`, `qualification`, `provider`, `volume`, `update_date`) and the actual column names in your input CSV file.
    *   `columns.input.update_date`: Specifies the column in the raw data containing the data update timestamp (e.g., `tietojoukkoPaivitettyPvm`), which is used in plot captions.
*   `analysis`:
    *   `min_market_size_threshold`: Minimum total market size for a qualification in the reference year(s) to be considered significant. Qualifications below this threshold (in both latest year and previous year) will be excluded **only** from the Year-over-Year market growth results (`qualification_market_yoy_growth`) to focus that analysis on more substantial markets. It does **not** filter the main `detailed_providers_market` DataFrame based on this threshold.
    *   `active_qualification_min_volume_sum`: Minimum summed volume the target institution must have across the last two full years for a qualification to be considered "active" for plot filtering (e.g., Heatmap, Line charts). Default is 3 (requiring summed volume > 2).
    *   `gainers_losers`:
        *   `min_market_share_threshold`: Optional. Minimum market share (%) a provider needs in the reference year to be included in the Gainers/Losers plot. Defaults to `null` (disabled).
        *   `min_market_rank_percentile`: Optional. Minimum market rank percentile (based on share) a provider needs in the reference year to be included in the Gainers/Losers plot. Defaults to `null` (disabled).

Always refer to the comments within `config.yaml` for the most up-to-date details on available settings.

## Institution Name Variants

Educational institutions often have variants of their name in the data. You can specify multiple name variants to ensure all data is captured:

```bash
python run_analysis.py --institution "Rastor-instituutti ry" --variant "Rastor Oy"
```

## Data Filtering Options

### Filter by Qualification Types

To analyze only ammattitutkinto and erikoisammattitutkinto qualifications:

```bash
python run_analysis.py --institution "Rastor-instituutti ry" --filter-qual-types
```

## Using Dummy Data

For testing or demonstration purposes, you can use dummy data:

```bash
python run_analysis.py --institution "Example Institute" --use-dummy
```

## Output Directory

By default, outputs are saved to `data/reports/`. A subdirectory named `education_market_[institution_short_name]` is created within this base directory to store the analysis results for the specific institution.

Within this subdirectory:
- An Excel file (e.g., `ri_market_analysis_[timestamp].xlsx`) is generated.
- A PDF file (e.g., `ri_visualizations_[timestamp].pdf`) containing all generated plots is created.

You can specify a different *base* output directory:

```bash
python run_analysis.py --institution "Rastor-instituutti ry" --output-dir "my_analysis_results"
```
This will create `my_analysis_results/education_market_ri/` etc.

## Complete Example

A complete example with multiple options:

```bash
python run_analysis.py \
  --data-file amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv \
  --institution "Rastor-instituutti ry" \
  --short-name "RI" \
  --variant "Rastor Oy" \
  --variant "Rastor" \
  --output-dir "reports/rastor_analysis_2023" \
  --filter-qual-types
```

This command will:
1. Load data from the specified CSV file
2. Analyze data for "Rastor-instituutti" and its variants
3. Use "RI" as a short name in titles and filenames
4. Save results to "reports/rastor_analysis_2023/education_market_ri/"
5. Filter to only include ammattitutkinto and erikoisammattitutkinto 