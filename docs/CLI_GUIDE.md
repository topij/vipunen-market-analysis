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
python run_analysis.py --data-file data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv --institution "Rastor-instituutti" --short-name "RI"
```

## Available Arguments

### `run_analysis.py` Arguments

| Argument | Short Flag | Description | Default Value |
|----------|------------|-------------|---------------|
| `--data-file` | `-d` | Path to the input CSV data file | `data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv` |
| `--institution` | `-i` | Main name of the institution to analyze | `Rastor-instituutti ry` |
| `--short-name` | `-s` | Short name for the institution (used in titles and file names) | `RI` |
| `--variant` | `-v` | Name variant for the institution (can be specified multiple times) | `[]` |
| `--output-dir` | `-o` | Base directory for output files | `data/reports/[institution_short_name]/` |
| `--use-dummy` | `-u` | Use dummy data instead of loading from file | `False` |
| `--filter-qual-types` | N/A | Filter data to include only ammattitutkinto and erikoisammattitutkinto | `False` |
| `--filter-by-institution-quals` | N/A | Filter data to include only qualifications offered by the institution | `False` |

### `fetch_data.py` Arguments

This script fetches the latest data directly from the Vipunen API.

| Argument | Short Flag | Description | Default Value |
|----------|------------|-------------|---------------|
| `--dataset` | N/A | Name of the dataset to fetch from the API | Value of `api.default_dataset` in `config.yaml` |
| `--output-dir`| N/A | Directory to save the output CSV and `.metadata` file | Parent directory of `paths.data` in `config.yaml` (or `data/raw/` if not set) |
| `--force-download` | N/A | Download data even if the update date hasn't changed | `False` |

See [Data Requirements](DATA_REQUIREMENTS.md#obtaining-data-from-vipunen-api) for more details on data fetching and configuration.

## Institution Name Variants

Educational institutions often have variants of their name in the data. You can specify multiple name variants to ensure all data is captured:

```bash
python run_analysis.py --institution "Rastor-instituutti" --variant "Rastor-instituutti ry" --variant "Rastor" --variant "Rastor Oy"
```

## Data Filtering Options

### Filter by Qualification Types

To analyze only ammattitutkinto and erikoisammattitutkinto qualifications:

```bash
python run_analysis.py --institution "Rastor-instituutti" --filter-qual-types
```

### Filter by Institution's Qualifications

To focus only on qualifications that the institution offers:

```bash
python run_analysis.py --institution "Rastor-instituutti" --filter-by-institution-quals
```

## Using Dummy Data

For testing or demonstration purposes, you can use dummy data:

```bash
python run_analysis.py --institution "Example Institute" --use-dummy
```

## Output Directory

By default, outputs are saved to `data/reports/[institution_short_name]/`. Within this directory, an Excel file (see [Excel Export](EXCEL_EXPORT.md)) and a `plots` subdirectory (see [Visualization Features](VISUALIZATION.md)) will be created.

You can specify a different *base* output directory:

```bash
python run_analysis.py --institution "Rastor-instituutti" --output-dir "my_analysis_results"
```
This will create `my_analysis_results/education_market_ri/` etc.

## Complete Example

A complete example with multiple options:

```bash
python run_analysis.py \
  --data-file data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv \
  --institution "Rastor-instituutti" \
  --short-name "RI" \
  --variant "Rastor-instituutti ry" \
  --variant "Rastor" \
  --output-dir "reports/rastor_analysis_2023" \
  --filter-qual-types \
  --filter-by-institution-quals
```

This command will:
1. Load data from the specified CSV file
2. Analyze data for "Rastor-instituutti" and its variants
3. Use "RI" as a short name in titles and filenames
4. Save results to "reports/rastor_analysis_2023"
5. Filter to only include ammattitutkinto and erikoisammattitutkinto
6. Filter to focus only on qualifications offered by Rastor-instituutti 