# Excel Export Functionality

This document describes the Excel export functionality in the Vipunen project, which allows analysis results to be saved as formatted Excel workbooks for further review and sharing.

## Overview

The Vipunen project provides a robust Excel export system that:

1. Organizes analysis results into structured worksheets
2. Applies basic formatting for readability
3. Handles special data types and values
4. Integrates with the FileUtils package for consistent file management

## Basic Usage

The simplest way to export data to Excel is using the FileUtils integration:

```python
from src.vipunen.utils.file_utils_config import get_file_utils
from FileUtils import OutputFileType

# Get the configured FileUtils instance
file_utils = get_file_utils()

# Prepare data for Excel export (dictionary of DataFrames)
# Note: Sheet names are defined in config.yaml['excel']['sheets']
# The keys used here are placeholders for the corresponding DataFrames
excel_data = {
    "Sheet Name 1 (from config)": total_volumes_df,
    "Sheet Name 2 (from config)": volumes_by_qual_df,
    "Sheet Name 3 (from config)": detailed_providers_market_df, # Renamed key for clarity
    "Sheet Name 4 (from config)": cagr_data_df
    # Add other sheets like qual_market_yoy_growth_df if needed
}

# Export to Excel
excel_path = file_utils.save_data_to_storage(
    data=excel_data,
    file_name="institution_market_analysis",
    output_type="reports",
    output_filetype=OutputFileType.XLSX,
    index=False
)
```

## Export Helper Function

For more convenient Excel export, the project includes a helper function in the `analyze_education_market.py` script:

```python
def export_to_excel(data_dict, file_name, output_type="reports", **kwargs):
    """
    Export multiple DataFrames to Excel using FileUtils directly.
    
    Args:
        data_dict: Dictionary mapping sheet names to DataFrames
        file_name: Base name for the output file
        output_type: Output directory type
        **kwargs: Additional arguments for Excel export
        
    Returns:
        Path: Path to the saved Excel file
    """
    # Filter out empty DataFrames
    filtered_data = {
        sheet_name: df for sheet_name, df in data_dict.items() 
        if isinstance(df, pd.DataFrame) and not df.empty
    }
    
    # Use FileUtils to save the Excel file
    file_utils = get_file_utils()
    path_result = file_utils.save_data_to_storage(
        data=filtered_data,
        file_name=file_name,
        output_type=output_type,
        output_filetype=OutputFileType.XLSX,
        index=False,
        **kwargs
    )
    
    return Path(path_result)
```

## Standard Excel Worksheets

The standard Excel export, generated using the results from `MarketAnalyzer.analyze()` (see [Market Analysis Features](MARKET_ANALYSIS.md) for details on filtering), includes the following worksheets.

**Important Note:** The default implementation of `export_analysis_results` in `analyze_cli.py` currently only maps the core analysis results (`total_volumes`, `volumes_by_qualification`, `detailed_providers_market`, `qualification_cagr`) to the first four sheet names defined in `config.yaml['excel']['sheets']`.
To include additional results like `bcg_data` or `provider_counts_by_year` in the Excel output, you would need to either:
*   Modify the mapping logic within the `export_analysis_results` function in `src/vipunen/cli/analyze_cli.py`.
*   Ensure the `config.yaml` sheet configuration (`excel.sheets`) explicitly lists and maps these additional result keys to desired sheet names, and update the mapping logic in `export_analysis_results` accordingly.

The actual names of these worksheets are defined in `config.yaml` under `excel.sheets`. The names listed below describe the *content* of each sheet typically exported in the default configuration.

| Default Worksheet Name     | Content                                                                                                   | Filtering Applied (based on `analyze()` logic) |
| :------------------------- | :-------------------------------------------------------------------------------------------------------- | :--------------------------------------------- |
| **NOM Yhteensä**           | Institution's total volumes by year, broken down by provider vs. subcontractor roles.                       | None                                           |
| **Oppilaitoksen NOM tutkinnoittain** | Student volumes for the target institution for each qualification by year.                                  | None (shows all qualifications institution participated in) |
| **Oppilaitoksen koko markkina** | Comprehensive market data (shares, ranks, volumes) for **all providers** across **all years** for qualifications relevant to the target institution. | Rows with `Total Volume == 0` removed. **Not** filtered by low market size or institution inactivity. |
| **CAGR analyysi**          | Detailed qualification history based *only* on the target institution's volumes, including CAGR calculation. | None (shows all qualifications institution ever offered with sufficient data) |
| *(Potentially) **Qual Market YoY Growth** * | Year-over-Year growth (%) of the *total market size* for each qualification.                              | **Filtered** to exclude qualifications identified in `analyze()` as low volume OR inactive for the target institution. (**Note:** Not exported by default) |
| *(Potentially) **BCG Data** *         | Data for BCG plot (Market Growth, Relative Share, Volume).                                                | Implicitly filtered by latest year data. (**Note:** Not exported by default) |
| *(Potentially) **Provider Counts** *  | Yearly count of unique providers/subcontractors in institution's markets.                             | None. (**Note:** Not exported by default) |
| *(Other sheets may be present depending on the analysis results and configuration)* |

## Example Output Structure (Column Names)

Column names within each sheet are also configurable via `config.yaml` under `columns.output`. The structure below shows the *default Finnish* column names for each sheet content type:

1.  **Total Volumes Sheet Content**:
    *   `Vuosi`
    *   `NOM järjestäjänä`
    *   `NOM hankintana`
    *   `NOM yhteensä`

2.  **Volumes by Qualification Sheet Content**:
    *   `Vuosi`
    *   `Tutkinto`
    *   `NOM järjestäjänä`
    *   `NOM hankintana`
    *   `NOM yhteensä`
    *   `Markkina yhteensä`
    *   `Markkinaosuus (%)`

3.  **Detailed Provider Market Sheet Content**:
    *   `Vuosi`
    *   `Tutkinto`
    *   `Oppilaitos`
    *   `NOM järjestäjänä`
    *   `NOM hankintana`
    *   `NOM yhteensä`
    *   `Markkina yhteensä`
    *   `Markkinaosuus (%)`
    *   `Sijoitus markkinaosuuden mukaan`
    *   `Markkinaosuuden kasvu (%)`
    *   `Sijoitus markkinaosuuden kasvun mukaan`

4.  **CAGR Analysis Sheet Content**:
    *   `Tutkinto`
    *   `CAGR (%)`
    *   `Aloitusvuosi`
    *   `Viimeinen vuosi`
    *   `Aloitusvuode volyymi`
    *   `Viimeisen vuoden volyymi`
    *   `Vuosia datassa`

5.  **(Potential) Qual Market YoY Growth Sheet Content**:
    *   `Tutkinto`
    *   `Vuosi`
    *   `Markkina yhteensä`
    *   `Markkina yhteensä YoY Growth (%)`

6.  **(Potential) BCG Data Sheet Content**:
    *   `Tutkinto`
    *   `Market Growth (%)`
    *   `Relative Market Share`
    *   `Institution Volume`

7.  **(Potential) Provider Counts Sheet Content**:
    *   `Vuosi`
    *   `Unique_Providers_Count` (or config name)
    *   `Unique_Subcontractors_Count` (or config name)

## Export File Path Structure

The Excel files are saved using the FileUtils directory structure:

```
data/\
└── reports/\
    └── education_market_[institution_short_name]/\
        └── [institution_short_name]_market_analysis_[timestamp].xlsx\
```

The timestamp format in the filename is configurable through the FileUtils settings.

## Handling Special Values

The export system automatically handles special values that might cause Excel export issues:

1. **Infinite values**: Replaced with NaN to avoid Excel errors
2. **NaN values**: Retained as empty cells in Excel
3. **Large numbers**: Formatted to maintain precision

## Customizing Excel Output

For advanced customization, you can pass additional parameters to the export function:

```python
export_analysis_results(\
    analysis_results, \
    config, \
    institution_short_name, \
    base_output_path,\
    # Additional kwargs for FileUtils.save_data_to_storage\
    index=False,\
    float_format="%.2f",  # Format floating-point numbers
    freeze_panes=(1, 0)   # Freeze header row
)\
```

## Troubleshooting Excel Export

Common issues and solutions:

1. **Missing FileUtils package**: Ensure FileUtils is installed (`pip install FileUtils`)
2. **Export path errors**: Check that FileUtils is properly configured with valid directories
3. **Data format errors**: Ensure DataFrames don't contain problematic data types
4. **Large file warnings**: For very large datasets, consider splitting the export into multiple files
5. **Sheet Name/Key Mismatches**: Ensure the sheet names defined in `config.yaml` correspond correctly to the keys in the `analysis_results` dictionary if customizing the export logic.