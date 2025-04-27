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
excel_data = {
    "Total Volumes": total_volumes_df,
    "Volumes by Qualification": volumes_by_qual_df,
    "Provider's Market": qual_growth_df,
    "CAGR Analysis": cagr_data_df
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

The standard Excel export, generated using the results from `MarketAnalyzer.analyze()` (see [Market Analysis Features](MARKET_ANALYSIS.md) for details on filtering), includes the following worksheets:

| Worksheet Name             | Content                                                                                                   | Filtering Applied (based on `analyze()` logic) |
| :------------------------- | :-------------------------------------------------------------------------------------------------------- | :--------------------------------------------- |
| **Total Volumes**          | Institution's total volumes by year, broken down by provider vs. subcontractor roles.                       | None                                           |
| **Volumes by Qualification** | Student volumes for the target institution for each qualification by year.                                  | None (shows all qualifications institution participated in) |
| **Detailed Provider Market** | Comprehensive market data (shares, ranks, volumes) for **all providers** across **all years** for qualifications relevant to the target institution. | Rows with `Total Volume == 0` removed. **Not** filtered by low market size or institution inactivity. |
| **CAGR Analysis**          | Detailed qualification history based *only* on the target institution's volumes, including CAGR calculation. | None (shows all qualifications institution ever offered with sufficient data) |
| **Qual Market YoY Growth** | Year-over-Year growth (%) of the *total market size* for each qualification.                              | **Filtered** to exclude qualifications identified in `analyze()` as low volume OR inactive for the target institution. |
| *(Other sheets like `market_shares` or `overall_total_market_volume` may also be present depending on the `analyze()` results)* |

## Example Output Structure

A typical Excel export contains:

1.  **Total Volumes** worksheet:
    *   Year
    *   Total Volume
    *   Volume as Provider
    *   Volume as Subcontractor
    *   Year-over-Year Growth

2.  **Volumes by Qualification** worksheet:
    *   Year
    *   Qualification
    *   Provider Amount
    *   Subcontractor Amount
    *   Total Amount
    *   Market Total
    *   Market Share (%)

3.  **Detailed Provider Market** worksheet (Formerly 'Provider's Market'):
    *   Year
    *   Qualification
    *   Provider
    *   Provider Amount
    *   Subcontractor Amount
    *   Total Volume (Provider's total volume for that qual/year)
    *   Market Total (Qualification's total market volume for that year)
    *   Market Share (%)
    *   Market Rank
    *   Market Share Growth (%) (YoY change in provider's market share)
    *   Market Gainer Rank

4.  **CAGR Analysis** worksheet:
    *   Qualification
    *   CAGR (%)
    *   First Year
    *   Last Year
    *   First Year Volume
    *   Last Year Volume
    *   Years Present

5.  **Qual Market YoY Growth** worksheet:
    *   Qualification
    *   Year
    *   Market Total
    *   Market Total YoY Growth (%) (YoY growth of the total market for the qualification)

## Export File Path Structure

The Excel files are saved using the FileUtils directory structure:

```
data/
└── reports/
    └── [institution_short_name]_education_market_analysis_[timestamp].xlsx
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
export_to_excel(
    excel_data,
    f"{institution_short_name}_market_analysis",
    output_type="reports",
    index=False,
    float_format="%.2f",  # Format floating-point numbers
    freeze_panes=(1, 0)   # Freeze header row
)
```

## Troubleshooting Excel Export

Common issues and solutions:

1. **Missing FileUtils package**: Ensure FileUtils is installed (`pip install FileUtils`)
2. **Export path errors**: Check that FileUtils is properly configured with valid directories
3. **Data format errors**: Ensure DataFrames don't contain problematic data types
4. **Large file warnings**: For very large datasets, consider splitting the export into multiple files 