# Vipunen Analysis Notebooks

This directory contains Jupyter Notebooks for analyzing Vipunen education market data.

## Market Analysis Notebook (`vipunen_market_analysis_notebook.ipynb`)

This Jupyter Notebook provides a step-by-step workflow for analyzing the market position of a specific educational institution based on Vipunen data.

### Purpose

The notebook aims to replicate and potentially extend the analysis performed by the main `run_analysis.py` script, but in a more interactive format suitable for exploration and visualization of intermediate steps. It allows users to:

*   Load and prepare raw Vipunen data.
*   Perform market analysis for a chosen institution (calculating volumes, market shares, growth, etc.).
*   Generate visualizations (plots saved to a PDF report).
*   Export detailed results to an Excel file.
*   Provide a starting point for custom analysis using the prepared data and results.

### Configuration

The analysis is configured using the `ANALYSIS_PARAMS` dictionary defined in one of the initial code cells within the notebook. Key parameters include:

*   `data_file`: Path to the input CSV data file. Set to `None` to use the default path from `config.yaml`, or provide a specific filename (e.g., `"my_data.csv"`) relative to the `data/raw` directory or an absolute path.
*   `institution`: The **exact name** of the institution to analyze (e.g., `"AEL-Amiedu Oy"`, `"Rastor-instituutti ry"`). This name is used directly for analysis unless it matches a key under `institutions:` in `config.yaml`.
*   `institution_short_name`: An optional shorter name used for output filenames (e.g., `"AEL"`, `"RI"`). If `None`, it defaults to the full `institution` name.
*   `institution_variants`: A **list of strings** containing alternative names the institution might have in the data (e.g., `["Amiedu", "AEL"]`, `["Rastor Oy"]`). This is crucial for accurately capturing all data related to the institution, especially if its name has changed or varies in the source data. Include the main `institution` name in this list if it should also be matched directly.
*   `use_dummy`: Set to `True` to use dummy data if available (primarily for testing).
*   `filter_qual_types`: Set to `True` to filter the analysis to include only specific qualification types defined in `config.yaml` (e.g., "Ammattitutkinnot", "Erikoisammattitutkinnot").
*   `output_dir`: Base directory for saving output files (Excel, PDF report). Set to `None` to use the default path from `config.yaml` (typically `data/reports`), or provide a specific path (absolute or relative to the project root).
*   `include_timestamp`: Set to `True` (default) to include a timestamp in the output filenames.

**Example Configuration:**

```python
ANALYSIS_PARAMS = {
    'data_file': None, # Use default from config
    'institution': "Rastor-instituutti ry",
    'institution_short_name': "RI",
    'institution_variants': ["Rastor Oy"], # Match these names
    'use_dummy': False,
    'filter_qual_types': False,
    'output_dir': None, # Use default from config
    'include_timestamp': True
}
```

### Execution

1.  Ensure you have the required environment activated (see main project `README.md`).
2.  Start Jupyter Lab or Jupyter Notebook from your project's root directory.
3.  Open `notebooks/vipunen_market_analysis_notebook.ipynb`.
4.  Modify the `ANALYSIS_PARAMS` cell near the top to set your desired institution, variants, and other parameters.
5.  Run all cells sequentially from top to bottom (e.g., using "Run > Run All Cells" or by executing each cell individually).

### Outputs

The notebook generates the following outputs in the specified `output_dir` (or the default `data/reports`