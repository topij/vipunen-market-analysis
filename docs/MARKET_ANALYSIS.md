# Market Analysis Features

This document provides an overview of the market analysis features available in the Vipunen project, primarily through the `MarketAnalyzer` class. These features help educational institutions understand their position in the Finnish vocational education market.

## Core Analysis Capabilities

The `MarketAnalyzer` class provides several core analysis capabilities:

1.  **Volume Analysis**: Track overall student volumes and volumes per qualification for the target institution.
2.  **Market Share Analysis**: Calculate market shares, growth, and rankings for all providers within qualifications relevant to the target institution (`detailed_providers_market`). It also provides a simplified year-over-year market share overview (`market_shares`). See [Excel Export](EXCEL_EXPORT.md) for sheet details.
3.  **Growth Analysis**: Analyze year-over-year growth trends for the total market size of each qualification (`qualification_market_yoy_growth`).
4.  **Compound Annual Growth Rate (CAGR)**: Calculate the long-term growth rate (CAGR) for qualifications based *only* on the target institution's historical volume (`qualification_cagr`).
5.  **Filtering**: Identifies qualifications with very low total market volume or where the target institution has become inactive, allowing for selective filtering of results.

## MarketAnalyzer Class

The primary class for performing the analysis is `MarketAnalyzer` located in `src/vipunen/analysis/market_analyzer.py`.

```python
from src.vipunen.analysis.market_analyzer import MarketAnalyzer

# Assuming 'processed_data' is the cleaned DataFrame
analyzer = MarketAnalyzer(processed_data)

# Set institution names (variants used for finding data)
analyzer.institution_names = institution_variants 
# Set the primary name (used for filtering checks and logging)
analyzer.institution_short_name = institution_short_name 

# Perform the analysis
analysis_results = analyzer.analyze(min_market_size_threshold=5) 
```

### `analyze()` Method

The main method is `analyze()`, which orchestrates all calculations and returns a dictionary of DataFrames.

**Key Steps in `analyze()`:**

1.  **Initial Calculations**: Calculates various metrics like total volumes, volumes by qualification, detailed provider market shares across all years, CAGR for the institution's qualifications, overall market volume per year, and year-over-year growth for qualification markets.
2.  **Identify Qualifications for Exclusion**:
    *   **Low Total Market Volume**: Identifies qualifications where the *total market volume* (across all providers) is below `min_market_size_threshold` in *both* the last full year and the current year.
    *   **Institution Inactivity**: Identifies qualifications where the *target institution's total volume* (across all its name variants) was less than 1 (`< 1`) in *both* the last full year and the current year.
3.  **Selective Filtering**:
    *   The combined list of qualifications identified in step 2 (low volume OR inactive) is used to filter **only** the `qualification_market_yoy_growth` results. This focuses growth trend analysis on more relevant qualifications.
    *   The `detailed_providers_market` DataFrame is **not** filtered based on the combined list, preserving the historical view of all qualifications the institution participated in. However, individual rows where a provider's `Total Volume` is exactly 0 *are* removed from this specific DataFrame to clean the output.
    *   The `volumes_by_qualification` and `qualification_cagr` DataFrames remain **unfiltered** by the low volume/inactivity criteria, providing a complete historical perspective for the institution.
4.  **Return Results**: Returns a dictionary containing all calculated DataFrames (some filtered, some not).

### Output DataFrames from `analyze()`

| Key in Results Dictionary       | Description                                                                                                | Filtering Applied by `analyze()`                                  |
| :------------------------------ | :--------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------- |
| `total_volumes`                 | Total student volumes (provider + subcontractor) for the target institution by year.                         | None                                                              |
| `volumes_by_qualification`      | Student volumes for the target institution, broken down by qualification and year.                           | None (by low volume/inactivity criteria)                          |
| `market_shares`                 | Simplified Year-over-Year market share comparison for all providers (based on the two most recent years).  | None                                                              |
| `detailed_providers_market`     | Detailed market shares, volumes, ranks for **all providers** within qualifications relevant to the target institution, across **all years**. | Rows with `Total Volume == 0` removed. **Not** filtered by low market size or institution inactivity. |
| `qualification_cagr`            | Compound Annual Growth Rate for qualifications based *only* on the target institution's historical volumes.    | None                                                              |
| `overall_total_market_volume`   | Series showing the total market volume across all providers/qualifications for each year.                    | None                                                              |
| `qualification_market_yoy_growth` | Year-over-Year growth (%) of the *total market size* for each qualification.                               | **Filtered** to exclude qualifications identified as low volume OR inactive for the target institution. |

*(Note: The underlying calculation methods like `calculate_providers_market` inherently focus on qualifications relevant to the target institution based on the input data provided to the `MarketAnalyzer`.)*

## Analysis Workflow Example (using `analyze_cli.py`)

The `src/vipunen/cli/analyze_cli.py` script (see [CLI Guide](CLI_GUIDE.md)) orchestrates the typical workflow:

1.  **Load Configuration**: Reads `config/config.yaml` (or specified file).
2.  **Load and Prepare Data**: Uses `data_loader` and `data_processor` (see [Data Requirements](DATA_REQUIREMENTS.md)).
3.  **Initialize `MarketAnalyzer`**: Passes cleaned data and sets `institution_names` and `institution_short_name` from config.
4.  **Run Analysis**: Calls `analyzer.analyze(min_market_size_threshold=...)`.
5.  **Export to Excel**: Uses `export_to_excel` to save the results dictionary from `analyze()` into different sheets (see [Excel Export](EXCEL_EXPORT.md)).
6.  **Generate Visualizations**: Uses `EducationVisualizer` with the results from `analyze()` to create plots (see [Visualization Features](VISUALIZATION.md)).

```python
# Simplified flow from analyze_cli.py

# ... load config, load data, prepare data ...

# Initialize analyzer
analyzer = MarketAnalyzer(processed_data)
analyzer.institution_names = institution_variants 
analyzer.institution_short_name = institution_short_name 

# Run analysis
min_market_size = config.get('analysis', {}).get('min_market_size_threshold', 5)
analysis_results = analyzer.analyze(min_market_size_threshold=min_market_size)

# Export results
excel_data = {
    "Total Volumes": analysis_results.get('total_volumes'),
    # ... map other results ...
    "Detailed Provider Market": analysis_results.get("detailed_providers_market"),
    "CAGR Analysis": analysis_results.get('qualification_cagr'),
    "Qual Market YoY Growth": analysis_results.get('qualification_market_yoy_growth') 
}
# ... call export_to_excel ...

# Generate visualizations
visualizer = EducationVisualizer(...)
generate_visualizations(analysis_results, visualizer, analyzer, config)

```

This revised structure centralizes the analysis logic within `MarketAnalyzer.analyze`, providing consistent and selectively filtered results for downstream use in exports and visualizations.

## EducationMarketAnalyzer

The primary class for performing market analysis is `EducationMarketAnalyzer`, which provides a comprehensive set of analysis methods:

```python
from src.vipunen.analysis.education_market import EducationMarketAnalyzer

analyzer = EducationMarketAnalyzer(
    data=df_filtered,
    institution_names=institution_variants,
    filter_degree_types=True
)
```

### Key Analysis Methods

| Method | Description | Return Value |
|--------|-------------|--------------|
| `analyze_total_volume()` | Calculates total student volumes by year | DataFrame with yearly totals |
| `analyze_volumes_by_qualification()` | Analyzes volumes for each qualification | DataFrame with qualification volumes |
| `analyze_institution_roles()` | Analyzes institution's roles as provider and subcontractor | DataFrame with role breakdowns |
| `analyze_qualification_growth()` | Analyzes qualification growth trends | DataFrame with growth metrics |
| `calculate_qualification_cagr()` | Calculates CAGR for qualifications | DataFrame with CAGR values |

## Volume Analysis

Volume analysis calculates the total student volumes for an institution, breaking it down by qualifications and roles:

```python
# Get total volumes by year
total_volumes = analyzer.analyze_total_volume()

# Get volumes by qualification
volumes_by_qual = analyzer.analyze_volumes_by_qualification()
```

Example output for `total_volumes`:

| tilastovuosi | Total Volume | Volume as Provider | Volume as Subcontractor |
|--------------|--------------|-------------------|-------------------------|
| 2018 | 456.7 | 398.2 | 58.5 |
| 2019 | 489.3 | 425.6 | 63.7 |
| 2020 | 512.8 | 445.1 | 67.7 |

## Market Share Analysis

Market share analysis calculates the percentage of students an institution has in each qualification market:

```python
from src.vipunen.analysis.market_share_analyzer import calculate_market_shares

market_shares = calculate_market_shares(
    df=df_filtered,
    institution_variants=institution_variants
)
```

The market share calculations include:

1. **Total Market Volume**: Total number of students in a qualification across all providers
2. **Institution's Volume**: Number of students at the specific institution
3. **Market Share**: Percentage of the market held by the institution
4. **Market Rank**: Institution's rank in the market (1st, 2nd, etc.)
5. **Provider Count**: Total number of providers in the market

## Growth Analysis

Growth analysis identifies trends in qualification volumes and market shares:

```python
# Calculate Year-over-Year (YoY) growth
yearly_growth = analyzer.calculate_yearly_growth(
    target_column="Total Volume", 
    year=2020,
    time_window=3
)
```

Growth metrics include:

1. **Year-over-Year (YoY) Growth**: Percentage change from one year to the next
2. **Compound Annual Growth Rate (CAGR)**: Average annual growth rate over multiple years
3. **Growth Trend Classification**: Categorization of qualifications as "Growing" or "Declining"

## CAGR Calculation

CAGR provides a smoothed growth rate over multiple years, useful for identifying long-term trends:

```python
cagr_results = analyzer.calculate_qualification_cagr(
    start_year=2018,
    end_year=2022
)
```

The CAGR formula used is:

```
CAGR = (End Value / Start Value)^(1 / Number of Years) - 1
```

CAGR analysis helps identify:
- Fastest growing qualifications
- Declining qualifications that may need attention
- Stable qualifications with consistent enrollment

## Provider Ranking

Provider ranking identifies the top institutions in each qualification market:

```python
top_providers = analyzer.get_top_providers_by_qualification(
    year=2020,
    top_n=5
)
```

This returns a dictionary mapping each qualification to a list of the top providers, allowing institutions to identify:
- Market leaders in each qualification
- Competitive landscape
- Potential collaboration partners or competitors

## Market Analysis Workflow Example

A typical market analysis workflow includes:

1. **Data Preparation**:
   ```python
   from src.vipunen.data.data_loader import load_data
   from src.vipunen.data.data_processor import clean_and_prepare_data
   
   raw_data = load_data(data_file)
   df_clean = clean_and_prepare_data(raw_data, institution_names=institution_variants)
   ```

2. **Volume Analysis**:
   ```python
   analyzer = EducationMarketAnalyzer(
       data=df_clean,
       institution_names=institution_variants
   )
   
   total_volumes = analyzer.analyze_total_volume()
   volumes_by_qual = analyzer.analyze_volumes_by_qualification()
   ```

3. **Market Share Analysis**:
   ```python
   institution_roles = analyzer.analyze_institution_roles()
   ```

4. **Growth Analysis**:
   ```python
   qual_growth = analyzer.analyze_qualification_growth()
   ```

5. **CAGR Calculation**:
   ```python
   cagr_data = analyzer.calculate_qualification_cagr(
       start_year=min(years),
       end_year=max(years)
   )
   ```

6. **Export Results**:
   ```python
   excel_data = {
       "Total Volumes": total_volumes,
       "Volumes by Qualification": volumes_by_qual,
       "Provider's Market": qual_growth,
       "CAGR Analysis": cagr_data,
       "Institution Roles": institution_roles
   }
   
   export_to_excel(excel_data, f"{institution_short_name}_market_analysis")
   ```

## Specialized Analysis Functions

In addition to the `EducationMarketAnalyzer` class, the package includes specialized analysis functions:

### Market Share Functions

```python
from src.vipunen.analysis.market_share_analyzer import (
    calculate_market_shares,
    calculate_market_share_changes,
    calculate_total_volumes
)
```

### Qualification Analysis Functions

```python
from src.vipunen.analysis.qualification_analyzer import (
    analyze_qualification_growth,
    calculate_cagr_for_groups
)
```

These functions can be used directly for more customized analysis when needed. 