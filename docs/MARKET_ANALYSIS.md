# Market Analysis Features

This document provides an overview of the market analysis features available in the Vipunen project. These features help educational institutions understand their position in the Finnish vocational education market.

## Core Analysis Capabilities

The Vipunen project provides several core analysis capabilities:

1. **Volume Analysis**: Track student volumes across years and qualifications
2. **Market Share Analysis**: Calculate market share percentages for providers in each qualification
3. **Growth Analysis**: Identify growing and declining qualifications and markets
4. **Compound Annual Growth Rate (CAGR)**: Calculate long-term growth rates for qualifications
5. **Provider Ranking**: Identify top providers in each qualification market

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