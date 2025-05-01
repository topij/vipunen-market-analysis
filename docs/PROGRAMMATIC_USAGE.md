# Programmatic Usage Guide

This guide demonstrates how to use the Vipunen library programmatically for customized education market analysis. By directly importing the modules, you have complete control over the analysis workflow.

## Basic Import Structure

Start by importing the necessary modules:

```python
# Core data handling
from src.vipunen.data.data_loader import load_data
from src.vipunen.data.data_processor import clean_and_prepare_data, shorten_qualification_names

# Analysis modules
from src.vipunen.analysis.market_share_analyzer import calculate_market_shares, calculate_market_share_changes
from src.vipunen.analysis.qualification_analyzer import analyze_qualification_growth, calculate_cagr_for_groups

# Visualization (optional)
from src.vipunen.visualization.volume_plots import plot_total_volumes
from src.vipunen.visualization.market_plots import plot_market_share_heatmap

# Export functionality
from src.vipunen.export.excel_exporter import ExcelExporter

# FileUtils integration
from src.vipunen.utils.file_utils_config import get_file_utils
```

## Complete Analysis Workflow

Here's a complete workflow example that demonstrates the key components of the analysis:

```python
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 1. Define parameters
data_file = "data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv"
institution_name = "Rastor-instituutti ry"
institution_variants = ["Rastor-instituutti", "Rastor", "Rastor Oy"]
institution_short_name = "RI"
output_dir = "data/reports"
filter_qual_types = True

# 2. Load and prepare data
from src.vipunen.data.data_loader import load_data
from src.vipunen.data.data_processor import clean_and_prepare_data

raw_data = load_data(data_file)
logger.info(f"Loaded raw data with {len(raw_data)} rows")

df_clean = clean_and_prepare_data(raw_data, institution_names=institution_variants)
logger.info(f"Cleaned data has {len(df_clean)} rows")

# 3. Apply optional filters
if filter_qual_types:
    df_filtered = df_clean[df_clean["tutkintotyyppi"].isin(["Ammattitutkinnot", "Erikoisammattitutkinnot"])]
    logger.info(f"Filtered data has {len(df_filtered)} rows")
else:
    df_filtered = df_clean

# 4. Use the EducationMarketAnalyzer for comprehensive analysis
from src.vipunen.analysis.education_market import EducationMarketAnalyzer

analyzer = EducationMarketAnalyzer(
    data=df_filtered,
    institution_names=institution_variants,
    filter_degree_types=filter_qual_types
)

# 5. Perform various analyses
total_volumes = analyzer.analyze_total_volume()
logger.info(f"Analyzed total volumes: {len(total_volumes)} rows")

volumes_by_qual = analyzer.analyze_volumes_by_qualification()
logger.info(f"Analyzed volumes by qualification: {len(volumes_by_qual)} rows")

institution_roles = analyzer.analyze_institution_roles()
logger.info(f"Analyzed institution roles: {len(institution_roles)} rows")

# 6. Calculate growth metrics
qual_growth = analyzer.analyze_qualification_growth()
logger.info(f"Analyzed qualification growth: {len(qual_growth)} rows")

# Calculate CAGR for qualifications
all_years = df_filtered['tilastovuosi'].unique()
cagr_data = analyzer.calculate_qualification_cagr(
    start_year=min(all_years),
    end_year=max(all_years)
)
logger.info(f"Calculated CAGR for {len(cagr_data)} qualifications")

# 7. Export results to Excel
from src.vipunen.utils.file_utils_config import get_file_utils
from FileUtils import OutputFileType

file_utils = get_file_utils()

# Prepare data for Excel export
excel_data = {
    "Total Volumes": total_volumes,
    "Volumes by Qualification": volumes_by_qual,
    "Provider's Market": qual_growth,
    "CAGR Analysis": cagr_data,
    "Institution Roles": institution_roles
}

# Export to Excel
excel_path = file_utils.save_data_to_storage(
    data=excel_data,
    file_name=f"{institution_short_name}_education_market_analysis",
    output_type="reports",
    output_filetype=OutputFileType.XLSX,
    index=False
)
logger.info(f"Exported Excel file to {excel_path}")
```

## Working with Market Share Analysis

To perform a focused market share analysis:

```python
from src.vipunen.analysis.market_share_analyzer import calculate_market_shares

# Calculate market shares for a specific institution
market_shares = calculate_market_shares(
    df=df_filtered,
    institution_variants=institution_variants,
    year_col='tilastovuosi',
    qualification_col='tutkinto',
    provider_col='koulutuksenJarjestaja',
    subcontractor_col='hankintakoulutuksenJarjestaja',
    volume_col='nettoopiskelijamaaraLkm'
)

# Get the top providers for each qualification
top_providers = {}
for qualification in market_shares['tutkinto'].unique():
    qual_data = market_shares[market_shares['tutkinto'] == qualification]
    top_providers[qualification] = qual_data.sort_values('Total Volume', ascending=False)['kouluttaja'].head(5).tolist()

print(top_providers)
```

## Calculating CAGR for Qualifications

To calculate Compound Annual Growth Rate for qualifications:

```python
from src.vipunen.analysis.qualification_analyzer import calculate_cagr_for_groups

# Get all available years
all_years = df_filtered['tilastovuosi'].unique()
start_year = min(all_years)
end_year = max(all_years)

# Group data by qualification
grouped_data = df_filtered.groupby(['tutkinto', 'tilastovuosi'])['nettoopiskelijamaaraLkm'].sum().reset_index()

# Calculate CAGR for each qualification
cagr_results = calculate_cagr_for_groups(
    df=grouped_data,
    group_col='tutkinto',
    year_col='tilastovuosi',
    value_col='nettoopiskelijamaaraLkm',
    start_year=start_year,
    end_year=end_year
)

# Sort qualifications by CAGR to identify fastest-growing ones
fastest_growing = cagr_results.sort_values('cagr', ascending=False)
print(fastest_growing.head(10))
```

## Working with Dummy Data

For testing or demo purposes, you can use the dummy data generator:

```python
from src.vipunen.data.dummy_generator import create_dummy_dataset

# Create a dummy dataset with configurable parameters
dummy_data = create_dummy_dataset(
    num_years=5,
    num_qualifications=20,
    num_providers=15,
    start_year=2018
)

# Proceed with analysis using the dummy data
df_clean = clean_and_prepare_data(dummy_data, institution_names=["Example Institute"])

# Continue with analysis as normal
analyzer = EducationMarketAnalyzer(
    data=df_clean,
    institution_names=["Example Institute"]
)
```

## Customizing FileUtils Integration

To customize the FileUtils configuration:

```python
from src.vipunen.utils.file_utils_config import configure_file_utils
from FileUtils import FileUtils

# Create a custom configuration
custom_config = {
    "base_dir": "/custom/path/to/data",
    "raw_dir": "raw_inputs",
    "processed_dir": "processed_data",
    "reports_dir": "analysis_outputs"
}

# Configure FileUtils with custom settings
file_utils = configure_file_utils(custom_config)

# Use the custom configuration for data loading
raw_data = file_utils.load_data_from_storage(
    file_path="your_data_file.csv",
    input_type="raw"
)
```

## New Example Code

```python
from src.vipunen.data.data_processor import clean_and_prepare_data
from src.vipunen.analysis.market_analyzer import MarketAnalyzer
from src.vipunen.export.excel_exporter import export_to_excel # Example exporter
# Import config loader
from src.vipunen.config.config_loader import get_config

# Configuration (example)
data_path = "amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv"
institution_name = "Rastor-instituutti ry"
institution_variants = ["Rastor-instituutti ry", "Rastor Oy"]
institution_short_name = "RI"
min_market_size_threshold = 5
output_dir = "data/reports/education_market_ri"

# Load config
config = get_config()

# Load and prepare data
raw_data = load_data(data_path)
processed_data = clean_and_prepare_data(raw_data) # Apply cleaning as needed

# Initialize analyzer, passing the config
analyzer = MarketAnalyzer(processed_data, cfg=config)
analyzer.institution_names = institution_variants
analyzer.institution_short_name = institution_short_name

# Run analysis
analysis_results = analyzer.analyze(min_market_size_threshold=min_market_size_threshold)

# Export (example)
# Note: Sheet names and column names in the output are controlled by config.yaml
excel_data = {
    config['excel']['sheets'][0]['name']: analysis_results.get('total_volumes'),
    config['excel']['sheets'][1]['name']: analysis_results.get('volumes_by_qualification'),
    config['excel']['sheets'][2]['name']: analysis_results.get('detailed_providers_market'),
    config['excel']['sheets'][3]['name']: analysis_results.get('qualification_cagr')
}

excel_file = export_to_excel(
    excel_data,
    f"{institution_short_name}_analysis",
    output_dir=output_dir
)
print(f"Exported results to {excel_file}")
``` 