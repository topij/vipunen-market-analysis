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
from src.vipunen.visualization.education_visualizer import EducationVisualizer

# Export functionality
from src.vipunen.export.excel_exporter import export_to_excel

# FileUtils integration
from src.vipunen.utils.file_utils_config import get_file_utils
from FileUtils import OutputFileType

# Add config loader import
from src.vipunen.config.config_loader import get_config
```

## Complete Analysis Workflow

Here's a complete workflow example that demonstrates the key components of the analysis:

```python
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 1. Define parameters (better to load from config or args)
config = get_config() # Load config
data_file = config['paths']['data']
institution_key = 'default' # Example
institution_variants = config['institutions'][institution_key].get('variants', []) + [config['institutions'][institution_key]['name']]
institution_short_name = config['institutions'][institution_key]['short_name']
min_market_size = config.get('analysis', {}).get('min_market_size_threshold', 5)
output_base_dir = config['paths']['output']
# Create institution-specific output dir path
output_dir_path = Path(output_base_dir) / f"education_market_{institution_short_name.lower()}"
output_dir_path.mkdir(parents=True, exist_ok=True)

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

# 4. Use the MarketAnalyzer for comprehensive analysis
from src.vipunen.analysis.market_analyzer import MarketAnalyzer

analyzer = MarketAnalyzer(df_filtered, cfg=config) # Pass config
analyzer.institution_names = institution_variants
analyzer.institution_short_name = institution_short_name

# 5. Perform various analyses
analysis_results = analyzer.analyze(min_market_size_threshold=min_market_size)
logger.info(f"Analysis complete. Results keys: {list(analysis_results.keys())}")

# 6. Export results to Excel
# Use sheet names from config
excel_data = {}
sheet_configs = config.get('excel', {}).get('sheets', [])
analysis_keys = ['total_volumes', 'volumes_by_qualification', 'detailed_providers_market', 'qualification_cagr', 'qualification_market_yoy_growth', 'provider_counts_by_year']
num_sheets = min(len(sheet_configs), len(analysis_keys))
for i in range(num_sheets):
    sheet_name = sheet_configs[i].get('name', f'Sheet{i+1}')
    analysis_key = analysis_keys[i]
    excel_data[sheet_name] = analysis_results.get(analysis_key, pd.DataFrame()).reset_index(drop=True)

# Use the helper function for export, passing the Path object directly might work
excel_path = export_to_excel(
    data_dict=excel_data,
    file_name=f"{institution_short_name}_programmatic_analysis",
    output_dir=str(output_dir_path), # Pass the specific dir path as string
    include_timestamp=True
)
logger.info(f"Exported Excel file to {excel_path}")

# 7. Visualize Data (Example: Generate one plot)

visualizer = EducationVisualizer() # Initialize (no output needed for inline)

# Example: Create the volume/provider count plot
volume_df = analysis_results.get('total_volumes')
count_df = analysis_results.get('provider_counts_by_year')
year_col_name = config['columns']['output']['year']
provider_amount_col_name = config['columns']['output']['provider_amount']
subcontractor_amount_col_name = config['columns']['output']['subcontractor_amount']
provider_count_col_name = config['columns']['output'].get('unique_providers_count', 'Unique_Providers_Count')
subcontractor_count_col_name = config['columns']['output'].get('unique_subcontractors_count', 'Unique_Subcontractors_Count')

if volume_df is not None and count_df is not None:
    fig, axes = visualizer.create_volume_and_provider_count_plot(
        volume_data=volume_df,
        count_data=count_df,
        title=f"{institution_short_name}: Volume & Providers",
        volume_title="Volume",
        count_title="Providers",
        year_col=year_col_name,
        vol_provider_col=provider_amount_col_name,
        vol_subcontractor_col=subcontractor_amount_col_name,
        count_provider_col=provider_count_col_name,
        count_subcontractor_col=subcontractor_count_col_name
    )
    # In a notebook, fig would display here. To save:
    # save_path = output_dir_path / "volume_provider_plot.png"
    # visualizer.save_visualization(fig, str(save_path)) 
    # logger.info(f"Saved example plot to {save_path}")
    
    # Make sure to close the figure if not relying on notebook display
    import matplotlib.pyplot as plt
    plt.close(fig)

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