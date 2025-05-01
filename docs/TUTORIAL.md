# Vipunen Tutorial

This tutorial will guide you through using the Vipunen project to analyze Finnish vocational education market data, from setup to generating insights.

## Prerequisites

Before starting this tutorial, ensure you have:

1. Python 3.8 or newer installed
2. Git (for cloning the repository)
3. Basic knowledge of Python and pandas
4. Sample data file (or use the dummy data generator)

## Step 1: Setup and Installation

First, clone the repository and set up the environment:

```bash
# Clone the repository
git clone <repository-url>
cd vipunen-project

# Create and activate the conda environment
conda env create -f environment.yaml
conda activate vipunen-analytics
```

If you prefer not to use conda, you can install the dependencies manually:

```bash
pip install pandas numpy matplotlib seaborn pathlib PyYAML FileUtils
```

## Step 2: Prepare Your Data

Place your data file in the appropriate directory:

```bash
mkdir -p data/raw
# Copy your data file to data/raw/
cp your_data_file.csv data/raw/
```

If you don't have data, you can use the dummy data generator in the next step.

## Step 3: Run a Basic Analysis

The quickest way to run an analysis is using the command-line interface:

```bash
# Using real data
python run_analysis.py --data-file data/raw/your_data_file.csv --institution "Your Institution" --short-name "YI"

# Using dummy data
python run_analysis.py --use-dummy --institution "Example Institute" --short-name "EI"
```

This will generate an Excel file with analysis results in the `data/reports` directory.

## Step 4: Customize the Analysis

For more control, you can create a custom script:

```python
# custom_analysis.py
import pandas as pd
import logging
from src.vipunen.data.data_loader import load_data
from src.vipunen.data.data_processor import clean_and_prepare_data
# Use the core MarketAnalyzer
from src.vipunen.analysis.market_analyzer import MarketAnalyzer 
# Use the standard Excel export helper
from src.vipunen.export.excel_exporter import export_to_excel 
# Import config loader
from src.vipunen.config.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Load Configuration
config = get_config()

# 2. Define parameters (can be overridden or taken from config)
data_file = config['paths']['data']
# Use default institution from config for this example
institution_key = 'default' 
institution_variants = config['institutions'][institution_key].get('variants', []) + [config['institutions'][institution_key]['name']]
institution_short_name = config['institutions'][institution_key]['short_name']
qual_types_to_filter = config.get('qualification_types', []) # Example: use types from config
min_market_size = config.get('analysis', {}).get('min_market_size_threshold', 5)
output_dir = Path(config['paths']['output']) / f"education_market_{institution_short_name.lower()}"
output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

# 3. Load data
try:
    raw_data = load_data(data_file)
    logger.info(f"Loaded {len(raw_data)} rows from {data_file}")
except FileNotFoundError:
    logger.error(f"Data file not found: {data_file}")
    raise

# 4. Clean and prepare data
df_clean = clean_and_prepare_data(raw_data, institution_names=institution_variants)
logger.info(f"Cleaned data has {len(df_clean)} rows")

# 5. Filter data (Example: by qualification type from config)
if qual_types_to_filter:
    df_filtered = df_clean[df_clean["tutkintotyyppi"].isin(qual_types_to_filter)]
    logger.info(f"Filtered data to types {qual_types_to_filter}. Rows remaining: {len(df_filtered)}")
else:
    df_filtered = df_clean
    logger.info("No qualification type filter applied.")
    
# 6. Create and run analyzer
logger.info("Initializing MarketAnalyzer...")
analyzer = MarketAnalyzer(df_filtered, cfg=config) # Pass config
analyzer.institution_names = institution_variants
analyzer.institution_short_name = institution_short_name

logger.info("Running analysis...")
analysis_results = analyzer.analyze(min_market_size_threshold=min_market_size)
logger.info("Analysis complete.")

# 7. Export to Excel (using sheet names from config)
logger.info("Preparing data for Excel export...")
excel_data = {}
sheet_configs = config.get('excel', {}).get('sheets', [])
analysis_keys = ['total_volumes', 'volumes_by_qualification', 'detailed_providers_market', 'qualification_cagr', 'qualification_market_yoy_growth'] # Match analyze() output

if len(sheet_configs) != len(analysis_keys):
     logger.warning(f"Config sheet count mismatch. Adjusting export based on available analysis keys and config sheets.")
     # Simple mapping: Use first N sheet names for first N results
     num_sheets = min(len(sheet_configs), len(analysis_keys))
     for i in range(num_sheets):
         sheet_name = sheet_configs[i].get('name', f'Sheet{i+1}')
         analysis_key = analysis_keys[i]
         excel_data[sheet_name] = analysis_results.get(analysis_key, pd.DataFrame()).reset_index(drop=True)
else:
     # Assumed 1:1 mapping based on order
     for i, sheet_info in enumerate(sheet_configs):
         sheet_name = sheet_info.get('name', f'Sheet{i+1}') 
         analysis_key = analysis_keys[i]
         excel_data[sheet_name] = analysis_results.get(analysis_key, pd.DataFrame()).reset_index(drop=True)

# Ensure the output directory path is a string for the exporter
output_dir_str = str(output_dir)

try:
    excel_path = export_to_excel(
        data_dict=excel_data,
        file_name=f"{institution_short_name}_custom_analysis",
        output_dir=output_dir_str, # Pass output directory string
        include_timestamp=True
    )
    logger.info(f"Exported Excel file to {excel_path}")
except Exception as export_err:
     logger.error(f"Failed to export Excel: {export_err}", exc_info=True)
```

Run your custom script:

```bash
python custom_analysis.py
```

## Step 5: Analyze the Results

Open the generated Excel file to analyze the results. The file contains several worksheets:

1. **Total Volumes**: Shows your institution's yearly student volumes
2. **Volumes by Qualification**: Detailed volume breakdown by qualification
3. **Provider's Market**: Market share and competitive analysis
4. **CAGR Analysis**: Long-term growth trends for qualifications
5. **Institution Roles**: Analysis of your institution's roles as provider and subcontractor

## Step 6: Visualize the Data (Optional)

If you want to generate visualizations:

```python
# Add to your custom script
from src.vipunen.visualization.volume_plots import plot_total_volumes
from src.vipunen.visualization.market_plots import plot_market_share_heatmap

# Create visualizations
plot_total_volumes(
    volumes_df=total_volumes,
    institution_short_name=institution_short_name,
    output_dir="data/reports/figures"
)

plot_market_share_heatmap(
    market_shares_df=qual_growth,
    institution_name=institution_variants[0],
    output_dir="data/reports/figures"
)
```

## Step 7: Interpreting the Results

Here's how to interpret key metrics in the analysis:

1. **Market Share**: Percentage of students your institution has in a qualification market. Higher values indicate stronger market position.
2. **Market Rank**: Your institution's position in the market (1st, 2nd, etc.).
3. **CAGR (Compound Annual Growth Rate)**: Average annual growth rate over the time period. Positive values indicate growth, negative values indicate decline.
4. **Growth Trend**: Classification of qualifications as "Growing" or "Declining" based on recent data.

## Step 8: Extending the Analysis

To extend the analysis for your specific needs:

1. **Filter for specific qualifications**:
   ```python
   # Filter for specific qualifications
   specific_quals = ["Liiketoiminnan ammattitutkinto", "Johtamisen erikoisammattitutkinto"]
   df_specific = df_filtered[df_filtered["tutkinto"].isin(specific_quals)]
   
   # Create a new analyzer with the filtered data
   specific_analyzer = EducationMarketAnalyzer(
       data=df_specific,
       institution_names=institution_variants
   )
   ```

2. **Focus on a specific time period**:
   ```python
   # Filter for a specific time period
   recent_years = [2020, 2021, 2022]
   df_recent = df_filtered[df_filtered["tilastovuosi"].isin(recent_years)]
   
   # Create a new analyzer with the filtered data
   recent_analyzer = EducationMarketAnalyzer(
       data=df_recent,
       institution_names=institution_variants
   )
   ```

## Troubleshooting

If you encounter issues:

1. **Missing data errors**: Ensure your data file follows the required format (see [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md))
2. **FileUtils errors**: Check that the FileUtils package is installed and properly configured
3. **Empty results**: Verify that the institution names match exactly what's in your data file
4. **Import errors**: Ensure you're running the code from the project root directory

For more detailed information, refer to the other documentation files in the `docs/` directory. 