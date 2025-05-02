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

Place your data file in the default directory (`data/raw/`) or note its path.

If you don't have data, you can fetch the latest data from the Vipunen API:
```bash
python src/scripts/fetch_data.py
```
This will download the data to `data/raw/` (or as configured in `config.yaml`). See [Data Requirements](DATA_REQUIREMENTS.md) for details.

Alternatively, you can use dummy data for testing (see next step).

## Step 3: Run a Basic Analysis

The quickest way to run an analysis is using the command-line interface:

```bash
# Using real data (assuming data is in default path and default institution in config is desired)
python run_analysis.py

# Specify institution and data file
python run_analysis.py --data-file path/to/your_data_file.csv --institution "Your Institution" --short-name "YI"

# Using dummy data
python run_analysis.py --use-dummy --institution "Example Institute" --short-name "EI"
```

This will generate an Excel file and a PDF report with visualizations in the `data/reports/education_market_[short_name]/` directory.

## Step 4: Customize the Analysis Programmatically

For more control, create a Python script (`custom_analysis.py`) and use the core `MarketAnalyzer`:

```python
# custom_analysis.py
import pandas as pd
import logging
from pathlib import Path
from src.vipunen.data.data_loader import load_data
from src.vipunen.data.data_processor import clean_and_prepare_data
# Use the core MarketAnalyzer
from src.vipunen.analysis.market_analyzer import MarketAnalyzer 
# Use the standard Excel export helper
from src.vipunen.export.excel_exporter import export_to_excel 
# Import config loader
from src.vipunen.config.config_loader import get_config
# Import visualizer
from src.vipunen.visualization.education_visualizer import EducationVisualizer
import matplotlib.pyplot as plt

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
output_base_dir = config['paths']['output']
# Create institution-specific output dir path
output_dir_path = Path(output_base_dir) / f"education_market_{institution_short_name.lower()}"
output_dir_path.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

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
logger.info(f"Analysis complete. Results keys: {list(analysis_results.keys())}")

# 7. Export to Excel (using sheet names from config)
logger.info("Preparing data for Excel export...")
excel_data = {}
sheet_configs = config.get('excel', {}).get('sheets', [])
# Include all relevant keys from analyze() output
analysis_keys = ['total_volumes', 'volumes_by_qualification', 'detailed_providers_market', 'qualification_cagr', 'qualification_market_yoy_growth', 'provider_counts_by_year']
num_sheets = min(len(sheet_configs), len(analysis_keys)) # Map based on shorter list

for i in range(num_sheets):
    sheet_name = sheet_configs[i].get('name', f'Sheet{i+1}')
    analysis_key = analysis_keys[i]
    # Ensure we handle potential non-DataFrame results (like overall_total_market_volume Series)
    result_data = analysis_results.get(analysis_key)
    if isinstance(result_data, pd.DataFrame):
        excel_data[sheet_name] = result_data.reset_index(drop=True)
    elif isinstance(result_data, pd.Series):
        # Convert Series to DataFrame for Excel export
        excel_data[sheet_name] = result_data.reset_index()
    else:
        excel_data[sheet_name] = pd.DataFrame() # Add empty if missing or wrong type

# Ensure the output directory path is a string for the exporter
output_dir_str = str(output_dir_path)

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
     
# 8. Generate and Save Visualizations (Example)
logger.info("Generating example visualizations...")
visualizer = EducationVisualizer(output_dir=output_dir_path, output_format='pdf')

# Get required dataframes from results
volume_df = analysis_results.get('total_volumes')
count_df = analysis_results.get('provider_counts_by_year')

# Get column names from config
year_col_name = config['columns']['output']['year']
provider_amount_col_name = config['columns']['output']['provider_amount']
subcontractor_amount_col_name = config['columns']['output']['subcontractor_amount']
provider_count_col_name = config['columns']['output'].get('unique_providers_count', 'Unique_Providers_Count')
subcontractor_count_col_name = config['columns']['output'].get('unique_subcontractors_count', 'Unique_Subcontractors_Count')

# Generate the volume/count plot
if volume_df is not None and count_df is not None:
    try:
        fig, _ = visualizer.create_volume_and_provider_count_plot(
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
        visualizer.save_visualization(fig, f"{institution_short_name}_example_plot")
        plt.close(fig)
        logger.info("Generated and saved example volume/provider count plot.")
    except Exception as plot_err:
        logger.error(f"Failed to generate example plot: {plot_err}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)

# Close the visualizer if PDF output was used
visualizer.close_pdf()
logger.info("Custom analysis script finished.")

```

Run your custom script:

```bash
python custom_analysis.py
```

## Step 5: Analyze the Results

Open the generated Excel file (`*_custom_analysis.xlsx`) and the PDF file (`*_visualizations.pdf`) in the `data/reports/education_market_[short_name]/` directory to analyze the results.

The Excel file contains several worksheets:

1. **Total Volumes**: Shows your institution's yearly student volumes
2. **Volumes by Qualification**: Detailed volume breakdown by qualification
3. **Provider's Market**: Market share and competitive analysis
4. **CAGR Analysis**: Long-term growth trends for qualifications
5. **Institution Roles**: Analysis of your institution's roles as provider and subcontractor

## Step 6: Visualize the Data (Optional)

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