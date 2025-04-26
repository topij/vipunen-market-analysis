# %% [markdown]
# # Vipunen Analysis Example
# 
# This notebook demonstrates how to use the Vipunen library to analyze Finnish vocational education market data. It implements the same analysis that the CLI exports to Excel.

# %% [markdown]
# ## 1. Setup and Imports
# 
# First, let's import the necessary modules from the Vipunen package.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Configure matplotlib
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# %% [markdown]
# Import Vipunen-specific modules:

# %%
# Core data handling
from src.vipunen.data.data_loader import load_data
from src.vipunen.data.data_processor import clean_and_prepare_data
from src.vipunen.data.dummy_generator import create_dummy_dataset

# Analysis modules
from src.vipunen.analysis.education_market import EducationMarketAnalyzer
from src.vipunen.analysis.market_share_analyzer import calculate_market_shares
from src.vipunen.analysis.qualification_analyzer import calculate_cagr_for_groups

# Visualization modules
from src.vipunen.visualization.volume_plots import plot_total_volumes, plot_top_qualifications
from src.vipunen.visualization.market_plots import plot_market_share_heatmap
from src.vipunen.visualization.growth_plots import plot_qualification_growth

# %% [markdown]
# ## 2. Define Parameters
# 
# Set up the parameters for our analysis.

# %%
# Use dummy data for this example
use_dummy_data = True

# Institution parameters
institution_name = "Example Institute"
institution_variants = ["Example Institute", "Example Institute Oy", "Example-Institute Ltd"]
institution_short_name = "EI"

# Data parameters
data_file = "data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv"
filter_qual_types = True

# Output parameters
output_dir = "data/reports"
figures_dir = Path(output_dir) / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 3. Load and Prepare Data
# 
# Here we load the data, either using the dummy data generator or from a file.

# %%
if use_dummy_data:
    # Generate dummy data for demonstration purposes
    raw_data = create_dummy_dataset(
        num_years=5,
        num_qualifications=20,
        num_providers=15,
        start_year=2018,
        provider_name=institution_name
    )
    logger.info(f"Generated dummy dataset with {len(raw_data)} rows")
else:
    # Load data from file
    try:
        raw_data = load_data(data_file)
        logger.info(f"Loaded {len(raw_data)} rows from {data_file}")
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_file}")
        raise

# Display the first few rows
raw_data.head()

# %% [markdown]
# Clean and prepare the data:

# %%
# Clean and prepare data
df_clean = clean_and_prepare_data(raw_data, institution_names=institution_variants)
logger.info(f"Cleaned data has {len(df_clean)} rows")

# Apply filter for qualification types if requested
if filter_qual_types:
    df_filtered = df_clean[df_clean["tutkintotyyppi"].isin(["Ammattitutkinnot", "Erikoisammattitutkinnot"])]
    logger.info(f"Filtered data has {len(df_filtered)} rows")
else:
    df_filtered = df_clean

# Display the cleaned and filtered data
df_filtered.head()

# %% [markdown]
# ## 4. Analyze Data with EducationMarketAnalyzer
# 
# Create an analyzer and perform various analyses.

# %%
# Create the analyzer
analyzer = EducationMarketAnalyzer(
    data=df_filtered,
    institution_names=institution_variants,
    filter_degree_types=filter_qual_types
)

# Get available years
all_years = df_filtered['tilastovuosi'].unique()
start_year = min(all_years)
end_year = max(all_years)
logger.info(f"Analysis period: {start_year}-{end_year}")

# %% [markdown]
# ### 4.1 Total Volumes Analysis
# 
# Analyze total student volumes by year for the institution.

# %%
# Analyze total volumes
total_volumes = analyzer.analyze_total_volume()
logger.info(f"Total volumes analysis: {len(total_volumes)} rows")

# Display the results
total_volumes

# %% [markdown]
# Visualize the total volumes:

# %%
plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

# Plot stacked bar chart
volumes = total_volumes.sort_values('tilastovuosi')
bottom_vals = volumes['Volume as Provider']
years = volumes['tilastovuosi']

plt.bar(years, bottom_vals, color='#1f77b4', label='As Provider')
plt.bar(years, volumes['Volume as Subcontractor'], bottom=bottom_vals, color='#ff7f0e', label='As Subcontractor')

# Add total values as text
for i, year in enumerate(years):
    total = volumes.loc[volumes['tilastovuosi'] == year, 'Total Volume'].values[0]
    plt.text(year, total + 5, f"{total:.1f}", ha='center')

plt.title(f"{institution_short_name} Total Student Volumes by Year", fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.2 Volumes by Qualification
# 
# Analyze student volumes broken down by qualification.

# %%
# Analyze volumes by qualification
volumes_by_qual = analyzer.analyze_volumes_by_qualification()
logger.info(f"Volumes by qualification: {len(volumes_by_qual)} rows")

# Display the first few rows
volumes_by_qual.head(10)

# %% [markdown]
# Visualize the top qualifications by volume:

# %%
# Get the most recent year
recent_year = max(volumes_by_qual['tilastovuosi'])
recent_data = volumes_by_qual[volumes_by_qual['tilastovuosi'] == recent_year]

# Sort and get top 10 qualifications by volume
top_quals = recent_data.sort_values('Total Volume', ascending=False).head(10)

plt.figure(figsize=(12, 8))
ax = plt.subplot(111)

# Plot horizontal bar chart
ax.barh(top_quals['tutkinto'], top_quals['Total Volume'], color='#1f77b4')

# Add volume values as text
for i, vol in enumerate(top_quals['Total Volume']):
    ax.text(vol + 1, i, f"{vol:.1f}", va='center')

plt.title(f"Top 10 Qualifications by Volume in {recent_year}", fontsize=14)
plt.xlabel('Number of Students', fontsize=12)
plt.ylabel('Qualification', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.3 Institution Roles
# 
# Analyze the institution's roles as main provider vs. subcontractor.

# %%
# Analyze institution roles
institution_roles = analyzer.analyze_institution_roles()
logger.info(f"Institution roles analysis: {len(institution_roles)} rows")

# Display the results
institution_roles

# %% [markdown]
# Visualize the institution roles over time:

# %%
plt.figure(figsize=(12, 6))
roles_data = institution_roles.sort_values('tilastovuosi')

# Create a pie chart for each year
num_years = len(roles_data)
for i, (idx, row) in enumerate(roles_data.iterrows()):
    plt.subplot(1, num_years, i+1)
    
    # Data for the pie chart
    values = [row['Volume as Provider'], row['Volume as Subcontractor']]
    labels = ['As Provider', 'As Subcontractor']
    explode = (0.1, 0)  # Explode the first slice
    
    plt.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=['#1f77b4', '#ff7f0e'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(f"{row['tilastovuosi']}\nTotal: {row['Total Volume']:.1f}")

plt.suptitle(f"{institution_short_name} Roles Distribution by Year", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.4 Qualification Growth Analysis
# 
# Analyze qualification growth trends.

# %%
# Analyze qualification growth
qual_growth = analyzer.analyze_qualification_growth()
logger.info(f"Qualification growth analysis: {len(qual_growth)} rows")

# Display the first few rows
qual_growth.head(10)

# %% [markdown]
# Visualize the market share and growth:

# %%
# Filter for the most recent year
recent_year = max(qual_growth['tilastovuosi'])
recent_growth = qual_growth[qual_growth['tilastovuosi'] == recent_year]

# Only include qualifications with non-zero market share
recent_growth = recent_growth[recent_growth['Market Share'] > 0]

plt.figure(figsize=(14, 10))

# Create scatter plot of market share vs growth
plt.scatter(recent_growth['Market Share'], recent_growth['Growth'], 
            s=recent_growth['Total Market Volume'] * 2, # Size by market volume
            alpha=0.6, c=recent_growth['Growth'].apply(lambda x: 'green' if x > 0 else 'red'))

# Add qualification labels to points
for i, row in recent_growth.iterrows():
    plt.annotate(row['tutkinto'], 
                 (row['Market Share'], row['Growth']),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')

# Add reference lines
plt.axhline(y=0, color='grey', linestyle='--')
plt.axvline(x=recent_growth['Market Share'].median(), color='grey', linestyle='--')

# Add quadrant labels
median_share = recent_growth['Market Share'].median()
max_share = recent_growth['Market Share'].max() * 1.1
max_growth = recent_growth['Growth'].max() * 1.1
min_growth = recent_growth['Growth'].min() * 1.1

plt.text(median_share + (max_share-median_share)/2, max_growth*0.9, "Stars\n(High Share, High Growth)", ha='center')
plt.text(median_share/2, max_growth*0.9, "Question Marks\n(Low Share, High Growth)", ha='center')
plt.text(median_share + (max_share-median_share)/2, min_growth*0.9, "Cash Cows\n(High Share, Low Growth)", ha='center')
plt.text(median_share/2, min_growth*0.9, "Dogs\n(Low Share, Low Growth)", ha='center')

plt.title(f"Market Share vs. Growth for {institution_short_name} Qualifications ({recent_year})", fontsize=16)
plt.xlabel('Market Share (%)', fontsize=14)
plt.ylabel('Growth (%)', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.5 CAGR Analysis
# 
# Calculate Compound Annual Growth Rate for qualifications.

# %%
# Calculate CAGR
cagr_data = analyzer.calculate_qualification_cagr(
    start_year=start_year,
    end_year=end_year
)
logger.info(f"CAGR analysis: {len(cagr_data)} rows")

# Display the results
cagr_data

# %% [markdown]
# Visualize the CAGR for top qualifications:

# %%
# Sort by CAGR and filter for qualifications with at least 2 years of data
cagr_filtered = cagr_data[cagr_data['Years in market'] >= 2].sort_values('cagr', ascending=False)
top_growing = cagr_filtered.head(10)
bottom_growing = cagr_filtered.tail(10)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot top growing qualifications
ax1.barh(top_growing['tutkinto'], top_growing['cagr'] * 100, color='green')
ax1.set_title('Top 10 Fastest Growing Qualifications (CAGR)', fontsize=14)
ax1.set_xlabel('CAGR (%)', fontsize=12)
ax1.set_ylabel('Qualification', fontsize=12)
ax1.grid(axis='x', linestyle='--', alpha=0.7)

# Add CAGR values as text
for i, cagr in enumerate(top_growing['cagr']):
    ax1.text(cagr * 100 + 1, i, f"{cagr * 100:.1f}%", va='center')

# Plot bottom growing (declining) qualifications
ax2.barh(bottom_growing['tutkinto'], bottom_growing['cagr'] * 100, color='red')
ax2.set_title('Top 10 Fastest Declining Qualifications (CAGR)', fontsize=14)
ax2.set_xlabel('CAGR (%)', fontsize=12)
ax2.set_ylabel('Qualification', fontsize=12)
ax2.grid(axis='x', linestyle='--', alpha=0.7)

# Add CAGR values as text
for i, cagr in enumerate(bottom_growing['cagr']):
    ax2.text(cagr * 100 - 3, i, f"{cagr * 100:.1f}%", va='center', ha='right')

plt.suptitle(f"CAGR Analysis for {institution_short_name} Qualifications ({start_year}-{end_year})", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Market Share Analysis
# 
# Calculate and analyze market shares.

# %%
# Calculate market shares
market_shares = calculate_market_shares(
    df=df_filtered,
    institution_variants=institution_variants
)
logger.info(f"Market shares: {len(market_shares)} rows")

# Display the first few rows
market_shares.head(10)

# %% [markdown]
# Visualize market share for top qualifications:

# %%
# Filter for most recent year
recent_year = max(market_shares['tilastovuosi'])
recent_shares = market_shares[market_shares['tilastovuosi'] == recent_year]

# Get top 10 qualifications by volume
top_vol_quals = recent_shares.sort_values('Total Volume', ascending=False).head(10)['tutkinto'].unique()
top_quals_shares = recent_shares[recent_shares['tutkinto'].isin(top_vol_quals)]

# Pivot data for heatmap
heatmap_data = top_quals_shares.pivot(index='tutkinto', columns='kouluttaja', values='Volume')
heatmap_data = heatmap_data.fillna(0)

# Calculate percentage of total for each qualification
for qual in heatmap_data.index:
    total = heatmap_data.loc[qual].sum()
    if total > 0:
        heatmap_data.loc[qual] = heatmap_data.loc[qual] / total * 100

plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=.5)
plt.title(f"Market Share (%) by Provider for Top Qualifications ({recent_year})", fontsize=16)
plt.ylabel('Qualification', fontsize=14)
plt.xlabel('Provider', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Summary of Analysis
# 
# Let's summarize the key findings from our analysis.

# %%
# Get latest year
latest_year = max(all_years)

# Get total volume in latest year
latest_volume = total_volumes[total_volumes['tilastovuosi'] == latest_year]['Total Volume'].values[0]

# Get number of qualifications
num_quals = len(cagr_data)

# Get growing qualifications
growing_quals = cagr_data[cagr_data['cagr'] > 0]
num_growing = len(growing_quals)

# Get top qualification by volume
top_qual_row = volumes_by_qual[
    volumes_by_qual['tilastovuosi'] == latest_year
].sort_values('Total Volume', ascending=False).iloc[0]
top_qual = top_qual_row['tutkinto']
top_qual_volume = top_qual_row['Total Volume']

# Print summary
print(f"Summary for {institution_name} ({start_year}-{end_year})")
print(f"")
print(f"Total student volume (latest year): {latest_volume:.1f}")
print(f"Number of qualifications: {num_quals}")
print(f"Growing qualifications: {num_growing} ({num_growing/num_quals*100:.1f}%)")
print(f"Top qualification: {top_qual} ({top_qual_volume:.1f} students)")

# Provider role breakdown
latest_roles = institution_roles[institution_roles['tilastovuosi'] == latest_year].iloc[0]
provider_pct = latest_roles['Volume as Provider'] / latest_roles['Total Volume'] * 100
subcontractor_pct = latest_roles['Volume as Subcontractor'] / latest_roles['Total Volume'] * 100
print(f"")
print(f"Role breakdown (latest year):")
print(f"  As Provider: {latest_roles['Volume as Provider']:.1f} students ({provider_pct:.1f}%)")
print(f"  As Subcontractor: {latest_roles['Volume as Subcontractor']:.1f} students ({subcontractor_pct:.1f}%)")

# %% [markdown]
# ## 7. Conclusion
# 
# This notebook has demonstrated how to use the Vipunen library to analyze Finnish vocational education market data. We've covered:
# 
# 1. Loading and preparing data
# 2. Analyzing total volumes
# 3. Analyzing volumes by qualification
# 4. Analyzing institution roles
# 5. Calculating qualification growth
# 6. Calculating CAGR for qualifications
# 7. Visualizing the results
# 
# This analysis provides educational institutions with insights into their market position, helping them make informed decisions about which qualifications to focus on and where there are opportunities for growth.

# %% [markdown]
# ## 8. Export to Excel (Optional)
# 
# If you want to export the analysis results to Excel, similar to what the CLI does:

# %%
# Import FileUtils
try:
    from src.vipunen.utils.file_utils_config import get_file_utils
    from FileUtils import OutputFileType
    
    # Get the configured FileUtils instance
    file_utils = get_file_utils()
    
    # Prepare data for Excel export
    excel_data = {
        "Total Volumes": total_volumes,
        "Volumes by Qualification": volumes_by_qual,
        "Provider's Market": qual_growth,
        "CAGR Analysis": cagr_data,
        "Institution Roles": institution_roles,
        "Market Shares": market_shares
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
    
except ImportError:
    logger.warning("FileUtils not installed or properly configured. Excel export skipped.")
    logger.info("To install FileUtils: pip install FileUtils") 