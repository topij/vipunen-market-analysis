# Visualization Features

This document provides a brief overview of the visualization capabilities in the Vipunen project. Note that visualization features are still being upgraded in the current version.

## Available Visualization Modules

The Vipunen project includes three main visualization modules:

1. **Volume Plots** (`src/vipunen/visualization/volume_plots.py`): Visualizations for student volume data
2. **Market Plots** (`src/vipunen/visualization/market_plots.py`): Visualizations for market share analysis
3. **Growth Plots** (`src/vipunen/visualization/growth_plots.py`): Visualizations for growth trends and time series

## Basic Usage

Visualization functions can be imported directly and used with analyzed data:

```python
from src.vipunen.visualization.volume_plots import plot_total_volumes
from src.vipunen.visualization.market_plots import plot_market_share_heatmap
from src.vipunen.visualization.growth_plots import plot_qualification_growth

# Plot total volumes
plot_total_volumes(
    volumes_df=total_volumes,
    institution_short_name="RI",
    output_dir="reports/figures"
)

# Plot market share heatmap
plot_market_share_heatmap(
    market_shares_df=market_shares,
    institution_name="Rastor-instituutti",
    output_dir="reports/figures"
)
```

## Key Visualization Functions

| Function | Module | Description |
|----------|--------|-------------|
| `plot_total_volumes` | volume_plots | Bar chart of total student volumes by year |
| `plot_top_qualifications` | volume_plots | Bar chart of volumes for top qualifications |
| `plot_market_share_heatmap` | market_plots | Heatmap of market shares across qualifications |
| `plot_qualification_market_shares` | market_plots | Bar chart of market shares for specific qualifications |
| `plot_qualification_growth` | growth_plots | Scatter plot of qualification growth rates |
| `plot_qualification_time_series` | growth_plots | Line chart of qualification volumes over time |

## Output Formats

Visualizations are saved as PNG files by default in the specified output directory. The filenames include:
- The institution name (or short name)
- The type of visualization
- A timestamp (optional)

## Customization

Note that visualization features in the current version have limited customization options. Future versions will include more extensive styling and customization capabilities. 