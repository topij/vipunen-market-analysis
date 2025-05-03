# Visualization Features

This document provides a brief overview of the visualization capabilities provided by the `EducationVisualizer` class in the Vipunen project.

## `EducationVisualizer` Class

The `src/vipunen/visualization/education_visualizer.py` module contains the `EducationVisualizer` class, which offers methods to create various standard plots based on the analysis results from `MarketAnalyzer` (see [Market Analysis Features](MARKET_ANALYSIS.md)).

The CLI script (`src/vipunen/cli/analyze_cli.py`) uses this class in its `generate_visualizations` function to automatically create a suite of plots.
It's designed to be reusable: plotting methods return Matplotlib Figure/Axes objects, which can then be displayed inline (e.g., in Jupyter) or saved to a file/PDF using the `save_visualization()` method.

## Key Visualization Methods in `EducationVisualizer`

| Method Name                       | Description                                                                          | Key Customizations/Notes                                                                                                |
| :-------------------------------- | :----------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------- |
| `create_area_chart`               | Stacked area chart showing total volume breakdown (e.g., Provider vs Subcontractor). | Accepts column names as args. Colors, labels configurable.                                                              |
| `create_line_chart`               | Line chart comparing market share evolution over time for different providers.       | Accepts column names as args. Generated for **all** active qualifications. Shows top 6 providers per qualification by default. |
| `create_heatmap`                  | Heatmap showing the target institution's market share across qualifications and years. | Accepts column names as args. Filtered to show only active qualifications for the institution.                      |
| `create_horizontal_bar_chart`   | Horizontal bar chart used for Provider Gainer/Loser plots.                           | Accepts column names as args. **Styling**: Labels appear to the right of bars; spines and horizontal grid removed; vertical line at 0 if needed.<br>**Filtering**: Gainers/Losers plot can be filtered via `config.yaml`.<br>**Captions**: Gainers/Losers caption indicates if filtering was applied. |
| `create_treemap`                  | Treemap visualizing market share vs. market size for the institution's qualifications. | Accepts column names as args. Static plot using Matplotlib/Squarify. Filtered to active qualifications.           |
| `create_volume_and_provider_count_plot` | Combined plot showing institution volume (left) and market provider counts (right). | Accepts column names as args. Provider counts based on market for qualifications offered by the institution. |
| `create_bcg_matrix`               | BCG Growth-Share Matrix plot showing qualification growth vs. relative market share. | Accepts column names as args. Uses data from `bcg_data` in analysis results. Bubble size = institution volume. **Note:** Interpret with care, see [BCG article](https://www.bcg.com/publications/2014/growth-share-matrix-bcg-classics-revisited). |

## Visualization Generation in `analyze_cli.py`