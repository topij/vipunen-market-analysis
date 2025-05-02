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
| `create_heatmap_with_marginals` | Combines the institution share heatmap with marginal plots for total market volume.    | Accepts column names as args. Filtered to show only active qualifications for the institution.                        |
| `create_horizontal_bar_chart`   | Horizontal bar chart used for Qualification Growth and Provider Gainer/Loser plots.  | Accepts column names as args. **Styling**: Labels appear to the right of bars; spines and horizontal grid removed; vertical line at 0 if needed.<br>**Filtering**: Gainers/Losers plot can be filtered via `config.yaml`.<br>**Captions**: Gainers/Losers caption indicates if filtering was applied. |
| `create_treemap`                  | Treemap visualizing market share vs. market size for the institution's qualifications. | Accepts column names as args. Static plot using Matplotlib/Squarify. Filtered to active qualifications.           |
| `create_volume_and_provider_count_plot` | Combined plot showing institution volume (left) and market provider counts (right). | Accepts column names as args. Provider counts based on market for qualifications offered by the institution. |
| `create_bcg_matrix`               | Bubble chart plotting Market Growth vs. Relative Market Share (BCG style).           | Uses latest year data. Bubble size represents institution volume. Quadrants defined by avg growth & share=1.0. Accepts column names. |

## Visualization Generation in `analyze_cli.py`

The `generate_visualizations` function in `analyze_cli.py` (see [CLI Guide](CLI_GUIDE.md)) uses the results from `MarketAnalyzer.analyze()` and calls the appropriate `EducationVisualizer` methods.

**Key Points:**

*   **Active Qualifications**: The function determines "active" qualifications based on criteria defined in the code (around lines 90-115) and configurable via `config.yaml` (`analysis.active_qualification_min_volume_sum`). The current logic requires the institution to have a summed `Total Volume` greater than a threshold (default: > 2) across the last two relevant years, AND the institution must not have had 100% market share in *both* of those years. This list is used to filter the Heatmaps and select which qualifications get individual Line and Bar charts.
*   **Caption Date**: Plot captions display the data update date extracted from the source file (column specified in `config.yaml` under `columns.input.update_date`, defaults to `tietojoukkoPaivitettyPvm`), providing context for the data's freshness.
*   **Gainers/Losers Filtering**: The Provider Gainer/Loser plot (Plot 6) can be filtered based on minimum market share and/or minimum market rank percentile using settings in `config.yaml` under `analysis.gainers_losers`. If filters are applied and remove providers, a note is added to the plot's caption (e.g., `Suodatus: Markkinaosuus < 0.5%.`).
*   **Qualification Scope**: Line charts (`create_line_chart`) and Gainer/Loser bar charts (`create_horizontal_bar_chart` for Plot 6) are generated for **all** identified active qualifications, providing a comprehensive view.
*   **Data Source**: All plots use the potentially filtered data returned by `MarketAnalyzer.analyze()`, ensuring consistency (e.g., `detailed_providers_market` used for many plots already has zero-volume rows removed).

## Output Formats

When run via the CLI, visualizations are typically saved into a multi-page PDF file in the institution-specific output directory (e.g., `data/reports/education_market_ri/ri_visualizations_[timestamp].pdf`).
If the `EducationVisualizer` is configured with `output_format='png'`, individual PNG files will be saved instead.

## Customization

Basic customization (titles, captions, colors) is handled within the `EducationVisualizer` methods. Further styling can be adjusted by modifying the methods in `education_visualizer.py` or potentially through matplotlib style sheets if configured. 