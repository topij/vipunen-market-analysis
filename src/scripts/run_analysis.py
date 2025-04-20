#!/usr/bin/env python
"""Script to run the Vipunen analysis."""
from pathlib import Path
import pandas as pd
from datetime import datetime
from vipunen.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RI_INSTITUTIONS,
    BRAND_COLORS
)
from vipunen.utils.file_utils import load_data, save_data
from vipunen.analysis.growth import calculate_cagr, calculate_yoy_growth
from vipunen.analysis.market import analyze_market_trends
from vipunen.analysis.visualizer import (
    plot_market_shares,
    plot_growth_trends,
    plot_provider_counts
)

def main():
    """Run the Vipunen analysis."""
    # Create output directories
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DATA_DIR / "plots").mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_data("amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv", RAW_DATA_DIR)
    
    # Calculate market trends
    print("Calculating market trends...")
    market_analysis = analyze_market_trends(
        df,
        groupby_cols=['tilastovuosi', 'tutkinto'],
        value_col='nettoopiskelijamaaraLkm_as_jarjestaja'
    )
    
    # Save market analysis results
    for name, data in market_analysis.items():
        save_data(
            data,
            f"market_{name}.csv",
            PROCESSED_DATA_DIR
        )
    
    # Calculate growth metrics
    print("Calculating growth metrics...")
    cagr_df = calculate_cagr(
        df,
        groupby_columns=['tutkinto'],
        value_column='nettoopiskelijamaaraLkm_as_jarjestaja'
    )
    
    yoy_df = calculate_yoy_growth(
        df,
        groupby_col='tutkinto',
        target_col='nettoopiskelijamaaraLkm_as_jarjestaja',
        output_col='opiskelijoiden_maaran_kasvu',
        end_year=2022,
        time_window=3
    )
    
    # Save growth metrics
    save_data(cagr_df, "cagr.csv", PROCESSED_DATA_DIR)
    save_data(yoy_df, "yoy_growth.csv", PROCESSED_DATA_DIR)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Market shares plot
    market_shares_plot = plot_market_shares(
        market_analysis['market_shares'],
        x_col='tutkinto',
        y_col='market_share',
        hue_col='koulutuksenJarjestaja',
        title="Market Shares by Qualification",
        x_label="Qualification",
        y_label="Market Share (%)",
        save_path=PROCESSED_DATA_DIR / "plots/market_shares.png"
    )
    
    # Growth trends plot
    growth_plot = plot_growth_trends(
        yoy_df,
        x_col='tutkinto',
        y_col='opiskelijoiden_maaran_kasvu',
        trend_col='opiskelijoiden_maaran_kasvu_trendi',
        title="Growth Trends by Qualification",
        x_label="Qualification",
        y_label="Growth (%)",
        save_path=PROCESSED_DATA_DIR / "plots/growth_trends.png"
    )
    
    # Provider counts plot
    provider_plot = plot_provider_counts(
        market_analysis['provider_counts'],
        x_col='tilastovuosi',
        y_col='provider_count',
        title="Number of Providers Over Time",
        x_label="Year",
        y_label="Number of Providers",
        save_path=PROCESSED_DATA_DIR / "plots/provider_counts.png"
    )
    
    print("Analysis complete!")
    print(f"Results saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main() 