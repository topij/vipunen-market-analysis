# Education Market Analysis

This project analyzes the Finnish education market from one provider's perspective, focusing specifically on Rastor-instituutti and the vocational qualifications they offer.

## Project Structure

- `src/vipunen/analysis/education_market.py`: Core module containing the `EducationMarketAnalyzer` class
- `src/scripts/education_market_analysis.py`: Entry point script to run the analysis
- `data/raw/amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto.csv`: Raw data file
- `output/`: Directory where results are saved

## Analysis Steps

The analysis follows these steps:

1. Calculate the total volume of students for Rastor-instituutti
2. Break down how much volume is done as "koulutuksenJarjestaja" (main provider) vs "hankintakoulutuksenJarjestaja" (subcontractor)

Future steps will include:
3. Analyzing volumes by qualification
4. Year-over-Year changes in volumes
5. Market share analysis over time
6. Ranking and market share analysis for all institutions
7. Year-over-Year changes in market shares

## Running the Analysis

### Prerequisites

Make sure you have all dependencies installed:

```bash
# Using conda
conda env create -f environment.yaml
conda activate vipunen

# Or using pip
pip install -r requirements.txt
```

### Running the Script

To run the analysis, execute:

```bash
python src/scripts/education_market_analysis.py
```

This will:
1. Load the raw data
2. Filter for relevant degree types
3. Calculate volume metrics for Rastor-instituutti
4. Save the results to `output/ri_volumes_summary.csv`
5. Generate a visualization at `output/ri_volumes_by_role.png`

## Results

The script produces a summary table with the following columns:

- `tilastovuosi`: Year
- `kouluttaja`: Institution name (always "RI" for Rastor-instituutti)
- `järjestäjänä`: Student volume as main provider
- `hankintana`: Student volume as subcontractor
- `Yhteensä`: Total student volume
- `järjestäjä_osuus (%)`: Percentage of volume as main provider

Example output:

```
tilastovuosi  kouluttaja  järjestäjänä  hankintana  Yhteensä  järjestäjä_osuus (%)
       2018         RI      2053.09     1787.36   3840.45                 53.46
       2019         RI      1939.84     1338.59   3278.44                 59.17
       2020         RI      1770.85     1234.77   3005.61                 58.92
       2021         RI      2192.98      874.52   3067.49                 71.49
```

## Notes

- The data focuses only on "ammattitutkinto" and "erikoisammattitutkinto" degrees.
- The institution appears under two names: "Rastor-instituutti ry" and "Rastor Oy", which are combined as "RI" in the output. 