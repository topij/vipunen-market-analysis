# Vipunen API Data Fetching

This document describes how to obtain data using the Vipunen API.

## Obtaining Data from Vipunen API

Vipunen data can be accessed through the public REST/JSON API provided by the Finnish education administrations. This project contains fetching script which interacts directly with the Vipunen API to get the latest vocational education market data.

**Script:** `src/scripts/fetch_data.py`

**Functionality:**
- Connects to the Vipunen API.
- Downloads data for a specified dataset (defaults to `amm_opiskelijat_ja_tutkinnot_vuosi_tutkinto`).
- Saves the data into a CSV file in the configured raw data directory (default: `data/raw/`).
- **Update Check:** Before downloading, the script checks a metadata file (`.metadata/{dataset_name}_metadata.json`) for the last known update timestamp of the data on the API. If the timestamp hasn't changed, the download is skipped to save time and resources.
- **Backup:** If a new download occurs, the previous version of the CSV file is backed up into a subdirectory (default: `old_api_calls_output`).

**Configuration:**
API connection details (base URL, caller ID, retry settings, timeouts) and output formatting (CSV separator, encoding, backup directory name) are configured in the `api` section of `config/config.yaml`. When fetching data, it is good practice to identify your organization with the caller ID. You can define that in the config.yaml file with `caller_id: "organisaatio_oid.organisaationimi"`


**Usage:**
```bash
# Fetch default dataset specified in config
python src/scripts/fetch_data.py

# Fetch a specific dataset
python src/scripts/fetch_data.py --dataset other_dataset_name

# Specify output directory
python src/scripts/fetch_data.py --output-dir path/to/save

# Force download even if update date hasn't changed
python src/scripts/fetch_data.py --force-download
```

See more info about the Vipunen API in the Finnish education administrations site: [Vipunen API info pages (in Finnish)](https://vipunen.fi/fi-fi/Sivut/Vipunen-API.aspx). There you can also find the original Python code examples on using the API.

Refer to the [CLI Guide](CLI_GUIDE.md) for more script details. 