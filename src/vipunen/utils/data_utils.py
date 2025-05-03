import datetime
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def extract_data_update_date(df: pd.DataFrame, config: dict) -> str:
    """
    Extracts and formats the data update date from a DataFrame based on config.

    Args:
        df: The input DataFrame (expected to contain the date column).
        config: The project configuration dictionary.

    Returns:
        A string representing the formatted date (DD.MM.YYYY) or the
        current date as a fallback if extraction fails.
    """
    # Default value if extraction fails
    default_date_str = datetime.datetime.now().strftime("%d.%m.%Y")
    data_update_date_str = default_date_str

    try:
        input_cols = config.get('columns', {}).get('input', {})
        update_date_col = input_cols.get('update_date', 'tietojoukkoPaivitettyPvm') # Default column name

        if df is None or df.empty:
            logger.warning("Input DataFrame is empty or None. Cannot extract update date. Using current date.")
            return default_date_str # Return default if no data

        if update_date_col in df.columns:
            try:
                # Attempt to parse the date from the first row
                raw_date_str = str(df[update_date_col].iloc[0])

                # Handle potential NaT or empty strings before parsing
                if pd.notna(raw_date_str) and raw_date_str.strip():
                    parsed_date = pd.to_datetime(raw_date_str)
                    data_update_date_str = parsed_date.strftime("%d.%m.%Y")
                    logger.info(f"Extracted data update date: {data_update_date_str} from column '{update_date_col}'")
                else:
                    logger.warning(f"Update date value in column '{update_date_col}' is missing or empty. Using current date.")
                    data_update_date_str = default_date_str # Explicitly set default

            except Exception as date_err:
                logger.warning(f"Could not parse date from column '{update_date_col}' value '{raw_date_str}': {date_err}. Using current date.")
                data_update_date_str = default_date_str # Use default on parsing error
        else:
            logger.warning(f"Update date column '{update_date_col}' not found in DataFrame. Using current date.")
            data_update_date_str = default_date_str # Use default if column missing

    except Exception as e:
         logger.error(f"An unexpected error occurred during date extraction: {e}", exc_info=True)
         data_update_date_str = default_date_str # Ensure default is returned on any unexpected error

    return data_update_date_str

# Add other data utility functions here if needed 