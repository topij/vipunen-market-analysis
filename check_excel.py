import pandas as pd

excel_file = "/Users/topi/data-science/repos/vipunen-project/data/reports/ri_market_analysis_20250426_145802.xlsx"

# Get the sheet names
sheet_names = pd.ExcelFile(excel_file).sheet_names
print("Sheet names:", sheet_names)

# Print basic info about each sheet
for sheet in sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet)
    print(f"\nSheet: {sheet}")
    print(f"Shape: {df.shape}")
    if not df.empty:
        print("Columns:", df.columns.tolist())
        print("First few rows:")
        print(df.head(2)) 