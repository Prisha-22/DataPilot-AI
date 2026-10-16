import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.cleaning.quality import check_data_quality


df = pd.read_csv("data/raw/retail_store_inventory.csv")

quality_report = check_data_quality(df)

print("\n--- DATA QUALITY REPORT ---")

print("\nMissing Values:")
print(quality_report["missing_values"])

print("\nDuplicate Rows:")
print(quality_report["duplicate_rows"])

print("\nNegative Values:")
print(quality_report["negative_values"])

print("\nInvalid Values:")
print(quality_report["invalid_values"])